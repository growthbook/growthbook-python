#!/usr/bin/env python
import inspect
import json
from dataclasses import dataclass, field
import random
import logging
from types import TracebackType
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple, Union, Callable, Awaitable, cast
from typing import Set

if TYPE_CHECKING:
    from .plugins.base import PluginLike
import asyncio
import threading
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from growthbook import FeatureRepository, feature_repo
from contextlib import asynccontextmanager

from .core import eval_feature as core_eval_feature, run_experiment
from .common_types import (
    T,
    AsyncEventLogger,
    Feature,
    GlobalContext,
    Options,
    Result,
    UserContext,
    EvaluationContext,
    StackContext,
    FeatureResult,
    FeatureRefreshStrategy,
    AbstractAsyncStickyBucketService,
    Experiment,
    build_remote_eval_payload,
    features_from_dict,
    tracking_user_context,
    validate_remote_eval_options,
)

logger = logging.getLogger("growthbook.growthbook_client")

class SingletonMeta(type):
    """One instance per (class, api_host, client_key). Two GrowthBookClients
    talking to different proxies — or one CDN-mode and one remote-eval-mode
    client in the same process — used to silently share a single repo with the
    first caller's config: wrong host, wrong client_key, wrong `_remote_eval`
    flag (so SSE invalidation took the CDN path on the remote-eval client).

    Keying on the constructor's identity args fixes that. The first two
    positional args of `EnhancedFeatureRepository.__init__` are `(api_host,
    client_key)`; we extract them here and fall back to kwargs for safety."""
    _instances: Dict[Any, Any] = {}
    _lock = threading.Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        api_host = args[0] if len(args) > 0 else kwargs.get("api_host", "")
        client_key = args[1] if len(args) > 1 else kwargs.get("client_key", "")
        key = (cls, api_host, client_key)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = super().__call__(*args, **kwargs)
        return cls._instances[key]

class BackoffStrategy:
    """Exponential backoff with jitter for failed requests"""
    def __init__(
        self, 
        initial_delay: float = 1.0, 
        max_delay: float = 60.0, 
        multiplier: float = 2.0,
        jitter: float = 0.1
    ) -> None:
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.current_delay = initial_delay
        self.attempt = 0

    def next_delay(self) -> float:
        """Calculate next delay with jitter"""
        delay = min(
            self.current_delay * (self.multiplier ** self.attempt), 
            self.max_delay
        )
        # Add random jitter
        jitter_amount = delay * self.jitter
        delay = delay + (random.random() * 2 - 1) * jitter_amount
        self.attempt += 1
        return max(delay, self.initial_delay)

    def reset(self) -> None:
        """Reset backoff state"""
        self.current_delay = self.initial_delay
        self.attempt = 0

class WeakRefWrapper:
    """A wrapper class to allow weak references for otherwise non-weak-referenceable objects."""
    def __init__(self, obj: Any) -> None:
        self.obj = obj

class FeatureCache:
    """Thread-safe feature cache"""
    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {
            'features': {},
            'savedGroups': {},
            'contextualBandits': {}
        }
        self._lock = threading.Lock()

    def update(self, features: Optional[Dict[str, Any]],
               saved_groups: Optional[Dict[str, Any]] = None,
               contextual_bandits: Optional[Dict[str, Any]] = None) -> None:
        """Thread-safe update of cache with new API data.

        A section passed as None (the payload omitted that key) keeps its
        current value — a partial or broken refresh must not wipe state that
        evaluations depend on (mirrors JS setPayload). Pass an explicit empty
        dict to clear a section."""
        with self._lock:
            if features is not None:
                self._cache['features'] = dict(features)
            if saved_groups is not None:
                self._cache['savedGroups'] = dict(saved_groups)
            if contextual_bandits is not None:
                self._cache['contextualBandits'] = dict(contextual_bandits)

    def get_current_state(self) -> Dict[str, Any]:
        """Get current cache state"""
        with self._lock:
            return {
                "features": dict(self._cache['features']),
                "savedGroups": self._cache['savedGroups'],
                "contextualBandits": self._cache['contextualBandits']
            }

class EnhancedFeatureRepository(FeatureRepository, metaclass=SingletonMeta):
    def __init__(self,
                 api_host: str,
                 client_key: str,
                 decryption_key: str = "",
                 cache_ttl: int = 60,
                 http_connect_timeout: Optional[int] = None,
                 http_read_timeout: Optional[int] = None,
                 remote_eval_cache_size: int = 1000,
                 remote_eval: bool = False) -> None:
        FeatureRepository.__init__(self)
        self._api_host = api_host
        self._client_key = client_key
        self._decryption_key = decryption_key
        self._cache_ttl = cache_ttl
        # Whether this repo serves a remote-eval client. Drives SSE handling:
        # remote-eval invalidation is a cache flush, NOT a CDN re-fetch.
        # Stored as an explicit flag rather than inferred from cache contents
        # because `_remote_eval_cache` is empty until the first user is
        # evaluated — inferring from emptiness would silently route SSE events
        # down the CDN path on a fresh client.
        self._remote_eval = remote_eval
        self._refresh_lock = threading.Lock()
        self._refresh_task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self._backoff = BackoffStrategy()
        self._feature_cache = FeatureCache()
        self._callbacks: List[Callable[[Dict[str, Any]], Awaitable[None]]] = []
        self._last_successful_refresh: Optional[datetime] = None
        self._refresh_in_progress = asyncio.Lock()
        self.http_connect_timeout = http_connect_timeout
        self.http_read_timeout = http_read_timeout

        # Remote-eval per-user response cache (LRU-bounded, distinct from
        # the GET-path self.cache to keep eviction strategies clean).
        # Entries are (response, stale_at) where stale_at = time.monotonic() +
        # effective_stale_ttl at write time. The max_age boundary is derived
        # JS-style: an entry is valid iff `stale_at > now - cache_ttl +
        # effective_stale_ttl` (algebraically: write_time > now - cache_ttl).
        # Storing a single timestamp mirrors `packages/sdk-js/src/feature-repository.ts`.
        #
        # No lock guards this dict: the critical sections in fetch_remote_eval
        # are pure dict ops with no `await` between them, and asyncio
        # coroutines only yield at await points — so the operations are
        # already atomic from asyncio's perspective.
        self._remote_eval_cache: "OrderedDict[str, tuple[Dict[str, Any], float]]" = OrderedDict()
        self._remote_eval_cache_max = remote_eval_cache_size
        # Inflight coalescing AND SWR-dedup: a key in this dict means a POST
        # for that cache key is already in flight, so concurrent foreground
        # callers and background-SWR refresh tasks both observe and skip.
        self._remote_eval_inflight: Dict[str, "asyncio.Future[Optional[Dict[str, Any]]]"] = {}
        # Fire-and-forget SWR refresh tasks. Tracked so `stop_refresh()` can
        # cancel them on shutdown — otherwise they may try to write to a
        # closed aiohttp session / event loop and emit
        # "Task was destroyed but it is pending" / "Event loop is closed".
        self._swr_tasks: Set["asyncio.Task[Any]"] = set()

    async def fetch_remote_eval(
        self,
        api_host: str,
        client_key: str,
        payload: Dict[str, Any],
        cache_key_attributes: Optional[List[str]] = None,
        cache_ttl: int = 60,
        stale_ttl: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """POST to /api/eval/{client_key} with JS-SDK-style cache lifecycle.

        Age windows for the per-payload cache (matches
        `packages/sdk-js/src/feature-repository.ts`):
            age < stale_ttl              → serve cached, no refresh
            stale_ttl <= age < cache_ttl → serve cached + fire-and-forget bg refresh
            age >= cache_ttl             → cache miss, await network

        With `stale_ttl=None` the SWR window collapses — entries are served
        until cache_ttl and then refetched synchronously.

        Concurrent foreground callers AND background SWR refreshes coalesce
        on a single inflight POST per cache key.
        """
        cache_key = self._compute_cache_key(api_host, client_key, True, payload, cache_key_attributes)
        effective_stale_ttl = stale_ttl if stale_ttl is not None else cache_ttl
        now = time.monotonic()

        cached = self._remote_eval_cache.get(cache_key)
        if cached is not None:
            response, stale_at = cached
            # JS-style max_age check: write_time > now - cache_ttl, i.e.
            # stale_at > now - cache_ttl + effective_stale_ttl.
            if stale_at > now - cache_ttl + effective_stale_ttl:
                self._remote_eval_cache.move_to_end(cache_key)
                # SWR: past stale_ttl, under cache_ttl — schedule a
                # background refetch unless one's already in flight.
                if (
                    stale_ttl is not None
                    and stale_at <= now
                    and cache_key not in self._remote_eval_inflight
                ):
                    task = asyncio.create_task(self._post_and_cache(
                        api_host, client_key, payload, cache_key, effective_stale_ttl
                    ))
                    # Track so stop_refresh() can cancel pending SWR work
                    # before the loop / aiohttp session closes.
                    self._swr_tasks.add(task)
                    task.add_done_callback(self._swr_tasks.discard)
                return response
            # Past max_age — evict and fall through to a synchronous refetch.
            self._remote_eval_cache.pop(cache_key, None)

        # Foreground POST (or join an existing inflight one).
        existing = self._remote_eval_inflight.get(cache_key)
        if existing is not None:
            return await existing
        return await self._post_and_cache(
            api_host, client_key, payload, cache_key, effective_stale_ttl
        )

    async def _post_and_cache(
        self,
        api_host: str,
        client_key: str,
        payload: Dict[str, Any],
        cache_key: str,
        effective_stale_ttl: int,
    ) -> Optional[Dict[str, Any]]:
        """POST + cache-write helper shared by foreground misses and background
        SWR refreshes. Registers `cache_key` in the inflight map for the POST's
        duration so duplicate POSTs are coalesced. Bumps `stale_at` on every
        successful write — even an unchanged payload refreshes freshness
        (matches JS SDK)."""
        # Re-check inflight: a concurrent caller might have raced us.
        existing = self._remote_eval_inflight.get(cache_key)
        if existing is not None:
            return await existing

        inflight: "asyncio.Future[Optional[Dict[str, Any]]]" = asyncio.Future()
        self._remote_eval_inflight[cache_key] = inflight
        try:
            response = await self._fetch_and_decode_post_async(api_host, client_key, payload)
        except BaseException as e:
            # `except Exception` would miss `asyncio.CancelledError` (Python 3.8+
            # derives it from BaseException). Without cleanup on cancellation
            # the inflight map leaks the cache_key forever and any waiters
            # blocked on `await existing` hang indefinitely.
            self._remote_eval_inflight.pop(cache_key, None)
            if not inflight.done():
                if isinstance(e, asyncio.CancelledError):
                    # Cancel the inflight Future instead of `set_exception(e)`.
                    # Both propagate CancelledError to any concurrent waiters,
                    # but a cancelled Future never triggers the
                    # "Future exception was never retrieved" warning when
                    # garbage-collected without an observer — `set_exception`
                    # does, producing stderr noise on every cancellation.
                    inflight.cancel()
                else:
                    inflight.set_exception(e)
            raise

        # If `flush_remote_eval_cache()` ran while we were awaiting the POST,
        # the cache_key is no longer in `_remote_eval_inflight` — the response
        # we just got is from BEFORE the invalidation signal, so writing it to
        # the cache would silently re-populate stale data the proxy just told
        # us to drop. Still resolve waiters with the response (they observed
        # the pre-flush request and have a right to its answer), but skip the
        # cache write so future eval calls trigger a fresh POST.
        was_flushed = self._remote_eval_inflight.pop(cache_key, None) is None
        if response is not None and not was_flushed:
            stale_at = time.monotonic() + effective_stale_ttl
            self._remote_eval_cache[cache_key] = (response, stale_at)
            self._remote_eval_cache.move_to_end(cache_key)
            # `len > 0` guard prevents popitem() raising KeyError when an
            # operator misconfigures `remote_eval_cache_size` to a negative
            # value (the while-loop would otherwise keep trying to evict
            # past the already-empty dict).
            while self._remote_eval_cache and len(self._remote_eval_cache) > self._remote_eval_cache_max:
                self._remote_eval_cache.popitem(last=False)
        if not inflight.done():
            inflight.set_result(response)
        return response

    def flush_remote_eval_cache(self) -> None:
        """Drop all cached remote-eval responses. Called when the proxy
        emits a features-updated SSE event.

        Also clears the inflight map so any POSTs in flight at this moment
        know (via the `was_flushed` check in `_post_and_cache`) that their
        result predates the proxy's invalidation signal and must NOT be
        written back to the cache."""
        self._remote_eval_cache.clear()
        self._remote_eval_inflight.clear()

    @asynccontextmanager
    async def refresh_operation(self) -> AsyncIterator[bool]:
        """Context manager for feature refresh with proper cleanup"""
        if self._refresh_in_progress.locked():
            yield False
            return

        # async with self._refresh_in_progress:
        try:
            await self._refresh_in_progress.acquire()
            yield True
            self._backoff.reset()
            self._last_successful_refresh = datetime.now()
        except Exception as e:
            delay = self._backoff.next_delay()
            logger.error(f"Refresh failed, next attempt in {delay:.2f}s: {str(e)}")
            traceback.print_exc()
            raise
        finally:
            if self._refresh_in_progress.locked():
                self._refresh_in_progress.release()

    async def _handle_feature_update(self, data: Dict[str, Any]) -> None:
        """Update features with memory optimization"""
        # Directly update with new features
        # Sections absent from the payload are passed as None so the cache
        # preserves their current values (see FeatureCache.update).
        self._feature_cache.update(
            data.get("features"),
            data.get("savedGroups"),
            data.get("contextualBandits")
        )

        # Create a copy of callbacks to avoid modification during iteration
        with self._refresh_lock:
            callbacks = self._callbacks.copy()

        for callback in callbacks:
            try:
                await callback(dict(self._feature_cache.get_current_state()))
            except Exception:
                traceback.print_exc()

    def add_callback(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """Add callback to the list"""
        with self._refresh_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """Remove callback from the list"""
        with self._refresh_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    """
    _start_sse_refresh flow mimics a bridge pattern to connect a blocking, synchronous background thread 
    (the SSEClient) with your non-blocking, async main loop.

    Bridge - _maintain_sse_connection - runs on the main async loop, calls `startAutoRefresh` (which in turn spawns a thread)
    and waits indefinitely. (Awaiting a Future suspends the coroutine, costing zero CPU)

    The SSEClient runs in a separate thread, makes a blocking HTTP request, and invokes `on_event` synchronously.

    The Hand off - when the event arrives (we're still on the background thread), sse_handler uses `asyncio.run_coroutine_threadsafe` 
    to schedule the async processing `_handle_sse_event` onto the main event loop.
    """

    async def _handle_sse_event(self, event_data: Dict[str, Any]) -> None:
        """Process an event received from the SSE connection"""
        try:
            event_type = event_data.get("type")
            if event_type == "features-updated":
                # In remote-eval mode the proxy emits this event without a
                # payload; the right response is to flush our per-user cache so
                # subsequent evals re-POST lazily. We do NOT proactively re-POST
                # for every cached user (could be millions in a busy service).
                if self._remote_eval:
                    self.flush_remote_eval_cache()
                    return
                response = await self.load_features_async(
                    self._api_host, self._client_key, self._decryption_key, self._cache_ttl
                )
                if response is not None:
                    await self._handle_feature_update(response)
            elif event_type == "features":
                # Remote-eval mode shouldn't receive inline payloads (they
                # wouldn't be user-filtered), but if one arrives defensively
                # flush the per-user cache instead of caching a bogus payload.
                if self._remote_eval:
                    self.flush_remote_eval_cache()
                    return

                data = event_data.get("data", "{}")
                if isinstance(data, str):
                    data = json.loads(data)

                if self._decryption_key and isinstance(data, dict) and (
                    "encryptedFeatures" in data
                    or "encryptedSavedGroups" in data
                    or "encryptedContextualBandits" in data
                ):
                    logger.debug("Decrypting SSE payload...")
                    data = self.decrypt_response(data, self._decryption_key)
                    if data is None:
                        logger.warning("Failed to decrypt SSE payload, skipping update")
                        return
                    logger.debug(f"🟢 Decrypted. Features keys: {list(data.get('features', {}).keys())}")

                await self._handle_feature_update(data)
        except Exception:
            logger.exception("Error handling SSE event")

    async def _start_sse_refresh(self) -> None:
        """Start SSE-based feature refresh"""
        with self._refresh_lock:
            if self._refresh_task is not None:  # Already running
                return

            # SSEClient invokes `on_event` synchronously from a background thread.
            main_loop = asyncio.get_running_loop()

            # We must not pass an `async def` callback here (it would never be awaited).
            def sse_handler(event_data: Dict[str, Any]) -> None:
                # Schedule async processing onto the main event loop.
                try:
                    asyncio.run_coroutine_threadsafe(self._handle_sse_event(event_data), main_loop)
                except Exception:
                    logger.exception("Failed to schedule SSE event handler")

            async def _maintain_sse_connection() -> None:
                """
                Start SSE streaming and keep the task alive until cancelled.
                """
                try:
                    # NOTE: `startAutoRefresh` is synchronous and starts a background thread.
                    self.startAutoRefresh(self._api_host, self._client_key, sse_handler)
                    
                    # Wait indefinitely until the task is cancelled - basically saying "Keep this service 'active' until someone cancels me."
                    # reconnection logic is handled inside SSEClient's thread
                    await asyncio.Future()
                except asyncio.CancelledError:
                    # Normal shutdown flow
                    raise
                except Exception:
                    logger.exception("Unexpected error in SSE lifecycle task")
                finally:
                    try:
                        # stopAutoRefresh blocks joining a thread, so it needs to be run in executor
                        # to avoid blocking the async event loop
                        await main_loop.run_in_executor(
                            None, 
                            lambda: self.stopAutoRefresh(timeout=10)
                        )
                    except Exception:
                        logger.exception("Failed to stop SSE auto-refresh")

            # Start a task that owns the SSE lifecycle and cleanup.
            self._refresh_task = asyncio.create_task(_maintain_sse_connection())

    async def _start_http_refresh(self, interval: int = 60) -> None:
        """Enhanced HTTP polling with backoff"""
        if self._refresh_task:
            return

        async def refresh_loop() -> None:
            try:
                while not self._stop_event.is_set():
                    async with self.refresh_operation() as should_refresh:
                        if should_refresh:
                            try:
                                response = await self.load_features_async(
                                    api_host=self._api_host,
                                    client_key=self._client_key,
                                    decryption_key=self._decryption_key,
                                    ttl=self._cache_ttl
                                )
                                if response is not None:
                                    await self._handle_feature_update(response)
                                # On success, reset backoff and use normal interval
                                self._backoff.reset()
                                try:
                                    await asyncio.sleep(interval)
                                except asyncio.CancelledError:
                                    # Allow cancellation during sleep
                                    raise
                            except Exception as e:
                                # On failure, use backoff delay
                                delay = self._backoff.next_delay()
                                logger.error(f"Refresh failed, next attempt in {delay:.2f}s: {str(e)}")
                                traceback.print_exc()
                                try:
                                    await asyncio.sleep(delay)
                                except asyncio.CancelledError:
                                    # Allow cancellation during sleep
                                    raise
            except asyncio.CancelledError:
                # Clean exit on cancellation
                raise
            finally:
                # Ensure we're marked as stopped
                self._stop_event.set()

        self._refresh_task = asyncio.create_task(refresh_loop())

    async def start_feature_refresh(self, strategy: FeatureRefreshStrategy, callback: Optional[Callable[..., Any]] = None) -> None:
        """Initialize feature refresh based on strategy"""
        self._refresh_callback = callback
        
        if strategy == FeatureRefreshStrategy.SERVER_SENT_EVENTS:
            await self._start_sse_refresh()
        else:
            await self._start_http_refresh()

    async def stop_refresh(self) -> None:
        """Clean shutdown of refresh tasks"""
        self._stop_event.set()
        # Ensure any SSE background thread is stopped as well.
        # stopAutoRefresh blocks joining a thread (up to `timeout` seconds),
        # so run it in the executor to keep the event loop free — same as the
        # SSE lifecycle teardown above.
        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.stopAutoRefresh(timeout=10)
            )
        except Exception:
            # Best-effort cleanup; task cancellation below will proceed.
            logger.exception("Error stopping SSE auto-refresh")
        # Cancel any pending SWR (stale-while-revalidate) background refresh
        # tasks before the event loop / aiohttp session closes.
        for task in list(self._swr_tasks):
            task.cancel()
        if self._swr_tasks:
            try:
                await asyncio.gather(*self._swr_tasks, return_exceptions=True)
            except Exception:
                logger.exception("Error draining SWR background tasks")
            self._swr_tasks.clear()
        if self._refresh_task:
            # Cancel the task
            self._refresh_task.cancel()
            try:
                # Wait for it to actually finish
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error during refresh task cleanup: {e}")
            finally:
                self._refresh_task = None
                self._backoff.reset()
        self._stop_event.clear()

    async def __aenter__(self) -> "EnhancedFeatureRepository":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.stop_refresh()
    
    async def load_features_async(
        self,
        api_host: str,
        client_key: str,
        decryption_key: str = "",
        ttl: int = 60,
        remote_eval: bool = False,
        payload: Optional[Dict[str, Any]] = None,
        cache_key_attributes: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> Optional[Dict[str, Any]]:
        # Use stored values when called internally
        if api_host == self._api_host and client_key == self._client_key:
            decryption_key = self._decryption_key
            ttl = self._cache_ttl
        return await super().load_features_async(
            api_host, client_key, decryption_key, ttl,
            remote_eval=remote_eval, payload=payload,
            cache_key_attributes=cache_key_attributes,
            force_refresh=force_refresh,
        )

class GrowthBookClient:
    def __init__(
        self,
        options: Optional[Union[Dict[str, Any], Options]] = None
    ) -> None:
        self.options = (
            options if isinstance(options, Options)
            else Options(**options) if options
            else Options()
        )

        if self.options.remote_eval:
            validate_remote_eval_options(
                self.options.client_key,
                self.options.decryption_key,
                self.options.sticky_bucket_service,
                self.options.api_host,
            )
            if self.options.refresh_strategy == FeatureRefreshStrategy.STALE_WHILE_REVALIDATE:
                # HTTP polling has no per-user payload to send, so it would
                # silently no-op. Sync GrowthBook raises on the equivalent
                # stale_while_revalidate=True; keep the two clients consistent.
                raise ValueError(
                    "refresh_strategy=STALE_WHILE_REVALIDATE is not compatible with remote_eval; "
                    "use SERVER_SENT_EVENTS or pass refresh_strategy=None"
                )
        
        # Thread-safe tracking state
        self._tracked: Dict[str, bool] = {}  # Access only within async context
        self._tracked_lock = threading.Lock()
        
        # Thread-safe subscription management
        self._subscriptions: Set[Callable[[Experiment[Any], Result[Any]], Union[None, Awaitable[None]]]] = set()
        self._subscriptions_lock = threading.Lock()

        # Per-attributes-key inflight sticky bucket fetches. Concurrent evals
        # with identical attributes coalesce onto one service fetch; distinct
        # attributes fetch in parallel. No cross-eval result cache by default
        # — assignments are fetched per evaluation context, matching the JS
        # SDK's server-side GrowthBookClient.applyStickyBuckets.
        self._sticky_bucket_inflight: Dict[str, "asyncio.Future[Dict[str, Any]]"] = {}
        # Opt-in TTL cache (Options.sticky_bucket_cache_ttl > 0): trades
        # bounded cross-worker staleness for fewer service round-trips.
        # Non-positive ttl or size disables caching entirely.
        self._sticky_cache_enabled = (
            (self.options.sticky_bucket_cache_ttl or 0) > 0
            and (self.options.sticky_bucket_cache_size or 0) > 0
        )
        self._sticky_bucket_cache: "OrderedDict[str, Any]" = OrderedDict()
        # Authoritative map of every sticky assignment doc THIS process has
        # written: doc key ("attributeName||attributeValue") -> doc. This is
        # the merge base for saves and is overlaid onto every fetched
        # snapshot, so an assignment made under one attributes snapshot is
        # never lost when a different snapshot for the same identifier
        # generates the next save (two snapshots for one id are otherwise
        # disjoint dicts, and unordered last-write-wins would drop data).
        # LRU-bounded; only clean (saved, not dirty/in-flight) entries evict.
        self._sticky_docs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        # Per-doc-key save pipeline: at most ONE in-flight save per key, with
        # a dirty flag that triggers a trailing save of the latest merged doc.
        # Serializing per key is what makes persistence converge — even
        # superset docs would regress if an older write landed last.
        self._sticky_save_inflight: Dict[str, "asyncio.Future[Any]"] = {}
        self._sticky_save_dirty: Set[str] = set()
        # Strong refs to scheduled async user callbacks (tracking, feature
        # usage, subscriptions). Drained in close().
        self._callback_tasks: Set["asyncio.Future[Any]"] = set()
        
        # Plugin support
        self._tracking_plugins: List["PluginLike"] = self.options.tracking_plugins or []
        self._initialized_plugins: List["PluginLike"] = []

        self._features_repository = (
            EnhancedFeatureRepository(
                self.options.api_host or "https://cdn.growthbook.io",
                self.options.client_key or "",
                self.options.decryption_key or "",
                self.options.cache_ttl,
                self.options.http_connect_timeout,
                self.options.http_read_timeout,
                self.options.remote_eval_cache_size,
                self.options.remote_eval,
            )
            if self.options.client_key
            else None
        )
        
        self._global_context: Optional[GlobalContext] = None
        self._context_lock = asyncio.Lock()
        
        # Initialize plugins
        self._initialize_plugins()


    def _spawn_tracked(self, fut: "asyncio.Future[Any]", task_set: Set["asyncio.Future[Any]"], error_msg: str) -> None:
        """Keep a strong ref to a fire-and-forget future until it completes;
        log (never raise) its exception."""
        task_set.add(fut)

        def _done(f: "asyncio.Future[Any]") -> None:
            task_set.discard(f)
            if not f.cancelled() and f.exception():
                logger.error(error_msg, exc_info=f.exception())

        fut.add_done_callback(_done)

    def _run_user_callback(self, callback: Callable[..., Any], args: Tuple[Any, ...], what: str,
                           on_error: Optional[Callable[[], None]] = None,
                           kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Invoke a user callback that may be sync or async.

        Called from synchronous eval paths, so a returned awaitable cannot be
        awaited here; it is scheduled fire-and-forget on the running loop
        (drained in close()). Sync exceptions propagate to the caller's
        existing try/except. `on_error` fires if the SCHEDULED coroutine
        fails or cannot be scheduled — sync failures don't need it because
        they propagate."""
        result = callback(*args, **(kwargs or {}))
        if inspect.isawaitable(result):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                if asyncio.iscoroutine(result):
                    result.close()
                logger.error("Async %s callback requires a running event loop; dropped", what)
                if on_error:
                    on_error()
                return
            fut = asyncio.ensure_future(result)
            if on_error is not None:
                def _fire_on_error(f: "asyncio.Future[Any]") -> None:
                    if not f.cancelled() and f.exception():
                        on_error()
                fut.add_done_callback(_fire_on_error)
            self._spawn_tracked(fut, self._callback_tasks, f"Error in {what} callback")

    def _track(self, experiment: Experiment[Any], result: Result[Any], user_context: UserContext) -> None:
        """Thread-safe tracking implementation"""
        if not self.options.on_experiment_viewed:
            return

        # Create unique key for this tracking event
        key = (
            result.hashAttribute
            + str(result.hashValue)
            + experiment.key
            + str(result.variationId)
        )

        with self._tracked_lock:
            if not self._tracked.get(key):
                try:
                    self._run_user_callback(
                        self.options.on_experiment_viewed,
                        (),
                        "tracking",
                        # An async tracking callback is deduped at schedule
                        # time; if it later fails, un-mark so the impression
                        # is retried on the next eval — same retry semantics
                        # as a sync callback that raises.
                        on_error=lambda: self._untrack(key),
                        # Tracking callbacks are invoked by keyword (same
                        # contract as the sync client): implementations must
                        # name their params experiment/result/user_context.
                        # user_context is snapshotted so the logged attributes
                        # are exactly the ones used for bucketing, even if the
                        # caller mutates them afterwards.
                        kwargs={
                            "experiment": experiment,
                            "result": result,
                            "user_context": tracking_user_context(user_context),
                        },
                    )
                    self._tracked[key] = True
                except Exception:
                    logger.exception("Error in tracking callback")

    def _untrack(self, key: str) -> None:
        with self._tracked_lock:
            self._tracked.pop(key, None)

    def subscribe(self, callback: Callable[[Experiment[Any], Result[Any]], Union[None, Awaitable[None]]]) -> Callable[[], None]:
        """Thread-safe subscription management"""
        with self._subscriptions_lock:
            self._subscriptions.add(callback)
            def unsubscribe() -> None:
                with self._subscriptions_lock:
                    self._subscriptions.discard(callback)
            return unsubscribe

    def _fire_subscriptions(self, experiment: Experiment[Any], result: Result[Any]) -> None:
        """Thread-safe subscription notifications"""
        with self._subscriptions_lock:
            subscriptions = self._subscriptions.copy()

        for callback in subscriptions:
            try:
                self._run_user_callback(callback, (experiment, result), "subscription")
            except Exception:
                logger.exception("Error in subscription callback")


    def set_event_logger(self, fn: AsyncEventLogger) -> None:
        """Register a callable that will be invoked by log_event.

        The callable receives (event_name: str, properties: dict, user_context: UserContext).
        Typically set by GrowthBookTrackingPlugin rather than called directly.
        """
        self.options.event_logger = fn

    async def log_event(
        self,
        event_name: str,
        properties: Optional[Dict[str, Any]] = None,
        user_context: Optional[UserContext] = None,
    ) -> None:
        """Log a custom event to the GrowthBook ingestor.

        Requires GrowthBookTrackingPlugin to be configured; without it a warning
        is emitted and the call is a no-op.

        Args:
            event_name: Name of the event (e.g. ``"button_clicked"``).
            properties: Optional dict of event-specific properties.
            user_context: User context for the event; uses an empty context if omitted.
        """
        if not self.options.event_logger:
            logger.warning(
                "log_event called but no event logger is configured. "
                "Add GrowthBookTrackingPlugin to enable event logging."
            )
            return
        ctx = user_context or UserContext()
        try:
            result = self.options.event_logger(event_name, properties or {}, ctx)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.exception("Error in event logger: %s", e)

    async def set_features(self, features: Dict[str, Any]) -> None:
        await self._feature_update_callback({"features": features})

    async def set_payload(self, payload: Dict[str, Any]) -> None:
        """Set features, saved groups, and contextual bandits from a full SDK
        payload, e.g. one fetched out-of-band from the GrowthBook API.
        Mirrors the JS SDK's setPayload: only the sections present in the
        payload are overwritten, and encrypted sections are decrypted with
        the configured decryption_key."""
        # decrypt_payload_sections is stateless, so the module singleton is a
        # safe stand-in when set_payload is called before initialize().
        repo: FeatureRepository = self._features_repository or feature_repo
        data = repo.decrypt_payload_sections(
            payload, self.options.decryption_key or ""
        )
        if data is not None:
            await self._feature_update_callback(data)
        
    
    async def _refresh_sticky_buckets(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch sticky bucket assignments for these attributes.

        Never blocks the event loop: async services are awaited natively, sync
        services are offloaded to the default executor. Concurrent evals with
        identical attributes share one inflight fetch; waiters are shielded so
        one cancelled waiter cannot poison the shared future, and if the OWNER
        is cancelled, waiters retry (one becomes the new owner).
        """
        service = self.options.sticky_bucket_service
        if not service:
            return {}

        key = json.dumps(attributes, sort_keys=True, default=str)

        if self._sticky_cache_enabled:
            entry = self._sticky_bucket_cache.get(key)
            if entry is not None:
                cached_assignments, expires_at = entry
                if time.monotonic() < expires_at:
                    self._sticky_bucket_cache.move_to_end(key)
                    # Re-apply local writes: another snapshot for the same
                    # identifier may have assigned since this entry was cached.
                    return self._overlay_local_sticky_docs(attributes, cached_assignments)
                del self._sticky_bucket_cache[key]

        while True:
            inflight = self._sticky_bucket_inflight.get(key)
            if inflight is None:
                break
            try:
                return await asyncio.shield(inflight)
            except asyncio.CancelledError:
                if not inflight.cancelled():
                    raise  # WE were cancelled; the owner fetch continues
                continue  # owner was cancelled; retry (maybe become owner)

        loop = asyncio.get_running_loop()
        fut: "asyncio.Future[Dict[str, Any]]" = loop.create_future()
        self._sticky_bucket_inflight[key] = fut
        try:
            if isinstance(service, AbstractAsyncStickyBucketService):
                assignments = await service.get_all_assignments(attributes)
            else:
                assignments = await loop.run_in_executor(
                    None, service.get_all_assignments, attributes
                )
            self._overlay_local_sticky_docs(attributes, assignments)
        except asyncio.CancelledError:
            if not fut.cancelled():
                fut.cancel()
            raise
        except Exception as e:
            if not fut.done():
                fut.set_exception(e)
                fut.exception()  # mark retrieved: no GC warning if unawaited
            raise
        finally:
            self._sticky_bucket_inflight.pop(key, None)

        if self._sticky_cache_enabled:
            self._sticky_bucket_cache[key] = (
                assignments,
                time.monotonic() + self.options.sticky_bucket_cache_ttl,
            )
            self._sticky_bucket_cache.move_to_end(key)
            while len(self._sticky_bucket_cache) > self.options.sticky_bucket_cache_size:
                self._sticky_bucket_cache.popitem(last=False)

        if not fut.done():
            fut.set_result(assignments)
        return assignments

    _STICKY_DOCS_MAX = 1000  # LRU bound for the authoritative doc map

    @staticmethod
    def _sticky_doc_key(doc: Dict[str, Any]) -> str:
        return f"{doc['attributeName']}||{doc['attributeValue']}"

    def _overlay_local_sticky_docs(self, attributes: Dict[str, Any],
                                   assignments: Dict[str, Any]) -> Dict[str, Any]:
        """Merge this process's authoritative assignment docs (local wins per
        experiment key) into a fetched/cached snapshot, for the doc keys this
        attributes dict can address. Keeps snapshots for the same identifier
        consistent with local writes even while their saves are in flight."""
        for name, value in attributes.items():
            key = f"{name}||{value}"
            local = self._sticky_docs.get(key)
            if local is None:
                continue
            service_doc = assignments.get(key) or {}
            assignments[key] = {
                "attributeName": name,
                "attributeValue": value,
                "assignments": {
                    **service_doc.get("assignments", {}),
                    **local["assignments"],
                },
            }
        return assignments

    def _schedule_sticky_bucket_save(self, doc: Dict[str, Any]) -> None:
        """Fire-and-forget persistence of a sticky bucket assignment doc.

        The doc is first merged into the authoritative per-process map, and
        what gets persisted is always the merged doc — so a save can never
        drop assignments made under a different attributes snapshot. Saves
        are serialized per doc key (one in flight, dirty flag for a trailing
        save), so completion order cannot regress the stored doc.

        Called synchronously from core's run_experiment via
        EvaluationContext.save_sticky_bucket_doc; the in-memory snapshot doc
        is already updated by core (read-your-writes).
        """
        service = self.options.sticky_bucket_service
        if not service:
            return

        key = self._sticky_doc_key(doc)
        local = self._sticky_docs.get(key)
        merged_assignments = {
            **(local["assignments"] if local else {}),
            **doc.get("assignments", {}),
        }
        self._sticky_docs[key] = {
            "attributeName": doc["attributeName"],
            "attributeValue": doc["attributeValue"],
            "assignments": merged_assignments,
        }
        self._sticky_docs.move_to_end(key)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (hand-rolled context outside the public API):
            # persist inline if the service is sync, otherwise drop with a log.
            if isinstance(service, AbstractAsyncStickyBucketService):
                logger.error(
                    "Cannot persist sticky bucket doc: async service but no "
                    "running event loop"
                )
            else:
                service.save_assignments(self._sticky_docs[key])
            return

        self._sticky_save_dirty.add(key)
        if key not in self._sticky_save_inflight:
            self._kick_sticky_save(key, loop)

    def _kick_sticky_save(self, key: str, loop: asyncio.AbstractEventLoop) -> None:
        """Start one save for `key` with the current merged doc. On completion,
        re-kick if the doc changed meanwhile (dirty), so the service converges
        to the latest merged state without ever having two writes for one key
        in flight."""
        self._sticky_save_dirty.discard(key)
        local = self._sticky_docs.get(key)
        if local is None:
            return
        # Snapshot for the write: the authoritative doc may be merged into
        # again while the (threaded or awaited) save is in flight.
        doc = {**local, "assignments": dict(local["assignments"])}

        service = self.options.sticky_bucket_service
        if service is None:  # unreachable via _schedule; keeps types honest
            return
        if isinstance(service, AbstractAsyncStickyBucketService):
            fut: "asyncio.Future[Any]" = loop.create_task(service.save_assignments(doc))
        else:
            fut = loop.run_in_executor(None, service.save_assignments, doc)
        self._sticky_save_inflight[key] = fut

        def _done(f: "asyncio.Future[Any]") -> None:
            self._sticky_save_inflight.pop(key, None)
            if not f.cancelled() and f.exception():
                logger.error("Sticky bucket save failed", exc_info=f.exception())
            if key in self._sticky_save_dirty:
                self._kick_sticky_save(key, loop)
            else:
                self._prune_sticky_docs()

        fut.add_done_callback(_done)

    def _prune_sticky_docs(self) -> None:
        """Evict least-recently-used CLEAN docs beyond the bound. Dirty or
        in-flight docs are never evicted — they are unsaved local truth."""
        excess = len(self._sticky_docs) - self._STICKY_DOCS_MAX
        if excess <= 0:
            return
        for key in list(self._sticky_docs.keys()):
            if excess <= 0:
                break
            if key in self._sticky_save_dirty or key in self._sticky_save_inflight:
                continue
            del self._sticky_docs[key]
            excess -= 1

    async def flush_sticky_bucket_saves(self) -> None:
        """Wait until every sticky bucket doc is persisted (including trailing
        saves triggered by writes that arrived mid-save).

        Useful in short-lived environments (e.g. serverless) where the event
        loop may be torn down before background writes finish. close() calls
        this automatically. Failures are logged, never raised.
        """
        while self._sticky_save_inflight or self._sticky_save_dirty:
            pending = list(self._sticky_save_inflight.values())
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            # ALWAYS yield one loop iteration: awaiting already-done futures
            # does not yield, and a completed save can still have its
            # inflight-popping done-callback queued — without this the loop
            # spins forever and that callback never runs.
            await asyncio.sleep(0)

    async def initialize(self) -> bool:
        """Initialize client with features and start refresh"""
        if not self._features_repository:
            logger.error("No features repository available")
            return False

        try:
            if self.options.remote_eval:
                # Remote-eval mode: do NOT fetch global features (responses are
                # per-user, so there is no meaningful "global" payload). Skip
                # callback registration (would cross-pollute cached per-user
                # state). Establish a default empty global context so
                # create_evaluation_context doesn't raise; features come from
                # fetch_remote_eval at eval time.
                async with self._context_lock:
                    if self._global_context is None:
                        self._global_context = GlobalContext(
                            options=self.options, features={}, saved_groups={}
                        )
                # Only SSE is meaningful in remote-eval mode (the validation
                # guard already rejects STALE_WHILE_REVALIDATE). None means
                # "no background refresh; rely on cache TTL".
                if self.options.refresh_strategy == FeatureRefreshStrategy.SERVER_SENT_EVENTS:
                    await self._features_repository.start_feature_refresh(
                        self.options.refresh_strategy,
                        callback=self._feature_update_callback,
                    )
                return True

            # Initial feature load
            initial_features = await self._features_repository.load_features_async(
                self.options.api_host or "https://cdn.growthbook.io",
                self.options.client_key or "",
                self.options.decryption_key or "",
                self.options.cache_ttl
            )
            if not initial_features:
                logger.error("Failed to load initial features")
                return False

            # Create global context with initial features
            await self._feature_update_callback(initial_features)

            # Set up callback for future updates
            self._features_repository.add_callback(self._feature_update_callback)

            # Start feature refresh
            refresh_strategy = self.options.refresh_strategy or FeatureRefreshStrategy.STALE_WHILE_REVALIDATE
            await self._features_repository.start_feature_refresh(refresh_strategy)
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}", exc_info=True)
            traceback.print_exc()
            return False

    def _remote_eval_payload(self, user_context: UserContext) -> Dict[str, Any]:
        return build_remote_eval_payload(
            user_context.attributes,
            user_context.forced_variations,
            user_context.url,
            forced_features=user_context.forced_features,
        )

    async def preload_remote_eval(self, user_context: UserContext) -> None:
        """Warm the remote-eval cache for this user context. No-op when
        remote_eval is disabled. After this returns, subsequent eval_feature /
        is_on / etc. calls for the same UserContext are cache hits and make no
        network requests."""
        if not self.options.remote_eval or not self._features_repository:
            return
        await self._features_repository.fetch_remote_eval(
            self.options.api_host or "https://cdn.growthbook.io",
            self.options.client_key or "",
            self._remote_eval_payload(user_context),
            self.options.cache_key_attributes,
            cache_ttl=self.options.cache_ttl,
            stale_ttl=self.options.stale_ttl,
        )

    async def _feature_update_callback(self, features_data: Dict[str, Any]) -> None:
        """Handle feature updates and manage global context"""
        if not features_data:
            logger.warning("Warning: Received empty features data")
            return

        async with self._context_lock:  # serializes concurrent updaters only
            prev = self._global_context
            # Mirror JS setPayload: sections absent from the update carry
            # over from the previous snapshot, so a partial update (e.g.
            # set_features) doesn't silently wipe savedGroups or
            # contextualBandits. Full refreshes carry all three keys.
            features = (
                features_from_dict(features_data["features"])
                if "features" in features_data
                else (prev.features if prev else {})
            )
            saved_groups = features_data.get(
                "savedGroups", prev.saved_groups if prev else {}
            )
            contextual_bandits = features_data.get(
                "contextualBandits", prev.contextual_bandits if prev else {}
            )

            # Build a NEW immutable snapshot and swap the reference atomically
            # (single assignment). In-flight evaluations captured the previous
            # snapshot and finish against it; new evaluations see this one.
            # This is what lets evaluations run without any lock.
            self._global_context = GlobalContext(
                options=self.options, features=features, saved_groups=saved_groups,
                contextual_bandits=contextual_bandits
            )

    async def __aenter__(self) -> "GrowthBookClient":
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.close()

    async def create_evaluation_context(self, user_context: UserContext) -> EvaluationContext:
        """Create evaluation context for feature evaluation"""
        # Capture the snapshot once; feature updates swap the reference, so
        # this evaluation runs against a consistent view without locking.
        global_context = self._global_context
        if global_context is None:
            raise RuntimeError("GrowthBook client not properly initialized")

        if self.options.remote_eval and self._features_repository:
            # Per-user POST + cache: features come from the proxy filtered for
            # this UserContext, not from self._global_context.features.
            response = await self._features_repository.fetch_remote_eval(
                self.options.api_host or "https://cdn.growthbook.io",
                self.options.client_key or "",
                self._remote_eval_payload(user_context),
                self.options.cache_key_attributes,
                cache_ttl=self.options.cache_ttl,
                stale_ttl=self.options.stale_ttl,
            ) or {}
            global_ctx = GlobalContext(
                options=self.options,
                features=features_from_dict(response.get("features")),
                saved_groups=response.get("savedGroups") or {},
                contextual_bandits=response.get("contextualBandits") or {},
            )
            return EvaluationContext(
                user=user_context,
                global_ctx=global_ctx,
                stack=StackContext(evaluated_features=set()),
            )

        # Get sticky bucket assignments if needed
        sticky_assignments = await self._refresh_sticky_buckets(user_context.attributes)

        # Intentionally the SHARED cached dict, not a copy: core mutates it in
        # place when an experiment assigns a new sticky bucket, which is what
        # gives read-your-writes semantics while persistence happens
        # asynchronously (same mechanism as the JS SDK's
        # stickyBucketAssignmentDocs).
        user_context.sticky_bucket_assignment_docs = sticky_assignments

        return EvaluationContext(
            user=user_context,
            global_ctx=global_context,
            stack=StackContext(evaluated_features=set()),
            save_sticky_bucket_doc=(
                self._schedule_sticky_bucket_save
                if self.options.sticky_bucket_service else None
            ),
        )

    async def eval_feature(self, key: str, user_context: UserContext) -> FeatureResult[Any]:
        """Evaluate a feature. Lock-free: the evaluation context captures an
        immutable feature snapshot, so concurrent evaluations never contend
        with each other or with feature updates."""
        context = await self.create_evaluation_context(user_context)
        result = core_eval_feature(key=key, evalContext=context, tracking_cb=self._track)
        # Call feature usage callback if provided
        if self.options.on_feature_usage:
            try:
                self._run_user_callback(
                    self.options.on_feature_usage,
                    (key, result, tracking_user_context(user_context)),
                    "feature usage",
                )
            except Exception:
                logger.exception("Error in feature usage callback")
        return result

    async def is_on(self, key: str, user_context: UserContext) -> bool:
        """Check if a feature is enabled with proper async context management"""
        result = await self.eval_feature(key, user_context)
        return result.on

    async def is_off(self, key: str, user_context: UserContext) -> bool:
        """Check if a feature is set to off with proper async context management"""
        result = await self.eval_feature(key, user_context)
        return result.off

    async def get_feature_value(self, key: str, fallback: T, user_context: UserContext) -> T:
        result = await self.eval_feature(key, user_context)
        return cast(T, result.value) if result.value is not None else fallback

    async def run(self, experiment: Experiment[T], user_context: UserContext) -> Result[T]:
        """Run experiment with tracking. Lock-free, same as eval_feature."""
        context = await self.create_evaluation_context(user_context)
        result = run_experiment(
            experiment=experiment,
            evalContext=context,
            tracking_cb=self._track
        )
        # Fire subscriptions synchronously
        self._fire_subscriptions(experiment, result)
        return result
        
    async def close(self) -> None:
        """Clean shutdown with proper cleanup"""
        # Let in-flight sticky bucket writes finish (sub-ms typically);
        # cancelling them would lose assignments.
        await self.flush_sticky_bucket_saves()

        # Drain any scheduled async user callbacks (tracking, feature usage,
        # subscriptions); failures are logged by their done-callbacks.
        while self._callback_tasks:
            await asyncio.gather(*list(self._callback_tasks), return_exceptions=True)
            # Yield so completed tasks' set-discarding done-callbacks run
            # (awaiting done futures alone never yields — see flush above).
            await asyncio.sleep(0)

        if self._features_repository:
            await self._features_repository.stop_refresh()

        # Clear tracking and subscription state
        with self._tracked_lock:
            self._tracked.clear()
        with self._subscriptions_lock:
            self._subscriptions.clear()
        # Clear context
        async with self._context_lock:
            self._global_context = None
            
        # Cleanup plugins
        self._cleanup_plugins()

    @property
    def user_agent_suffix(self) -> Optional[str]:
        """Get the suffix appended to the User-Agent header"""
        return feature_repo.user_agent_suffix
        
    @user_agent_suffix.setter
    def user_agent_suffix(self, value: Optional[str]) -> None:
        """Set a suffix to be appended to the User-Agent header"""
        feature_repo.user_agent_suffix = value

    def _initialize_plugins(self) -> None:
        """Initialize all tracking plugins with this GrowthBookClient instance."""
        for plugin in self._tracking_plugins:
            try:
                # getattr (not hasattr+access) keeps duck-typed plugin objects
                # working while narrowing cleanly under both checkers.
                initialize = getattr(plugin, "initialize", None)
                if callable(initialize):
                    # Plugin is a class instance with initialize method
                    initialize(self)
                    self._initialized_plugins.append(plugin)
                    logger.debug(f"Initialized plugin: {plugin.__class__.__name__}")
                elif callable(plugin):
                    # Plugin is a callable function
                    plugin(self)
                    self._initialized_plugins.append(plugin)
                    logger.debug(f"Initialized callable plugin: {getattr(plugin, '__name__', plugin)}")
                else:
                    logger.warning(f"Plugin {plugin} is neither callable nor has initialize method")
            except Exception as e:
                logger.error(f"Failed to initialize plugin {plugin}: {e}")

    def _cleanup_plugins(self) -> None:
        """Cleanup all initialized plugins."""
        for plugin in self._initialized_plugins:
            try:
                cleanup = getattr(plugin, "cleanup", None)
                if callable(cleanup):
                    cleanup()
                    logger.debug(f"Cleaned up plugin: {plugin.__class__.__name__}")
            except Exception as e:
                logger.error(f"Error cleaning up plugin {plugin}: {e}")
        self._initialized_plugins.clear()