#!/usr/bin/env python
import json
from dataclasses import dataclass, field
import random
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from typing import Set
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

    def __call__(cls, *args, **kwargs):
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
    ):
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
    def __init__(self, obj):
        self.obj = obj

class FeatureCache:
    """Thread-safe feature cache"""
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {
            'features': {},
            'savedGroups': {}
        }
        self._lock = threading.Lock()

    def update(self, features: Dict[str, Any], saved_groups: Dict[str, Any]) -> None:
        """Simple thread-safe update of cache with new API data"""
        with self._lock:
            self._cache['features'] = dict(features)
            self._cache['savedGroups'] = dict(saved_groups)

    def get_current_state(self) -> Dict[str, Any]:
        """Get current cache state"""
        with self._lock:
            return {
                "features": dict(self._cache['features']),
                "savedGroups": self._cache['savedGroups']
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
                 remote_eval: bool = False):
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
        self._refresh_task: Optional[asyncio.Task] = None
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
    async def refresh_operation(self):
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
        self._feature_cache.update(
            data.get("features", {}),
            data.get("savedGroups", {})
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
                    "encryptedFeatures" in data or "encryptedSavedGroups" in data
                ):
                    logger.debug("Decrypting SSE payload...")
                    data = self.decrypt_response(data, self._decryption_key)
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

    async def start_feature_refresh(self, strategy: FeatureRefreshStrategy, callback=None):
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

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
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
    ) -> Optional[Dict]:
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
    ):
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
        self._subscriptions: Set[Callable[[Experiment, Result], None]] = set()
        self._subscriptions_lock = threading.Lock()

        # Add sticky bucket cache
        self._sticky_bucket_cache: Dict[str, Dict[str, Any]] = {
            'attributes': {},
            'assignments': {}
        }
        self._sticky_bucket_lock = asyncio.Lock()
        # Strong refs to in-flight fire-and-forget save_assignments futures;
        # bare create_task results are only weakly held by the loop and can be
        # garbage-collected mid-write. Drained in close()/flush.
        self._sticky_save_tasks: Set["asyncio.Future[Any]"] = set()
        # Strong refs to scheduled async user callbacks (tracking, feature
        # usage, subscriptions). Drained in close().
        self._callback_tasks: Set["asyncio.Future[Any]"] = set()
        
        # Plugin support
        self._tracking_plugins: List[Any] = self.options.tracking_plugins or []
        self._initialized_plugins: List[Any] = []

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

    def _run_user_callback(self, callback: Callable, args: tuple, what: str) -> None:
        """Invoke a user callback that may be sync or async.

        Called from synchronous eval paths, so a returned coroutine cannot be
        awaited here; it is scheduled fire-and-forget on the running loop
        (drained in close()). Sync exceptions propagate to the caller's
        existing try/except."""
        result = callback(*args)
        if asyncio.iscoroutine(result):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                result.close()
                logger.error("Async %s callback requires a running event loop; dropped", what)
                return
            self._spawn_tracked(
                loop.create_task(result),
                self._callback_tasks,
                f"Error in {what} callback",
            )

    def _track(self, experiment: Experiment, result: Result, user_context: UserContext) -> None:
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
                        (experiment, result, user_context),
                        "tracking",
                    )
                    self._tracked[key] = True
                except Exception:
                    logger.exception("Error in tracking callback")

    def subscribe(self, callback: Callable[[Experiment, Result], None]) -> Callable[[], None]:
        """Thread-safe subscription management"""
        with self._subscriptions_lock:
            self._subscriptions.add(callback)
            def unsubscribe():
                with self._subscriptions_lock:
                    self._subscriptions.discard(callback)
            return unsubscribe

    def _fire_subscriptions(self, experiment: Experiment, result: Result) -> None:
        """Thread-safe subscription notifications"""
        with self._subscriptions_lock:
            subscriptions = self._subscriptions.copy()

        for callback in subscriptions:
            try:
                self._run_user_callback(callback, (experiment, result), "subscription")
            except Exception:
                logger.exception("Error in subscription callback")


    def set_event_logger(self, fn) -> None:
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

    async def set_features(self, features: dict) -> None:
        await self._feature_update_callback({"features": features})
        
    
    async def _refresh_sticky_buckets(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Refresh sticky bucket assignments only if attributes have changed.

        Never blocks the event loop: async services are awaited natively, sync
        services are offloaded to the default executor. The lock also coalesces
        concurrent refreshes for identical attributes — waiters hit the cache
        check after the first fetch completes.
        """
        service = self.options.sticky_bucket_service
        if not service:
            return {}

        async with self._sticky_bucket_lock:
            if attributes == self._sticky_bucket_cache['attributes']:
                return self._sticky_bucket_cache['assignments']

            if isinstance(service, AbstractAsyncStickyBucketService):
                assignments = await service.get_all_assignments(attributes)
            else:
                loop = asyncio.get_running_loop()
                assignments = await loop.run_in_executor(
                    None, service.get_all_assignments, attributes
                )
            self._sticky_bucket_cache['attributes'] = attributes.copy()
            self._sticky_bucket_cache['assignments'] = assignments
            return assignments

    def _schedule_sticky_bucket_save(self, doc: Dict) -> None:
        """Fire-and-forget persistence of a sticky bucket assignment doc,
        mirroring the JS SDK: the in-memory doc is already updated
        synchronously by core (read-your-writes), so the service write can
        complete in the background without blocking the event loop.

        Called synchronously from core's run_experiment via
        EvaluationContext.save_sticky_bucket_doc.
        """
        service = self.options.sticky_bucket_service
        if not service:
            return
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
                service.save_assignments(doc)
            return

        if isinstance(service, AbstractAsyncStickyBucketService):
            fut: "asyncio.Future[Any]" = loop.create_task(service.save_assignments(doc))
        else:
            fut = loop.run_in_executor(None, service.save_assignments, doc)
        self._spawn_tracked(fut, self._sticky_save_tasks, "Sticky bucket save failed")

    async def flush_sticky_bucket_saves(self) -> None:
        """Wait for all in-flight sticky bucket saves to complete.

        Useful in short-lived environments (e.g. serverless) where the event
        loop may be torn down before background writes finish. close() calls
        this automatically. Failures are logged, never raised.
        """
        while self._sticky_save_tasks:
            await asyncio.gather(*list(self._sticky_save_tasks), return_exceptions=True)

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

        async with self._context_lock:
            features = features_from_dict(features_data.get("features"))
            saved_groups = features_data.get("savedGroups", {})

            if self._global_context is None:
                self._global_context = GlobalContext(
                    options=self.options, features=features, saved_groups=saved_groups
                )
            else:
                self._global_context.features = features
                self._global_context.saved_groups = saved_groups

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def create_evaluation_context(self, user_context: UserContext) -> EvaluationContext:
        """Create evaluation context for feature evaluation"""
        if self._global_context is None:
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
            global_ctx=self._global_context,
            stack=StackContext(evaluated_features=set()),
            save_sticky_bucket_doc=(
                self._schedule_sticky_bucket_save
                if self.options.sticky_bucket_service else None
            ),
        )

    @asynccontextmanager
    async def _eval_lock(self):
        """Lock for the duration of an evaluation.

        In CDN mode this guards against `_global_context` mutations (the
        shared features dict) during `create_evaluation_context` +
        `core_eval_feature`.

        In remote-eval mode the EvaluationContext is built fresh per-call
        from the per-user POST response — no shared state to guard, and
        holding the lock across the network round-trip would serialize all
        evaluations through one POST even for unrelated users (a real
        throughput cliff on busy services)."""
        if self.options.remote_eval:
            yield
        else:
            async with self._context_lock:
                yield

    async def eval_feature(self, key: str, user_context: UserContext) -> FeatureResult:
        """Evaluate a feature with proper async context management"""
        async with self._eval_lock():
            context = await self.create_evaluation_context(user_context)
            result = core_eval_feature(key=key, evalContext=context, tracking_cb=self._track)
            # Call feature usage callback if provided
            if self.options.on_feature_usage:
                try:
                    self._run_user_callback(
                        self.options.on_feature_usage,
                        (key, result, user_context),
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

    async def get_feature_value(self, key: str, fallback: Any, user_context: UserContext) -> Any:
        result = await self.eval_feature(key, user_context)
        return result.value if result.value is not None else fallback

    async def run(self, experiment: Experiment, user_context: UserContext) -> Result:
        """Run experiment with tracking"""
        async with self._eval_lock():
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
                if hasattr(plugin, 'initialize'):
                    # Plugin is a class instance with initialize method
                    plugin.initialize(self)
                    self._initialized_plugins.append(plugin)
                    logger.debug(f"Initialized plugin: {plugin.__class__.__name__}")
                elif callable(plugin):
                    # Plugin is a callable function
                    plugin(self)
                    self._initialized_plugins.append(plugin)
                    logger.debug(f"Initialized callable plugin: {plugin.__name__}")
                else:
                    logger.warning(f"Plugin {plugin} is neither callable nor has initialize method")
            except Exception as e:
                logger.error(f"Failed to initialize plugin {plugin}: {e}")

    def _cleanup_plugins(self) -> None:
        """Cleanup all initialized plugins."""
        for plugin in self._initialized_plugins:
            try:
                if hasattr(plugin, 'cleanup'):
                    plugin.cleanup()
                    logger.debug(f"Cleaned up plugin: {plugin.__class__.__name__}")
            except Exception as e:
                logger.error(f"Error cleaning up plugin {plugin}: {e}")
        self._initialized_plugins.clear()