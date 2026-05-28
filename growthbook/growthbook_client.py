#!/usr/bin/env python
import json
from dataclasses import dataclass, field
import random
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from typing import Set
import asyncio
import threading
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
    Experiment
)

logger = logging.getLogger("growthbook.growthbook_client")

class SingletonMeta(type):
    """Thread-safe implementation of Singleton pattern"""
    _instances: Dict[type, Any] = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

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
                 remote_eval_cache_size: int = 1000):
        FeatureRepository.__init__(self)
        self._api_host = api_host
        self._client_key = client_key
        self._decryption_key = decryption_key
        self._cache_ttl = cache_ttl
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
        # NOTE: no asyncio.Lock here on purpose — the critical sections in
        # fetch_remote_eval are pure dict ops with no `await`, so they're
        # already atomic from asyncio's perspective. Binding an asyncio.Lock
        # at __init__ time would tie it to whichever event loop happened to
        # be running at construction (a footgun with the SingletonMeta +
        # pytest-asyncio's per-test loops — leads to cross-loop awaits that
        # hang forever).
        self._remote_eval_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._remote_eval_cache_max = remote_eval_cache_size
        # Inflight coalescing: when concurrent calls land for the same cache
        # key, the second caller awaits the first's future instead of issuing
        # a duplicate POST.
        self._remote_eval_inflight: Dict[str, "asyncio.Future[Optional[Dict[str, Any]]]"] = {}

    async def fetch_remote_eval(
        self,
        api_host: str,
        client_key: str,
        payload: Dict[str, Any],
        cache_key_attributes: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """POST to /api/eval/{client_key}, caching per-payload with LRU eviction
        and coalescing concurrent identical requests onto a single inflight POST.

        The cache and inflight dicts are touched only between `await` points,
        so no explicit lock is needed — asyncio coroutines yield only at
        `await`, and the CPython interpreter makes single dict ops atomic."""
        cache_key = self._compute_cache_key(api_host, client_key, True, payload, cache_key_attributes)

        # Cache lookup — no await between get and the on-hit handling.
        cached: Optional[Dict[str, Any]] = self._remote_eval_cache.get(cache_key)
        if cached is not None:
            self._remote_eval_cache.move_to_end(cache_key)
            return cached

        # Inflight check — if another coroutine is already POSTing for this
        # cache key, await its future instead of issuing a duplicate POST.
        existing = self._remote_eval_inflight.get(cache_key)
        if existing is not None:
            return await existing

        # Leader path: create the future bound to the CURRENT running loop
        # (using asyncio.Future() rather than capturing a loop reference),
        # register inflight, then perform the POST.
        inflight: "asyncio.Future[Optional[Dict[str, Any]]]" = asyncio.Future()
        self._remote_eval_inflight[cache_key] = inflight

        try:
            response = await self._fetch_and_decode_post_async(api_host, client_key, payload)
        except Exception as e:
            self._remote_eval_inflight.pop(cache_key, None)
            if not inflight.done():
                inflight.set_exception(e)
            raise

        self._remote_eval_inflight.pop(cache_key, None)
        if response is not None:
            self._remote_eval_cache[cache_key] = response
            self._remote_eval_cache.move_to_end(cache_key)
            while len(self._remote_eval_cache) > self._remote_eval_cache_max:
                self._remote_eval_cache.popitem(last=False)
        if not inflight.done():
            inflight.set_result(response)
        return response

    async def flush_remote_eval_cache(self) -> None:
        """Drop all cached remote-eval responses. Called when the proxy
        emits a features-updated SSE event."""
        self._remote_eval_cache.clear()

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
                if self._remote_eval_cache_max and self._remote_eval_cache:
                    await self.flush_remote_eval_cache()
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
                if self._remote_eval_cache_max and self._remote_eval_cache:
                    await self.flush_remote_eval_cache()
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
        try:
            self.stopAutoRefresh(timeout=10)
        except Exception:
            # Best-effort cleanup; task cancellation below will proceed.
            logger.exception("Error stopping SSE auto-refresh")
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
    ) -> Optional[Dict]:
        # Use stored values when called internally
        if api_host == self._api_host and client_key == self._client_key:
            decryption_key = self._decryption_key
            ttl = self._cache_ttl
        return await super().load_features_async(
            api_host, client_key, decryption_key, ttl,
            remote_eval=remote_eval, payload=payload,
            cache_key_attributes=cache_key_attributes,
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
            if not self.options.client_key:
                raise ValueError("Must specify client_key for remote eval")
            if self.options.decryption_key:
                raise ValueError("Encryption is not available for remote eval")
            if self.options.sticky_bucket_service is not None:
                raise ValueError(
                    "sticky_bucket_service is not compatible with remote_eval; "
                    "the proxy handles sticky bucketing server-side"
                )
            from urllib.parse import urlparse as _urlparse
            host = _urlparse(self.options.api_host or "").hostname or ""
            if host == "growthbook.io" or host.endswith(".growthbook.io"):
                raise ValueError(
                    "Cloud host does not support remote eval; use a self-hosted proxy/edge"
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
        self._sticky_bucket_cache_lock = False
        
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
            )
            if self.options.client_key
            else None
        )
        
        self._global_context: Optional[GlobalContext] = None
        self._context_lock = asyncio.Lock()
        
        # Initialize plugins
        self._initialize_plugins()


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
                    self.options.on_experiment_viewed(experiment, result, user_context)
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
                callback(experiment, result)
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
        """Refresh sticky bucket assignments only if attributes have changed"""
        if not self.options.sticky_bucket_service:
            return {}

        # Use compare-and-swap pattern
        while not self._sticky_bucket_cache_lock:
            if attributes == self._sticky_bucket_cache['attributes']:
                return self._sticky_bucket_cache['assignments']
            
            self._sticky_bucket_cache_lock = True
            try:
                assignments = self.options.sticky_bucket_service.get_all_assignments(attributes)
                self._sticky_bucket_cache['attributes'] = attributes.copy()
                self._sticky_bucket_cache['assignments'] = assignments
                return assignments
            finally:
                self._sticky_bucket_cache_lock = False
        
        # Fallback return for edge case where loop condition is never satisfied
        return {}

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
                refresh_strategy = self.options.refresh_strategy or FeatureRefreshStrategy.STALE_WHILE_REVALIDATE
                if refresh_strategy == FeatureRefreshStrategy.SERVER_SENT_EVENTS:
                    # SSE: features-updated event will flush the remote-eval cache
                    await self._features_repository.start_feature_refresh(
                        refresh_strategy, callback=self._feature_update_callback
                    )
                # HTTP polling refresh is intentionally not started in remote-eval
                # mode — the polling loop has no per-user payload to send.
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

    def _build_remote_eval_payload(self, user_context: UserContext) -> Dict[str, Any]:
        return {
            "attributes": user_context.attributes or {},
            "forcedFeatures": [],  # not exposed on UserContext today; future extension
            "forcedVariations": user_context.forced_variations or {},
            "url": user_context.url or "",
        }

    def _materialize_features(self, response: Dict[str, Any]) -> Dict[str, Feature]:
        features: Dict[str, Feature] = {}
        for key, feature in (response.get("features") or {}).items():
            if isinstance(feature, Feature):
                features[key] = feature
            else:
                features[key] = Feature(
                    rules=feature.get("rules", []),
                    defaultValue=feature.get("defaultValue", None),
                )
        return features

    async def preload_remote_eval(self, user_context: UserContext) -> None:
        """Warm the remote-eval cache for this user context. No-op when
        remote_eval is disabled. After this returns, subsequent eval_feature /
        is_on / etc. calls for the same UserContext are cache hits and make no
        network requests."""
        if not self.options.remote_eval or not self._features_repository:
            return
        payload = self._build_remote_eval_payload(user_context)
        await self._features_repository.fetch_remote_eval(
            self.options.api_host or "https://cdn.growthbook.io",
            self.options.client_key or "",
            payload,
            self.options.cache_key_attributes,
        )

    async def _feature_update_callback(self, features_data: Dict[str, Any]) -> None:
        """Handle feature updates and manage global context"""
        if not features_data:
            logger.warning("Warning: Received empty features data")
            return

        async with self._context_lock:
            features = {}

            for key, feature in features_data.get("features", {}).items():
                if isinstance(feature, Feature):
                    features[key] = feature
                else:
                    features[key] = Feature(
                        rules=feature.get("rules", []),
                        defaultValue=feature.get("defaultValue", None),
                    )

            if self._global_context is None:
                # Initial creation of global context
                self._global_context = GlobalContext(
                        options=self.options,
                        features=features,
                        saved_groups=features_data.get("savedGroups", {})
                )
            else:
                # Update existing global context
                self._global_context.features = features
                self._global_context.saved_groups = features_data.get("savedGroups", {})

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
            payload = self._build_remote_eval_payload(user_context)
            response = await self._features_repository.fetch_remote_eval(
                self.options.api_host or "https://cdn.growthbook.io",
                self.options.client_key or "",
                payload,
                self.options.cache_key_attributes,
            ) or {}
            features = self._materialize_features(response)
            saved_groups = response.get("savedGroups") or {}
            global_ctx = GlobalContext(
                options=self.options,
                features=features,
                saved_groups=saved_groups,
            )
            return EvaluationContext(
                user=user_context,
                global_ctx=global_ctx,
                stack=StackContext(evaluated_features=set()),
            )

        # Get sticky bucket assignments if needed
        sticky_assignments = await self._refresh_sticky_buckets(user_context.attributes)

        # update user context with sticky bucket assignments
        user_context.sticky_bucket_assignment_docs = sticky_assignments

        return EvaluationContext(
            user=user_context,
            global_ctx=self._global_context,
            stack=StackContext(evaluated_features=set())
        )

    async def eval_feature(self, key: str, user_context: UserContext) -> FeatureResult:
        """Evaluate a feature with proper async context management"""
        async with self._context_lock:
            context = await self.create_evaluation_context(user_context)
            result = core_eval_feature(key=key, evalContext=context, tracking_cb=self._track)
            # Call feature usage callback if provided
            if self.options.on_feature_usage:
                try:
                    self.options.on_feature_usage(key, result, user_context)
                except Exception:
                    logger.exception("Error in feature usage callback")
            return result

    async def is_on(self, key: str, user_context: UserContext) -> bool:
        """Check if a feature is enabled with proper async context management"""
        async with self._context_lock:
            context = await self.create_evaluation_context(user_context)
            result = core_eval_feature(key=key, evalContext=context, tracking_cb=self._track)
            # Call feature usage callback if provided
            if self.options.on_feature_usage:
                try:
                    self.options.on_feature_usage(key, result, user_context)
                except Exception:
                    logger.exception("Error in feature usage callback")
            return result.on
    
    async def is_off(self, key: str, user_context: UserContext) -> bool:
        """Check if a feature is set to off with proper async context management"""
        async with self._context_lock:
            context = await self.create_evaluation_context(user_context)
            result = core_eval_feature(key=key, evalContext=context, tracking_cb=self._track)
            # Call feature usage callback if provided
            if self.options.on_feature_usage:
                try:
                    self.options.on_feature_usage(key, result, user_context)
                except Exception:
                    logger.exception("Error in feature usage callback")
            return result.off
    
    async def get_feature_value(self, key: str, fallback: Any, user_context: UserContext) -> Any:
        async with self._context_lock:
            context = await self.create_evaluation_context(user_context)
            result = core_eval_feature(key=key, evalContext=context, tracking_cb=self._track)
            # Call feature usage callback if provided
            if self.options.on_feature_usage:
                try:
                    self.options.on_feature_usage(key, result, user_context)
                except Exception:
                    logger.exception("Error in feature usage callback")
            return result.value if result.value is not None else fallback

    async def run(self, experiment: Experiment, user_context: UserContext) -> Result:
        """Run experiment with tracking"""
        async with self._context_lock:
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