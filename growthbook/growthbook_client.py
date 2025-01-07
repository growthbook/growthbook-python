#!/usr/bin/env python

from dataclasses import dataclass, field
import random
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from typing import Set
import asyncio
import threading
import traceback
from datetime import datetime
from growthbook import FeatureRepository
from contextlib import asynccontextmanager
from .core import eval_feature as core_eval_feature
from .common_types import *

logger = logging.getLogger("growthbook.growthbook_client")

class SingletonMeta(type):
    """Thread-safe implementation of Singleton pattern"""
    _instances = {}
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
        self._cache = {
            'features': {},
            'savedGroups': {}
        }
        self._lock = threading.Lock()

    def update(self, features: Dict[str, Any], saved_groups: Dict[str, Any]) -> None:
        """Simple thread-safe update of cache with new API data"""
        with self._lock:
            self._cache['features'].update(features)
            self._cache['savedGroups'].update(saved_groups)

    def get_current_state(self) -> Dict[str, Any]:
        """Get current cache state"""
        with self._lock:
            return {
                "features": dict(self._cache['features']),
                "savedGroups": self._cache['savedGroups']
            }

class EnhancedFeatureRepository(FeatureRepository, metaclass=SingletonMeta):
    def __init__(self, api_host: str, client_key: str, decryption_key: str = "", cache_ttl: int = 60):
        FeatureRepository.__init__(self)
        self._api_host = api_host
        self._client_key = client_key
        self._decryption_key = decryption_key
        self._cache_ttl = cache_ttl
        self._refresh_lock = threading.Lock()
        self._refresh_task = None
        self._stop_event = asyncio.Event()
        self._backoff = BackoffStrategy()
        self._feature_cache = FeatureCache()
        self._callbacks = []
        self._last_successful_refresh = None
        self._refresh_in_progress = asyncio.Lock()

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

    def add_callback(self, callback: Callable) -> None:
        """Add callback to the list"""
        with self._refresh_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove callback from the list"""
        with self._refresh_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    async def _start_sse_refresh(self) -> None:
        """Start SSE-based feature refresh"""
        with self._refresh_lock:
            if self._refresh_task is not None:  # Already running
                return

            async def sse_handler(event_data: Dict[str, Any]) -> None:
                try:
                    if event_data['type'] == 'features-updated':
                        response = await self.load_features_async(
                            self._api_host, self._client_key, self._decryption_key, self._cache_ttl
                        )
                        await self._handle_feature_update(response)
                    elif event_data['type'] == 'features':
                        await self._handle_feature_update(event_data['data'])
                except Exception:
                    traceback.print_exc()

            # Start the SSE connection task
            self._refresh_task = asyncio.create_task(
                self._maintain_sse_connection(sse_handler)
            )

    async def _maintain_sse_connection(self, handler: Callable) -> None:
        """Maintain SSE connection with automatic reconnection"""
        while not self._stop_event.is_set():
            try:
                await self.startAutoRefresh(self._api_host, self._client_key, handler)
            except Exception as e:
                if not self._stop_event.is_set():
                    delay = self._backoff.next_delay()
                    logger.error(f"SSE connection lost, reconnecting in {delay:.2f}s: {str(e)}")
                    await asyncio.sleep(delay)

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
        self, api_host: str, client_key: str, decryption_key: str = "", ttl: int = 60
    ) -> Optional[Dict]:
        # Use stored values when called internally
        if api_host == self._api_host and client_key == self._client_key:
            decryption_key = self._decryption_key
            ttl = self._cache_ttl
        return await super().load_features_async(api_host, client_key, decryption_key, ttl)

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
        
        self._features_repository = (
            EnhancedFeatureRepository(self.options.api_host, self.options.client_key, self.options.decryption_key)
            if self.options.client_key
            else None
        )
        
        self._global_context = None
        # Change to asyncio.Lock for async operations
        self._context_lock = asyncio.Lock()  
    
    async def initialize(self) -> bool:
        """Initialize client with features and start refresh"""
        if not self._features_repository:
            return False

        try:
            # Initial feature load
            initial_features = await self._features_repository.load_features_async(
                self.options.api_host, self.options.client_key, self.options.decryption_key, self.options.cache_ttl
            )
            if not initial_features:
                logger.error("Failed to load initial features")
                return False

            # Create global context with initial features
            await self._feature_update_callback(initial_features)
            
            # Set up callback for future updates
            self._features_repository.add_callback(self._feature_update_callback)
            
            # Start feature refresh
            await self._features_repository.start_feature_refresh(self.options.refresh_strategy)
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}", exc_info=True)
            traceback.print_exc()
            return False

    async def _feature_update_callback(self, features_data: Dict[str, Any]) -> None:
        """Handle feature updates and manage global context"""
        if not features_data:
            logger.warning("Warning: Received empty features data")
            return

        async with self._context_lock:
            if self._global_context is None:
                # Initial creation of global context
                self._global_context = GlobalContext(
                        options=self.options,
                        features=features_data.get("features", {}),
                        saved_groups=features_data.get("savedGroups", {})
                )
            else:
                # Update existing global context
                self._global_context.features = features_data.get("features", {})
                self._global_context.saved_groups = features_data.get("savedGroups", {})
    
    async def close(self) -> None:
        """Clean resource cleanup"""
        if self._features_repository:
            await self._features_repository.stop_refresh()
        
        # Clear context to help garbage collection
        with self._context_lock:
            self._global_context = None

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def create_evaluation_context(self, user_context: UserContext) -> EvaluationContext:
        """Create evaluation context for feature evaluation"""
        with self._context_lock:
            if self._global_context is None:
                raise RuntimeError("GrowthBook client not properly initialized")
                
            return EvaluationContext(
                user=user_context,
                global_ctx=self._global_context,
                options=self.options,
                stack=StackContext(evaluted_features=set())
            )

    async def eval_feature(self, key: str, user_context: Optional[Dict[str, Any]] = None) -> Any:
        """Evaluate a feature with proper async context management"""
        if self._global_context is None:
            raise RuntimeError("GrowthBook client not properly initialized")

        async with self._context_lock:
            context = self.create_evaluation_context(
                UserContext(**user_context) if user_context else UserContext()
            )
            result = self._get_feature_result(key, context)
            context.stack.evaluted_features.add(key)
            return result["value"]

    async def is_on(self, key: str, user_context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a feature is enabled with proper async context management"""
        if self._global_context is None:
            raise RuntimeError("GrowthBook client not properly initialized")

        async with self._context_lock:
            context = self.create_evaluation_context(
                UserContext(**user_context) if user_context else UserContext()
            )
            result = self._get_feature_result(key, context)
            context.stack.evaluted_features.add(key)
            return result["on"]

    def _get_feature_result(self, key: str, context: EvaluationContext) -> Dict[str, Any]:
        """Get feature evaluation result"""
        
        # Use GrowthBook instance to evaluate
        result = core_eval_feature(
            key=key,
            features=context.global_ctx.features,
            attributes=context.user.attributes,
            saved_groups=context.global_ctx.saved_groups,
            stack=context.stack.evaluted_features
        )
        return {
            "value": result.value,
            "on": result.on,
            "off": result.off,
            "source": result.source
        }

    async def close(self) -> None:
        """Clean shutdown"""
        if self._features_repository:
            await self._features_repository.stop_refresh()