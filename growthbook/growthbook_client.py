#!/usr/bin/env python

from dataclasses import dataclass, field
import random
from typing import Any, Dict, List, Optional, Union, Callable
import json
import hashlib
import time
from enum import Enum
from typing import Set
import asyncio
import threading
import traceback
from datetime import datetime
from growthbook import AbstractStickyBucketService, FeatureRepository
import weakref
from contextlib import asynccontextmanager

@dataclass
class StackContext: 
    id: Optional[str] = None
    evaluted_features: Set[str] = field(default_factory=set)

class FeatureRefreshStrategy(Enum):
    STALE_WHILE_REVALIDATE = 'HTTP_REFRESH'
    SERVER_SENT_EVENTS = 'SSE'

@dataclass
class Options:
    url: Optional[str] = None
    api_host: Optional[str] = "https://cdn.growthbook.io"
    client_key: Optional[str] = None
    decryption_key: Optional[str] = None
    cache_ttl: int = 60
    enabled: bool = True
    qa_mode: bool = False
    enable_dev_mode: bool = False
    forced_variations: Dict[str, Any] = field(default_factory=dict)
    refresh_strategy: Optional[FeatureRefreshStrategy] = FeatureRefreshStrategy.STALE_WHILE_REVALIDATE
    sticky_bucket_service: AbstractStickyBucketService = None
    sticky_bucket_identifier_attributes: List[str] = None
    on_experiment_viewed=None

@dataclass
class UserContext:
    user_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    groups: Dict[str, str] = field(default_factory=dict)
    forced_variations: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GlobalContext:
    options: Options
    features: Dict[str, Any] = field(default_factory=dict)
    saved_groups: Dict[str, Any] = field(default_factory=dict)
    forced_variations: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationContext:
    user_context: UserContext
    global_context: GlobalContext
    options: Options
    stack: StackContext

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

class FeatureCache:
    """Memory-efficient feature cache using weak references"""
    def __init__(self):
        self._features = weakref.WeakValueDictionary()
        self._saved_groups = weakref.WeakValueDictionary()
        self._lock = threading.Lock()

    def update(self, features: Dict[str, Any], saved_groups: Dict[str, Any]) -> None:
        """Update cache with new values"""
        with self._lock:
            self._features.clear()
            self._saved_groups.clear()
            
            # Store dictionaries as weak references
            for key, value in features.items():
                self._features[key] = value
            for key, value in saved_groups.items():
                self._saved_groups[key] = value

    def get_current_state(self) -> Dict[str, Any]:
        """Get current cache state"""
        with self._lock:
            return {
                "features": dict(self._features),
                "savedGroups": dict(self._saved_groups)
            }

class EnhancedFeatureRepository(FeatureRepository, metaclass=SingletonMeta):
    def __init__(self, api_host: str, client_key: str, decryption_key: str = "", cache_ttl: int = 60):
        FeatureRepository.__init__(self)
        print(f"creating EnhancedFeatureRepository: {self}")
        print(f"Arguments: {api_host}, {client_key}, {decryption_key}, {cache_ttl}")
        print(f"Parent class: {FeatureRepository.__init__}")
        self._api_host = api_host
        self._client_key = client_key
        self._decryption_key = decryption_key
        self._cache_ttl = cache_ttl
        self._refresh_lock = threading.Lock()
        self._refresh_task = None
        self._stop_event = asyncio.Event()
        self._backoff = BackoffStrategy()
        self._feature_cache = FeatureCache()
        self._callbacks = weakref.WeakSet()  # Use WeakSet for callbacks
        self._last_successful_refresh = None
        self._refresh_in_progress = asyncio.Lock()

    @asynccontextmanager
    async def refresh_operation(self):
        """Context manager for feature refresh with proper cleanup"""
        if self._refresh_in_progress.locked():
            yield False
            return

        async with self._refresh_in_progress:
            try:
                yield True
                self._backoff.reset()
                self._last_successful_refresh = datetime.now()
            except Exception as e:
                delay = self._backoff.next_delay()
                print(f"Refresh failed, next attempt in {delay:.2f}s: {str(e)}")
                traceback.print_exc()
                raise

    async def _handle_feature_update(self, features_data: Dict[str, Any]) -> None:
        """Update features with memory optimization"""
        self._feature_cache.update(
            features_data.get("features", {}),
            features_data.get("savedGroups", {})
        )
        
        # Notify callbacks
        current_state = self._feature_cache.get_current_state()
        for callback in list(self._callbacks):
            try:
                await callback(current_state)
            except Exception:
                traceback.print_exc()

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
                await self.startAutoRefresh(self.api_host, self.client_key, handler)
            except Exception as e:
                if not self._stop_event.is_set():
                    delay = self._backoff.next_delay()
                    print(f"SSE connection lost, reconnecting in {delay:.2f}s: {str(e)}")
                    await asyncio.sleep(delay)

    async def _start_http_refresh(self, interval: int = 60) -> None:
        """Enhanced HTTP polling with backoff"""
        if self._refresh_task:
            return

        async def refresh_loop() -> None:
            while not self._stop_event.is_set():
                async with self.refresh_operation() as should_refresh:
                    if should_refresh:
                        response = await self.load_features_async(
                            self._api_host, self._client_key, self._decryption_key, self._cache_ttl
                        )
                        await self._handle_feature_update(response)
                    await asyncio.sleep(interval)

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
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            finally:
                self._refresh_task = None
                self._backoff.reset()
        self._stop_event.clear()

    def add_callback(self, callback: Callable) -> None:
        """Add callback using weak reference"""
        self._callbacks.add(callback)

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
        print(f"creating GrowthBookClient: {self}")
        
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
        self._context_lock = threading.Lock()
    
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
                print("Failed to load initial features")
                return False

            # Create global context with initial features
            await self._feature_update_callback(initial_features)
            
            # Set up callback for future updates
            self._features_repository.add_callback(self._feature_update_callback)
            
            # Start feature refresh
            await self._features_repository.start_feature_refresh(self.options.refresh_strategy)
            return True
            
        except Exception as e:
            print(f"Initialization failed: {str(e)}")
            traceback.print_exc()
            return False

    async def _feature_update_callback(self, features_data: Dict[str, Any]) -> None:
        """Handle feature updates and manage global context"""
        if not features_data:
            print("Warning: Received empty features data")
            return

        with self._context_lock:
            if self._global_context is None:
                # Initial creation of global context
                self._global_context = GlobalContext(
                        options=self.options,
                        features=features_data.get("features", {}),
                        saved_groups=features_data.get("savedGroups", {}),
                        forced_variations=self.options.forced_variations
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
                user_context=user_context,
                global_context=self._global_context,
                options=self.options,
                stack=StackContext(evaluted_features=set())
            )

    def eval_feature(self, key: str, user_context: Optional[Dict[str, Any]] = None) -> Any:
        """Evaluate a feature with proper context management"""
        if self._global_context is None:
            raise RuntimeError("GrowthBook client not properly initialized")

        with self._context_lock:
            context = self.create_evaluation_context(
                UserContext(**user_context) if user_context else UserContext()
            )
            result = self._get_feature_result(key, context)
            context.stack.evaluted_features.add(key)
            return result["value"]

    def is_on(self, key: str, user_context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a feature is enabled with proper context management"""
        if self._global_context is None:
            raise RuntimeError("GrowthBook client not properly initialized")

        with self._context_lock:
            context = self.create_evaluation_context(
                UserContext(**user_context) if user_context else UserContext()
            )
            result = self._get_feature_result(key, context)
            context.stack.evaluted_features.add(key)
            return result["on"]

    def _get_feature_result(self, key: str, context: EvaluationContext) -> Dict[str, Any]:
        """Get feature evaluation result"""
        if not self._global_context.options.enabled:
            return {"value": None, "on": False, "off": True, "source": "defaultValue"}

        feature = self._global_context.features.get(key)
        if not feature:
            return {"value": None, "on": False, "off": True, "source": "unknownFeature"}

        # Check forced variations from context
        if key in context.user_context.forced_variations:
            forced_value = context.user_context.forced_variations[key]
            return {
                "value": forced_value,
                "on": bool(forced_value),
                "off": not bool(forced_value),
                "source": "force"
            }

        # Use default value for now until we implement full evaluation logic
        default_value = feature.get("defaultValue")
        return {
            "value": default_value,
            "on": bool(default_value),
            "off": not bool(default_value),
            "source": "defaultValue"
        }

    async def close(self) -> None:
        """Clean shutdown"""
        if self._features_repository:
            await self._features_repository.stop_refresh()