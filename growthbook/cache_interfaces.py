import asyncio
from abc import abstractmethod, ABC
from time import time
from typing import Optional, Dict

class AbstractFeatureCache(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Dict]:
        pass

    @abstractmethod
    def set(self, key: str, value: Dict, ttl: int) -> None:
        pass

    def clear(self) -> None:
        pass

class AbstractAsyncFeatureCache(ABC):
    """Abstract base class for async feature caching implementations"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Dict]:
        """
        Retrieve cached features by key.

        Args:
            key: Cache key

        Returns:
            Cached dictionary or None if not found/expired
        """
        pass

    @abstractmethod
    async def set(self, key: str, value: Dict, ttl: int) -> None:
        """
        Store features in cache with TTL.

        Args:
            key: Cache key
            value: Features dictionary to cache
            ttl: Time to live in seconds
        """
        pass

    async def clear(self) -> None:
        """Clear all cached entries (optional to override)"""
        pass

class CacheEntry(object):
    def __init__(self, value: Dict, ttl: int) -> None:
        self.value = value
        self.ttl = ttl
        self.expires = time() + ttl

    def update(self, value: Dict):
        self.value = value
        self.expires = time() + self.ttl


class InMemoryFeatureCache(AbstractFeatureCache):
    def __init__(self) -> None:
        self.cache: Dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            entry = self.cache[key]
            if entry.expires >= time():
                return entry.value
        return None

    def set(self, key: str, value: Dict, ttl: int) -> None:
        if key in self.cache:
            self.cache[key].update(value)
        else:
            self.cache[key] = CacheEntry(value, ttl)

    def clear(self) -> None:
        self.cache.clear()


class InMemoryAsyncFeatureCache(AbstractAsyncFeatureCache):
    """
    Async in-memory cache implementation.
    Uses the same CacheEntry structure but with async interface.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Dict]:
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry.expires >= time():
                    return entry.value
        return None

    async def set(self, key: str, value: Dict, ttl: int) -> None:
        async with self._lock:
            if key in self._cache:
                self._cache[key].update(value)
            else:
                self._cache[key] = CacheEntry(value, ttl)

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
