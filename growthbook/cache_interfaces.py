from abc import abstractmethod, ABC
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
