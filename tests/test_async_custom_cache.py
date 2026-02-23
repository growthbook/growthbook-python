import pytest
import asyncio
from typing import Dict, Optional
from growthbook import GrowthBookClient, AbstractAsyncFeatureCache, InMemoryAsyncFeatureCache, Options, EnhancedFeatureRepository

@pytest.fixture(autouse=True)
async def cleanup():
    EnhancedFeatureRepository._instances = {}
    yield
    EnhancedFeatureRepository._instances = {}

class CustomAsyncCache(AbstractAsyncFeatureCache):
    def __init__(self):
        self.cache = {}
        self.get_calls = 0
        self.set_calls = 0

    async def get(self, key: str) -> Optional[Dict]:
        self.get_calls += 1
        return self.cache.get(key)

    async def set(self, key: str, value: Dict, ttl: int) -> None:
        self.set_calls += 1
        self.cache[key] = value
    
    async def clear(self) -> None:
        self.cache.clear()

@pytest.mark.asyncio
async def test_default_async_cache():
    # Test that default cache is InMemoryAsyncFeatureCache
    client = GrowthBookClient(options=Options(client_key="123", api_host="http://localhost"))
    # Access private repo to check cache (white-box testing)
    assert isinstance(client._features_repository.cache, InMemoryAsyncFeatureCache)
    
    # Clean up
    await client.close()

@pytest.mark.asyncio
async def test_custom_async_cache():
    # Force cleanup manually to debug singleton issue
    EnhancedFeatureRepository._instances = {}
    
    custom_cache = CustomAsyncCache()
    options = Options(
        client_key="123", 
        api_host="http://localhost",
        cache=custom_cache
    )
    client = GrowthBookClient(options=options)
    
    # Debug info
    print(f"DEBUG: Client cache option: {client.options.cache}")
    print(f"DEBUG: FeatureRepo cache: {client._features_repository.cache}")
    
    # Ensure options passed correctly
    assert client.options.cache is custom_cache
    
    assert client._features_repository.cache is custom_cache
    
    # Simulate loading features (mocking fetch would be better, but checking instance is good step 1)
    # Let's try to set something manually in the cache and see if load_features finds it
    key = "http://localhost::123"
    features_data = {"features": {"foo": {"defaultValue": True}}}
    await custom_cache.set(key, features_data, 60)
    
    # This should hit the cache and not fail due to network (since localhost might not be reachable)
    # load_features_async is what we want to test
    loaded = await client._features_repository.load_features_async("http://localhost", "123")
    assert loaded == features_data
    assert custom_cache.get_calls > 0
    
    await client.close()
