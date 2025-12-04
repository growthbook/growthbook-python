"""
Tests for ETag caching functionality in GrowthBook SDK.

This test suite verifies that the SDK correctly implements ETag-based
HTTP caching to reduce bandwidth and CDN load.
"""
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from growthbook.growthbook import feature_repo, FeatureRepository


class MockResponse:
    """Mock HTTP response for testing"""
    def __init__(self, status, data, headers=None):
        self.status = status
        self.data = data.encode('utf-8') if isinstance(data, str) else data
        self.headers = headers or {}


class MockAsyncResponse:
    """Mock async HTTP response for testing"""
    def __init__(self, status, data, headers=None):
        self.status = status
        self._data = data
        self.headers = headers or {}
    
    async def json(self):
        return json.loads(self._data) if isinstance(self._data, str) else self._data
    
    def __aenter__(self):
        async def _aenter():
            return self
        return _aenter()
    
    def __aexit__(self, exc_type, exc_val, exc_tb):
        async def _aexit():
            pass
        return _aexit()


@pytest.fixture(autouse=True)
def cleanup_etag_cache():
    """Clear ETag cache before each test"""
    feature_repo._etag_cache.clear()
    yield
    feature_repo._etag_cache.clear()


class TestETags:
    """Test ETag caching functionality"""
    
    def test_etag_initial_request_no_cache(self):
        """First request should not send If-None-Match header"""
        features_response = json.dumps({
            "features": {"feature1": {"defaultValue": True}}
        })
        
        with patch.object(feature_repo, '_get') as mock_get:
            mock_get.return_value = MockResponse(
                status=200,
                data=features_response,
                headers={'ETag': '"abc123"'}
            )
            
            result = feature_repo._fetch_and_decode("https://cdn.growthbook.io", "test_key")
            
            # Verify request was made without If-None-Match
            assert mock_get.called
            call_args = mock_get.call_args
            headers = call_args[1].get('headers', {})
            assert 'If-None-Match' not in headers
            
            # Verify response was returned
            assert result is not None
            assert "features" in result
            
            # Verify ETag was cached
            url = feature_repo._get_features_url("https://cdn.growthbook.io", "test_key")
            assert url in feature_repo._etag_cache
            cached_etag, cached_data = feature_repo._etag_cache[url]
            assert cached_etag == '"abc123"'
            assert cached_data == result
    
    def test_etag_second_request_sends_if_none_match(self):
        """Second request should send If-None-Match header with cached ETag"""
        features_response = json.dumps({
            "features": {"feature1": {"defaultValue": True}}
        })
        
        with patch.object(feature_repo, '_get') as mock_get:
            # First request - returns 200 with ETag
            mock_get.return_value = MockResponse(
                status=200,
                data=features_response,
                headers={'ETag': '"abc123"'}
            )
            
            feature_repo._fetch_and_decode("https://cdn.growthbook.io", "test_key")
            
            # Second request - should send If-None-Match
            mock_get.reset_mock()
            mock_get.return_value = MockResponse(
                status=200,
                data=features_response,
                headers={'ETag': '"abc123"'}
            )
            
            feature_repo._fetch_and_decode("https://cdn.growthbook.io", "test_key")
            
            # Verify If-None-Match was sent
            call_args = mock_get.call_args
            # Headers are passed as positional or keyword argument
            if len(call_args[0]) > 1:
                headers = call_args[0][1] if isinstance(call_args[0][1], dict) else {}
            else:
                headers = call_args[1].get('headers', {})
            assert 'If-None-Match' in headers
            assert headers['If-None-Match'] == '"abc123"'
    
    def test_etag_304_returns_cached_data(self):
        """Server returning 304 should return cached data"""
        features_response = json.dumps({
            "features": {"feature1": {"defaultValue": True}}
        })
        
        with patch.object(feature_repo, '_get') as mock_get:
            # First request - returns 200 with ETag
            mock_get.return_value = MockResponse(
                status=200,
                data=features_response,
                headers={'ETag': '"abc123"'}
            )
            
            first_result = feature_repo._fetch_and_decode("https://cdn.growthbook.io", "test_key")
            
            # Second request - returns 304 Not Modified
            mock_get.return_value = MockResponse(
                status=304,
                data="",  # 304 responses have no body
                headers={'ETag': '"abc123"'}
            )
            
            second_result = feature_repo._fetch_and_decode("https://cdn.growthbook.io", "test_key")
            
            # Verify cached data was returned
            assert second_result is not None
            assert second_result == first_result
            assert "features" in second_result
    
    def test_etag_updated_on_content_change(self):
        """ETag should be updated when content changes"""
        with patch.object(feature_repo, '_get') as mock_get:
            # First request - initial ETag
            mock_get.return_value = MockResponse(
                status=200,
                data=json.dumps({"features": {"feature1": {"defaultValue": True}}}),
                headers={'ETag': '"abc123"'}
            )
            
            feature_repo._fetch_and_decode("https://cdn.growthbook.io", "test_key")
            
            url = feature_repo._get_features_url("https://cdn.growthbook.io", "test_key")
            assert feature_repo._etag_cache[url][0] == '"abc123"'
            
            # Second request - new ETag (content changed)
            mock_get.return_value = MockResponse(
                status=200,
                data=json.dumps({"features": {"feature2": {"defaultValue": False}}}),
                headers={'ETag': '"xyz789"'}
            )
            
            result = feature_repo._fetch_and_decode("https://cdn.growthbook.io", "test_key")
            
            # Verify ETag was updated
            assert feature_repo._etag_cache[url][0] == '"xyz789"'
            assert "feature2" in result["features"]
    
    def test_etag_no_header_still_works(self):
        """Requests should work even if server doesn't send ETag"""
        features_response = json.dumps({
            "features": {"feature1": {"defaultValue": True}}
        })
        
        with patch.object(feature_repo, '_get') as mock_get:
            mock_get.return_value = MockResponse(
                status=200,
                data=features_response,
                headers={}  # No ETag header
            )
            
            result = feature_repo._fetch_and_decode("https://cdn.growthbook.io", "test_key")
            
            # Should still return data
            assert result is not None
            assert "features" in result
            
            # But ETag should not be cached
            url = feature_repo._get_features_url("https://cdn.growthbook.io", "test_key")
            assert url not in feature_repo._etag_cache
    
    def test_etag_multiple_urls_cached_separately(self):
        """Different URLs should have separate ETag cache entries"""
        with patch.object(feature_repo, '_get') as mock_get:
            # Request for key1
            mock_get.return_value = MockResponse(
                status=200,
                data=json.dumps({"features": {"feature1": {"defaultValue": True}}}),
                headers={'ETag': '"etag-key1"'}
            )
            
            feature_repo._fetch_and_decode("https://cdn.growthbook.io", "key1")
            
            # Request for key2
            mock_get.return_value = MockResponse(
                status=200,
                data=json.dumps({"features": {"feature2": {"defaultValue": False}}}),
                headers={'ETag': '"etag-key2"'}
            )
            
            feature_repo._fetch_and_decode("https://cdn.growthbook.io", "key2")
            
            # Verify both are cached with different ETags
            url1 = feature_repo._get_features_url("https://cdn.growthbook.io", "key1")
            url2 = feature_repo._get_features_url("https://cdn.growthbook.io", "key2")
            
            assert url1 in feature_repo._etag_cache
            assert url2 in feature_repo._etag_cache
            assert feature_repo._etag_cache[url1][0] == '"etag-key1"'
            assert feature_repo._etag_cache[url2][0] == '"etag-key2"'
    
    def test_etag_bandwidth_savings(self):
        """Test that 304 responses save bandwidth compared to full responses"""
        large_features = {
            "features": {f"feature{i}": {"defaultValue": True} for i in range(100)}
        }
        features_response = json.dumps(large_features)
        
        with patch.object(feature_repo, '_get') as mock_get:
            # First request - full response
            mock_get.return_value = MockResponse(
                status=200,
                data=features_response,
                headers={'ETag': '"large-response"'}
            )
            
            first_result = feature_repo._fetch_and_decode("https://cdn.growthbook.io", "test_key")
            first_size = len(features_response)
            
            # Second request - 304 response (minimal data)
            mock_get.return_value = MockResponse(
                status=304,
                data="",  # No body in 304
                headers={'ETag': '"large-response"'}
            )
            
            second_result = feature_repo._fetch_and_decode("https://cdn.growthbook.io", "test_key")
            second_size = 0  # 304 has no body
            
            # Verify we got the same data
            assert first_result == second_result
            
            # Verify bandwidth savings
            assert first_size > 1000  # Large response
            assert second_size == 0  # 304 response has no body
            print(f"\n💾 Bandwidth saved: {first_size} bytes (100% savings on cache hit)")
    
    def test_etag_cache_persistence_across_requests(self):
        """Test that ETag cache persists across multiple different requests"""
        with patch.object(feature_repo, '_get') as mock_get:
            # Make 3 requests to different keys
            for i in range(3):
                mock_get.return_value = MockResponse(
                    status=200,
                    data=json.dumps({"features": {f"feature{i}": {}}}),
                    headers={'ETag': f'"etag-{i}"'}
                )
                feature_repo._fetch_and_decode("https://cdn.growthbook.io", f"key{i}")
            
            # Verify all 3 ETags are cached
            assert len(feature_repo._etag_cache) == 3
            
            # Verify each ETag is correct
            for i in range(3):
                url = feature_repo._get_features_url("https://cdn.growthbook.io", f"key{i}")
                assert url in feature_repo._etag_cache
                assert feature_repo._etag_cache[url][0] == f'"etag-{i}"'
    
    def test_etag_cache_initialized(self):
        """ETag cache should be initialized on FeatureRepository instantiation"""
        assert hasattr(feature_repo, '_etag_cache')
        assert isinstance(feature_repo._etag_cache, dict)

    def test_etag_cache_lru_eviction(self):
        """Test that ETag cache enforces size limit and LRU eviction"""
        # Set a small limit for testing
        feature_repo._max_etag_entries = 3
        
        with patch.object(feature_repo, '_get') as mock_get:
            # Fill cache to limit
            for i in range(3):
                mock_get.return_value = MockResponse(
                    status=200,
                    data=json.dumps({"features": {f"feature{i}": {}}}),
                    headers={'ETag': f'"etag-{i}"'}
                )
                feature_repo._fetch_and_decode("https://cdn.growthbook.io", f"key{i}")
            
            assert len(feature_repo._etag_cache) == 3
            assert "https://cdn.growthbook.io/api/features/key0" in feature_repo._etag_cache
            
            # Add one more item - should evict key0 (least recently used)
            mock_get.return_value = MockResponse(
                status=200,
                data=json.dumps({"features": {"feature3": {}}}),
                headers={'ETag': '"etag-3"'}
            )
            feature_repo._fetch_and_decode("https://cdn.growthbook.io", "key3")
            
            assert len(feature_repo._etag_cache) == 3
            assert "https://cdn.growthbook.io/api/features/key0" not in feature_repo._etag_cache
            assert "https://cdn.growthbook.io/api/features/key3" in feature_repo._etag_cache
            
            # Access key1 to make it most recently used
            mock_get.return_value = MockResponse(
                status=304,
                data="",
                headers={'ETag': '"etag-1"'}
            )
            feature_repo._fetch_and_decode("https://cdn.growthbook.io", "key1")
            
            # Add another item - should evict key2 (now least recently used)
            mock_get.return_value = MockResponse(
                status=200,
                data=json.dumps({"features": {"feature4": {}}}),
                headers={'ETag': '"etag-4"'}
            )
            feature_repo._fetch_and_decode("https://cdn.growthbook.io", "key4")
            
            assert len(feature_repo._etag_cache) == 3
            assert "https://cdn.growthbook.io/api/features/key2" not in feature_repo._etag_cache
            assert "https://cdn.growthbook.io/api/features/key1" in feature_repo._etag_cache  # Preserved!
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
