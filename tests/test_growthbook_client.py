from datetime import datetime 
from unittest.mock import patch, AsyncMock
import pytest
import asyncio
import os
from growthbook.growthbook_client import (
    GrowthBookClient, 
    Options, 
    UserContext, 
    FeatureRefreshStrategy,
    EnhancedFeatureRepository,
    WeakRefWrapper
)

@pytest.fixture
def mock_features_response():
    return {
        "features": {
            "test-feature": {
                "defaultValue": True,
                "rules": []
            }
        },
        "savedGroups": {}
    }

@pytest.fixture
def mock_options():
    return Options(
        api_host="https://test.growthbook.io",
        client_key="test_key",
        decryption_key="test_decrypt",
        cache_ttl=60,
        enabled=True,
        refresh_strategy=FeatureRefreshStrategy.STALE_WHILE_REVALIDATE
    )


@pytest.fixture
def mock_sse_data():
    return {
        'type': 'features',
        'data': {
            'features': {
                'feature-1': {'defaultValue': True},
                'feature-2': {'defaultValue': False}
            }
        }
    }

@pytest.mark.asyncio
async def test_initialization_for_failure(mock_options):
    with patch('growthbook.growthbook_client.EnhancedFeatureRepository.load_features_async') as mock_load:
        mock_load.side_effect = Exception("Network error")
        client = GrowthBookClient(mock_options)
        success = await client.initialize()
        assert success == False
        assert mock_load.call_count == 1

@pytest.mark.asyncio
async def test_sse_connection_lifecycle(mock_options, mock_features_response):
    with patch('growthbook.growthbook_client.EnhancedFeatureRepository.load_features_async') as mock_load:
        mock_load.return_value = mock_features_response
        
        client = GrowthBookClient(
            Options(**{**mock_options.__dict__, 
                     "refresh_strategy": FeatureRefreshStrategy.SERVER_SENT_EVENTS})
        )
        
        with patch('growthbook.growthbook_client.EnhancedFeatureRepository._maintain_sse_connection') as mock_sse:
            await client.initialize()
            assert mock_sse.called
            await client.close()

@pytest.mark.asyncio
async def test_feature_repository_load():
    repo = EnhancedFeatureRepository(
        api_host="https://test.growthbook.io",
        client_key="test_key"
    )
    features_response = {
        "features": {"test-feature": {"defaultValue": True}},
        "savedGroups": {}
    }
    
    with patch('growthbook.FeatureRepository.load_features_async') as mock_load:
        mock_load.return_value = features_response
        result = await repo.load_features_async(api_host="", client_key="")
        assert result == features_response

@pytest.mark.asyncio
async def test_initialize_success(mock_options, mock_features_response):
    with patch('growthbook.growthbook_client.EnhancedFeatureRepository.load_features_async') as mock_load, \
         patch('growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh', return_value=None):
        mock_load.return_value = mock_features_response
        
        client = GrowthBookClient(mock_options)
        success = await client.initialize()

        # result = client.eval_feature('test-feature')
        # print(f'result= {result}')
        assert success == True

@pytest.mark.asyncio
async def test_refresh_operation_lock():
    """Verify refresh_operation lock prevents concurrent refreshes"""
    repo = EnhancedFeatureRepository(
        api_host="https://test.growthbook.io",
        client_key="test_key"
    )
    
    results = []
    async def refresh_task():
        async with repo.refresh_operation() as should_refresh:
            results.append(should_refresh)
            await asyncio.sleep(0.1)  # Simulate work
            return should_refresh
            
    await asyncio.gather(*[refresh_task() for _ in range(5)])
    assert sum(1 for r in results if r) == 1  # Only one task should get True
    assert sum(1 for r in results if not r) == 4  # Rest should get False


@pytest.mark.asyncio
async def test_concurrent_feature_updates():
    """Verify FeatureCache thread safety during concurrent updates"""
    repo = EnhancedFeatureRepository(
        api_host="https://test.growthbook.io",
        client_key="test_key"
    )
    features = {f"feature-{i}": {"defaultValue": i} for i in range(10)}
    
    async def update_features(feature_subset):
        await repo._handle_feature_update({"features": feature_subset, "savedGroups": {}})
            
    await asyncio.gather(*[
        update_features({k: features[k]}) 
        for k in features
    ])
    
    cache_state = repo._feature_cache.get_current_state()
    # Verify all features were properly stored
    assert cache_state["features"] == features
    assert cache_state["savedGroups"] == {}

@pytest.mark.asyncio
async def test_callback_thread_safety():
    """Verify callback invocations are thread-safe"""
    repo = EnhancedFeatureRepository(
        api_host="https://test.growthbook.io",
        client_key="test_key"
    )
    
    received_callbacks = []
    async def test_callback(features):
        received_callbacks.append(features)
        
    repo.add_callback(test_callback)
    test_features = [{"features": {f"f{i}": {"value": i}}, "savedGroups": {}} for i in range(5)]
    
    await asyncio.gather(*[
        repo._handle_feature_update(update) 
        for update in test_features
    ])
    
    assert len(received_callbacks) == 5

@pytest.mark.asyncio
async def test_http_refresh():
    """Verify HTTP refresh mechanism works correctly"""
    repo = EnhancedFeatureRepository(
        api_host="https://test.growthbook.io",
        client_key="test_key"
    )
    
    print(f"BEFORE TEST: Current cache state: {repo._feature_cache.get_current_state()}")
    # Mock responses for load_features_async
    feature_updates = [
        {"features": {"feature1": {"defaultValue": 1}}, "savedGroups": {}},
        {"features": {"feature1": {"defaultValue": 2}}, "savedGroups": {}}
    ]
    
    mock_load = AsyncMock()
    mock_load.side_effect = [feature_updates[0], feature_updates[1], *[feature_updates[1]] * 10]  # Provide more responses
    
    with patch('growthbook.FeatureRepository.load_features_async', mock_load):
        # Start HTTP refresh with a short interval for testing
        refresh_task = asyncio.create_task(repo._start_http_refresh(interval=0.1))
        
        # Wait for two refresh cycles
        await asyncio.sleep(0.3)
        
        # Stop the refresh
        await repo.stop_refresh()
        
        # Verify load_features_async was called at least twice
        assert mock_load.call_count == 3
        
        # Verify the latest feature state
        cache_state = repo._feature_cache.get_current_state()
        assert cache_state["features"]["feature1"] == {"defaultValue": 2}