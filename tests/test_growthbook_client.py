from datetime import datetime 
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from growthbook_client import (
    GrowthBookClient, 
    Options, 
    UserContext, 
    FeatureRefreshStrategy,
    EnhancedFeatureRepository
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
async def test_initialization_with_retries(mock_options, mock_features_response):
    with patch('growthbook_client.EnhancedFeatureRepository.load_features_async') as mock_load:
        mock_load.side_effect = [
            Exception("Network error"),
            mock_features_response
        ]
        
        client = GrowthBookClient(mock_options)
        success = await client.initialize()
        
        assert success == False
        assert mock_load.call_count == 1

@pytest.mark.asyncio
async def test_sse_connection_lifecycle(mock_options, mock_features_response):
    with patch('growthbook_client.EnhancedFeatureRepository.load_features_async') as mock_load:
        mock_load.return_value = mock_features_response
        
        client = GrowthBookClient(
            Options(**{**mock_options.__dict__, 
                     "refresh_strategy": FeatureRefreshStrategy.SERVER_SENT_EVENTS})
        )
        
        with patch('growthbook_client.EnhancedFeatureRepository._maintain_sse_connection') as mock_sse:
            await client.initialize()
            assert mock_sse.called
            await client.close()