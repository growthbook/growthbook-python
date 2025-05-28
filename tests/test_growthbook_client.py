from datetime import datetime 
from unittest.mock import patch


try:
    from unittest.mock import AsyncMock
except ImportError:
    # For Python 3.7 compatibility
    from unittest.mock import MagicMock
    class AsyncMock(MagicMock):
        async def __call__(self, *args, **kwargs):
            return super(AsyncMock, self).__call__(*args, **kwargs)

from growthbook import InMemoryStickyBucketService
import pytest
import asyncio
import os
import json

from growthbook.common_types import Experiment, Options
from growthbook.growthbook_client import (
    GrowthBookClient, 
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

@pytest.fixture(autouse=True)
async def cleanup_singleton():
    """Clean up singleton instance between tests"""
    yield
    # Clear singleton instances after each test
    EnhancedFeatureRepository._instances = {}
    await asyncio.sleep(0.1)  # Allow tasks to clean up

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
    
    # Mock responses for load_features_async
    feature_updates = [
        {"features": {"feature1": {"defaultValue": 1}}, "savedGroups": {}},
        {"features": {"feature1": {"defaultValue": 2}}, "savedGroups": {}}
    ]
    
    mock_load = AsyncMock()
    mock_load.side_effect = [feature_updates[0], feature_updates[1], *[feature_updates[1]] * 10]
    
    try:
        with patch('growthbook.FeatureRepository.load_features_async', mock_load):
            # Start HTTP refresh with a short interval for testing
            refresh_task = asyncio.create_task(repo._start_http_refresh(interval=0.1))
            
            # Wait for two refresh cycles
            await asyncio.sleep(0.3)
            
            # Verify load_features_async was called at least twice
            assert mock_load.call_count == 3
            
            # Verify the latest feature state
            cache_state = repo._feature_cache.get_current_state()
            assert cache_state["features"]["feature1"] == {"defaultValue": 2}
    finally:
        # Ensure cleanup happens even if test fails
        await repo.stop_refresh()
        # Wait a bit to ensure task is fully cleaned up
        await asyncio.sleep(0.1)

@pytest.mark.asyncio
async def test_initialization_state_verification(mock_options, mock_features_response):
    """Verify feature state and callback registration after initialization"""
    callback_called = False
    features_received = None

    async def test_callback(features):
        nonlocal callback_called, features_received
        callback_called = True
        features_received = features

    with patch('growthbook.FeatureRepository.load_features_async') as mock_load:
        mock_load.return_value = mock_features_response
        
        client = GrowthBookClient(mock_options)
        client._features_repository.add_callback(test_callback)
        
        success = await client.initialize()
        await asyncio.sleep(0)
        
        assert success == True
        assert callback_called == True
        assert features_received == mock_features_response
        # Convert Feature objects to dict for comparison
        features_dict = {
            key: {"defaultValue": feature.defaultValue, "rules": feature.rules}
            for key, feature in client._global_context.features.items()
        }
        assert features_dict == mock_features_response["features"]

@pytest.mark.asyncio
async def test_sse_event_handling(mock_options):
    """Test SSE event handling and reconnection logic"""
    events = [
        {'type': 'features', 'data': {'features': {'feature1': {'defaultValue': 1}}}},
        {'type': 'ping', 'data': {}},  # Should be ignored
        {'type': 'features', 'data': {'features': {'feature1': {'defaultValue': 2}}}}
    ]

    async def mock_sse_handler(event_data):
        """Mock the SSE event handler to directly update feature cache"""
        if event_data['type'] == 'features':
            await client._features_repository._handle_feature_update(event_data['data'])

    with patch('growthbook.FeatureRepository.load_features_async') as mock_load:
        mock_load.return_value = {"features": {}, "savedGroups": {}}

        # Create options with SSE strategy
        sse_options = Options(
            api_host=mock_options.api_host,
            client_key=mock_options.client_key,
            refresh_strategy=FeatureRefreshStrategy.SERVER_SENT_EVENTS
        )
        
        client = GrowthBookClient(sse_options)

        try:
            await client.initialize()

            # Simulate SSE events directly
            for event in events:
                if event['type'] == 'features':
                    await client._features_repository._handle_feature_update(event['data'])

            # print(f"AFTER TEST: Current cache state: {client._features_repository._feature_cache.get_current_state()}")
            # Verify feature update happened
            assert client._features_repository._feature_cache.get_current_state()["features"]["feature1"]["defaultValue"] == 2
        finally:
            # Ensure we clean up the SSE connection
            await client.close()

@pytest.mark.asyncio
async def test_http_refresh_backoff():
    """Test HTTP refresh backoff strategy"""
    repo = EnhancedFeatureRepository(
        api_host="https://test.growthbook.io",
        client_key="test_key"
    )
    
    call_times = []
    success_time = None
    done = asyncio.Event()
    
    async def mock_load(*args, **kwargs):
        current_time = asyncio.get_event_loop().time()
        call_times.append(current_time)
        if len(call_times) < 3:
            raise ConnectionError("Network error")
        nonlocal success_time
        if not success_time:
            success_time = current_time
            if len(call_times) >= 4:
                done.set()
        return {"features": {}, "savedGroups": {}}
    
    try:
        with patch('growthbook.FeatureRepository.load_features_async', side_effect=mock_load):
            refresh_task = asyncio.create_task(repo._start_http_refresh(interval=0.1))
            try:
                await asyncio.wait_for(done.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                pass
            
            # Verify backoff behavior
            backoff_delays = [call_times[i] - call_times[i-1] for i in range(1, 3)]
            assert all(backoff_delays[i] > backoff_delays[i-1] for i in range(1, len(backoff_delays)))
            
            assert len(call_times) >= 4
            first_normal_delay = call_times[3] - call_times[2]
            assert 0.09 <= first_normal_delay <= 0.11
    finally:
        # Ensure cleanup happens even if test fails
        await repo.stop_refresh()
        # Wait a bit to ensure task is fully cleaned up
        await asyncio.sleep(0.1)

@pytest.mark.asyncio
async def test_concurrent_initialization():
    """Test concurrent initialization attempts"""
    shared_response = {
        "features": {
            "test-feature": {"defaultValue": 0}
        },
        "savedGroups": {}
    }
    loading_started = asyncio.Event()
    loading_wait = asyncio.Event()
    load_count = 0

    async def mock_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        loading_started.set()
        await loading_wait.wait()
        shared_response["features"]["test-feature"]["defaultValue"] += 1
        return shared_response

    with patch('growthbook.FeatureRepository.load_features_async', side_effect=mock_load):
        client = GrowthBookClient(Options(
            api_host="https://test.growthbook.io",
            client_key="test_key"
        ))
        
        try:
            # Start concurrent initializations
            init_tasks = [asyncio.create_task(client.initialize()) for _ in range(5)]
            
            # Wait for the first load attempt to start
            await loading_started.wait()
            await asyncio.sleep(0.1)
            loading_wait.set()
            
            results = await asyncio.gather(*init_tasks, return_exceptions=True)
            
            # Verify results
            assert all(r == True for r in results)
            assert load_count > 1
            final_cache = client._features_repository._feature_cache.get_current_state()
            assert final_cache["features"]["test-feature"]["defaultValue"] == 6
        finally:
            # Ensure proper cleanup
            await client.close()
            # Wait for any pending tasks to complete
            await asyncio.sleep(0.1)
            # Get all tasks and cancel any remaining ones
            for task in asyncio.all_tasks():
                if not task.done() and task != asyncio.current_task():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

def pytest_generate_tests(metafunc):
    """Generate test cases from cases.json"""
    # Skip if the test doesn't need case data
    if not any(x.endswith('_data') for x in metafunc.fixturenames):
        return

    folder = os.path.abspath(os.path.dirname(__file__))
    jsonfile = os.path.join(folder, "cases.json")
    with open(jsonfile) as file:
        data = json.load(file)

    # Map test functions to their data
    test_data_map = {
        'test_eval_feature': 'feature',
        'test_experiment_run': 'run',
        'test_sticky_bucket': 'stickyBucket'
    }

    for func, data_key in test_data_map.items():
        fixture_name = f"{func}_data"
        if fixture_name in metafunc.fixturenames:
            metafunc.parametrize(fixture_name, data.get(data_key, []))

@pytest.mark.asyncio
async def test_eval_feature(test_eval_feature_data, base_client_setup):
    """Test feature evaluation similar to test_feature in test_growthbook.py"""
    _, ctx, key, expected = test_eval_feature_data
   
    # Get base setup
    user_attrs, client_opts, features_data = base_client_setup(ctx)

    # Clear any existing singleton instances
    EnhancedFeatureRepository._instances = {}
    
    try:
        # Set up mocks for both FeatureRepository and EnhancedFeatureRepository
        with patch('growthbook.FeatureRepository.load_features_async', 
                  new_callable=AsyncMock, return_value=features_data), \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh',
                  new_callable=AsyncMock), \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh',
                  new_callable=AsyncMock):
            
            # Create and initialize client
            async with GrowthBookClient(Options(**client_opts)) as client:
                result = await client.eval_feature(key, UserContext(**user_attrs))
                
                if "experiment" in expected:
                    expected["experiment"] = Experiment(**expected["experiment"]).to_dict()
                
                assert result.to_dict() == expected
    except Exception as e:
        print(f"Error during test execution: {str(e)}")
        raise
    finally:
        await client.close()
        await asyncio.sleep(0.1)

@pytest.mark.asyncio
async def test_experiment_run(test_experiment_run_data, base_client_setup):
    """Test experiment running similar to test_run in test_growthbook.py"""
    _, ctx, exp, value, inExperiment, hashUsed = test_experiment_run_data
    
    # Get base setup
    user_attrs, client_opts, features_data = base_client_setup(ctx)

    # Clear any existing singleton instances
    EnhancedFeatureRepository._instances = {}
    
    try:
        # Set up mocks for both FeatureRepository and EnhancedFeatureRepository
        with patch('growthbook.FeatureRepository.load_features_async', 
                  new_callable=AsyncMock, return_value=features_data), \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh',
                  new_callable=AsyncMock), \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh',
                  new_callable=AsyncMock):
            
            # Create and initialize client
            async with GrowthBookClient(Options(**client_opts)) as client:
                result = await client.run(Experiment(**exp), UserContext(**user_attrs))
            
                # Verify experiment results
                assert result.value == value
                assert result.inExperiment == inExperiment
                assert result.hashUsed == hashUsed
    except Exception as e:
        print(f"Error during test execution: {str(e)}")
        raise
    finally:
        await client.close()
        await asyncio.sleep(0.1)

@pytest.mark.asyncio
async def test_feature_methods():
    """Test feature helper methods (isOn, isOff, getFeatureValue)"""
    features_data = {
        "features": {
            "featureOn": {"defaultValue": 12},
            "featureNone": {"defaultValue": None},
            "featureOff": {"defaultValue": 0}
        },
        "savedGroups": {}
    }
    
    # Simple client options
    client_opts = {
        'api_host': "https://localhost.growthbook.io",
        'client_key': "test-key",
        'enabled': True
    }

    # Clear any existing singleton instances
    EnhancedFeatureRepository._instances = {}
    user_context = UserContext(attributes={"id": "user-1"})

    try:
        # Set up mocks for both FeatureRepository and EnhancedFeatureRepository
        with patch('growthbook.FeatureRepository.load_features_async', 
                  new_callable=AsyncMock, return_value=features_data), \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh',
                  new_callable=AsyncMock), \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh',
                  new_callable=AsyncMock):
            
            # Create and initialize client
            async with GrowthBookClient(Options(**client_opts)) as client:
                # Test isOn
                assert await client.is_on("featureOn", user_context) is True
                assert await client.is_on("featureOff", user_context) is False
                assert await client.is_on("featureNone", user_context) is False

                # Test isOff
                assert await client.is_off("featureOn", user_context) is False
                assert await client.is_off("featureOff", user_context) is True
                assert await client.is_off("featureNone", user_context) is True

                # Test getFeatureValue
                assert await client.get_feature_value("featureOn", 15, user_context) == 12
                assert await client.get_feature_value("featureOff", 10, user_context) == 0
                assert await client.get_feature_value("featureNone", 10, user_context) == 10
                assert await client.get_feature_value("nonexistent", "default", user_context) == "default"
    except Exception as e:
        print(f"Error during test execution: {str(e)}")
        raise
    finally:
        await client.close()
        await asyncio.sleep(0.1)

@pytest.fixture
def base_client_setup():
    """Common setup for client tests"""
    def _setup(ctx):
        # Separate client options from user context
        user_attrs = {
            "attributes": ctx.get("attributes", {}),
            "url": ctx.get("url", ""),
            "groups": ctx.get("groups", {}),
            "forced_variations": ctx.get("forcedVariations", {})
        }
        
        # Base client options
        client_opts = {
            'api_host': "https://localhost.growthbook.io",
            'client_key': "test-key",
            'enabled': ctx.get("enabled", True),
            'qa_mode': ctx.get("qaMode", False)
        }
        
        # Features data structure
        features_data = {
            "features": ctx.get("features", {}),
            "savedGroups": ctx.get("savedGroups", {})
        }
        
        return user_attrs, client_opts, features_data
    return _setup

@pytest.mark.asyncio
async def test_sticky_bucket(test_sticky_bucket_data, base_client_setup):
    """Test sticky bucket functionality in GrowthBookClient"""
    _, ctx, initial_docs, key, expected_result, expected_docs = test_sticky_bucket_data

    # Initialize sticky bucket service with test data
    service = InMemoryStickyBucketService()
    
    # Add initial documents to the service
    for doc in initial_docs:
        service.save_assignments(doc)
    
    # Handle sticky bucket identifier attributes mapping
    if 'stickyBucketIdentifierAttributes' in ctx:
        ctx['sticky_bucket_identifier_attributes'] = ctx['stickyBucketIdentifierAttributes']
        ctx.pop('stickyBucketIdentifierAttributes')
        
    # Handle sticky bucket assignment docs
    if 'stickyBucketAssignmentDocs' in ctx:
        service.docs = ctx['stickyBucketAssignmentDocs']
        ctx.pop('stickyBucketAssignmentDocs')
    
    # Get base setup
    user_attrs, client_opts, features_data = base_client_setup(ctx)
    
    # Add sticky bucket service to client options
    client_opts['sticky_bucket_service'] = service
    
    # Clear any existing singleton instances
    EnhancedFeatureRepository._instances = {}
    
    try:
        # Set up mocks
        with patch('growthbook.FeatureRepository.load_features_async', 
                  new_callable=AsyncMock, return_value=features_data), \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh',
                  new_callable=AsyncMock), \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh',
                  new_callable=AsyncMock):
             
            # Create and initialize client
            async with GrowthBookClient(Options(**client_opts)) as client:
                # Evaluate feature
                result = await client.eval_feature(key, UserContext(**user_attrs))
                
                # Verify experiment result
                if not result.experimentResult:
                    assert None == expected_result
                else:
                    assert result.experimentResult.to_dict() == expected_result
  
                # Verify sticky bucket assignments - check each expected doc individually
                for doc_key, expected_doc in expected_docs.items():
                    assert service.docs[doc_key] == expected_doc
    except Exception as e:
        print(f"Error during test execution: {str(e)}")
        raise
    finally:
        await client.close()
        service.destroy()
        await asyncio.sleep(0.1)

async def getTrackingMock(client: GrowthBookClient):
    """Helper function to mock tracking for tests"""
    calls = []

    def track(experiment, result):
        calls.append([experiment, result])

    client.options.on_experiment_viewed = track
    return lambda: calls

@pytest.mark.asyncio
async def test_tracking():
    """Test experiment tracking behavior"""
    # Create client with minimal options
    client = GrowthBookClient(Options(
        api_host="https://localhost.growthbook.io",
        client_key="test-key",
        enabled=True
    ))

    getMockedCalls = await getTrackingMock(client)

    # Create test experiments
    exp1 = Experiment(
        key="my-tracked-test",
        variations=[0, 1],
    )
    exp2 = Experiment(
        key="my-other-tracked-test",
        variations=[0, 1],
    )

    # Create user context
    user_context = UserContext(attributes={"id": "1"})

    try:
        # Set up mocks for feature repository
        with patch('growthbook.FeatureRepository.load_features_async', 
                  new_callable=AsyncMock, return_value={"features": {}, "savedGroups": {}}), \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh',
                  new_callable=AsyncMock), \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh',
                  new_callable=AsyncMock):
            
            # Initialize client
            await client.initialize()

            # Run experiments
            res1 = await client.run(exp1, user_context)
            await client.run(exp1, user_context)  # Should not track duplicate
            await client.run(exp1, user_context)  # Should not track duplicate
            res4 = await client.run(exp2, user_context)
            
            # Change user attributes
            user_context.attributes = {"id": "2"}
            res5 = await client.run(exp2, user_context)

            # Verify tracking calls
            calls = getMockedCalls()
            assert len(calls) == 3, "Expected exactly 3 tracking calls"
            assert calls[0] == [exp1, res1], "First tracking call mismatch"
            assert calls[1] == [exp2, res4], "Second tracking call mismatch"
            assert calls[2] == [exp2, res5], "Third tracking call mismatch"

    finally:
        await client.close()

@pytest.mark.asyncio
async def test_handles_tracking_errors():
    """Test graceful handling of tracking callback errors"""
    client = GrowthBookClient(Options(
        api_host="https://localhost.growthbook.io",
        client_key="test-key",
        enabled=True
    ))

    # Set up tracking callback that raises an error
    def failing_track(experiment, result):
        raise Exception("Tracking failed")

    client.options.on_experiment_viewed = failing_track

    # Create test experiment
    exp = Experiment(
        key="error-test",
        variations=[0, 1],
    )
    user_context = UserContext(attributes={"id": "1"})

    try:
        # Set up mocks
        with patch('growthbook.FeatureRepository.load_features_async', 
                  new_callable=AsyncMock, return_value={"features": {}, "savedGroups": {}}), \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh',
                  new_callable=AsyncMock), \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh',
                  new_callable=AsyncMock):
            
            await client.initialize()

            # Should not raise exception despite tracking error
            result = await client.run(exp, user_context)
            assert result is not None, "Experiment should run despite tracking error"

    finally:
        await client.close()