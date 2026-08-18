from datetime import datetime 
from unittest.mock import patch

import pytest_asyncio

try:
    from unittest.mock import AsyncMock
except ImportError:
    # For Python 3.7 compatibility
    from unittest.mock import MagicMock
    class AsyncMock(MagicMock):
        async def __call__(self, *args, **kwargs):
            return super(AsyncMock, self).__call__(*args, **kwargs)

from growthbook import InMemoryStickyBucketService, AbstractAsyncStickyBucketService
import pytest
import asyncio
import os
import json
import threading
from typing import Any, Dict, Optional

from growthbook.common_types import Experiment, Options
from growthbook.growthbook_client import (
    GrowthBookClient,
    UserContext,
    FeatureRefreshStrategy,
    EnhancedFeatureRepository
)


class AsyncInMemoryStickyBucketService(AbstractAsyncStickyBucketService):
    """Async mirror of InMemoryStickyBucketService, instrumented for tests.

    Optional asyncio.Event gates make concurrency tests deterministic:
    a gated call parks until the test sets the event — no wall-clock sleeps.
    """

    def __init__(self,
                 fetch_gate: Optional[asyncio.Event] = None,
                 save_gate: Optional[asyncio.Event] = None) -> None:
        self.docs: Dict[str, Dict] = {}
        self.fetch_gate = fetch_gate
        self.save_gate = save_gate
        self.get_all_calls = 0
        self.save_calls = 0

    async def get_assignments(self, attributeName: str, attributeValue: str) -> Optional[Dict]:
        return self.docs.get(self.get_key(attributeName, attributeValue), None)

    async def save_assignments(self, doc: Dict) -> None:
        self.save_calls += 1
        if self.save_gate:
            await self.save_gate.wait()
        self.docs[self.get_key(doc["attributeName"], doc["attributeValue"])] = doc

    async def get_all_assignments(self, attributes: Dict[str, str]) -> Dict[str, Dict]:
        self.get_all_calls += 1
        if self.fetch_gate:
            await self.fetch_gate.wait()
        return await super().get_all_assignments(attributes)

    def destroy(self) -> None:
        self.docs.clear()


class CountingStickyBucketService(InMemoryStickyBucketService):
    """Sync service instrumented with call counts and optional threading.Event
    gates. A gated call BLOCKS its thread until the test sets the event, which
    lets tests prove the call is not running on the event loop: if it were,
    the loop-side code that sets the event could never run and the gate would
    time out."""

    GATE_TIMEOUT = 5  # generous upper bound; only reached on real deadlock

    def __init__(self,
                 fetch_gate: Optional[threading.Event] = None,
                 save_gate: Optional[threading.Event] = None) -> None:
        super().__init__()
        self.fetch_gate = fetch_gate
        self.save_gate = save_gate
        self.fetch_started = threading.Event()
        self.get_all_calls = 0

    def get_all_assignments(self, attributes: Dict[str, str]) -> Dict[str, Dict]:
        self.get_all_calls += 1
        self.fetch_started.set()
        if self.fetch_gate:
            assert self.fetch_gate.wait(timeout=self.GATE_TIMEOUT), \
                "fetch gate never released — event loop was blocked"
        return super().get_all_assignments(attributes)

    def save_assignments(self, doc: Dict) -> None:
        if self.save_gate:
            assert self.save_gate.wait(timeout=self.GATE_TIMEOUT), \
                "save gate never released — event loop was blocked"
        super().save_assignments(doc)

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

@pytest_asyncio.fixture(autouse=True)
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
    with patch('growthbook.growthbook_client.EnhancedFeatureRepository.load_features_async', 
               new_callable=AsyncMock, return_value=mock_features_response) as mock_load:
        
        client = GrowthBookClient(
            Options(**{**mock_options.__dict__, 
                     "refresh_strategy": FeatureRefreshStrategy.SERVER_SENT_EVENTS})
        )
        
        # `startAutoRefresh` is synchronous and should be invoked as part of SSE start-up.
        # `stopAutoRefresh` should be called during shutdown to stop/join the SSE thread.
        with patch('growthbook.growthbook_client.EnhancedFeatureRepository.startAutoRefresh') as mock_start, \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.stopAutoRefresh') as mock_stop:
            await client.initialize()
            # Allow the SSE lifecycle task to start and invoke startAutoRefresh
            await asyncio.sleep(0.1)
            assert mock_start.called
            
            # Verify the thread created is a daemon thread (if possible without real start)
            # Since we mock startAutoRefresh, we can't check the real thread here.
            # But we can check that SSEClient is initialized correctly if we don't mock it all.
            
            await client.close()
            assert mock_stop.called

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
    
    with patch('growthbook.FeatureRepository.load_features_async', 
               new_callable=AsyncMock, return_value=features_response) as mock_load:
        result = await repo.load_features_async(api_host="", client_key="")
        assert result == features_response

@pytest.mark.asyncio
async def test_initialize_success(mock_options, mock_features_response):
    with patch('growthbook.growthbook_client.EnhancedFeatureRepository.load_features_async', 
               new_callable=AsyncMock, return_value=mock_features_response) as mock_load, \
         patch('growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh', 
               new_callable=AsyncMock, return_value=None):
        
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
    assert len(cache_state["features"]) == 1
    assert cache_state["savedGroups"] == {}
    feature_key = list(cache_state["features"].keys())[0]
    assert feature_key in features
    assert cache_state["features"][feature_key] == features[feature_key]


@pytest.mark.asyncio
async def test_feature_cache_thread_safety(cache):
    """Verify FeatureCache is thread-safe during concurrent updates"""
    repo = EnhancedFeatureRepository(
        api_host="https://test.growthbook.io",
        client_key="test_key"
    )

    feature_sets = [
        {f"set-{i}-feature-{j}": {"value": j} for j in range(3)}
        for i in range(5)
    ]

    async def update_full_set(feature_set):
        await repo._handle_feature_update({
            "features": feature_set,
            "savedGroups": {}
        })

    # Concurrent updates
    await asyncio.gather(*[update_full_set(fs) for fs in feature_sets])

    cache = repo._feature_cache.get_current_state()

    # One complete set should be in cache (race condition winner)
    assert len(cache["features"]) == 3
    cache_keys = set(cache["features"].keys())
    assert any(cache_keys == set(fs.keys()) for fs in feature_sets)


@pytest.mark.asyncio
async def test_disabled_features_removed_from_cache(cache):
    """
    Regression test: disabled features must be removed from cache.

    Previously, FeatureCache.update() used dict.update() which only
    adds/modifies entries but never removes them. This caused disabled
    features to persist in the cache indefinitely.
    """
    repo = EnhancedFeatureRepository(
        api_host="https://test.growthbook.io",
        client_key="test_key"
    )

    # Initial state: 2 features enabled
    await repo._handle_feature_update({
        "features": {
            "feature-a": {"defaultValue": True},
            "feature-b": {"defaultValue": False}
        },
        "savedGroups": {}
    })

    cache = repo._feature_cache.get_current_state()
    assert "feature-a" in cache["features"]
    assert "feature-b" in cache["features"]

    # User disables feature-b in Growthbook UI
    # API now returns only active features
    await repo._handle_feature_update({
        "features": {
            "feature-a": {"defaultValue": True}
        },
        "savedGroups": {}
    })

    cache = repo._feature_cache.get_current_state()
    assert "feature-a" in cache["features"]
    assert "feature-b" not in cache["features"]  # Must be removed!

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

    with patch('growthbook.FeatureRepository.load_features_async', 
               new_callable=AsyncMock, return_value=mock_features_response) as mock_load:
        
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
    """Test SSE event handling including JSON parsing"""
    events = [
        # Real SSE payload is a raw string in 'data'
        {'type': 'features', 'data': json.dumps({'features': {'feature1': {'defaultValue': 1}}})},
        {'type': 'ping', 'data': '{}'},  # Should be ignored
        {'type': 'features', 'data': json.dumps({'features': {'feature1': {'defaultValue': 2}}})}
    ]

    with patch('growthbook.FeatureRepository.load_features_async', 
               new_callable=AsyncMock, return_value={"features": {}, "savedGroups": {}}) as mock_load:

        # Create options with SSE strategy
        sse_options = Options(
            api_host=mock_options.api_host,
            client_key=mock_options.client_key,
            refresh_strategy=FeatureRefreshStrategy.SERVER_SENT_EVENTS
        )
        
        client = GrowthBookClient(sse_options)

        try:
            await client.initialize()

            # Simulate SSE events using the actual handler method
            # This now tests the json.loads parsing logic!
            for event in events:
                await client._features_repository._handle_sse_event(event)

            # Verify feature update happened
            state = client._features_repository._feature_cache.get_current_state()
            assert state["features"]["feature1"]["defaultValue"] == 2
        finally:
            await client.close()

@pytest.mark.asyncio
async def test_http_refresh_backoff():
    """Test HTTP refresh backoff strategy"""
    repo = EnhancedFeatureRepository(
        api_host="https://test.growthbook.io",
        client_key="test_key"
    )
    
    call_times = []
    failure_count = 0
    success_time = None
    done = asyncio.Event()
    
    async def mock_load(*args, **kwargs):
        nonlocal failure_count
        current_time = asyncio.get_event_loop().time()
        call_times.append(current_time)
        
        if failure_count < 3:
            failure_count += 1
            raise ConnectionError("Network error")
        
        nonlocal success_time
        if not success_time:
            success_time = current_time
            # Wait for at least one more call after success to verify normal interval
            if len(call_times) >= 5:
                done.set()
        return {"features": {}, "savedGroups": {}}
    
    try:
        with patch('growthbook.FeatureRepository.load_features_async', side_effect=mock_load):
            refresh_task = asyncio.create_task(repo._start_http_refresh(interval=0.1))
            try:
                await asyncio.wait_for(done.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            
            # Verify we had failures followed by success
            assert failure_count == 3, f"Expected 3 failures, got {failure_count}"
            assert len(call_times) >= 4, f"Expected at least 4 calls, got {len(call_times)}"
            
            # Verify backoff behavior - delays should generally increase during failures
            if len(call_times) >= 3:
                first_delay = call_times[1] - call_times[0]
                second_delay = call_times[2] - call_times[1]
                # Allow some flexibility in CI environments
                assert second_delay >= first_delay * 0.8, f"Second delay ({second_delay:.3f}) should be >= 80% of first delay ({first_delay:.3f})"
            
            # After success, verify we have reasonable timing for normal operation
            if len(call_times) >= 5:
                post_success_delay = call_times[4] - call_times[3]
                assert 0.05 <= post_success_delay <= 0.2, f"Post-success delay should be near 0.1s, got {post_success_delay:.3f}"
                
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
@pytest.mark.parametrize("service_flavor", ["sync", "async"])
async def test_sticky_bucket(test_sticky_bucket_data, base_client_setup, service_flavor):
    """Test sticky bucket functionality in GrowthBookClient.

    Runs every cases.json stickyBucket case against BOTH service flavors:
    the sync AbstractStickyBucketService (executor-offloaded) and the async
    AbstractAsyncStickyBucketService (awaited natively).
    """
    _, ctx, initial_docs, key, expected_result, expected_docs = test_sticky_bucket_data

    # Initialize sticky bucket service with test data
    if service_flavor == "sync":
        service = InMemoryStickyBucketService()
        for doc in initial_docs:
            service.save_assignments(doc)
    else:
        service = AsyncInMemoryStickyBucketService()
        for doc in initial_docs:
            await service.save_assignments(doc)

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

                # Persistence is fire-and-forget; settle before asserting
                await client.flush_sticky_bucket_saves()

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


# Rule-free feature: evaluating it still triggers the sticky bucket READ
# (prefetch happens per evaluation context), but never a write.
STICKY_READ_FEATURES = {
    "features": {
        "read-feature": {"defaultValue": "control"}
    },
    "savedGroups": {}
}


def _sticky_client_ctx(service, features=STICKY_READ_FEATURES, **opt_kwargs):
    return patch('growthbook.FeatureRepository.load_features_async',
                 new_callable=AsyncMock, return_value=features), \
           patch('growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh',
                 new_callable=AsyncMock), \
           patch('growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh',
                 new_callable=AsyncMock), \
           Options(
               api_host="https://localhost.growthbook.io",
               client_key="test-key",
               sticky_bucket_service=service,
               **opt_kwargs,
           )


@pytest.mark.asyncio
async def test_sticky_bucket_sync_service_does_not_block_loop():
    """Deterministic proof the SYNC-service fetch runs OFF the event loop:
    the fetch blocks its thread on a gate that only a coroutine running on
    the loop can release. If the fetch ran on the loop, the releaser
    coroutine could never be scheduled and the gate would time out."""
    fetch_gate = threading.Event()
    service = CountingStickyBucketService(fetch_gate=fetch_gate)
    EnhancedFeatureRepository._instances = {}
    p1, p2, p3, opts = _sticky_client_ctx(service)

    async def releaser():
        # Wait (on the loop) until the fetch has actually started in the
        # executor, then release it — impossible if the loop is blocked.
        while not service.fetch_started.is_set():
            await asyncio.sleep(0.001)
        fetch_gate.set()

    with p1, p2, p3:
        async with GrowthBookClient(opts) as client:
            releaser_task = asyncio.ensure_future(releaser())
            await client.eval_feature("read-feature", UserContext(attributes={"id": "user-1"}))
            await releaser_task

    assert service.get_all_calls == 1


@pytest.mark.asyncio
async def test_sticky_bucket_concurrent_refresh_coalesced():
    """Concurrent evals with identical attributes must trigger exactly one
    get_all_assignments fetch. The first fetch is gated so all ten evals are
    provably in flight (queued on the refresh lock) before it completes."""
    fetch_gate = asyncio.Event()
    service = AsyncInMemoryStickyBucketService(fetch_gate=fetch_gate)
    EnhancedFeatureRepository._instances = {}
    p1, p2, p3, opts = _sticky_client_ctx(service)

    with p1, p2, p3:
        async with GrowthBookClient(opts) as client:
            tasks = [
                asyncio.ensure_future(
                    client.eval_feature("read-feature", UserContext(attributes={"id": "user-1"}))
                )
                for _ in range(10)
            ]
            # Let every task advance to its await point (first holds the lock
            # awaiting the gate, the rest park on the lock).
            for _ in range(5):
                await asyncio.sleep(0)
            fetch_gate.set()
            await asyncio.gather(*tasks)

    assert service.get_all_calls == 1


@pytest.mark.asyncio
async def test_sticky_bucket_fetched_per_evaluation():
    """No cross-eval result cache by default (parity with the JS SDK's
    server-side GrowthBookClient.applyStickyBuckets): every evaluation
    fetches assignments for its context, so writes from other workers are
    visible on the next eval instead of being masked indefinitely."""
    service = AsyncInMemoryStickyBucketService()
    EnhancedFeatureRepository._instances = {}
    p1, p2, p3, opts = _sticky_client_ctx(service)

    with p1, p2, p3:
        async with GrowthBookClient(opts) as client:
            await client.eval_feature("read-feature", UserContext(attributes={"id": "user-1"}))
            assert service.get_all_calls == 1
            await client.eval_feature("read-feature", UserContext(attributes={"id": "user-1"}))
            assert service.get_all_calls == 2  # sequential evals refetch
            await client.eval_feature("read-feature", UserContext(attributes={"id": "user-2"}))
            assert service.get_all_calls == 3


@pytest.mark.asyncio
async def test_sticky_bucket_distinct_users_fetch_in_parallel():
    """Fetches for DIFFERENT attribute sets must overlap, not serialize.
    With the first user's fetch still parked behind the gate, the other
    users' fetches must also have started — impossible under a global
    refresh lock or an eval-wide lock."""
    fetch_gate = asyncio.Event()
    service = AsyncInMemoryStickyBucketService(fetch_gate=fetch_gate)
    EnhancedFeatureRepository._instances = {}
    p1, p2, p3, opts = _sticky_client_ctx(service)

    with p1, p2, p3:
        async with GrowthBookClient(opts) as client:
            tasks = [
                asyncio.ensure_future(
                    client.eval_feature("read-feature", UserContext(attributes={"id": uid}))
                )
                for uid in ("user-1", "user-2", "user-3")
            ]
            for _ in range(10):
                await asyncio.sleep(0)
            # All three fetches are in flight concurrently
            assert service.get_all_calls == 3
            fetch_gate.set()
            await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_sticky_bucket_opt_in_ttl_cache():
    """With sticky_bucket_cache_ttl > 0, fetched assignments are reused per
    attributes dict (bounded staleness), LRU-bounded by
    sticky_bucket_cache_size."""
    service = AsyncInMemoryStickyBucketService()
    EnhancedFeatureRepository._instances = {}
    p1, p2, p3, opts = _sticky_client_ctx(
        service, sticky_bucket_cache_ttl=60, sticky_bucket_cache_size=2)

    async def ev(uid):
        await client.eval_feature("read-feature", UserContext(attributes={"id": uid}))

    with p1, p2, p3:
        async with GrowthBookClient(opts) as client:
            await ev("user-1")
            await ev("user-1")
            assert service.get_all_calls == 1  # cached
            await ev("user-2")
            await ev("user-3")  # evicts user-1 (size bound = 2)
            assert service.get_all_calls == 3
            await ev("user-2")  # still cached
            assert service.get_all_calls == 3
            await ev("user-1")  # was evicted -> refetch
            assert service.get_all_calls == 4


@pytest.mark.asyncio
async def test_sticky_bucket_nonpositive_cache_size_disables_caching():
    """Regression: sticky_bucket_cache_size=-1 used to crash evaluation with
    KeyError (popitem on an empty cache). Non-positive size (or ttl) now
    simply disables caching."""
    service = AsyncInMemoryStickyBucketService()
    EnhancedFeatureRepository._instances = {}
    p1, p2, p3, opts = _sticky_client_ctx(
        service, sticky_bucket_cache_ttl=60, sticky_bucket_cache_size=-1)

    with p1, p2, p3:
        async with GrowthBookClient(opts) as client:
            await client.eval_feature("read-feature", UserContext(attributes={"id": "user-1"}))
            await client.eval_feature("read-feature", UserContext(attributes={"id": "user-2"}))
    assert service.get_all_calls == 2  # no crash; per-eval fetch


@pytest.mark.asyncio
async def test_sticky_bucket_waiter_cancellation_does_not_poison_owner():
    """Regression: a cancelled coalesced waiter used to propagate its
    cancellation into the shared inflight future, making the OWNER's
    successful fetch die with InvalidStateError. Waiters are shielded now."""
    fetch_gate = asyncio.Event()
    service = AsyncInMemoryStickyBucketService(fetch_gate=fetch_gate)
    EnhancedFeatureRepository._instances = {}
    p1, p2, p3, opts = _sticky_client_ctx(service)

    with p1, p2, p3:
        async with GrowthBookClient(opts) as client:
            user = {"id": "user-1"}
            owner = asyncio.ensure_future(
                client.eval_feature("read-feature", UserContext(attributes=dict(user))))
            for _ in range(3):
                await asyncio.sleep(0)
            waiter = asyncio.ensure_future(
                client.eval_feature("read-feature", UserContext(attributes=dict(user))))
            for _ in range(3):
                await asyncio.sleep(0)
            waiter.cancel()
            for _ in range(3):
                await asyncio.sleep(0)
            fetch_gate.set()

            result = await owner  # owner must complete normally
            assert result.value == "control"
            with pytest.raises(asyncio.CancelledError):
                await waiter
    assert service.get_all_calls == 1


@pytest.mark.asyncio
async def test_sticky_bucket_owner_cancellation_waiter_retries():
    """If the fetch OWNER is cancelled, a coalesced waiter must not be
    collaterally cancelled: it retries, becomes the new owner, and its
    evaluation succeeds."""
    fetch_gate = asyncio.Event()
    service = AsyncInMemoryStickyBucketService(fetch_gate=fetch_gate)
    EnhancedFeatureRepository._instances = {}
    p1, p2, p3, opts = _sticky_client_ctx(service)

    with p1, p2, p3:
        async with GrowthBookClient(opts) as client:
            user = {"id": "user-1"}
            owner = asyncio.ensure_future(
                client.eval_feature("read-feature", UserContext(attributes=dict(user))))
            for _ in range(3):
                await asyncio.sleep(0)
            waiter = asyncio.ensure_future(
                client.eval_feature("read-feature", UserContext(attributes=dict(user))))
            for _ in range(3):
                await asyncio.sleep(0)
            owner.cancel()
            for _ in range(3):
                await asyncio.sleep(0)
            fetch_gate.set()

            result = await waiter  # waiter retried as the new owner
            assert result.value == "control"
            with pytest.raises(asyncio.CancelledError):
                await owner
    assert service.get_all_calls == 2  # aborted owner fetch + waiter's retry


@pytest.mark.asyncio
async def test_feature_update_swaps_snapshot_without_disrupting_eval():
    """A feature update mid-evaluation must not affect the in-flight eval
    (it finishes against the snapshot it captured) and must be visible to
    the next eval — lock-free consistency via immutable snapshot swap."""
    fetch_gate = asyncio.Event()
    service = AsyncInMemoryStickyBucketService(fetch_gate=fetch_gate)
    EnhancedFeatureRepository._instances = {}
    p1, p2, p3, opts = _sticky_client_ctx(service)

    with p1, p2, p3:
        async with GrowthBookClient(opts) as client:
            inflight = asyncio.ensure_future(
                client.eval_feature("read-feature", UserContext(attributes={"id": "user-1"}))
            )
            for _ in range(5):
                await asyncio.sleep(0)
            # Swap features while the first eval is parked on the sticky fetch
            await client.set_features({"read-feature": {"defaultValue": "v2"}})
            fetch_gate.set()

            result = await inflight
            assert result.value == "control"  # finished against its captured snapshot

            result2 = await client.eval_feature("read-feature", UserContext(attributes={"id": "user-1"}))
            assert result2.value == "v2"  # new snapshot visible to the next eval


# Experiment forcing variation1 (weights [0, 1]) so a sticky bucket
# assignment doc is written deterministically on first evaluation.
STICKY_WRITE_FEATURES = {
    "features": {
        "exp-feature": {
            "defaultValue": 0,
            "rules": [{
                "key": "exp",
                "variations": [0, 1],
                "weights": [0, 1],
                "meta": [{"key": "control"}, {"key": "variation1"}],
            }]
        }
    },
    "savedGroups": {}
}


@pytest.mark.asyncio
@pytest.mark.parametrize("service_flavor", ["sync", "async"])
async def test_sticky_bucket_write_fire_and_forget(service_flavor):
    """The in-memory assignment doc must be visible immediately after eval
    (read-your-writes) while the service write is still parked behind a gate
    — proving eval never waits on persistence and (for the sync flavor) that
    the blocking write runs off the event loop. Releasing the gate and
    flushing makes the write observable."""
    if service_flavor == "sync":
        save_gate: Any = threading.Event()
        service: Any = CountingStickyBucketService(save_gate=save_gate)
    else:
        save_gate = asyncio.Event()
        service = AsyncInMemoryStickyBucketService(save_gate=save_gate)
    EnhancedFeatureRepository._instances = {}
    p1, p2, p3, opts = _sticky_client_ctx(service, STICKY_WRITE_FEATURES)

    with p1, p2, p3:
        async with GrowthBookClient(opts) as client:
            user = UserContext(attributes={"id": "user-1"})
            result = await client.eval_feature("exp-feature", user)
            assert result.value == 1

            # Read-your-writes: in-memory doc updated synchronously during eval
            assert "id||user-1" in user.sticky_bucket_assignment_docs

            # The write is still gated: eval returned without waiting for it
            assert "id||user-1" not in service.docs

            save_gate.set()
            await client.flush_sticky_bucket_saves()
            assert service.docs["id||user-1"]["assignments"] == {"exp__0": "variation1"}


@pytest.mark.asyncio
async def test_sticky_bucket_save_failure_is_logged_not_raised(caplog):
    """A failing save must never propagate into eval; it is logged and the
    task set is drained."""

    class FailingAsyncService(AsyncInMemoryStickyBucketService):
        async def save_assignments(self, doc):
            self.save_calls += 1
            raise RuntimeError("backend down")

    service = FailingAsyncService()
    EnhancedFeatureRepository._instances = {}
    p1, p2, p3, opts = _sticky_client_ctx(service, STICKY_WRITE_FEATURES)

    with p1, p2, p3:
        async with GrowthBookClient(opts) as client:
            result = await client.eval_feature("exp-feature", UserContext(attributes={"id": "user-1"}))
            assert result.value == 1  # eval unaffected
            await client.flush_sticky_bucket_saves()
            assert client._sticky_save_inflight == {}
            assert client._sticky_save_dirty == set()

    assert service.save_calls == 1
    assert any("Sticky bucket save failed" in r.message for r in caplog.records)


# Two experiment features so one user can accumulate two assignments in
# one sticky doc.
STICKY_TWO_EXP_FEATURES = {
    "features": {
        "feature-a": {"defaultValue": 0, "rules": [{
            "key": "exp_a", "variations": [0, 1], "weights": [0, 1],
            "meta": [{"key": "0"}, {"key": "v"}]}]},
        "feature-b": {"defaultValue": 0, "rules": [{
            "key": "exp_b", "variations": [0, 1], "weights": [0, 1],
            "meta": [{"key": "0"}, {"key": "v"}]}]},
    },
    "savedGroups": {}
}


@pytest.mark.asyncio
async def test_sticky_bucket_same_id_different_attributes_loses_no_assignments():
    """Regression: two evaluations with the SAME sticky identifier but
    DIFFERENT surrounding attributes fetch separate snapshots. Before the
    authoritative doc map, each snapshot generated a doc missing the other's
    assignment and unordered saves overwrote each other, silently losing one
    assignment. Both assignments must survive, regardless of save timing."""
    save_gate = asyncio.Event()
    service = AsyncInMemoryStickyBucketService(save_gate=save_gate)
    EnhancedFeatureRepository._instances = {}
    p1, p2, p3, opts = _sticky_client_ctx(service, STICKY_TWO_EXP_FEATURES)

    with p1, p2, p3:
        async with GrowthBookClient(opts) as client:
            # Same id, different surrounding attributes -> distinct snapshots.
            # Saves stay parked behind the gate the whole time, so the second
            # eval can never see the first one's write via the service.
            await client.eval_feature("feature-a", UserContext(attributes={"id": "1", "x": "1"}))
            await client.eval_feature("feature-b", UserContext(attributes={"id": "1", "x": "2"}))
            save_gate.set()
            await client.flush_sticky_bucket_saves()

    assert service.docs["id||1"]["assignments"] == {"exp_a__0": "v", "exp_b__0": "v"}


@pytest.mark.asyncio
async def test_sticky_bucket_saves_serialized_per_key():
    """At most one save per doc key is in flight; writes landing mid-save
    trigger a trailing save so the service converges to the merged doc."""
    inflight_peak = 0

    class ProbeService(AsyncInMemoryStickyBucketService):
        def __init__(self):
            super().__init__()
            self._inflight = 0
            self.release = asyncio.Event()

        async def save_assignments(self, doc):
            nonlocal inflight_peak
            self._inflight += 1
            inflight_peak = max(inflight_peak, self._inflight)
            self.save_calls += 1
            await self.release.wait()
            self.docs[self.get_key(doc["attributeName"], doc["attributeValue"])] = doc
            self._inflight -= 1

    service = ProbeService()
    EnhancedFeatureRepository._instances = {}
    p1, p2, p3, opts = _sticky_client_ctx(service, STICKY_TWO_EXP_FEATURES)

    with p1, p2, p3:
        async with GrowthBookClient(opts) as client:
            await client.eval_feature("feature-a", UserContext(attributes={"id": "1", "x": "1"}))
            await client.eval_feature("feature-b", UserContext(attributes={"id": "1", "x": "2"}))
            service.release.set()
            await client.flush_sticky_bucket_saves()

    assert inflight_peak == 1  # per-key serialization held under concurrent writes
    assert service.docs["id||1"]["assignments"] == {"exp_a__0": "v", "exp_b__0": "v"}


@pytest.mark.asyncio
async def test_sticky_bucket_unchanged_assignment_not_resaved():
    """Re-evaluating with an unchanged assignment must not schedule another
    save (core's changed=False gate). Regression test: core used to replace
    an EMPTY shared assignment-docs dict instead of mutating it in place,
    severing the client's cache reference so every re-eval looked 'changed'
    and re-saved."""
    service = AsyncInMemoryStickyBucketService()
    EnhancedFeatureRepository._instances = {}
    p1, p2, p3, opts = _sticky_client_ctx(service, STICKY_WRITE_FEATURES)

    with p1, p2, p3:
        async with GrowthBookClient(opts) as client:
            await client.eval_feature("exp-feature", UserContext(attributes={"id": "user-1"}))
            await client.flush_sticky_bucket_saves()
            assert service.save_calls == 1

            await client.eval_feature("exp-feature", UserContext(attributes={"id": "user-1"}))
            await client.flush_sticky_bucket_saves()
            assert service.save_calls == 1  # unchanged -> no new save


@pytest.mark.asyncio
async def test_async_user_callbacks_are_scheduled_and_drained():
    """Coroutine-function callbacks (on_experiment_viewed, on_feature_usage,
    subscriptions) must be scheduled on the loop instead of silently dropped,
    and drained by close()."""
    viewed, usage, subs = [], [], []

    async def on_viewed(experiment, result, user_context):
        viewed.append(experiment.key)

    async def on_usage(key, result, user_context):
        usage.append(key)

    async def on_sub(experiment, result):
        subs.append(experiment.key)

    EnhancedFeatureRepository._instances = {}
    opts = Options(
        api_host="https://localhost.growthbook.io",
        client_key="test-key",
        on_experiment_viewed=on_viewed,
        on_feature_usage=on_usage,
    )

    with patch('growthbook.FeatureRepository.load_features_async',
               new_callable=AsyncMock, return_value=STICKY_WRITE_FEATURES), \
         patch('growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh',
               new_callable=AsyncMock), \
         patch('growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh',
               new_callable=AsyncMock):
        async with GrowthBookClient(opts) as client:
            client.subscribe(on_sub)
            result = await client.eval_feature("exp-feature", UserContext(attributes={"id": "user-1"}))
            assert result.value == 1
            await client.run(
                Experiment(key="manual-exp", variations=[0, 1], weights=[0, 1]),
                UserContext(attributes={"id": "user-1"}),
            )
        # close() has drained all scheduled callback coroutines
        assert "exp" in viewed  # experiment key of the feature rule
        assert "manual-exp" in viewed
        assert usage == ["exp-feature"]
        assert "manual-exp" in subs


@pytest.mark.asyncio
async def test_failed_async_tracking_callback_is_retried():
    """An async on_experiment_viewed that fails must be un-deduped so the
    impression fires again on the next eval (parity with sync callbacks,
    whose exceptions also leave the event unmarked)."""
    calls = []

    async def flaky_tracker(experiment, result, user_context):
        calls.append(experiment.key)
        if len(calls) == 1:
            raise RuntimeError("collector down")

    EnhancedFeatureRepository._instances = {}
    opts = Options(
        api_host="https://localhost.growthbook.io",
        client_key="test-key",
        on_experiment_viewed=flaky_tracker,
    )

    with patch('growthbook.FeatureRepository.load_features_async',
               new_callable=AsyncMock, return_value=STICKY_WRITE_FEATURES), \
         patch('growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh',
               new_callable=AsyncMock), \
         patch('growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh',
               new_callable=AsyncMock):
        async with GrowthBookClient(opts) as client:
            user = UserContext(attributes={"id": "user-1"})

            await client.eval_feature("exp-feature", user)
            while client._callback_tasks:  # let the failing callback settle
                await asyncio.gather(*list(client._callback_tasks), return_exceptions=True)
            assert calls == ["exp"]

            # First attempt failed -> retried on the next eval
            await client.eval_feature("exp-feature", user)
            while client._callback_tasks:
                await asyncio.gather(*list(client._callback_tasks), return_exceptions=True)
            assert calls == ["exp", "exp"]

            # Second attempt succeeded -> now deduped
            await client.eval_feature("exp-feature", user)
            while client._callback_tasks:
                await asyncio.gather(*list(client._callback_tasks), return_exceptions=True)
            assert calls == ["exp", "exp"]


async def getTrackingMock(client: GrowthBookClient):
    """Helper function to mock tracking for tests"""
    calls = []

    def track(experiment, result, user_context):
        calls.append([experiment, result, user_context])

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
            assert calls[0] == [exp1, res1, user_context], "First tracking call mismatch"
            assert calls[1] == [exp2, res4, user_context], "Second tracking call mismatch"
            assert calls[2] == [exp2, res5, user_context], "Third tracking call mismatch"

    finally:
        await client.close()

async def getFailedTrackingMock(client: GrowthBookClient):
    """Helper function to mock tracking for tests"""
    calls = []
    # Set up tracking callback that raises an error
    def failing_track(experiment, result, user_context):
        calls.append([experiment, result, user_context])
        raise Exception("Tracking failed")

    client.options.on_experiment_viewed = failing_track
    return lambda: calls

@pytest.mark.asyncio
async def test_handles_tracking_errors():
    """Test graceful handling of tracking callback errors"""
    client = GrowthBookClient(Options(
        api_host="https://localhost.growthbook.io",
        client_key="test-key",
        enabled=True
    ))

    getMockedTrackingCalls = await getFailedTrackingMock(client)

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

            calls = getMockedTrackingCalls()
            assert len(calls) == 1, "Expected exactly 1 tracking call"

    finally:
        await client.close()


@pytest.mark.asyncio
async def test_feature_usage_callback():
    """Test that feature usage callback is called correctly"""
    calls = []
    
    def feature_usage_cb(key, result, user_context):
        calls.append([key, result, user_context])
    
    client = GrowthBookClient(Options(
        api_host="https://localhost.growthbook.io",
        client_key="test-key",
        enabled=True,
        on_feature_usage=feature_usage_cb
    ))
    
    user_context = UserContext(attributes={"id": "1"})
    
    try:
        # Set up mocks for feature repository
        mock_features = {
            "features": {
                "feature-1": {"defaultValue": True},
                "feature-2": {"defaultValue": False},
                "feature-3": {
                    "defaultValue": "blue",
                    "rules": [
                        {"force": "red", "condition": {"id": "1"}}
                    ]
                },
            },
            "savedGroups": {}
        }
        
        with patch('growthbook.FeatureRepository.load_features_async', 
                  new_callable=AsyncMock, return_value=mock_features), \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh',
                  new_callable=AsyncMock), \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh',
                  new_callable=AsyncMock):
            
            # Initialize client
            await client.initialize()
            
            # Test eval_feature
            result1 = await client.eval_feature("feature-1", user_context)
            assert len(calls) == 1
            assert calls[0][0] == "feature-1"
            assert calls[0][1].value is True
            assert calls[0][1].source == "defaultValue"
            assert calls[0][2].attributes == {"id": "1"}
            
            # Test is_on
            await client.is_on("feature-2", user_context)
            assert len(calls) == 2
            assert calls[1][0] == "feature-2"
            assert calls[1][1].value is False
            assert calls[1][2].attributes == {"id": "1"}
            
            # Test get_feature_value
            value = await client.get_feature_value("feature-3", "blue", user_context)
            assert len(calls) == 3
            assert calls[2][0] == "feature-3"
            assert calls[2][1].value == "red"
            assert value == "red"
            assert calls[2][2].attributes == {"id": "1"}
            
            # Test is_off
            await client.is_off("feature-1", user_context)
            assert len(calls) == 4
            assert calls[3][0] == "feature-1"
            assert calls[3][2].attributes == {"id": "1"}
            
            # Calling same feature multiple times should trigger callback each time
            await client.eval_feature("feature-1", user_context)
            await client.eval_feature("feature-1", user_context)
            assert len(calls) == 6
            
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_feature_usage_callback_error_handling():
    """Test that feature usage callback errors are handled gracefully"""
    
    def failing_callback(key, result, user_context):
        raise Exception("Callback error")
    
    client = GrowthBookClient(Options(
        api_host="https://localhost.growthbook.io",
        client_key="test-key",
        enabled=True,
        on_feature_usage=failing_callback
    ))
    
    user_context = UserContext(attributes={"id": "1"})
    
    try:
        # Set up mocks for feature repository
        mock_features = {
            "features": {
                "feature-1": {"defaultValue": True},
            },
            "savedGroups": {}
        }
        
        with patch('growthbook.FeatureRepository.load_features_async', 
                  new_callable=AsyncMock, return_value=mock_features), \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh',
                  new_callable=AsyncMock), \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh',
                  new_callable=AsyncMock):
            
            # Initialize client
            await client.initialize()
            
            # Should not raise an error even if callback fails
            result = await client.eval_feature("feature-1", user_context)
            assert result.value is True
            
            # Should work with is_on as well
            is_on = await client.is_on("feature-1", user_context)
            assert is_on is True
            
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_skip_all_experiments_flag():
    """Test that skip_all_experiments flag prevents users from being put into experiments"""
    
    client = GrowthBookClient(Options(
        api_host="https://localhost.growthbook.io",
        client_key="test-key",
        enabled=True
    ))
    
    # User context WITH skip_all_experiments=True
    user_context_skip = UserContext(
        attributes={"id": "1"},
        skip_all_experiments=True
    )
    
    # User context WITHOUT skip_all_experiments (normal behavior)
    user_context_normal = UserContext(
        attributes={"id": "1"},
        skip_all_experiments=False
    )
    
    try:
        # Set up mocks for feature repository
        mock_features = {
            "features": {
                "feature-with-experiment": {
                    "defaultValue": "control",
                    "rules": [
                        {
                            "key": "exp-123",
                            "variations": ["control", "variation"],
                            "weights": [0.5, 0.5]
                        }
                    ]
                }
            },
            "savedGroups": {}
        }
        
        with patch('growthbook.FeatureRepository.load_features_async', 
                  new_callable=AsyncMock, return_value=mock_features), \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh',
                  new_callable=AsyncMock), \
             patch('growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh',
                  new_callable=AsyncMock):
            
            # Initialize client
            await client.initialize()
            
            # Test with skip_all_experiments=True
            result_skip = await client.eval_feature("feature-with-experiment", user_context_skip)
            assert result_skip.value == "control"  # Should get default value
            assert result_skip.source == "defaultValue"
            assert result_skip.experiment is None
            assert result_skip.experimentResult is None
            
            # Test direct experiment run with skip_all_experiments=True
            exp = Experiment(key="direct-exp", variations=["a", "b"])
            exp_result_skip = await client.run(exp, user_context_skip)
            assert exp_result_skip.inExperiment is False
            assert exp_result_skip.value == "a"  # Should get first variation
            
            # Test with skip_all_experiments=False (normal)
            result_normal = await client.eval_feature("feature-with-experiment", user_context_normal)
            # User should be assigned to a variation
            assert result_normal.value in ["control", "variation"]
            assert result_normal.source == "experiment"
            
    finally:
        await client.close()