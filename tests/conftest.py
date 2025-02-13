import pytest
import asyncio
import os
import sys
from growthbook.growthbook_client import EnhancedFeatureRepository, SingletonMeta

# Only define the event_loop_policy fixture, and let pytest-asyncio handle event_loop
@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.get_event_loop_policy()

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the EnhancedFeatureRepository singleton between tests"""
    # Let the test run first
    yield
    # Only clear after test is completely done
    if hasattr(SingletonMeta, '_instances'):
        # Ensure any async operations are complete
        for instance in SingletonMeta._instances.values():
            if hasattr(instance, '_stop_event'):
                instance._stop_event.set()
        SingletonMeta._instances.clear()

@pytest.fixture(autouse=True)
async def cleanup_tasks():
    """Cleanup any pending tasks after each test."""
    yield
    loop = asyncio.get_event_loop()
    # Let any pending callbacks complete
    await asyncio.sleep(0)
    # Ensure singleton instances are cleaned up first
    if hasattr(SingletonMeta, '_instances'):
        for instance in SingletonMeta._instances.values():
            if hasattr(instance, 'stop_refresh'):
                await instance.stop_refresh()
    # Clear singleton instances
    SingletonMeta._instances.clear()

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))