import pytest
import asyncio
import os
import sys
from growthbook.growthbook_client import EnhancedFeatureRepository, SingletonMeta

# Only define the event_loop_policy fixture, and let pytest-asyncio handle event_loop
@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.get_event_loop_policy()

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))