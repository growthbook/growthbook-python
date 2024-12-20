import pytest
import asyncio

# Only define the event_loop_policy fixture, and let pytest-asyncio handle event_loop
@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.get_event_loop_policy()