"""Options.http_connect_timeout / http_read_timeout must reach every HTTP path."""
from unittest.mock import patch

import aiohttp
import pytest

from growthbook.growthbook import FeatureRepository


class _FakeResponse:
    status = 500

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Stands in for aiohttp.ClientSession and records its constructor kwargs."""

    created = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeSession.created.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def get(self, *args, **kwargs):
        return _FakeResponse()

    def post(self, *args, **kwargs):
        return _FakeResponse()


@pytest.fixture
def fake_session():
    _FakeSession.created = []
    with patch("growthbook.growthbook.aiohttp.ClientSession", _FakeSession):
        yield _FakeSession


def _repo(connect=None, read=None):
    repo = FeatureRepository()
    repo.http_connect_timeout = connect
    repo.http_read_timeout = read
    return repo


@pytest.mark.asyncio
async def test_async_get_applies_http_timeouts(fake_session):
    repo = _repo(connect=3, read=7)

    assert await repo._fetch_and_decode_async("https://cdn.growthbook.io", "sdk-abc123") is None

    (session,) = fake_session.created
    timeout = session.kwargs["timeout"]
    assert isinstance(timeout, aiohttp.ClientTimeout)
    assert timeout.sock_connect == 3
    assert timeout.sock_read == 7


@pytest.mark.asyncio
async def test_async_post_applies_http_timeouts(fake_session):
    repo = _repo(connect=3, read=7)

    assert await repo._fetch_and_decode_post_async(
        "https://cdn.growthbook.io", "sdk-abc123", {"attributes": {}}
    ) is None

    (session,) = fake_session.created
    timeout = session.kwargs["timeout"]
    assert timeout.sock_connect == 3
    assert timeout.sock_read == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("connect, read", [(3, 7), (3, None), (None, 7)])
async def test_async_keeps_aiohttp_defaults_for_unset_fields(fake_session, connect, read):
    default = aiohttp.client.DEFAULT_TIMEOUT
    repo = _repo(connect=connect, read=read)

    await repo._fetch_and_decode_async("https://cdn.growthbook.io", "sdk-abc123")

    timeout = fake_session.created[0].kwargs["timeout"]
    # The overall request limit must survive setting either option.
    assert timeout.total == default.total
    assert timeout.connect == default.connect
    assert timeout.sock_connect == (connect if connect is not None else default.sock_connect)
    assert timeout.sock_read == (read if read is not None else default.sock_read)


@pytest.mark.asyncio
async def test_async_keeps_aiohttp_defaults_without_timeouts(fake_session):
    repo = _repo()

    await repo._fetch_and_decode_async("https://cdn.growthbook.io", "sdk-abc123")

    assert "timeout" not in fake_session.created[0].kwargs


@pytest.mark.parametrize(
    "connect, read",
    [(3, 7), (3, None), (None, 7)],
)
def test_sync_pool_manager_applies_each_timeout(connect, read):
    repo = _repo(connect=connect, read=read)

    with patch("growthbook.growthbook.PoolManager") as pool_manager:
        repo._get("https://cdn.growthbook.io/api/features/sdk-abc123")

    timeout = pool_manager.call_args.kwargs["timeout"]
    assert timeout.connect_timeout == connect
    assert timeout.read_timeout == read


def test_sync_pool_manager_has_no_timeout_by_default():
    repo = _repo()

    with patch("growthbook.growthbook.PoolManager") as pool_manager:
        repo._get("https://cdn.growthbook.io/api/features/sdk-abc123")

    assert pool_manager.call_args.kwargs["timeout"] is None
