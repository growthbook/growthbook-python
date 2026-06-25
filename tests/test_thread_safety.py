#!/usr/bin/env python
"""Concurrency regressions for the feature repository.

Each test here fails on the unsynchronized version of the code: the shared
cache dict, the shared callback list, and the un-coalesced fetch path all
misbehave only under real concurrency, so they need real threads/tasks rather
than mocks of them.
"""

import asyncio
import json
import threading
from unittest.mock import patch

import pytest

from growthbook import FeatureRepository


FEATURES = {"features": {"flag": {"defaultValue": True}}}


class SlowHttpResp:
    """Response that takes long enough to fetch for every waiting thread to
    pile up on a cold cache — that's the stampede being tested."""

    def __init__(self, delay: float = 0.05) -> None:
        self.status = 200
        self.data = json.dumps(FEATURES).encode("utf-8")
        self.headers = {}
        self._delay = delay


def test_cache_miss_stampede_is_coalesced_sync():
    repo = FeatureRepository()
    calls = []

    def slow_get(url, headers=None):
        calls.append(url)
        threading.Event().wait(0.05)
        return SlowHttpResp()

    with patch.object(repo, "_get", side_effect=slow_get):
        threads = [
            threading.Thread(
                target=repo.load_features,
                args=("https://cdn.growthbook.io", "sdk-abc123"),
            )
            for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    # Without per-key coalescing every thread misses the cold cache and fetches.
    assert len(calls) == 1
    repo.clear_cache()


@pytest.mark.asyncio
async def test_cache_miss_stampede_is_coalesced_async():
    repo = FeatureRepository()
    calls = []

    async def slow_fetch(api_host, client_key):
        calls.append(client_key)
        await asyncio.sleep(0.05)
        return dict(FEATURES)

    with patch.object(repo, "_fetch_and_decode_async", side_effect=slow_fetch):
        await asyncio.gather(*[
            repo.load_features_async("https://cdn.growthbook.io", "sdk-abc123")
            for _ in range(10)
        ])

    assert len(calls) == 1
    repo.clear_cache()


def test_callbacks_can_be_mutated_during_notification():
    """A callback that unregisters itself must not corrupt the iteration —
    notification iterates a snapshot, not the live list."""
    repo = FeatureRepository()
    seen = []

    def self_removing(data):
        seen.append("self_removing")
        repo.remove_feature_update_callback(self_removing)

    def other(data):
        seen.append("other")

    repo.add_feature_update_callback(self_removing)
    repo.add_feature_update_callback(other)

    repo._notify_feature_update_callbacks(dict(FEATURES))
    assert seen == ["self_removing", "other"]

    # The self-removing one is gone; the other still fires.
    seen.clear()
    repo._notify_feature_update_callbacks(dict(FEATURES))
    assert seen == ["other"]

    repo.remove_feature_update_callback(other)


def test_async_lock_map_does_not_leak_across_event_loops():
    """Each asyncio.run() used to leave a lock — and the closed loop it kept a
    reference to — in the map, since the key carries a loop id and entries were
    never removed. An app driving async code per request grew without bound."""
    repo = FeatureRepository()

    async def fetch(api_host, client_key):
        return dict(FEATURES)

    with patch.object(repo, "_fetch_and_decode_async", side_effect=fetch):
        for _ in range(25):
            asyncio.run(
                repo.load_features_async("https://cdn.growthbook.io", "sdk-abc123")
            )
            repo.clear_cache()  # force a real fetch (and so a lock) every run

    assert repo._async_key_locks == {}, (
        f"leaked {len(repo._async_key_locks)} lock entries"
    )
    repo.clear_cache()


@pytest.mark.asyncio
async def test_async_lock_entry_survives_while_waiters_remain():
    """Refcounting, not just delete-after-use: an entry must outlive the first
    holder while others are still queued on it, or they'd coalesce onto
    different locks and both fetch."""
    repo = FeatureRepository()
    calls = []
    release = asyncio.Event()

    async def slow_fetch(api_host, client_key):
        calls.append(1)
        await release.wait()
        return dict(FEATURES)

    with patch.object(repo, "_fetch_and_decode_async", side_effect=slow_fetch):
        waiters = [
            asyncio.create_task(
                repo.load_features_async("https://cdn.growthbook.io", "sdk-abc123")
            )
            for _ in range(4)
        ]
        await asyncio.sleep(0.05)  # let them all queue on the same key
        assert len(repo._async_key_locks) == 1
        release.set()
        await asyncio.gather(*waiters)

    assert len(calls) == 1          # still coalesced
    assert repo._async_key_locks == {}   # and cleaned up afterwards
    repo.clear_cache()


def test_feature_update_callback_may_force_refresh_the_same_key(mocker):
    """Callbacks are user code and used to run while the per-key lock was
    held, so one that forced a refresh of the same key deadlocked."""
    repo = FeatureRepository()
    seen = []

    def callback(data):
        if len(seen) == 0:
            seen.append("outer")
            # Re-entering for the same key on this thread.
            repo.load_features("https://cdn.growthbook.io", "sdk-abc123", force_refresh=True)
        else:
            seen.append("inner")

    mocker.patch.object(repo, "_get", return_value=SlowHttpResp(delay=0))
    repo.add_feature_update_callback(callback)

    # daemon: if this regresses, the worker is wedged on the lock forever and
    # a non-daemon thread would stop the test process from ever exiting.
    worker = threading.Thread(
        target=repo.load_features,
        args=("https://cdn.growthbook.io", "sdk-abc123"),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive(), "load_features deadlocked on its own key lock"
    assert seen == ["outer", "inner"]

    repo.remove_feature_update_callback(callback)
    repo.clear_cache()
