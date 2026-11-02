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
