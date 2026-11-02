from unittest.mock import patch

import pytest

from growthbook import InMemoryFeatureCache


@pytest.mark.parametrize(
    "initial_ttl, updated_ttl, update_delay",
    [
        (600, 5, 10),
        (5, 600, 1),
        (5, 600, 10),
        (600, 0, 10),
        (0, 600, 10),
        (600, 600, 10),
    ],
)
def test_cache_overwrite_uses_new_ttl(initial_ttl, updated_ttl, update_delay):
    cache = InMemoryFeatureCache()
    original = {"features": {"flag": {"defaultValue": False}}}
    updated = {"features": {"flag": {"defaultValue": True}}}

    with patch("growthbook.growthbook.time", return_value=100) as clock:
        cache.set("sdk-key", original, initial_ttl)
        clock.return_value += update_delay
        cache.set("sdk-key", updated, updated_ttl)
        assert cache.get("sdk-key") == updated

        clock.return_value += updated_ttl
        assert cache.get("sdk-key") == updated

        clock.return_value += 1
        assert cache.get("sdk-key") is None


def test_cache_overwrite_does_not_change_other_keys():
    cache = InMemoryFeatureCache()
    original = {"features": {"flag": {"defaultValue": False}}}
    updated = {"features": {"flag": {"defaultValue": True}}}

    with patch("growthbook.growthbook.time", return_value=100) as clock:
        cache.set("first", original, 600)
        cache.set("second", original, 600)
        clock.return_value = 110
        cache.set("first", updated, 5)
        clock.return_value = 116
        assert cache.get("first") is None
        assert cache.get("second") == original
