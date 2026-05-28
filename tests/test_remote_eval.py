"""Tests for remote evaluation support.

Remote eval mode POSTs the user's targeting context to
`/api/eval/{client_key}` on a self-hosted proxy. The server returns the same
`{features, savedGroups}` shape (rules filtered server-side) and the SDK runs
its normal local `eval_feature` over the result.
"""

import asyncio
import json
import pytest
from unittest.mock import MagicMock, patch

from growthbook import GrowthBook, feature_repo, FeatureRepository, InMemoryStickyBucketService
from growthbook.common_types import Options, UserContext
from growthbook.growthbook_client import GrowthBookClient, SingletonMeta


def _make_post_response(body):
    """Build a urllib3-shaped HTTPResponse mock with JSON body."""
    resp = MagicMock()
    resp.status = 200
    resp.data = json.dumps(body).encode("utf-8")
    return resp


def _reset_repo():
    """Clear shared singleton repo state between tests."""
    feature_repo.clear_cache()
    # Drop registered callbacks so prior test instances don't fire on new ones
    feature_repo._feature_update_callbacks = []
    # Drop ETag entries from prior CDN-mode tests
    feature_repo._etag_cache.clear()


# ---------------------------------------------------------------------------
# Sync GrowthBook tests
# ---------------------------------------------------------------------------


class TestRemoteEvalSyncValidation:
    def setup_method(self):
        _reset_repo()

    def test_missing_client_key_raises(self):
        with pytest.raises(ValueError, match="client_key for remote eval"):
            GrowthBook(api_host="https://proxy.example.com", remoteEval=True)

    def test_decryption_key_raises(self):
        with pytest.raises(ValueError, match="Encryption is not available"):
            GrowthBook(
                api_host="https://proxy.example.com",
                client_key="k",
                decryption_key="d",
                remoteEval=True,
            )

    def test_sticky_bucket_service_raises(self):
        with pytest.raises(ValueError, match="sticky_bucket_service is not compatible"):
            GrowthBook(
                api_host="https://proxy.example.com",
                client_key="k",
                sticky_bucket_service=InMemoryStickyBucketService(),
                remoteEval=True,
            )

    def test_stale_while_revalidate_raises(self):
        with pytest.raises(ValueError, match="stale_while_revalidate is not compatible"):
            GrowthBook(
                api_host="https://proxy.example.com",
                client_key="k",
                stale_while_revalidate=True,
                remoteEval=True,
            )

    def test_cloud_host_raises(self):
        with pytest.raises(ValueError, match="Cloud host does not support remote eval"):
            GrowthBook(
                api_host="https://cdn.growthbook.io",
                client_key="k",
                remoteEval=True,
            )

    def test_cloud_subdomain_raises(self):
        with pytest.raises(ValueError, match="Cloud host does not support remote eval"):
            GrowthBook(
                api_host="https://anything.growthbook.io",
                client_key="k",
                remoteEval=True,
            )


class TestRemoteEvalSyncWireFormat:
    def setup_method(self):
        _reset_repo()

    def test_post_url_and_body_shape(self):
        body = {"features": {"flag1": {"defaultValue": True}}, "savedGroups": {}}
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)) as mock_post:
            GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1", "country": "US"},
                forced_variations={"exp-1": 2},
                url="/checkout",
                remoteEval=True,
            )

        assert mock_post.call_count == 1
        # _post(self, url, payload, headers)
        call = mock_post.call_args
        assert call.args[0] == "https://proxy.example.com/api/eval/sdk-test"
        assert call.args[1] == {
            "attributes": {"id": "u1", "country": "US"},
            "forcedFeatures": [],
            "forcedVariations": {"exp-1": 2},
            "url": "/checkout",
        }
        assert call.args[2]["Content-Type"] == "application/json"

    def test_eval_uses_response_features(self):
        body = {"features": {"flag1": {"defaultValue": True}}, "savedGroups": {}}
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)):
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            assert gb.is_on("flag1") is True

    def test_user_agent_header_present(self):
        body = {"features": {}, "savedGroups": {}}
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)) as mock_post:
            GrowthBook(
                api_host="https://proxy.example.com",
                client_key="abcd1234",
                attributes={"id": "u1"},
                remoteEval=True,
            )
        headers = mock_post.call_args.args[2]
        assert headers["User-Agent"].startswith("Gb-Python")


class TestRemoteEvalSyncRefetch:
    def setup_method(self):
        _reset_repo()

    def test_set_attributes_triggers_refetch(self):
        body = {"features": {"flag1": {"defaultValue": True}}, "savedGroups": {}}
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)) as mock_post:
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            assert mock_post.call_count == 1
            gb.set_attributes({"id": "u2"})
            assert mock_post.call_count == 2

    def test_set_forced_variations_triggers_refetch(self):
        body = {"features": {}, "savedGroups": {}}
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)) as mock_post:
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            assert mock_post.call_count == 1
            gb.set_forced_variations({"exp-1": 1})
            assert mock_post.call_count == 2

    def test_set_url_triggers_refetch(self):
        body = {"features": {}, "savedGroups": {}}
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)) as mock_post:
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            assert mock_post.call_count == 1
            gb.set_url("/different-page")
            assert mock_post.call_count == 2


class TestRemoteEvalSyncCaching:
    def setup_method(self):
        _reset_repo()

    def test_same_payload_hits_cache(self):
        body = {"features": {}, "savedGroups": {}}
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)) as mock_post:
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            initial = mock_post.call_count
            # set_attributes with same value: cache hit
            gb.set_attributes({"id": "u1"})
            assert mock_post.call_count == initial

    def test_different_attributes_cache_miss(self):
        body = {"features": {}, "savedGroups": {}}
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)) as mock_post:
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            gb.set_attributes({"id": "u2"})
            assert mock_post.call_count == 2

    def test_cache_key_attributes_narrows(self):
        """If cacheKeyAttributes=['id'], changing a non-listed attr is a cache HIT."""
        body = {"features": {}, "savedGroups": {}}
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)) as mock_post:
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1", "country": "US"},
                cacheKeyAttributes=["id"],
                remoteEval=True,
            )
            initial = mock_post.call_count
            # 'country' is not in cacheKeyAttributes — should be cache hit
            gb.set_attributes({"id": "u1", "country": "DE"})
            assert mock_post.call_count == initial


class TestRemoteEvalSyncRobustness:
    def setup_method(self):
        _reset_repo()

    def test_missing_saved_groups_doesnt_crash(self):
        body = {"features": {"flag1": {"defaultValue": True}}}  # no savedGroups key
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)):
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="k",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            assert gb.is_on("flag1") is True

    def test_empty_features_returns_off(self):
        body = {"features": {}, "savedGroups": {}}
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)):
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="k",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            # Unknown features return a falsy value -> is_on is False
            assert gb.is_on("nonexistent") is False

    def test_no_cross_pollution_between_instances(self):
        """Two GrowthBook instances with same client_key but different attributes
        must each see their own filtered features (the singleton repo's global
        callback path is skipped in remote-eval mode to prevent leakage)."""
        responses = {
            ("u1",): {"features": {"flag1": {"defaultValue": "for-u1"}}, "savedGroups": {}},
            ("u2",): {"features": {"flag1": {"defaultValue": "for-u2"}}, "savedGroups": {}},
        }

        def fake_post(self, url, payload, headers):
            return _make_post_response(responses[(payload["attributes"]["id"],)])

        with patch.object(FeatureRepository, "_post", fake_post):
            gb1 = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="shared",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            gb2 = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="shared",
                attributes={"id": "u2"},
                remoteEval=True,
            )
            assert gb1.get_feature_value("flag1", "fallback") == "for-u1"
            assert gb2.get_feature_value("flag1", "fallback") == "for-u2"


# ---------------------------------------------------------------------------
# Async GrowthBookClient tests
# ---------------------------------------------------------------------------


def _reset_singletons():
    """Drop the EnhancedFeatureRepository singleton so each test gets fresh state."""
    SingletonMeta._instances.clear()


@pytest.fixture(autouse=True)
def _reset_async_singletons():
    _reset_singletons()
    yield
    _reset_singletons()


class TestRemoteEvalAsyncValidation:
    def test_missing_client_key_raises(self):
        with pytest.raises(ValueError, match="client_key for remote eval"):
            GrowthBookClient(Options(api_host="https://proxy.example.com", remote_eval=True))

    def test_decryption_key_raises(self):
        with pytest.raises(ValueError, match="Encryption is not available"):
            GrowthBookClient(Options(
                api_host="https://proxy.example.com",
                client_key="k",
                decryption_key="d",
                remote_eval=True,
            ))

    def test_sticky_bucket_service_raises(self):
        with pytest.raises(ValueError, match="sticky_bucket_service is not compatible"):
            GrowthBookClient(Options(
                api_host="https://proxy.example.com",
                client_key="k",
                sticky_bucket_service=InMemoryStickyBucketService(),
                remote_eval=True,
            ))

    def test_cloud_host_raises(self):
        with pytest.raises(ValueError, match="Cloud host does not support remote eval"):
            GrowthBookClient(Options(
                api_host="https://cdn.growthbook.io",
                client_key="k",
                remote_eval=True,
            ))


async def _make_async_client(post_handler):
    """Build a GrowthBookClient with remote_eval on and the network method mocked."""
    client = GrowthBookClient(Options(
        api_host="https://proxy.example.com",
        client_key="sdk-async",
        remote_eval=True,
        refresh_strategy=None,
    ))
    client._features_repository._fetch_and_decode_post_async = post_handler
    await client.initialize()
    return client


@pytest.mark.asyncio
async def test_async_cache_hit_same_user_context():
    calls = []

    async def post_handler(api_host, client_key, payload):
        calls.append(payload)
        return {"features": {"flag1": {"defaultValue": True}}, "savedGroups": {}}

    client = await _make_async_client(post_handler)
    uc = UserContext(attributes={"id": "u1"})

    assert await client.is_on("flag1", uc) is True
    assert len(calls) == 1
    # Second eval with same context = cache hit
    assert await client.is_on("flag1", uc) is True
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_async_cache_miss_different_user_context():
    calls = []

    async def post_handler(api_host, client_key, payload):
        calls.append(payload)
        return {"features": {"flag1": {"defaultValue": True}}, "savedGroups": {}}

    client = await _make_async_client(post_handler)
    await client.is_on("flag1", UserContext(attributes={"id": "u1"}))
    await client.is_on("flag1", UserContext(attributes={"id": "u2"}))
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_async_inflight_coalescing():
    """Three concurrent evals for the same UserContext = exactly 1 POST."""
    calls = []
    started = asyncio.Event()
    finish = asyncio.Event()

    async def slow_post(api_host, client_key, payload):
        calls.append(payload)
        started.set()
        await finish.wait()
        return {"features": {"flag1": {"defaultValue": True}}, "savedGroups": {}}

    client = await _make_async_client(slow_post)
    uc = UserContext(attributes={"id": "u1"})

    t1 = asyncio.create_task(client.is_on("flag1", uc))
    t2 = asyncio.create_task(client.is_on("flag1", uc))
    t3 = asyncio.create_task(client.is_on("flag1", uc))
    await started.wait()
    # Give the other coroutines time to reach the inflight check
    await asyncio.sleep(0.05)
    finish.set()
    r1, r2, r3 = await asyncio.gather(t1, t2, t3)
    assert (r1, r2, r3) == (True, True, True)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_async_lru_eviction():
    """Cache size 2: three distinct users in order A,B,C then A again -> A is a miss."""
    calls = []

    async def post_handler(api_host, client_key, payload):
        calls.append(payload["attributes"]["id"])
        return {"features": {"flag1": {"defaultValue": True}}, "savedGroups": {}}

    client = GrowthBookClient(Options(
        api_host="https://proxy.example.com",
        client_key="sdk-async",
        remote_eval=True,
        refresh_strategy=None,
        remote_eval_cache_size=2,
    ))
    client._features_repository._fetch_and_decode_post_async = post_handler
    await client.initialize()

    await client.is_on("flag1", UserContext(attributes={"id": "A"}))
    await client.is_on("flag1", UserContext(attributes={"id": "B"}))
    await client.is_on("flag1", UserContext(attributes={"id": "C"}))  # evicts A
    await client.is_on("flag1", UserContext(attributes={"id": "A"}))  # cache miss
    assert calls == ["A", "B", "C", "A"]


@pytest.mark.asyncio
async def test_async_preload_warms_cache():
    calls = []

    async def post_handler(api_host, client_key, payload):
        calls.append(payload)
        return {"features": {"flag1": {"defaultValue": True}}, "savedGroups": {}}

    client = await _make_async_client(post_handler)
    uc = UserContext(attributes={"id": "u1"})

    await client.preload_remote_eval(uc)
    assert len(calls) == 1
    # Subsequent eval is a cache hit
    await client.is_on("flag1", uc)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_async_sse_features_updated_flushes_cache():
    calls = []

    async def post_handler(api_host, client_key, payload):
        calls.append(payload)
        return {"features": {"flag1": {"defaultValue": True}}, "savedGroups": {}}

    client = await _make_async_client(post_handler)
    uc = UserContext(attributes={"id": "u1"})

    await client.is_on("flag1", uc)
    assert len(calls) == 1

    # Simulate the proxy's features-updated SSE event
    await client._features_repository._handle_sse_event({"type": "features-updated"})

    await client.is_on("flag1", uc)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_async_preload_noop_in_cdn_mode():
    """preload_remote_eval is a safe no-op when remote_eval is False."""
    client = GrowthBookClient(Options(
        api_host="https://proxy.example.com",
        client_key="sdk-cdn",
        remote_eval=False,
        refresh_strategy=None,
    ))
    # Don't initialize (no features to load in a CDN-only smoke). Just make
    # sure preload doesn't raise or hit the network.
    await client.preload_remote_eval(UserContext(attributes={"id": "u1"}))


class TestRemoteEvalRuleTracks:
    """The proxy attaches a `tracks: [{experiment, result}]` array to rules so
    the SDK can fire trackingCallback for experiments that were evaluated
    server-side. Mirrors `if (rule.tracks)` in JS sdk-js/src/core.ts."""

    def setup_method(self):
        _reset_repo()

    def test_sync_fires_tracking_callback_for_each_track_entry(self):
        # Server-evaluated rule: force=True with two deferred tracking events.
        body = {
            "features": {
                "ab-feature": {
                    "defaultValue": False,
                    "rules": [{
                        "force": True,
                        "tracks": [
                            {
                                "experiment": {"key": "exp-A", "variations": [0, 1]},
                                "result": {
                                    "variationId": 1,
                                    "inExperiment": True,
                                    "value": 1,
                                    "hashUsed": True,
                                    "hashAttribute": "id",
                                    "hashValue": "u1",
                                    "featureId": "ab-feature",
                                    "key": "1",
                                },
                            },
                            {
                                "experiment": {"key": "exp-B", "variations": ["c", "t"]},
                                "result": {
                                    "variationId": 0,
                                    "inExperiment": True,
                                    "value": "c",
                                    "hashUsed": True,
                                    "hashAttribute": "id",
                                    "hashValue": "u1",
                                    "featureId": "ab-feature",
                                    "key": "0",
                                },
                            },
                        ],
                    }],
                },
            },
            "savedGroups": {},
        }

        tracked = []
        def cb(experiment, result, user_context):
            tracked.append((experiment.key, result.variationId, result.value))

        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)):
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                on_experiment_viewed=cb,
                remoteEval=True,
            )
            result = gb.eval_feature("ab-feature")
            assert result.value is True
            assert result.source == "force"
            assert tracked == [("exp-A", 1, 1), ("exp-B", 0, "c")]

    def test_sync_no_tracks_no_callback_fire(self):
        """Force rule with no tracks fires nothing."""
        body = {
            "features": {
                "f": {"defaultValue": False, "rules": [{"force": True}]},
            },
            "savedGroups": {},
        }
        tracked = []
        def cb(experiment, result, user_context):
            tracked.append((experiment.key,))

        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)):
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                on_experiment_viewed=cb,
                remoteEval=True,
            )
            gb.eval_feature("f")
            assert tracked == []

    def test_sync_malformed_track_entry_doesnt_crash(self):
        """A tracks entry missing required experiment fields is skipped, not raised."""
        body = {
            "features": {
                "f": {
                    "defaultValue": False,
                    "rules": [{
                        "force": True,
                        "tracks": [
                            {"experiment": {}, "result": {}},  # missing key/variations
                            {
                                "experiment": {"key": "exp", "variations": [0, 1]},
                                "result": {"variationId": 1, "inExperiment": True, "value": 1},
                            },
                        ],
                    }],
                },
            },
            "savedGroups": {},
        }
        tracked = []
        def cb(experiment, result, user_context):
            tracked.append(experiment.key)

        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)):
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                on_experiment_viewed=cb,
                remoteEval=True,
            )
            gb.eval_feature("f")
            assert tracked == ["exp"]  # only the valid one fired


@pytest.mark.asyncio
async def test_async_rule_tracks_fires_tracking_callback():
    """Same as sync — async client also fires rule.tracks on force-rule path."""
    body = {
        "features": {
            "ab-feature": {
                "defaultValue": False,
                "rules": [{
                    "force": True,
                    "tracks": [{
                        "experiment": {"key": "exp-A", "variations": [0, 1]},
                        "result": {
                            "variationId": 1, "inExperiment": True, "value": 1,
                            "hashUsed": True, "hashAttribute": "id", "hashValue": "u1",
                            "featureId": "ab-feature", "key": "1",
                        },
                    }],
                }],
            },
        },
        "savedGroups": {},
    }
    tracked = []

    async def post_handler(api_host, client_key, payload):
        return body

    def cb(experiment, result, *_):
        tracked.append((experiment.key, result.variationId))

    client = GrowthBookClient(Options(
        api_host="https://proxy.example.com",
        client_key="sdk-async",
        on_experiment_viewed=cb,
        remote_eval=True,
        refresh_strategy=None,
    ))
    client._features_repository._fetch_and_decode_post_async = post_handler
    await client.initialize()

    result = await client.eval_feature("ab-feature", UserContext(attributes={"id": "u1"}))
    assert result.value is True
    assert tracked == [("exp-A", 1)]


@pytest.mark.asyncio
async def test_async_non_remote_eval_still_works():
    """Regression: when remote_eval is False, initialize() does the CDN load
    and create_evaluation_context uses _global_context as before."""
    cdn_response = {"features": {"flag1": {"defaultValue": "cdn-value"}}, "savedGroups": {}}

    async def cdn_fetch(api_host, client_key, decryption_key=""):
        return cdn_response

    client = GrowthBookClient(Options(
        api_host="https://cdn.growthbook.io",
        client_key="k",
        refresh_strategy=None,
    ))
    client._features_repository._fetch_features_async = cdn_fetch
    feature_repo.clear_cache()
    client._features_repository.cache.cache.clear() if hasattr(client._features_repository.cache, "cache") else None
    assert await client.initialize() is True
    assert await client.get_feature_value("flag1", "fallback", UserContext(attributes={"id": "u1"})) == "cdn-value"
