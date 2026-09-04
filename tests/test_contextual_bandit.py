#!/usr/bin/env python
"""Contextual bandit tests that the shared cases.json corpus can't express:
payload ingestion in both clients, encrypted contextualBandits, and tracking
callback behavior. Evaluation semantics themselves are pinned by the
`contextualBandit` section of tests/cases.json."""

import json
import os
from base64 import b64decode, b64encode

import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from unittest.mock import patch, AsyncMock

from growthbook import (
    GrowthBook,
    GrowthBookClient,
    InMemoryStickyBucketService,
    feature_repo,
)
from growthbook.growthbook_client import EnhancedFeatureRepository, FeatureCache
from growthbook.common_types import Options, UserContext


CB_FEATURES = {
    "bandit-feature": {
        "defaultValue": "default",
        "rules": [
            {
                "key": "bandit-exp",
                "seed": "bandit-exp",
                "hashAttribute": "id",
                "hashVersion": 2,
                "coverage": 1,
                "contextualVariations": ["control", "treatment"],
                "weights": [0.5, 0.5],
                "meta": [{"key": "0"}, {"key": "1"}],
                "contextualBanditRef": "cb-bandit",
            }
        ],
    }
}

# Single catch-all leaf that sends everyone to variation 0.
CB_MAP = {
    "cb-bandit": {
        "banditVersion": 7,
        "contexts": [{"leafId": 1, "condition": {}, "weights": [1, 0]}],
    }
}


class MockHttpResp:
    def __init__(self, status: int, data: str) -> None:
        self.status = status
        self.data = data.encode("utf-8")
        self.headers: dict = {}


def _encrypt(payload: str, key_str: str) -> str:
    """Inverse of growthbook.decrypt (AES128-CBC, 'iv.ciphertext' base64)."""
    key = b64decode(key_str)
    iv = os.urandom(16)
    padder = padding.PKCS7(128).padder()
    padded = padder.update(payload.encode("utf-8")) + padder.finalize()
    cipher = Cipher(algorithms.AES128(key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    ct = encryptor.update(padded) + encryptor.finalize()
    return b64encode(iv).decode() + "." + b64encode(ct).decode()


def test_constructor_and_eval():
    gb = GrowthBook(
        attributes={"id": "1"},
        features=CB_FEATURES,
        contextualBandits=CB_MAP,
    )
    res = gb.eval_feature("bandit-feature")
    assert res.value == "control"
    assert res.experimentResult is not None
    assert res.experimentResult.leafId == 1
    assert res.experimentResult.variationWeights == [1, 0]
    assert res.experimentResult.banditVersion == 7
    gb.destroy()


def test_on_feature_update_ingestion():
    gb = GrowthBook(attributes={"id": "1"})
    gb._on_feature_update({"features": CB_FEATURES, "contextualBandits": CB_MAP})
    assert gb._global_ctx.contextual_bandits == CB_MAP
    res = gb.eval_feature("bandit-feature")
    assert res.value == "control"
    assert res.experimentResult.leafId == 1
    gb.destroy()


def test_load_features_ingestion(mocker):
    m = mocker.patch.object(feature_repo, "_get")
    m.return_value = MockHttpResp(
        200, json.dumps({"features": CB_FEATURES, "contextualBandits": CB_MAP})
    )
    gb = GrowthBook(
        api_host="https://cdn.growthbook.io",
        client_key="sdk-cb-ingest",
        attributes={"id": "1"},
    )
    gb.load_features()
    assert gb._global_ctx.contextual_bandits == CB_MAP
    assert gb.eval_feature("bandit-feature").value == "control"
    gb.destroy()
    feature_repo.clear_cache()


def test_decrypt_response_encrypted_contextual_bandits():
    key = "Zvwv/+uhpFDznZ6SX28Yjg=="
    data = {
        "features": CB_FEATURES,
        "encryptedContextualBandits": _encrypt(json.dumps(CB_MAP), key),
    }
    result = feature_repo.decrypt_response(data, key)
    assert result["contextualBandits"] == CB_MAP
    assert "encryptedContextualBandits" not in result

    # Missing decryption key raises like the other encrypted payload keys
    with pytest.raises(ValueError):
        feature_repo.decrypt_response(
            {"features": {}, "encryptedContextualBandits": "abc.def"}, ""
        )


def test_tracking_callback_receives_bandit_result():
    tracked = []

    def on_view(experiment, result, user_context):
        tracked.append((experiment, result, user_context))

    gb = GrowthBook(
        attributes={"id": "1", "country": "US"},
        features=CB_FEATURES,
        contextualBandits=CB_MAP,
        on_experiment_viewed=on_view,
    )
    gb.eval_feature("bandit-feature")
    assert len(tracked) == 1
    experiment, result, user_context = tracked[0]
    assert result.leafId == 1
    assert result.variationWeights == [1, 0]
    assert result.banditVersion == 7
    # The exact attributes used at bucketing time are available for logging
    assert user_context.attributes == {"id": "1", "country": "US"}
    gb.destroy()


def test_non_bandit_result_has_no_bandit_metadata():
    gb = GrowthBook(
        attributes={"id": "1"},
        features={
            "plain": {
                "defaultValue": 0,
                "rules": [{"key": "plain-exp", "variations": [0, 1]}],
            }
        },
    )
    res = gb.eval_feature("plain")
    assert res.experimentResult is not None
    assert res.experimentResult.leafId is None
    serialized = res.experimentResult.to_dict()
    assert "leafId" not in serialized
    assert "variationWeights" not in serialized
    assert "banditVersion" not in serialized
    gb.destroy()


def test_one_bandit_shared_by_two_features():
    features = dict(CB_FEATURES)
    features["bandit-feature-2"] = {
        "defaultValue": "default",
        "rules": [
            {
                "key": "bandit-exp-2",
                "seed": "bandit-exp-2",
                "hashAttribute": "id",
                "hashVersion": 2,
                "coverage": 1,
                "contextualVariations": ["a", "b"],
                "weights": [0.5, 0.5],
                "contextualBanditRef": "cb-bandit",
            }
        ],
    }
    gb = GrowthBook(
        attributes={"id": "1"},
        features=features,
        contextualBandits=CB_MAP,
    )
    res1 = gb.eval_feature("bandit-feature")
    res2 = gb.eval_feature("bandit-feature-2")
    assert res1.value == "control"
    assert res2.value == "a"
    assert res1.experimentResult.leafId == 1
    assert res2.experimentResult.leafId == 1
    gb.destroy()


def test_sync_set_payload_seeds_bandit_map():
    gb = GrowthBook(attributes={"id": "1"})
    gb.set_payload({"features": CB_FEATURES, "contextualBandits": CB_MAP})
    res = gb.eval_feature("bandit-feature")
    assert res.value == "control"
    assert res.experimentResult.leafId == 1
    # A later payload without the section preserves it
    gb.set_payload({"features": CB_FEATURES})
    assert gb.eval_feature("bandit-feature").experimentResult.leafId == 1
    gb.destroy()


@pytest.mark.asyncio
async def test_async_set_payload_seeds_bandit_map():
    EnhancedFeatureRepository._instances = {}
    with patch(
        "growthbook.FeatureRepository.load_features_async",
        new_callable=AsyncMock,
        return_value={"features": {}, "savedGroups": {}},
    ), patch(
        "growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh",
        new_callable=AsyncMock,
    ), patch(
        "growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh",
        new_callable=AsyncMock,
    ):
        async with GrowthBookClient(
            Options(api_host="https://localhost.growthbook.io", client_key="test-key")
        ) as client:
            await client.set_payload(
                {"features": CB_FEATURES, "contextualBandits": CB_MAP}
            )
            result = await client.eval_feature(
                "bandit-feature", UserContext(attributes={"id": "1"})
            )
            assert result.value == "control"
            assert result.experimentResult.leafId == 1


def test_feature_cache_round_trip():
    cache = FeatureCache()
    cache.update({"f": {"defaultValue": 1}}, {"sg": []}, CB_MAP)
    state = cache.get_current_state()
    assert state["contextualBandits"] == CB_MAP
    # A refresh whose payload omits a section (None) preserves its current
    # value; an explicit empty dict clears it.
    cache.update({"f": {"defaultValue": 2}})
    assert cache.get_current_state()["contextualBandits"] == CB_MAP
    assert cache.get_current_state()["savedGroups"] == {"sg": []}
    cache.update(None, {}, {})
    assert cache.get_current_state()["contextualBandits"] == {}
    assert cache.get_current_state()["features"] == {"f": {"defaultValue": 2}}


@pytest.mark.asyncio
async def test_refresh_without_bandit_key_preserves_map():
    """A second refresh whose payload lacks contextualBandits (partial or
    broken payload) must not wipe the map mid-flight — evals keep routing
    with the last known weights."""
    EnhancedFeatureRepository._instances = {}
    full_payload = {
        "features": CB_FEATURES,
        "savedGroups": {},
        "contextualBandits": CB_MAP,
    }
    with patch(
        "growthbook.FeatureRepository.load_features_async",
        new_callable=AsyncMock,
        return_value=full_payload,
    ), patch(
        "growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh",
        new_callable=AsyncMock,
    ), patch(
        "growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh",
        new_callable=AsyncMock,
    ):
        async with GrowthBookClient(
            Options(api_host="https://localhost.growthbook.io", client_key="test-key")
        ) as client:
            repo = client._features_repository
            # A full refresh populates the cache...
            await repo._handle_feature_update(full_payload)
            # ...then a partial refresh missing the CB section must not wipe it
            await repo._handle_feature_update({"features": CB_FEATURES})
            assert repo._feature_cache.get_current_state()["contextualBandits"] == CB_MAP
            result = await client.eval_feature(
                "bandit-feature", UserContext(attributes={"id": "1"})
            )
            assert result.value == "control"
            assert result.experimentResult.leafId == 1


@pytest.mark.asyncio
async def test_async_set_features_preserves_bandit_map():
    """A partial update (set_features) must not wipe the contextualBandits or
    savedGroups sections — mirrors JS setPayload, which only overwrites the
    sections present in the payload."""
    EnhancedFeatureRepository._instances = {}
    payload = {
        "features": CB_FEATURES,
        "savedGroups": {"sg": ["1"]},
        "contextualBandits": CB_MAP,
    }
    with patch(
        "growthbook.FeatureRepository.load_features_async",
        new_callable=AsyncMock,
        return_value=payload,
    ), patch(
        "growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh",
        new_callable=AsyncMock,
    ), patch(
        "growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh",
        new_callable=AsyncMock,
    ):
        async with GrowthBookClient(
            Options(api_host="https://localhost.growthbook.io", client_key="test-key")
        ) as client:
            await client.set_features(CB_FEATURES)
            assert client._global_context.contextual_bandits == CB_MAP
            assert client._global_context.saved_groups == {"sg": ["1"]}
            result = await client.eval_feature(
                "bandit-feature", UserContext(attributes={"id": "1"})
            )
            assert result.value == "control"
            assert result.experimentResult.leafId == 1


def test_tracking_callback_gets_exposure_time_attribute_snapshot():
    """Attributes mutated after evaluation must not leak into the tracked
    user context — the warehouse row has to carry the attributes used for
    leaf routing (JS SDK: getTrackingUserContext snapshot)."""
    tracked = []

    def on_view(experiment, result, user_context):
        tracked.append(user_context)

    gb = GrowthBook(
        attributes={"id": "1", "country": "US"},
        features=CB_FEATURES,
        contextualBandits=CB_MAP,
        on_experiment_viewed=on_view,
    )
    gb.eval_feature("bandit-feature")
    gb.set_attributes({"id": "1", "country": "DE"})
    assert tracked[0].attributes == {"id": "1", "country": "US"}
    gb.destroy()


@pytest.mark.asyncio
async def test_async_client_contextual_bandits():
    EnhancedFeatureRepository._instances = {}
    payload = {
        "features": CB_FEATURES,
        "savedGroups": {},
        "contextualBandits": CB_MAP,
    }
    with patch(
        "growthbook.FeatureRepository.load_features_async",
        new_callable=AsyncMock,
        return_value=payload,
    ), patch(
        "growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh",
        new_callable=AsyncMock,
    ), patch(
        "growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh",
        new_callable=AsyncMock,
    ):
        async with GrowthBookClient(
            Options(api_host="https://localhost.growthbook.io", client_key="test-key")
        ) as client:
            result = await client.eval_feature(
                "bandit-feature", UserContext(attributes={"id": "1"})
            )
            assert result.value == "control"
            assert result.experimentResult.leafId == 1
            assert result.experimentResult.variationWeights == [1, 0]
            assert result.experimentResult.banditVersion == 7


def test_sync_set_payload_bandits_only_republishes():
    """A payload carrying only contextualBandits must take effect immediately,
    not wait for the next features update to republish the eval context."""
    gb = GrowthBook(attributes={"id": "1"}, features=CB_FEATURES, contextualBandits=CB_MAP)
    assert gb.eval_feature("bandit-feature").value == "control"

    flipped = {
        "cb-bandit": {
            "banditVersion": 8,
            "contexts": [{"leafId": 2, "condition": {}, "weights": [0, 1]}],
        }
    }
    gb.set_payload({"contextualBandits": flipped})
    res = gb.eval_feature("bandit-feature")
    assert res.value == "treatment"
    assert res.experimentResult.leafId == 2
    assert res.experimentResult.banditVersion == 8

    # An explicit empty map clears it: the ref dangles, so the rule runs as a
    # plain experiment on aggregate weights with no bandit metadata.
    gb.set_payload({"contextualBandits": {}})
    assert gb.eval_feature("bandit-feature").experimentResult.leafId is None
    gb.destroy()


def test_sync_refresh_swaps_a_coherent_snapshot():
    """Refreshes rebind one new GlobalContext instead of mutating fields in
    place, so a concurrent eval holding the old snapshot never sees features
    from one payload generation with bandit weights from another."""
    gb = GrowthBook(attributes={"id": "1"}, features=CB_FEATURES, contextualBandits=CB_MAP)
    before = gb._global_ctx
    gb.set_payload(
        {
            "features": CB_FEATURES,
            "contextualBandits": {
                "cb-bandit": {"contexts": [{"leafId": 2, "condition": {}, "weights": [0, 1]}]}
            },
        }
    )
    assert gb._global_ctx is not before
    # The old snapshot is untouched — in-flight evals stay coherent.
    assert before.contextual_bandits == CB_MAP
    gb.destroy()


def test_malformed_bandit_payload_degrades_gracefully():
    """Malformed definitions/leaves must never crash evaluation: a matched
    leaf missing weights or leafId falls back to aggregate weights (leaf -1),
    like a definition of the wrong shape. Only a null definition is treated
    as a dangling ref (plain experiment, no metadata)."""
    missing_weights = {
        "cb-bandit": {"banditVersion": 7, "contexts": [{"leafId": 1, "condition": {}}]}
    }
    missing_leaf_id = {
        "cb-bandit": {"contexts": [{"condition": {}, "weights": [1, 0]}]}
    }

    def leaf_weights(weights):
        return {"cb-bandit": {"contexts": [{"leafId": 1, "condition": {}, "weights": weights}]}}

    for bad_map in (
        missing_weights,
        missing_leaf_id,
        {"cb-bandit": [1, 2]},
        {"cb-bandit": {}},
        # Weight vectors that bucketing would reject (or crash on) are
        # treated as malformed leaves, so reported propensities always match
        # the weights actually used.
        leaf_weights("ab"),          # not a list
        leaf_weights(5),             # not sized
        leaf_weights([1, 0, 0]),     # wrong length for 2 variations
        leaf_weights([1, "x"]),      # non-numeric entry
        leaf_weights([0.9, 0.9]),    # sum outside bucketing tolerance
        leaf_weights([10**1000, 0]),  # overflows float conversion — must not crash
    ):
        gb = GrowthBook(attributes={"id": "1"}, features=CB_FEATURES, contextualBandits=bad_map)
        res = gb.eval_feature("bandit-feature")
        assert res.experimentResult.leafId == -1
        assert res.experimentResult.variationWeights == [0.5, 0.5]
        gb.destroy()

    # banditVersion still reported when the definition carries one
    gb = GrowthBook(attributes={"id": "1"}, features=CB_FEATURES, contextualBandits=missing_weights)
    assert gb.eval_feature("bandit-feature").experimentResult.banditVersion == 7
    gb.destroy()

    gb = GrowthBook(attributes={"id": "1"}, features=CB_FEATURES, contextualBandits={"cb-bandit": None})
    res = gb.eval_feature("bandit-feature")
    assert res.experimentResult is not None
    assert res.experimentResult.leafId is None
    gb.destroy()


def test_sync_set_payload_accepts_encrypted_sections():
    """JS setPayload accepts encrypted payloads; the Python port decrypts
    encrypted sections with the configured decryption_key."""
    key = "Zvwv/+uhpFDznZ6SX28Yjg=="
    gb = GrowthBook(attributes={"id": "1"}, decryption_key=key)
    gb.set_payload(
        {
            "encryptedFeatures": _encrypt(json.dumps(CB_FEATURES), key),
            "encryptedContextualBandits": _encrypt(json.dumps(CB_MAP), key),
        }
    )
    res = gb.eval_feature("bandit-feature")
    assert res.value == "control"
    assert res.experimentResult.leafId == 1
    gb.destroy()


@pytest.mark.asyncio
async def test_async_set_payload_accepts_encrypted_sections():
    key = "Zvwv/+uhpFDznZ6SX28Yjg=="
    EnhancedFeatureRepository._instances = {}
    with patch(
        "growthbook.FeatureRepository.load_features_async",
        new_callable=AsyncMock,
        return_value={"features": {}, "savedGroups": {}},
    ), patch(
        "growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh",
        new_callable=AsyncMock,
    ), patch(
        "growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh",
        new_callable=AsyncMock,
    ):
        async with GrowthBookClient(
            Options(
                api_host="https://localhost.growthbook.io",
                client_key="test-key",
                decryption_key=key,
            )
        ) as client:
            await client.set_payload(
                {
                    "encryptedFeatures": _encrypt(json.dumps(CB_FEATURES), key),
                    "encryptedContextualBandits": _encrypt(json.dumps(CB_MAP), key),
                }
            )
            result = await client.eval_feature(
                "bandit-feature", UserContext(attributes={"id": "1"})
            )
            assert result.value == "control"
            assert result.experimentResult.leafId == 1


def test_failed_bandit_decryption_preserves_previous_map():
    """An undecryptable contextualBandits section is dropped (encrypted key
    removed, like JS decryptPayload) and, being absent, preserves the previous
    coherent map instead of wiping it."""
    key = "Zvwv/+uhpFDznZ6SX28Yjg=="
    out = feature_repo.decrypt_response(
        {"features": {}, "encryptedContextualBandits": "bad.cipher"}, key
    )
    assert out is not None
    assert "encryptedContextualBandits" not in out
    assert "contextualBandits" not in out

    gb = GrowthBook(
        attributes={"id": "1"},
        decryption_key=key,
        features=CB_FEATURES,
        contextualBandits=CB_MAP,
    )
    gb.set_payload({"features": CB_FEATURES, "encryptedContextualBandits": "bad.cipher"})
    assert gb.eval_feature("bandit-feature").experimentResult.leafId == 1
    gb.destroy()


def test_remote_eval_tracks_preserve_bandit_fields():
    """rule.tracks entries from the remote-eval proxy must reach the tracking
    callback with their contextual bandit fields intact — the JS SDK passes
    the proxy result through verbatim."""
    tracked = []

    def on_view(experiment, result, user_context):
        tracked.append(result)

    gb = GrowthBook(
        attributes={"id": "1"},
        on_experiment_viewed=on_view,
        features={
            "remote-feature": {
                "defaultValue": "x",
                "rules": [
                    {
                        "force": "treatment",
                        "tracks": [
                            {
                                "experiment": {
                                    "key": "bandit-exp",
                                    "variations": ["control", "treatment"],
                                },
                                "result": {
                                    "variationId": 1,
                                    "inExperiment": True,
                                    "hashUsed": True,
                                    "hashAttribute": "id",
                                    "hashValue": "1",
                                    "value": "treatment",
                                    "key": "1",
                                    "featureId": "remote-feature",
                                    "leafId": 8,
                                    "variationWeights": [0.2, 0.8],
                                    "banditVersion": 7,
                                },
                            }
                        ],
                    }
                ],
            }
        },
    )
    assert gb.eval_feature("remote-feature").value == "treatment"
    assert len(tracked) == 1
    assert tracked[0].leafId == 8
    assert tracked[0].variationWeights == [0.2, 0.8]
    assert tracked[0].banditVersion == 7
    gb.destroy()


@pytest.mark.asyncio
async def test_async_remote_eval_tracks_preserve_bandit_fields():
    tracked = []

    def on_view(experiment, result, user_context):
        tracked.append(result)

    EnhancedFeatureRepository._instances = {}
    with patch(
        "growthbook.FeatureRepository.load_features_async",
        new_callable=AsyncMock,
        return_value={"features": {}, "savedGroups": {}},
    ), patch(
        "growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh",
        new_callable=AsyncMock,
    ), patch(
        "growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh",
        new_callable=AsyncMock,
    ):
        async with GrowthBookClient(
            Options(
                api_host="https://localhost.growthbook.io",
                client_key="test-key",
                on_experiment_viewed=on_view,
            )
        ) as client:
            await client.set_payload(
                {
                    "features": {
                        "remote-feature": {
                            "defaultValue": "x",
                            "rules": [
                                {
                                    "force": "treatment",
                                    "tracks": [
                                        {
                                            "experiment": {
                                                "key": "bandit-exp",
                                                "variations": ["control", "treatment"],
                                            },
                                            "result": {
                                                "variationId": 1,
                                                "inExperiment": True,
                                                "hashUsed": True,
                                                "hashAttribute": "id",
                                                "hashValue": "1",
                                                "value": "treatment",
                                                "key": "1",
                                                "leafId": 8,
                                                "variationWeights": [0.2, 0.8],
                                                "banditVersion": 7,
                                            },
                                        }
                                    ],
                                }
                            ],
                        }
                    }
                }
            )
            result = await client.eval_feature(
                "remote-feature", UserContext(attributes={"id": "1"})
            )
            assert result.value == "treatment"
    assert len(tracked) == 1
    assert tracked[0].leafId == 8
    assert tracked[0].variationWeights == [0.2, 0.8]
    assert tracked[0].banditVersion == 7


def test_concurrent_payload_writers_publish_coherent_generations():
    """Two writers (e.g. set_payload and a background refresh) must not
    interleave section writes: every published snapshot pairs features and
    contextualBandits from the same payload generation. Deterministic: writer
    A is held mid-ingest while writer B runs; without writer serialization, A
    would then publish its features with B's bandit map."""
    import threading

    gen_a = {
        "features": {"gen": {"defaultValue": "A"}},
        "contextualBandits": {"gen": {"banditVersion": 1, "contexts": []}},
    }
    gen_b = {
        "features": {"gen": {"defaultValue": "B"}},
        "contextualBandits": {"gen": {"banditVersion": 2, "contexts": []}},
    }

    gb = GrowthBook(attributes={"id": "1"})
    published = []
    orig_publish = gb._publish_global_context

    def recording_publish():
        orig_publish()
        ctx = gb._global_ctx
        feature = ctx.features.get("gen")
        bandit = ctx.contextual_bandits.get("gen") or {}
        published.append((feature.defaultValue if feature else None, bandit.get("banditVersion")))

    gb._publish_global_context = recording_publish

    gate = threading.Event()
    held = threading.Event()
    orig_set_features = gb.set_features

    def slow_set_features(features):
        if not held.is_set():
            held.set()
            gate.wait(timeout=5)
        orig_set_features(features)

    gb.set_features = slow_set_features

    writer_a = threading.Thread(target=gb.set_payload, args=(gen_a,))
    writer_b = threading.Thread(target=gb.set_payload, args=(gen_b,))
    writer_a.start()
    assert held.wait(timeout=5)
    writer_b.start()
    gate.set()
    writer_a.join(timeout=5)
    writer_b.join(timeout=5)

    assert published, "writers must have published"
    for pair in published:
        assert pair in (("A", 1), ("B", 2)), f"mixed payload generation published: {pair}"
    gb.destroy()


# Leaf routing on both a top-level and a nested attribute, for the
# mutation-freezing tests below.
CB_NESTED_FEATURES = {
    "bandit-feature": {
        "defaultValue": "default",
        "rules": [
            {
                "key": "bandit-exp",
                "seed": "bandit-exp",
                "hashAttribute": "id",
                "hashVersion": 2,
                "coverage": 1,
                "contextualVariations": ["control", "treatment"],
                "weights": [0.5, 0.5],
                "contextualBanditRef": "cb-bandit",
            }
        ],
    }
}
CB_NESTED_MAP = {
    "cb-bandit": {
        "banditVersion": 7,
        "contexts": [
            {
                "leafId": 1,
                "condition": {"country": "US", "profile.tier": "pro"},
                "weights": [1, 0],
            },
            {"leafId": 2, "condition": {}, "weights": [0, 1]},
        ],
    }
}


@pytest.mark.asyncio
async def test_async_eval_freezes_attributes_before_awaits():
    """Attributes are snapshotted at the async eval boundary, before the
    first await: a caller (or another task) mutating the UserContext while
    sticky-bucket I/O is pending must not change leaf routing or what the
    tracking callback reports — for top-level or nested keys."""
    tracked = []

    def on_view(experiment, result, user_context):
        tracked.append((result, user_context))

    EnhancedFeatureRepository._instances = {}
    with patch(
        "growthbook.FeatureRepository.load_features_async",
        new_callable=AsyncMock,
        return_value={
            "features": CB_NESTED_FEATURES,
            "savedGroups": {},
            "contextualBandits": CB_NESTED_MAP,
        },
    ), patch(
        "growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh",
        new_callable=AsyncMock,
    ), patch(
        "growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh",
        new_callable=AsyncMock,
    ):
        async with GrowthBookClient(
            Options(
                api_host="https://localhost.growthbook.io",
                client_key="test-key",
                on_experiment_viewed=on_view,
                # Arms the boundary freeze: only evals that can yield
                # (sticky/remote I/O) snapshot their inputs up front.
                sticky_bucket_service=InMemoryStickyBucketService(),
            )
        ) as client:
            caller_attrs = {"id": "1", "country": "US", "profile": {"tier": "pro"}}
            user_context = UserContext(attributes=caller_attrs)

            async def mutating_refresh(attributes):
                # Simulates a concurrent task mutating the caller's context
                # while the eval awaits sticky-bucket I/O.
                caller_attrs["country"] = "DE"
                caller_attrs["profile"]["tier"] = "basic"
                return {}

            client._refresh_sticky_buckets = mutating_refresh

            result = await client.eval_feature("bandit-feature", user_context)
            # Routed with the attributes the call started with, not the
            # mutated ones (which match only the catch-all leaf 2).
            assert result.value == "control"
            assert result.experimentResult.leafId == 1

    (tracked_result, tracked_user) = tracked[0]
    assert tracked_result.leafId == 1
    assert tracked_user.attributes["country"] == "US"
    assert tracked_user.attributes["profile"]["tier"] == "pro"


def test_tracking_snapshot_preserves_nested_attributes():
    """Deferred tracking callbacks must see the nested attribute values used
    at exposure time, even after the caller mutates them (the snapshot copies
    containers recursively, not just the top level)."""
    tracked = []

    def on_view(experiment, result, user_context):
        tracked.append(user_context)

    attrs = {"id": "1", "country": "US", "profile": {"tier": "pro"}}
    gb = GrowthBook(
        attributes=attrs,
        features=CB_NESTED_FEATURES,
        contextualBandits=CB_NESTED_MAP,
        on_experiment_viewed=on_view,
    )
    assert gb.eval_feature("bandit-feature").experimentResult.leafId == 1
    attrs["profile"]["tier"] = "basic"
    assert tracked[0].attributes["profile"]["tier"] == "pro"
    gb.destroy()


def test_bandit_metadata_reports_weights_bucketing_uses():
    """Invalid vectors on the fallback and override paths are normalized the
    same way getBucketRanges normalizes them, so Result.variationWeights can
    never differ from the weights actually used. Negative, non-finite, and
    boolean leaf weights are rejected outright."""
    # Fallback path: rule weights [0.9, 0.9] are invalid (sum 1.8), so
    # bucketing uses equal weights — the metadata must say so too.
    bad_marginals = {
        "bandit-feature": {
            "defaultValue": "default",
            "rules": [
                {
                    "key": "bandit-exp",
                    "seed": "bandit-exp",
                    "hashAttribute": "id",
                    "hashVersion": 2,
                    "coverage": 1,
                    "contextualVariations": ["control", "treatment"],
                    "weights": [0.9, 0.9],
                    "contextualBanditRef": "cb-bandit",
                }
            ],
        }
    }
    gb = GrowthBook(
        attributes={"id": "1"},
        features=bad_marginals,
        contextualBandits={"cb-bandit": {"contexts": []}},
    )
    res = gb.eval_feature("bandit-feature")
    assert res.experimentResult.leafId == -1
    assert res.experimentResult.variationWeights == [0.5, 0.5]
    gb.destroy()

    # Matched-leaf path: negative / non-finite / boolean entries are malformed.
    for bad_weights in ([1.5, -0.5], [float("inf"), 1.0], [True, False]):
        bad_map = {
            "cb-bandit": {
                "contexts": [{"leafId": 1, "condition": {}, "weights": bad_weights}]
            }
        }
        gb = GrowthBook(attributes={"id": "1"}, features=CB_FEATURES, contextualBandits=bad_map)
        res = gb.eval_feature("bandit-feature")
        assert res.experimentResult.leafId == -1, bad_weights
        assert res.experimentResult.variationWeights == [0.5, 0.5]
        gb.destroy()


@pytest.mark.asyncio
async def test_plain_cdn_eval_skips_the_boundary_copy():
    """Without remote eval or a sticky bucket service, create_evaluation_context
    has no await that can yield, so it must not pay for an input snapshot —
    plain evaluations stay allocation-free (the eval context carries the
    caller's own UserContext object)."""
    EnhancedFeatureRepository._instances = {}
    with patch(
        "growthbook.FeatureRepository.load_features_async",
        new_callable=AsyncMock,
        return_value={"features": CB_FEATURES, "savedGroups": {}, "contextualBandits": CB_MAP},
    ), patch(
        "growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh",
        new_callable=AsyncMock,
    ), patch(
        "growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh",
        new_callable=AsyncMock,
    ):
        async with GrowthBookClient(
            Options(api_host="https://localhost.growthbook.io", client_key="test-key")
        ) as client:
            user_context = UserContext(attributes={"id": "1"})
            context = await client.create_evaluation_context(user_context)
            assert context.user is user_context
            result = await client.eval_feature("bandit-feature", user_context)
            assert result.value == "control"


@pytest.mark.asyncio
async def test_async_eval_freezes_forced_inputs_before_awaits():
    """The boundary snapshot covers every mutable evaluation input, not just
    attributes: forced variations and overrides mutated while sticky I/O is
    pending must not turn a hashed assignment into a forced one."""
    EnhancedFeatureRepository._instances = {}
    with patch(
        "growthbook.FeatureRepository.load_features_async",
        new_callable=AsyncMock,
        return_value={"features": CB_FEATURES, "savedGroups": {}, "contextualBandits": CB_MAP},
    ), patch(
        "growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh",
        new_callable=AsyncMock,
    ), patch(
        "growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh",
        new_callable=AsyncMock,
    ):
        async with GrowthBookClient(
            Options(
                api_host="https://localhost.growthbook.io",
                client_key="test-key",
                sticky_bucket_service=InMemoryStickyBucketService(),
            )
        ) as client:
            forced = {}
            overrides = {}
            user_context = UserContext(
                attributes={"id": "1"}, forced_variations=forced, overrides=overrides
            )

            async def mutating_refresh(attributes):
                forced["bandit-exp"] = 1
                overrides["bandit-exp"] = {"force": 1}
                return {}

            client._refresh_sticky_buckets = mutating_refresh

            result = await client.eval_feature("bandit-feature", user_context)
            # The catch-all leaf sends everyone to variation 0; the mid-await
            # forced inputs must not flip this eval to variation 1.
            assert result.value == "control"
            assert result.experimentResult.variationId == 0
            assert result.experimentResult.hashUsed


def test_invalid_aggregate_and_override_weights_are_sanitized():
    """Aggregate (fallback) and override weight vectors get the same strict
    validation as matched leaves: negative or boolean entries would bucket on
    nonsense ranges and report them as propensities, so both bucketing and
    the reported metadata normalize them to equal weights."""
    for bad_weights in ([1.5, -0.5], [True, False]):
        features = {
            "bandit-feature": {
                "defaultValue": "default",
                "rules": [
                    {
                        "key": "bandit-exp",
                        "seed": "bandit-exp",
                        "hashAttribute": "id",
                        "hashVersion": 2,
                        "coverage": 1,
                        "contextualVariations": ["control", "treatment"],
                        "weights": bad_weights,
                        "contextualBanditRef": "cb-bandit",
                    }
                ],
            }
        }
        gb = GrowthBook(
            attributes={"id": "1"},
            features=features,
            contextualBandits={"cb-bandit": {"contexts": []}},
        )
        res = gb.eval_feature("bandit-feature")
        assert res.experimentResult.leafId == -1, bad_weights
        assert res.experimentResult.variationWeights == [0.5, 0.5], bad_weights
        gb.destroy()

    # Override path: a context override replacing the weights with an
    # unusable vector is cleared during the re-sync.
    gb = GrowthBook(
        attributes={"id": "1"},
        features=CB_FEATURES,
        contextualBandits=CB_MAP,
        overrides={"bandit-exp": {"weights": [1.5, -0.5]}},
    )
    res = gb.eval_feature("bandit-feature")
    assert res.experimentResult is not None
    assert res.experimentResult.variationWeights == [0.5, 0.5]
    gb.destroy()


def test_bucket_ranges_reject_invalid_weight_vectors():
    """One weight-validation policy for every experiment: vectors bucketing
    cannot honor sanely (negative, non-finite, boolean, or non-numeric
    entries — not just wrong length/sum) normalize to equal weights, so
    bucket ranges can never be inverted. Deliberately stricter than the JS
    SDK, which checks only length and sum, on these invalid payloads."""
    from growthbook.core import getBucketRanges

    equal = getBucketRanges(2, 1, None)
    for bad in (
        [1.2, -0.2],
        [True, False],
        [float("inf"), 1.0],
        [float("nan"), 1.0],
        [0.5, "x"],
        # Arbitrary-precision ints overflow math.isfinite / float summation;
        # validation must stay total instead of raising OverflowError.
        [10**1000, 0],
        [10**1000, 0.5],
    ):
        assert getBucketRanges(2, 1, bad) == equal, bad
    # Valid vectors are untouched
    assert getBucketRanges(2, 1, [0.4, 0.6]) == [(0, 0.4), (0.4, 1.0)]
