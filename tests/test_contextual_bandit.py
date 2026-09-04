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
