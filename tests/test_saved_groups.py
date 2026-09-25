"""Saved group references v2 conformance and Python-specific regressions."""

import asyncio
import json
import os
import sys
from base64 import b64decode, b64encode
from copy import deepcopy
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from unittest.mock import AsyncMock

from growthbook import Experiment, FeatureRepository, GrowthBook, GrowthBookClient, Options, UserContext, feature_repo
from growthbook.core import evalCondition
from growthbook.growthbook_client import EnhancedFeatureRepository


CASES = json.loads(Path(__file__).with_name("cases.json").read_text())[
    "savedGroupReferencesV2"
]


@pytest.mark.parametrize("case", CASES["evalCondition"], ids=lambda case: case[0])
def test_condition_conformance(case):
    _, condition, attributes, expected, saved_groups = case
    assert evalCondition(attributes, condition, saved_groups) is expected


@pytest.mark.parametrize("case", CASES["feature"], ids=lambda case: case[0])
def test_sync_feature_conformance(case):
    _, context, key, expected = deepcopy(case)
    client = GrowthBook(**context)
    try:
        assert client.eval_feature(key).to_dict() == expected
    finally:
        client.destroy()


@pytest.mark.parametrize("case", CASES["run"], ids=lambda case: case[0])
def test_sync_experiment_conformance(case):
    _, context, experiment, value, in_experiment, hash_used = deepcopy(case)
    client = GrowthBook(**context)
    try:
        result = client.run(Experiment(**experiment))
        assert (result.value, result.inExperiment, result.hashUsed) == (
            value, in_experiment, hash_used
        )
    finally:
        client.destroy()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES["feature"], ids=lambda case: case[0])
async def test_async_feature_conformance(case):
    _, context, key, expected = deepcopy(case)
    client = GrowthBookClient(Options())
    try:
        await client.set_payload(context)
        result = await client.eval_feature(key, UserContext(attributes=context["attributes"]))
        assert result.to_dict() == expected
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES["run"], ids=lambda case: case[0])
async def test_async_experiment_conformance(case):
    _, context, experiment, value, in_experiment, hash_used = deepcopy(case)
    client = GrowthBookClient(Options())
    try:
        await client.set_payload(context)
        result = await client.run(Experiment(**experiment), UserContext(attributes=context["attributes"]))
        assert (result.value, result.inExperiment, result.hashUsed) == (
            value, in_experiment, hash_used
        )
    finally:
        await client.close()


@pytest.mark.parametrize("entry", [
    {"type": "list", "attributeKey": "id", "values": ["u1"]},
    {"type": "condition", "condition": {"id": "u1"}},
])
@pytest.mark.parametrize("reference", [
    None, False, 7, "g", [], ["g"], {},
    {"id": None}, {"id": 7}, {"id": []}, {"id": {}},
    *({"id": "g", "attributeKey": key} for key in [None, 7, False, [], {}]),
])
def test_invalid_references_fail_closed(reference, entry):
    assert evalCondition({"id": "u1"}, {"$savedGroup": reference}, {"g": entry}) is False


@pytest.mark.parametrize("entry", [
    None, False, 7, "g", {}, {"type": "future"}, {"type": []},
    {"type": "list", "attributeKey": "id"},
    *({"type": "list", "attributeKey": "id", "values": values} for values in [None, 7, {}, "u1"]),
    {"type": "condition"},
    *({"type": "condition", "condition": condition} for condition in [
        None, 7, [], "g", {"$and": 7}, {"$not": None}, {"id": {"": True}},
    ]),
])
def test_malformed_entries_fail_closed(entry):
    groups = {"bad": entry, "good": {"type": "condition", "condition": {}}}
    attributes = {"id": "u1"}
    for condition in [
        {"$savedGroup": {"id": "bad"}},
        {"id": {"$inGroup": "bad"}},
        {"id": {"$notInGroup": "bad"}},
    ]:
        assert evalCondition(attributes, condition, groups) is False
    assert evalCondition(attributes, {"$savedGroup": {"id": "good"}}, groups) is True
    assert evalCondition(attributes, {"id": {"$notInGroup": "missing"}}, groups) is True


@pytest.mark.parametrize("groups", [None, {}])
def test_legacy_operators_without_saved_groups(groups):
    assert evalCondition({"id": "u1"}, {"id": {"$inGroup": "missing"}}, groups) is False
    assert evalCondition({"id": "u1"}, {"id": {"$notInGroup": "missing"}}, groups) is True


@pytest.mark.parametrize("entry_key", [None, 7, "id"])
def test_override_and_legacy_operators_supply_the_attribute(entry_key):
    entry = {"type": "list", "attributeKey": entry_key, "values": ["u1"]}
    groups = {"g": entry}
    attributes = {"profile": {"id": "u1"}, "id": "u2"}
    assert evalCondition(attributes, {"$savedGroup": {"id": "g", "attributeKey": "profile.id"}}, groups)
    assert evalCondition(attributes, {"profile.id": {"$inGroup": "g"}}, groups)
    assert not evalCondition(attributes, {"$savedGroup": {"id": "g"}}, groups)


def test_overrides_and_visited_groups_do_not_leak_to_siblings():
    groups = {
        "a": {"type": "condition", "condition": {"$savedGroup": {"id": "b"}}},
        "b": {"type": "list", "attributeKey": "id", "values": ["u1"]},
    }
    original = deepcopy(groups)
    attributes = {"id": "u1", "backup": "u2"}
    reference = {"$savedGroup": {"id": "a", "future": {"key": True}}}
    assert evalCondition(attributes, {"$and": [reference, reference]}, groups)
    assert evalCondition(attributes, {"$or": [{"$and": [reference, {"missing": True}]}, reference]}, groups)
    assert evalCondition(attributes, {"$and": [
        {"$not": {"$savedGroup": {"id": "b", "attributeKey": "backup"}}},
        {"$savedGroup": {"id": "a", "attributeKey": "backup"}},
    ]}, groups)
    assert groups == original


@pytest.mark.parametrize("condition, expected", [
    ({"$savedGroup": {"id": "g"}}, False),
    ({"$and": [{"$savedGroup": {"id": "g"}}]}, False),
    ({"$or": [{"$savedGroup": {"id": "g"}}]}, False),
    ({"$nor": [{"$savedGroup": {"id": "g"}}]}, True),
    ({"$not": {"$savedGroup": {"id": "g"}}}, True),
    ({"$not": {"$not": {"$savedGroup": {"id": "g"}}}}, False),
])
def test_cycles_preserve_boolean_composition(condition, expected):
    groups = {"g": {"type": "condition", "condition": condition}}
    assert evalCondition({}, {"$savedGroup": {"id": "g"}}, groups) is expected


@pytest.mark.parametrize("operator", ["$elemMatch", "$all", "$alli", "$not"])
def test_visited_groups_survive_nested_value_evaluation(operator):
    nested = {"$elemMatch": {"marker": True, "$savedGroup": {"id": "g"}}}
    items = [{"marker": True, "terminal": True}]
    if operator in ("$all", "$alli"):
        nested = {operator: [nested]}
        items = [items]
    elif operator == "$not":
        nested = {"$not": {"$not": nested}}
    groups = {"g": {"type": "condition", "condition": {"$or": [
        {"terminal": True}, {"items": nested},
    ]}}}
    # Losing the visited set would let the nested element's terminal branch pass.
    assert not evalCondition({"items": items}, {"$savedGroup": {"id": "g"}}, groups)


def test_long_acyclic_chain_has_no_sdk_depth_limit():
    groups = {str(i): {"type": "condition", "condition": {"$savedGroup": {"id": str(i + 1)}}}
              for i in range(150)}
    groups["150"] = {"type": "condition", "condition": {"id": "u1"}}
    assert evalCondition({"id": "u1"}, {"$savedGroup": {"id": "0"}}, groups)


def test_cycle_longer_than_python_stack_fails_closed():
    count = sys.getrecursionlimit()
    groups = {str(i): {"type": "condition", "condition": {"$savedGroup": {"id": str((i + 1) % count)}}}
              for i in range(count)}
    assert not evalCondition({}, {"$savedGroup": {"id": "0"}}, groups)


@pytest.mark.parametrize("marker", ["__sgInvalid__", "__sgUnknown__", "__sgCycle__", "__sgMaxDepth__"])
def test_error_markers_keep_boolean_semantics(marker):
    condition = {marker: "g"}
    assert not evalCondition({}, condition)
    assert evalCondition({}, {"$not": condition})
    assert not evalCondition({}, {"$and": [{}, condition]})
    assert evalCondition({}, {"$or": [condition, {}]})


def test_saved_group_is_only_a_top_level_operator():
    groups = {"g": {"type": "condition", "condition": {}}}
    assert not evalCondition({"id": "u1"}, {"id": {"$savedGroup": {"id": "g"}}}, groups)
    assert not evalCondition({}, {"$savedGroups": None}, groups)


def test_array_membership_handles_unhashable_attributes():
    attributes = {"id": [{"nested": True}, ["u2"], "u1"]}
    groups = {"g": {"type": "list", "attributeKey": "id", "values": ["u1"]}}
    assert evalCondition(attributes, {"id": {"$in": ["u1"]}})
    assert evalCondition(attributes, {"id": {"$inGroup": "g"}}, groups)
    assert not evalCondition(attributes, {"id": {"$notInGroup": "g"}}, groups)
    assert evalCondition(attributes, {"$savedGroup": {"id": "g"}}, groups)


@pytest.mark.asyncio
@pytest.mark.parametrize("attributes, expected", [
    ({"id": "u1", "plan": "pro"}, True),
    ({"id": "u2", "plan": "pro"}, False),
    ({"id": "u1", "plan": "free"}, False),
])
async def test_switching_payload_formats_preserves_results(attributes, expected):
    legacy = {
        "savedGroups": {"beta": ["u1"]},
        "features": {"flag": {"defaultValue": False, "rules": [{
            "condition": {"id": {"$inGroup": "beta"}, "plan": "pro"}, "force": True,
        }]}},
    }
    typed = {
        "savedGroups": {
            "beta": {"type": "list", "attributeKey": "id", "values": ["u1"]},
            "power": {"type": "condition", "condition": {"plan": "pro"}},
        },
        "features": {"flag": {"defaultValue": False, "rules": [{
            "condition": {"$and": [{"$savedGroup": {"id": "beta"}}, {"$savedGroup": {"id": "power"}}]},
            "force": True,
        }]}},
    }
    sync_client = GrowthBook(attributes=attributes)
    async_client = GrowthBookClient(Options())
    user = UserContext(attributes=attributes)
    try:
        for payload in [legacy, typed, legacy]:
            sync_client.set_payload(deepcopy(payload))
            await async_client.set_payload(deepcopy(payload))
            assert sync_client.is_on("flag") is expected
            assert await async_client.is_on("flag", user) is expected
    finally:
        sync_client.destroy()
        await async_client.close()


def _encrypt_section(value, key):
    """Encode a test payload using the SDK's AES-CBC wire format."""
    iv = os.urandom(16)
    padder = padding.PKCS7(128).padder()
    padded = padder.update(json.dumps(value).encode()) + padder.finalize()
    encryptor = Cipher(algorithms.AES(b64decode(key)), modes.CBC(iv)).encryptor()
    ciphertext = encryptor.update(padded) + encryptor.finalize()
    return b64encode(iv).decode() + "." + b64encode(ciphertext).decode()


@pytest.mark.asyncio
@pytest.mark.parametrize("encrypted", [False, True])
@pytest.mark.parametrize("transport", ["set_payload", "fetch"])
async def test_both_clients_load_and_refresh_typed_groups(mocker, encrypted, transport):
    mocker.patch.object(EnhancedFeatureRepository, "_instances", {})
    key = b64encode(b"0123456789abcdef").decode()
    payload = {
        "features": {
            "flag": {"defaultValue": False, "rules": [{
                "condition": {"$savedGroup": {"id": "eligible"}}, "force": True,
            }]},
            "unsafe": {"defaultValue": False, "rules": [{
                "condition": {"id": {"$notInGroup": "bad"}}, "force": True,
            }]},
        },
        "savedGroups": {
            "members": {"type": "list", "attributeKey": "id", "values": ["u1"]},
            "eligible": {"type": "condition", "condition": {
                "$and": [{"$savedGroup": {"id": "members"}}, {"plan": "pro"}],
            }},
            "bad": None,
        },
    }
    wire = {}
    sync_fetch = mocker.patch.object(
        FeatureRepository, "_fetch_and_decode", side_effect=lambda *args: deepcopy(wire),
    )
    async_fetch = mocker.patch.object(
        FeatureRepository, "_fetch_and_decode_async", new_callable=AsyncMock,
        side_effect=lambda *args: deepcopy(wire),
    )
    mocker.patch.object(EnhancedFeatureRepository, "start_feature_refresh", new_callable=AsyncMock)
    sync_client = GrowthBook(client_key="sdk-saved-sync" if transport == "fetch" else "", decryption_key=key)
    async_client = GrowthBookClient(Options(client_key="sdk-saved-async" if transport == "fetch" else "", decryption_key=key))
    users = [{"id": "u1", "plan": "pro"}, {"id": "u2", "plan": "pro"}, {"id": "u1", "plan": "free"}]
    experiment = Experiment(key="saved-group", variations=[0, 1], condition={"$savedGroup": {"id": "eligible"}})

    async def check_user(attributes, member):
        user = UserContext(attributes=attributes)
        expected = attributes["id"] == member and attributes["plan"] == "pro"
        assert await async_client.is_on("flag", user) is expected
        assert await async_client.is_on("unsafe", user) is False
        assert (await async_client.run(experiment, user)).inExperiment is expected

    feature_repo.clear_cache()
    try:
        for index, member in enumerate(("u1", "u2")):
            payload["savedGroups"]["members"]["values"] = [member]
            wire = ({
                "encryptedFeatures": _encrypt_section(payload["features"], key),
                "encryptedSavedGroups": _encrypt_section(payload["savedGroups"], key),
            } if encrypted else deepcopy(payload))
            if transport == "set_payload":
                sync_client.set_payload(deepcopy(wire))
                await async_client.set_payload(deepcopy(wire))
            else:
                sync_client.load_features(force_refresh=True)
                if index == 0:
                    assert await async_client.initialize()
                else:
                    repo = async_client._features_repository
                    updated = await repo.load_features_async(
                        "https://cdn.growthbook.io", "sdk-saved-async", key, force_refresh=True,
                    )
                    await repo._handle_feature_update(updated)
            for attributes in users:
                expected = attributes["id"] == member and attributes["plan"] == "pro"
                sync_client.set_attributes(dict(attributes))
                assert sync_client.is_on("flag") is expected
                assert sync_client.is_on("unsafe") is False
                assert sync_client.run(experiment).inExperiment is expected
            await asyncio.gather(*(check_user(dict(attributes), member) for _ in range(10) for attributes in users))
        if transport == "fetch":
            assert sync_fetch.call_count == 2
            assert async_fetch.await_count == 2
        else:
            sync_fetch.assert_not_called()
            async_fetch.assert_not_called()
    finally:
        sync_client.destroy()
        await async_client.close()
        feature_repo.clear_cache()
