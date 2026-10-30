"""Remote saved groups (growthbook/growthbook#7277): the spec's
savedGroupReferencesRemote section and Python-specific behavior."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from growthbook import GrowthBook, GrowthBookClient, Options, UserContext
from growthbook.core import evalCondition, pruneCondition
from growthbook.remote_saved_groups import (
    REMOTE_GROUP_IDS_ATTRIBUTE,
    get_remote_saved_groups,
    redis_resolver,
    resolve_remote_group_ids,
)

CASES = json.loads(Path(__file__).with_name("cases.json").read_text())["savedGroupReferencesRemote"]

SAVED_GROUPS = {"grp_vip": {"type": "remote", "attributeKey": "account_id"}}
# "flag" is on for members of grp_vip; "excluded" is on for resolved non-members,
# with the $exists guard GrowthBook adds to "not in group" rules
FEATURES = {
    "flag": {
        "defaultValue": False,
        "rules": [{"condition": {"$savedGroup": {"id": "grp_vip"}}, "force": True}],
    },
    "excluded": {
        "defaultValue": False,
        "rules": [{
            "condition": {
                "$not": {"$savedGroup": {"id": "grp_vip"}},
                REMOTE_GROUP_IDS_ATTRIBUTE: {"$exists": True},
            },
            "force": True,
        }],
    },
}


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


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES["feature"], ids=lambda case: case[0])
async def test_async_feature_conformance(case):
    _, context, key, expected = deepcopy(case)
    client = GrowthBookClient(Options())
    try:
        await client.set_payload(context)
        user = UserContext(attributes=context["attributes"], remote_group_ids=context.get("remoteGroupIds"))
        assert (await client.eval_feature(key, user)).to_dict() == expected
    finally:
        await client.close()


def test_set_remote_group_ids_changes_evaluation():
    gb = GrowthBook(attributes={"account_id": "a1"}, features=deepcopy(FEATURES), saved_groups=SAVED_GROUPS)
    try:
        # Unresolved groups match nobody, and fail "not in group" too
        assert gb.is_on("flag") is False
        assert gb.is_on("excluded") is False

        gb.set_remote_group_ids(["grp_vip"])
        assert gb.get_remote_group_ids() == ["grp_vip"]
        assert gb.is_on("flag") is True
        assert gb.is_on("excluded") is False

        # [] means "in no group", which is what lets "not in group" match
        gb.set_remote_group_ids([])
        assert gb.is_on("flag") is False
        assert gb.is_on("excluded") is True

        gb.set_remote_group_ids(None)
        assert gb.is_on("flag") is False
        assert gb.is_on("excluded") is False
        # The caller's attributes are left as they were
        assert gb.get_attributes() == {"account_id": "a1"}
    finally:
        gb.destroy()


def test_remote_group_ids_win_over_the_attribute_and_reach_remote_eval():
    gb = GrowthBook(
        attributes={"account_id": "a1", REMOTE_GROUP_IDS_ATTRIBUTE: ["grp_vip"]},
        features=deepcopy(FEATURES), saved_groups=SAVED_GROUPS, remote_group_ids=[],
    )
    try:
        assert gb.is_on("flag") is False
        payload = gb._remote_eval_payload()
        assert payload["attributes"][REMOTE_GROUP_IDS_ATTRIBUTE] == []
        assert payload["attributes"]["account_id"] == "a1"
    finally:
        gb.destroy()


@pytest.mark.asyncio
async def test_async_client_reads_remote_group_ids_from_the_user_context():
    client = GrowthBookClient(Options())
    try:
        await client.set_payload({"features": deepcopy(FEATURES), "savedGroups": SAVED_GROUPS})
        user = UserContext(attributes={"account_id": "a1"})
        assert (await client.eval_feature("flag", user)).on is False

        member = UserContext(attributes={"account_id": "a1"}, remote_group_ids=["grp_vip"])
        assert (await client.eval_feature("flag", member)).on is True
        # The caller's attributes are left as they were
        assert member.attributes == {"account_id": "a1"}
        payload = client._remote_eval_payload(member)
        assert payload["attributes"][REMOTE_GROUP_IDS_ATTRIBUTE] == ["grp_vip"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_add_remote_saved_groups_resolves_with_only_the_attributes_groups_use():
    client = GrowthBookClient(Options())
    calls = []

    async def resolver(attributes, groups):
        calls.append((attributes, groups))
        return ["grp_vip"]

    try:
        await client.set_payload({"features": deepcopy(FEATURES), "savedGroups": SAVED_GROUPS})
        user = UserContext(attributes={"account_id": "a1", "email": "secret"})
        resolved = await client.add_remote_saved_groups(user, resolver)

        assert resolved.remote_group_ids == ["grp_vip"]
        assert user.remote_group_ids is None, "returns a copy"
        assert calls == [({"account_id": "a1"}, [{"id": "grp_vip", "attributeKey": "account_id"}])]
        assert (await client.eval_feature("flag", resolved)).on is True
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_add_remote_saved_groups_leaves_the_context_unchanged_without_remote_groups_or_on_error():
    client = GrowthBookClient(Options())

    async def failing(attributes, groups):
        raise RuntimeError("lookup failed")

    async def unused(attributes, groups):
        raise AssertionError("not called without remote groups")

    try:
        user = UserContext(attributes={"account_id": "a1"})
        await client.set_payload({"features": {}, "savedGroups": {"admins": ["u_1"]}})
        assert await client.add_remote_saved_groups(user, unused) is user

        await client.set_payload({"features": {}, "savedGroups": SAVED_GROUPS})
        assert await client.add_remote_saved_groups(user, failing) is user
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_add_remote_saved_groups_looks_up_again_when_the_payload_changes_mid_lookup():
    client = GrowthBookClient(Options())
    seen = []
    new_groups = {**SAVED_GROUPS, "grp_new": {"type": "remote", "attributeKey": "account_id"}}

    async def resolver(attributes, groups):
        seen.append([g["id"] for g in groups])
        if len(seen) == 1:
            await client.set_payload({"features": {}, "savedGroups": new_groups})
            return ["grp_vip"]
        return ["grp_vip", "grp_new"]

    try:
        await client.set_payload({"features": {}, "savedGroups": SAVED_GROUPS})
        resolved = await client.add_remote_saved_groups(UserContext(attributes={"account_id": "a1"}), resolver)
        assert seen == [["grp_vip"], ["grp_vip", "grp_new"]]
        assert resolved.remote_group_ids == ["grp_vip", "grp_new"]
    finally:
        await client.close()


class FakeRedis:
    def __init__(self, sets):
        self.sets = sets
        self.keys = []

    async def smembers(self, key):
        self.keys.append(key)
        return self.sets.get(key, set())


@pytest.mark.asyncio
async def test_redis_resolver():
    groups = get_remote_saved_groups({
        "grp_vip": {"type": "remote", "attributeKey": "account_id"},
        "grp_vip:parent_id": {"type": "remote", "attributeKey": "parent_id", "valuesFrom": "grp_vip"},
        "grp_ab": {"type": "remote", "attributeKey": "a:b"},
    })
    redis = FakeRedis({
        "gb:member:account_id:a1": {b"grp_vip"},
        "gb:member:account_id:p9": {"grp_vip"},
        "gb:member:account_id:2": {"grp_vip"},
        # `a:b` + `c` and `a` + `b:c` share a key; only the expected id matches
        "gb:member:a:b:c": {"grp_other"},
    })
    resolve = redis_resolver(redis)

    assert await resolve({"account_id": "a1", "parent_id": "p9", "a:b": "c"}, groups) == ["grp_vip", "grp_vip:parent_id"]
    # Arrays are looked up item by item; numbers as JS String(value)
    assert await resolve({"account_id": ["x", 2.0]}, groups) == ["grp_vip"]
    assert "gb:member:account_id:2" in redis.keys


@pytest.mark.asyncio
async def test_resolve_remote_group_ids_returns_empty_without_remote_groups():
    async def unused(attributes, groups):
        raise AssertionError("not called")

    assert await resolve_remote_group_ids({"id": "u"}, {"admins": ["u"]}, unused) == []


def test_in_group_on_a_remote_entry_is_undecided():
    attributes = {"account_id": "a1", REMOTE_GROUP_IDS_ATTRIBUTE: ["grp_vip"]}
    # $inGroup takes values from the payload, and a remote entry has none:
    # undecided, so it fails, and fails under $not as well
    assert evalCondition(attributes, {"account_id": {"$inGroup": "grp_vip"}}, SAVED_GROUPS) is False
    assert evalCondition(attributes, {"$not": {"account_id": {"$inGroup": "grp_vip"}}}, SAVED_GROUPS) is False


def test_prune_condition_keeps_a_remote_reference_while_the_ids_are_unknown():
    condition = {"$savedGroup": {"id": "grp_vip"}}
    assert pruneCondition({"account_id": "a1"}, condition, SAVED_GROUPS) == condition
    assert pruneCondition({REMOTE_GROUP_IDS_ATTRIBUTE: ["grp_vip"]}, condition, SAVED_GROUPS) is True
    assert pruneCondition({REMOTE_GROUP_IDS_ATTRIBUTE: []}, condition, SAVED_GROUPS) is False


@pytest.mark.asyncio
async def test_add_remote_saved_groups_leaves_groups_unresolved_while_the_payload_keeps_changing():
    client = GrowthBookClient(Options())
    calls = []

    async def resolver(attributes, groups):
        # Every lookup sees a refresh land before it returns
        calls.append(len(calls))
        await client.set_payload({"features": {}, "savedGroups": {**SAVED_GROUPS, f"grp_{len(calls)}": {
            "type": "remote", "attributeKey": "account_id"}}})
        return ["grp_vip"]

    try:
        await client.set_payload({"features": {}, "savedGroups": SAVED_GROUPS})
        user = UserContext(attributes={"account_id": "a1"})
        assert await client.add_remote_saved_groups(user, resolver) is user
        assert len(calls) == 3
    finally:
        await client.close()


def test_a_remote_group_without_an_attribute_is_undecided():
    saved_groups = {**SAVED_GROUPS, "grp_bad": {"type": "remote"}}
    attributes = {"account_id": "a1", REMOTE_GROUP_IDS_ATTRIBUTE: ["grp_vip"]}
    not_in_bad = {"$not": {"$savedGroup": {"id": "grp_bad"}}, REMOTE_GROUP_IDS_ATTRIBUTE: {"$exists": True}}

    # A resolver never looked it up, so its absence from the ids can't put the
    # user outside it
    assert evalCondition(attributes, {"$savedGroup": {"id": "grp_bad"}}, saved_groups) is False
    assert evalCondition(attributes, not_in_bad, saved_groups) is False
    assert pruneCondition(attributes, {"$savedGroup": {"id": "grp_bad"}}, saved_groups) == {
        "$savedGroup": {"id": "grp_bad"}}
    assert [g["id"] for g in get_remote_saved_groups(saved_groups)] == ["grp_vip"]


@pytest.mark.asyncio
async def test_a_resolver_cannot_change_what_later_lookups_see():
    async def greedy(attributes, groups):
        groups[0]["attributeKey"] = "changed"
        groups.clear()
        return []

    seen = []

    async def recording(attributes, groups):
        seen.append(groups)
        return []

    await resolve_remote_group_ids({"account_id": "a1"}, SAVED_GROUPS, greedy)
    await resolve_remote_group_ids({"account_id": "a1"}, SAVED_GROUPS, recording)
    assert seen == [[{"id": "grp_vip", "attributeKey": "account_id"}]]


def test_remote_saved_groups_cache_is_per_client(monkeypatch):
    import growthbook.remote_saved_groups as module

    scans = []
    original = module.get_remote_saved_groups

    def counting(saved_groups):
        scans.append(saved_groups)
        return original(saved_groups)

    monkeypatch.setattr(module, "get_remote_saved_groups", counting)
    first, second = module.RemoteSavedGroupsCache(), module.RemoteSavedGroupsCache()
    local_only = {"admins": {"type": "list", "attributeKey": "id", "values": ["u_1"]}}

    # Two clients with different payloads alternate without evicting each other
    for _ in range(3):
        assert first.has_remote_groups(SAVED_GROUPS) is True
        assert second.has_remote_groups(local_only) is False
    assert len(scans) == 2

    # A new payload is scanned again
    assert first.has_remote_groups(dict(local_only)) is False
    assert len(scans) == 3
