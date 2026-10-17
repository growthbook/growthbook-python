"""Literal JSON objects must not be mistaken for operator expressions."""

import pytest

from growthbook import GrowthBook, GrowthBookClient, UserContext
from growthbook.core import evalCondition


@pytest.mark.asyncio
@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize("condition, attribute, expected", [
    ({}, {}, True),
    ({}, {"other": 1}, False),
    ({}, None, False),
    ({}, "", False),
    ({"": "match"}, {"": "match"}, True),
    ({"": "match"}, {"": "other"}, False),
    ({"": "match", "$eq": "literal"}, {"": "match", "$eq": "literal"}, True),
    ({"$eq": "match"}, "match", True),
])
async def test_literal_object_feature_rule(async_client, condition, attribute, expected):
    features = {"targeted": {
        "defaultValue": False,
        "rules": [{"condition": {"profile": condition}, "force": True}],
    }}
    if async_client:
        client = GrowthBookClient()
        try:
            await client.set_features(features)
            assert await client.is_on("targeted", UserContext(attributes={"profile": attribute})) is expected
        finally:
            await client.close()
    else:
        sync_client = GrowthBook(attributes={"profile": attribute}, features=features)
        try:
            assert sync_client.is_on("targeted") is expected
        finally:
            sync_client.destroy()


@pytest.mark.asyncio
@pytest.mark.parametrize("attribute, expected", [({"": True}, False), ("u1", True)])
async def test_empty_key_literal_negation_in_saved_group(attribute, expected):
    # An empty string is a valid literal JSON key, not a malformed operator.
    literal = {"id": {"": True}}
    groups = {"literal": {"type": "condition", "condition": literal}}
    sync_client = GrowthBook(attributes={"id": attribute})
    async_client = GrowthBookClient()
    try:
        for target in (literal, {"$savedGroup": {"id": "literal"}}):
            condition = {"$not": target}
            payload = {"savedGroups": groups, "features": {"targeted": {
                "defaultValue": False,
                "rules": [{"condition": condition, "force": True}],
            }}}
            sync_client.set_payload(payload)
            await async_client.set_payload(payload)
            assert evalCondition({"id": attribute}, condition, groups) is expected
            assert sync_client.is_on("targeted") is expected
            assert await async_client.is_on("targeted", UserContext(attributes={"id": attribute})) is expected
    finally:
        sync_client.destroy()
        await async_client.close()
