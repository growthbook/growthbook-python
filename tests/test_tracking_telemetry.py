"""Exposure and feature-usage reporting for prerequisite and passthrough
evaluations, and the deferred tracking buffer used to forward exposures."""
import json

import pytest

from growthbook import GrowthBook
from growthbook.common_types import Experiment, Options, UserContext
from growthbook.growthbook_client import GrowthBookClient

# Weights of [1, 0] / [0, 1] make every assignment deterministic.
FEATURES = {
    "parent": {
        "defaultValue": "off",
        "rules": [
            {"key": "parent-exp", "coverage": 1, "variations": ["on", "off"], "weights": [1, 0]}
        ],
    },
    "child": {
        "defaultValue": "child-default",
        "rules": [
            {"parentConditions": [{"id": "parent", "condition": {"value": "on"}}], "force": "child-on"}
        ],
    },
    "child-twice": {
        "defaultValue": "child-default",
        "rules": [
            {"parentConditions": [{"id": "parent", "condition": {"value": "nope"}}], "force": "r1"},
            {"parentConditions": [{"id": "parent", "condition": {"value": "on"}}], "force": "r2"},
        ],
    },
    "ramped": {
        "defaultValue": "default",
        "rules": [
            {
                "key": "ramp",
                "coverage": 1,
                "variations": ["treatment", "default"],
                "weights": [0, 1],
                "meta": [{"key": "0"}, {"key": "1", "passthrough": True}],
            },
            {"force": "fallthrough"},
        ],
    },
}


def make_gb(**kwargs):
    tracked, usage = [], []

    def on_viewed(experiment, result, user_context):
        tracked.append((experiment.key, result.variationId))

    def on_usage(key, result, user_context):
        usage.append(key)

    gb = GrowthBook(
        attributes={"id": "user-1"},
        features=FEATURES,
        on_experiment_viewed=on_viewed,
        on_feature_usage=on_usage,
        **kwargs,
    )
    return gb, tracked, usage


def test_prerequisite_experiment_assignment_is_tracked():
    gb, tracked, usage = make_gb()
    res = gb.eval_feature("child")
    assert res.value == "child-on"
    assert tracked == [("parent-exp", 0)]
    assert usage == ["parent", "child"]
    gb.destroy()


def test_experiment_level_prerequisite_is_tracked():
    gb, tracked, usage = make_gb()
    exp = Experiment(
        key="direct",
        variations=["x", "y"],
        weights=[1, 0],
        parentConditions=[{"id": "parent", "condition": {"value": "on"}}],
    )
    res = gb.run(exp)
    assert res.inExperiment and res.value == "x"
    assert tracked == [("parent-exp", 0), ("direct", 0)]
    assert usage == ["parent"]
    gb.destroy()


def test_prerequisite_consulted_by_several_rules_is_reported_once():
    gb, tracked, usage = make_gb()
    res = gb.eval_feature("child-twice")
    assert res.value == "r2"
    assert tracked == [("parent-exp", 0)]
    assert usage == ["parent", "child-twice"]
    gb.destroy()


def test_passthrough_assignment_is_tracked():
    gb, tracked, usage = make_gb()
    res = gb.eval_feature("ramped")
    assert res.value == "fallthrough"
    assert tracked == [("ramp", 1)]
    gb.destroy()


def test_deferred_tracking_buffers_when_no_callback():
    gb = GrowthBook(attributes={"id": "user-1"}, features=FEATURES)
    gb.eval_feature("child")
    gb.eval_feature("child")  # same assignment, deduped

    calls = gb.get_deferred_tracking_calls()
    assert len(calls) == 1
    assert calls[0]["experiment"]["key"] == "parent-exp"
    assert calls[0]["result"]["variationId"] == 0
    assert calls[0]["user"] == {"attributes": {"id": "user-1"}, "url": ""}
    json.dumps(calls)  # forwardable as-is

    fired = []
    gb.set_tracking_callback(lambda experiment, result, user_context: fired.append(experiment.key))
    assert fired == ["parent-exp"]
    assert gb.get_deferred_tracking_calls() == []
    gb.destroy()


def test_deferred_tracking_calls_can_be_hydrated_and_fired():
    fired = []
    gb = GrowthBook(
        attributes={"id": "user-1"},
        on_experiment_viewed=lambda experiment, result, user_context: fired.append(
            (experiment.key, result.variationId)
        ),
    )
    gb.set_deferred_tracking_calls([
        {
            "experiment": {"key": "forwarded", "variations": ["a", "b"]},
            "result": {"variationId": 1, "inExperiment": True, "hashAttribute": "id", "hashValue": "user-1"},
        },
        {"experiment": {"key": "missing-variations"}, "result": {}},
    ])
    gb.fire_deferred_tracking_calls()
    assert fired == [("forwarded", 1)]
    assert gb.get_deferred_tracking_calls() == []
    gb.destroy()


def test_no_buffering_when_callback_configured():
    gb, tracked, _ = make_gb()
    gb.eval_feature("child")
    assert tracked == [("parent-exp", 0)]
    assert gb.get_deferred_tracking_calls() == []
    gb.destroy()


@pytest.mark.asyncio
async def test_async_client_buffers_per_user_context():
    client = GrowthBookClient(Options(api_host="https://localhost.growthbook.io", client_key="test"))
    await client.set_features(FEATURES)
    try:
        user1 = UserContext(attributes={"id": "user-1"})
        user2 = UserContext(attributes={"id": "user-2"})
        res = await client.eval_feature("child", user1)
        assert res.value == "child-on"

        calls = user1.get_deferred_tracking_calls()
        assert [c["experiment"]["key"] for c in calls] == ["parent-exp"]
        assert calls[0]["user"]["attributes"] == {"id": "user-1"}
        assert user2.get_deferred_tracking_calls() == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_client_tracks_prerequisites_through_callback():
    tracked, usage = [], []
    client = GrowthBookClient(Options(
        api_host="https://localhost.growthbook.io",
        client_key="test",
        on_experiment_viewed=lambda experiment, result, user_context: tracked.append(experiment.key),
        on_feature_usage=lambda key, result, user_context: usage.append(key),
    ))
    await client.set_features(FEATURES)
    try:
        user = UserContext(attributes={"id": "user-1"})
        await client.eval_feature("child", user)
        assert tracked == ["parent-exp"]
        assert usage == ["parent", "child"]
        assert user.get_deferred_tracking_calls() == []
    finally:
        await client.close()
