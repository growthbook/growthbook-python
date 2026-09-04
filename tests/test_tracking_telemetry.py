"""Exposure reporting for prerequisite and passthrough evaluations.

No cases.json coverage exists for tracking behavior in any SDK (the shared
spec asserts values/sources only), so these live as a standalone module.
"""
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
    "child-gated": {
        "defaultValue": "child-default",
        "rules": [
            {
                "parentConditions": [{"id": "parent", "condition": {"value": "nope"}, "gate": True}],
                "force": "never",
            }
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
    "unserializable": {"defaultValue": {1: "a", "b": 2}},
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


def test_prerequisite_tracked_even_when_gate_fails():
    # The prerequisite experiment is evaluated (and its exposure fired)
    # before the gate decision — matches the JS SDK.
    gb, tracked, usage = make_gb()
    res = gb.eval_feature("child-gated")
    assert res.value is None and res.source == "prerequisite"
    assert tracked == [("parent-exp", 0)]
    assert usage == ["parent", "child-gated"]
    gb.destroy()


def test_experiment_level_prerequisite_is_tracked():
    gb, tracked, usage = make_gb()
    exp = Experiment(
        key="direct",
        variations=["x", "y"],
        weights=[1, 0],
        parentConditions=[{"id": "parent", "condition": {"value": "on"}}],
    )
    seen = []
    gb.subscribe(lambda experiment, result: seen.append(experiment.key))
    res = gb.run(exp)
    assert res.inExperiment and res.value == "x"
    assert tracked == [("parent-exp", 0), ("direct", 0)]
    assert usage == ["parent"]
    assert seen == ["parent-exp", "direct"]
    gb.destroy()


def test_prerequisite_consulted_by_several_rules_is_reported_once():
    gb, tracked, usage = make_gb()
    res = gb.eval_feature("child-twice")
    assert res.value == "r2"
    assert tracked == [("parent-exp", 0)]
    assert usage == ["parent", "child-twice"]
    gb.destroy()


def test_passthrough_assignment_is_tracked():
    gb, tracked, _ = make_gb()
    res = gb.eval_feature("ramped")
    assert res.value == "fallthrough"
    assert tracked == [("ramp", 1)]
    gb.destroy()


def test_unserializable_feature_values_do_not_break_evaluation():
    gb, _, usage = make_gb()
    assert gb.eval_feature("unserializable").value == {1: "a", "b": 2}
    assert usage == ["unserializable"]
    gb.destroy()


def test_subscriptions_see_prerequisite_experiments():
    gb, _, _ = make_gb()
    seen = []
    gb.subscribe(lambda experiment, result: seen.append(experiment.key))
    gb.eval_feature("child")
    assert seen == ["parent-exp"]
    assert "parent-exp" in gb.get_all_results()
    gb.destroy()


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
        res = await client.eval_feature("child", UserContext(attributes={"id": "user-1"}))
        assert res.value == "child-on"
        assert tracked == ["parent-exp"]
        assert usage == ["parent", "child"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_client_subscriptions_fire_from_eval_feature():
    # Previously the async client only notified subscribers from run();
    # experiments discovered through eval_feature (prerequisites included)
    # were invisible. Now matches the sync client.
    client = GrowthBookClient(Options(
        api_host="https://localhost.growthbook.io",
        client_key="test",
    ))
    await client.set_features(FEATURES)
    seen = []
    try:
        client.subscribe(lambda experiment, result: seen.append(experiment.key))
        await client.eval_feature("child", UserContext(attributes={"id": "user-1"}))
    finally:
        await client.close()  # also drains scheduled subscription callbacks
    assert seen == ["parent-exp"]
