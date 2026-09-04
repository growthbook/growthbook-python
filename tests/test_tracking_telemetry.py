"""Exposure and feature-usage reporting for prerequisite and passthrough
evaluations, and the deferred tracking buffer used to forward exposures."""
import dataclasses
import json

import pytest

from growthbook import GrowthBook
from growthbook.common_types import Experiment, Options, Result, UserContext
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
    gb, tracked, usage = make_gb()
    res = gb.eval_feature("ramped")
    assert res.value == "fallthrough"
    assert tracked == [("ramp", 1)]
    gb.destroy()


def test_subscriptions_see_prerequisite_experiments():
    gb, _, _ = make_gb()
    seen = []
    gb.subscribe(lambda experiment, result: seen.append(experiment.key))
    gb.eval_feature("child")
    assert seen == ["parent-exp"]
    assert "parent-exp" in gb.get_all_results()
    gb.destroy()


def test_unserializable_feature_values_do_not_break_evaluation():
    gb, _, usage = make_gb()
    assert gb.eval_feature("unserializable").value == {1: "a", "b": 2}
    assert usage == ["unserializable"]
    gb.destroy()


def test_deferred_tracking_is_opt_in():
    gb = GrowthBook(attributes={"id": "user-1"}, features=FEATURES)
    gb.eval_feature("child")
    assert gb.get_deferred_tracking_calls() == []
    gb.destroy()


def test_deferred_tracking_buffers_when_no_callback():
    gb = GrowthBook(attributes={"id": "user-1"}, features=FEATURES, defer_tracking=True)
    gb.eval_feature("child")
    gb.eval_feature("child")  # same assignment, deduped

    calls = gb.get_deferred_tracking_calls()
    assert len(calls) == 1
    assert calls[0]["experiment"]["key"] == "parent-exp"
    assert calls[0]["result"]["variationId"] == 0
    assert calls[0]["user"] == {"attributes": {"id": "user-1"}, "url": ""}
    json.dumps(calls)  # forwardable as-is
    gb.destroy()


def test_buffering_is_independent_of_the_callback():
    gb, tracked, _ = make_gb(defer_tracking=True)
    gb.eval_feature("child")
    assert tracked == [("parent-exp", 0)]
    assert [c["experiment"]["key"] for c in gb.get_deferred_tracking_calls()] == ["parent-exp"]
    gb.destroy()


def test_forwarded_calls_replay_with_their_own_user():
    sender = GrowthBook(attributes={"id": "user-1"}, features=FEATURES, defer_tracking=True)
    sender.eval_feature("child")
    sender.set_attributes({"id": "user-2"})
    sender.eval_feature("child")
    forwarded = json.loads(json.dumps(sender.get_deferred_tracking_calls()))
    sender.destroy()

    seen = []
    receiver = GrowthBook(
        attributes={"id": "receiver"},
        on_experiment_viewed=lambda experiment, result, user_context: seen.append(
            (result.hashValue, user_context.attributes["id"])
        ),
    )
    receiver.set_deferred_tracking_calls(forwarded)
    receiver.fire_deferred_tracking_calls()
    assert seen == [("user-1", "user-1"), ("user-2", "user-2")]
    assert receiver.get_deferred_tracking_calls() == []
    receiver.destroy()


def test_hydration_skips_malformed_entries():
    fired = []
    gb = GrowthBook(
        attributes={"id": "user-1"},
        on_experiment_viewed=lambda experiment, result, user_context: fired.append(
            (experiment.key, result.variationId)
        ),
    )
    gb.set_deferred_tracking_calls([
        None,
        {"experiment": {"key": 123, "variations": ["a"]}, "result": {}},
        {"experiment": {"key": "no-variations", "variations": None}, "result": {}},
        {"experiment": {"key": "no-result", "variations": ["a", "b"]}},
        {
            "experiment": {"key": "bad-user", "variations": ["a", "b"]},
            "result": {"variationId": 0, "inExperiment": True, "hashAttribute": "id", "hashValue": "user-1"},
            "user": "not-a-dict",
        },
        {
            "experiment": {"key": "forwarded", "variations": ["a", "b"]},
            "result": {"variationId": 1, "inExperiment": True, "hashAttribute": "id", "hashValue": "user-1"},
        },
    ])
    assert len(gb.get_deferred_tracking_calls()) == 2
    gb.fire_deferred_tracking_calls()
    assert fired == [("bad-user", 0), ("forwarded", 1)]
    assert gb.get_deferred_tracking_calls() == []
    gb.destroy()


def test_replay_keeps_entries_the_callback_raised_on():
    sender = GrowthBook(attributes={"id": "user-1"}, features=FEATURES, defer_tracking=True)
    sender.eval_feature("child")
    sender.eval_feature("ramped")
    calls = sender.get_deferred_tracking_calls()
    sender.destroy()

    def flaky(experiment, result, user_context):
        if experiment.key == "ramp":
            raise RuntimeError("analytics down")

    receiver = GrowthBook(attributes={"id": "receiver"}, on_experiment_viewed=flaky)
    receiver.set_deferred_tracking_calls(calls)
    receiver.fire_deferred_tracking_calls()
    assert [c["experiment"]["key"] for c in receiver.get_deferred_tracking_calls()] == ["ramp"]
    receiver.destroy()


def test_user_context_buffer_is_not_a_dataclass_field():
    ctx = UserContext(attributes={"id": "u"}, defer_tracking=True)
    result = Result(
        variationId=0, inExperiment=True, value="a", hashUsed=True, hashAttribute="id", hashValue="u", featureId=None
    )
    ctx.defer_tracking_call(Experiment(key="e", variations=["a", "b"]), result)
    assert len(ctx.get_deferred_tracking_calls()) == 1

    rebuilt = UserContext(**dataclasses.asdict(ctx))
    assert rebuilt.defer_tracking is True
    assert rebuilt.get_deferred_tracking_calls() == []
    assert dataclasses.replace(ctx).get_deferred_tracking_calls() == []


@pytest.mark.asyncio
async def test_async_client_buffers_per_user_context():
    client = GrowthBookClient(Options(api_host="https://localhost.growthbook.io", client_key="test"))
    await client.set_features(FEATURES)
    try:
        user1 = UserContext(attributes={"id": "user-1"}, defer_tracking=True)
        user2 = UserContext(attributes={"id": "user-2"}, defer_tracking=True)
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
