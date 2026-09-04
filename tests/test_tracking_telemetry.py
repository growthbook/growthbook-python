"""Exposure reporting for prerequisite and passthrough evaluations, and the
deferred tracking buffer used to forward exposures to a client SDK.

No cases.json coverage exists for tracking behavior in any SDK (the shared
spec asserts values/sources only), so these live as a standalone module.
"""
import asyncio
import json
import threading

import pytest

from growthbook import GrowthBook, TrackingBuffer
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


def test_snapshot_failure_drops_the_exposure_not_the_evaluation():
    # A variation that deepcopy cannot handle makes the buffer's record-time
    # snapshot raise; the exposure is dropped (logged), evaluation and the
    # tracking callback are unaffected.
    lock = threading.Lock()
    features = {
        "locked": {
            "defaultValue": "off",
            "rules": [{"key": "locked-exp", "coverage": 1,
                       "variations": [lock, "off"], "weights": [1, 0]}],
        },
    }
    tracked = []
    gb = GrowthBook(
        attributes={"id": "user-1"}, features=features, defer_tracking=True,
        on_experiment_viewed=lambda experiment, result, user_context: tracked.append(experiment.key),
    )
    assert gb.eval_feature("locked").value is lock
    assert tracked == ["locked-exp"]
    assert gb.get_deferred_tracking_calls() == []
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


def test_deferred_tracking_is_opt_in():
    gb = GrowthBook(attributes={"id": "user-1"}, features=FEATURES)
    gb.eval_feature("child")
    assert gb.get_deferred_tracking_calls() == []
    gb.destroy()


def test_deferred_tracking_buffers_without_a_callback():
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


def test_buffered_entries_keep_the_exposure_time_user():
    gb = GrowthBook(attributes={"id": "user-1"}, features=FEATURES, defer_tracking=True)
    gb.eval_feature("child")
    gb.set_attributes({"id": "user-2"})
    gb.eval_feature("child")

    calls = gb.get_deferred_tracking_calls()
    assert [c["user"]["attributes"]["id"] for c in calls] == ["user-1", "user-2"]
    gb.destroy()


def test_get_deferred_tracking_calls_returns_detached_copies():
    gb = GrowthBook(attributes={"id": "user-1"}, features=FEATURES, defer_tracking=True)
    gb.eval_feature("child")
    calls = gb.get_deferred_tracking_calls()
    calls[0]["experiment"]["key"] = "mutated"
    calls[0]["user"]["attributes"]["id"] = "mutated"
    assert gb.get_deferred_tracking_calls()[0]["experiment"]["key"] == "parent-exp"
    assert gb.get_deferred_tracking_calls()[0]["user"]["attributes"] == {"id": "user-1"}
    gb.destroy()


def test_passthrough_and_gated_prerequisite_exposures_land_in_buffer():
    gb = GrowthBook(attributes={"id": "user-1"}, features=FEATURES, defer_tracking=True)
    gb.eval_feature("ramped")
    gb.eval_feature("child-gated")
    keys = [c["experiment"]["key"] for c in gb.get_deferred_tracking_calls()]
    assert keys == ["ramp", "parent-exp"]
    gb.destroy()


def test_rule_tracks_exposures_land_in_buffer():
    # Pre-evaluated exposures attached by the remote-eval proxy buffer too.
    features = {
        "remote": {
            "defaultValue": None,
            "rules": [{
                "force": "server-value",
                "tracks": [{
                    "experiment": {"key": "proxy-exp", "variations": ["a", "b"]},
                    "result": {
                        "variationId": 1, "inExperiment": True, "value": "b",
                        "hashUsed": True, "hashAttribute": "id", "hashValue": "user-1",
                    },
                }],
            }],
        },
    }
    gb = GrowthBook(attributes={"id": "user-1"}, features=features, defer_tracking=True)
    assert gb.eval_feature("remote").value == "server-value"
    calls = gb.get_deferred_tracking_calls()
    assert [c["experiment"]["key"] for c in calls] == ["proxy-exp"]
    gb.destroy()


def test_clear_deferred_tracking_calls():
    gb = GrowthBook(attributes={"id": "user-1"}, features=FEATURES, defer_tracking=True)
    gb.eval_feature("child")
    assert gb.get_deferred_tracking_calls()
    gb.clear_deferred_tracking_calls()
    assert gb.get_deferred_tracking_calls() == []
    gb.destroy()


def test_tracking_buffer_keeps_bandit_result_fields():
    buffer = TrackingBuffer()
    result = Result(
        variationId=1, inExperiment=True, value="b", hashUsed=True,
        hashAttribute="id", hashValue="u", featureId="f",
        leafId=3, variationWeights=[0.2, 0.8], banditVersion=7,
    )
    buffer.record(Experiment(key="e", variations=["a", "b"]), result, UserContext(attributes={"id": "u"}))
    (call,) = buffer.get_calls()
    assert call["result"]["leafId"] == 3
    assert call["result"]["variationWeights"] == [0.2, 0.8]
    assert call["result"]["banditVersion"] == 7


@pytest.mark.asyncio
async def test_async_client_buffers_per_request():
    client = GrowthBookClient(Options(api_host="https://localhost.growthbook.io", client_key="test"))
    await client.set_features(FEATURES)
    try:
        buf1, buf2 = TrackingBuffer(), TrackingBuffer()
        res1, res2 = await asyncio.gather(
            client.eval_feature("child", UserContext(attributes={"id": "user-1"}), tracking_buffer=buf1),
            client.eval_feature("child", UserContext(attributes={"id": "user-2"}), tracking_buffer=buf2),
        )
        assert res1.value == "child-on" and res2.value == "child-on"

        calls1, calls2 = buf1.get_calls(), buf2.get_calls()
        assert [c["experiment"]["key"] for c in calls1] == ["parent-exp"]
        assert [c["experiment"]["key"] for c in calls2] == ["parent-exp"]
        assert calls1[0]["user"]["attributes"] == {"id": "user-1"}
        assert calls2[0]["user"]["attributes"] == {"id": "user-2"}

        # No buffer passed → nothing collected anywhere.
        await client.eval_feature("child", UserContext(attributes={"id": "user-3"}))
        assert len(buf1.get_calls()) == 1 and len(buf2.get_calls()) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_client_subscriptions_fire_only_from_run():
    # Deliberate asymmetry with the sync client: the multi-user client fires
    # subscriptions only from run() (like the JS multi-user client, which has
    # no eval-time subscriptions). It has no per-user assignment
    # change-detection, so per-eval firing would repeat every subscriber
    # callback on every request.
    client = GrowthBookClient(Options(
        api_host="https://localhost.growthbook.io",
        client_key="test",
    ))
    await client.set_features(FEATURES)
    seen = []
    try:
        client.subscribe(lambda experiment, result: seen.append(experiment.key))
        await client.eval_feature("child", UserContext(attributes={"id": "user-1"}))
        assert seen == []
        exp = Experiment(key="direct", variations=["x", "y"], weights=[1, 0])
        await client.run(exp, UserContext(attributes={"id": "user-1"}))
    finally:
        await client.close()  # also drains scheduled subscription callbacks
    assert seen == ["direct"]
