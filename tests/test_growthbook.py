#!/usr/bin/env python

import json
import os
from growthbook import (
    GrowthBook,
    Experiment,
    getBucketRanges,
    gbhash,
    chooseVariation,
    getQueryStringOverride,
    inNamespace,
    getEqualWeights,
    evalCondition,
)


def pytest_generate_tests(metafunc):
    folder = os.path.abspath(os.path.dirname(__file__))
    jsonfile = os.path.join(folder, "cases.json")
    with open(jsonfile) as file:
        data = json.load(file)

    for func, cases in data.items():
        key = func + "_data"
        if key in metafunc.fixturenames:
            metafunc.parametrize(key, cases)


def test_hash(hash_data):
    hashValue, expected = hash_data
    assert gbhash(hashValue) == expected


def round_list(item):
    is_tuple = type(item) is tuple

    if is_tuple:
        item = list(item)

    for i, value in enumerate(item):
        item[i] = round(value, 6)

    return item


def round_list_of_lists(item):
    for i, value in enumerate(item):
        item[i] = round_list(value)
    return item


def test_get_bucket_range(getBucketRange_data):
    _, args, expected = getBucketRange_data
    numVariations, coverage, weights = args

    actual = getBucketRanges(numVariations, coverage, weights)

    assert round_list_of_lists(actual) == round_list_of_lists(expected)


def test_choose_variation(chooseVariation_data):
    _, n, ranges, expected = chooseVariation_data
    assert chooseVariation(n, ranges) == expected


def test_get_qs_override(getQueryStringOverride_data):
    _, id, url, numVariations, expected = getQueryStringOverride_data
    assert getQueryStringOverride(id, url, numVariations) == expected


def test_namespace(inNamespace_data):
    _, id, namespace, expected = inNamespace_data
    assert inNamespace(id, namespace) == expected


def test_equal_weights(getEqualWeights_data):
    numVariations, expected = getEqualWeights_data
    weights = getEqualWeights(numVariations)
    assert round_list(weights) == round_list(expected)


def test_conditions(evalCondition_data):
    _, condition, attributes, expected = evalCondition_data
    assert evalCondition(attributes, condition) == expected


def test_feature(feature_data):
    _, ctx, key, expected = feature_data
    gb = GrowthBook(**ctx)
    res = gb.evalFeature(key)

    if "experiment" in expected:
        expected["experiment"] = Experiment(**expected["experiment"]).to_dict()

    assert res.to_dict() == expected
    gb.destroy()


def test_run(run_data):
    _, ctx, exp, value, inExperiment = run_data
    gb = GrowthBook(**ctx)

    res = gb.run(Experiment(**exp))
    assert res.value == value
    assert res.inExperiment == inExperiment

    gb.destroy()


def getTrackingMock(gb: GrowthBook):
    calls = []

    def track(experiment, result):
        return calls.append([experiment, result])

    gb._trackingCallback = track
    return lambda: calls


def test_tracking():
    gb = GrowthBook(user={"id": "1"})

    getMockedCalls = getTrackingMock(gb)

    exp1 = Experiment(
        key="my-tracked-test",
        variations=[0, 1],
    )
    exp2 = Experiment(
        key="my-other-tracked-test",
        variations=[0, 1],
    )

    res1 = gb.run(exp1)
    gb.run(exp1)
    gb.run(exp1)
    res4 = gb.run(exp2)
    gb._user = {"id": "2"}
    res5 = gb.run(exp2)

    calls = getMockedCalls()
    assert len(calls) == 3
    assert calls[0] == [exp1, res1]
    assert calls[1] == [exp2, res4]
    assert calls[2] == [exp2, res5]

    gb.destroy()


def test_handles_weird_experiment_values():
    gb = GrowthBook(user={"id": "1"})

    assert (
        gb.run(
            Experiment(
                key="my-test",
                variations=[0, 1],
                include=lambda: 1 / 0,
            )
        ).inExperiment
        is False
    )

    # Should fail gracefully
    gb._trackingCallback = lambda experiment, result: 1 / 0
    assert gb.run(Experiment(key="my-test", variations=[0, 1])).value == 1

    gb.subscribe(lambda: 1 / 0)
    assert gb.run(Experiment(key="my-new-test", variations=[0, 1])).value == 0

    gb.destroy()


def test_force_variation():
    gb = GrowthBook(user={"id": "6"})
    exp = Experiment(key="forced-test", variations=[0, 1])
    assert gb.run(exp).value == 0

    getMockedCalls = getTrackingMock(gb)

    gb._overrides = {
        "forced-test": {
            "force": 1,
        },
    }
    assert gb.run(exp).value == 1

    calls = getMockedCalls()
    assert len(calls) == 0

    gb.destroy()


def test_uses_overrides():
    gb = GrowthBook(
        user={"id": "1"},
        overrides={
            "my-test": {
                "coverage": 0.01,
            },
        },
    )

    assert (
        gb.run(
            Experiment(
                key="my-test",
                variations=[0, 1],
            )
        ).inExperiment
        is False
    )

    gb._overrides = {
        "my-test": {
            "url": r"^\\/path",
        },
    }

    assert (
        gb.run(
            Experiment(
                key="my-test",
                variations=[0, 1],
            )
        ).inExperiment
        is False
    )

    gb.destroy()


def test_filters_user_groups():
    gb = GrowthBook(
        user={"id": "123"},
        groups={
            "alpha": True,
            "beta": True,
            "internal": False,
            "qa": False,
        },
    )

    assert (
        gb.run(
            Experiment(
                key="my-test",
                variations=[0, 1],
                groups=["internal", "qa"],
            )
        ).inExperiment
        is False
    )

    assert (
        gb.run(
            Experiment(
                key="my-test",
                variations=[0, 1],
                groups=["internal", "qa", "beta"],
            )
        ).inExperiment
        is True
    )

    assert (
        gb.run(
            Experiment(
                key="my-test",
                variations=[0, 1],
            )
        ).inExperiment
        is True
    )

    gb.destroy()


def test_runs_custom_include_callback():
    gb = GrowthBook(user={"id": "1"})
    assert (
        gb.run(
            Experiment(key="my-test", variations=[0, 1], include=lambda: False)
        ).inExperiment
        is False
    )

    gb.destroy()


def test_supports_custom_user_hash_keys():
    gb = GrowthBook(user={"id": "1", "company": "abc"})

    exp = Experiment(key="my-test", variations=[0, 1], hashAttribute="company")

    res = gb.run(exp)

    assert res.hashAttribute == "company"
    assert res.hashValue == "abc"

    gb.destroy()


def test_querystring_force_disabled_tracking():
    gb = GrowthBook(
        user={"id": "1"},
        url="http://example.com?forced-test-qs=1",
    )
    getMockedCalls = getTrackingMock(gb)

    exp = Experiment(
        key="forced-test-qs",
        variations=[0, 1],
    )
    gb.run(exp)

    calls = getMockedCalls()
    assert len(calls) == 0


def test_url_targeting():
    gb = GrowthBook(
        user={"id": "1"},
        url="http://example.com",
    )

    exp = Experiment(
        key="my-test",
        variations=[0, 1],
        url="^\\/post\\/[0-9]+",
    )

    res = gb.run(exp)
    assert res.inExperiment is False
    assert res.value == 0

    gb._url = "http://example.com/post/123"
    res = gb.run(exp)
    assert res.inExperiment is True
    assert res.value == 1

    exp.url = "http:\\/\\/example.com\\/post\\/[0-9]+"
    res = gb.run(exp)
    assert res.inExperiment is True
    assert res.value == 1

    gb.destroy()


def test_invalid_url_regex():
    gb = GrowthBook(
        user={"id": "1"},
        overrides={
            "my-test": {
                "url": "???***[)",
            },
        },
        url="http://example.com",
    )

    assert (
        gb.run(
            Experiment(
                key="my-test",
                variations=[0, 1],
            )
        ).value
        == 1
    )

    gb.destroy()


def test_ignores_draft_experiments():
    gb = GrowthBook(user={"id": "1"})
    exp = Experiment(
        key="my-test",
        status="draft",
        variations=[0, 1],
    )

    res1 = gb.run(exp)
    gb._url = "http://example.com/?my-test=1"
    res2 = gb.run(exp)

    assert res1.inExperiment is False
    assert res1.value == 0
    assert res2.inExperiment is False
    assert res2.value == 1

    gb.destroy()


def test_ignores_stopped_experiments_unless_forced():
    gb = GrowthBook(user={"id": "1"})
    expLose = Experiment(
        key="my-test",
        status="stopped",
        variations=[0, 1, 2],
    )
    expWin = Experiment(
        key="my-test",
        status="stopped",
        variations=[0, 1, 2],
        force=2,
    )

    res1 = gb.run(expLose)
    res2 = gb.run(expWin)

    assert res1.value == 0
    assert res1.inExperiment is False
    assert res2.value == 2
    assert res2.inExperiment is False

    gb.destroy()


fired = {}


def flagSubscription(experiment, result):
    fired["value"] = True


def hasFired():
    return fired.get("value", False)


def resetFiredFlag():
    fired["value"] = False


def test_destroy_removes_subscriptions():
    gb = GrowthBook(user={"id": "1"})

    resetFiredFlag()
    gb.subscribe(flagSubscription)

    gb.run(
        Experiment(
            key="my-test",
            variations=[0, 1],
        )
    )

    assert hasFired() is True

    resetFiredFlag()
    gb.destroy()

    gb.run(
        Experiment(
            key="my-other-test",
            variations=[0, 1],
        )
    )

    assert hasFired() is False

    gb.destroy()


def test_fires_subscriptions_correctly():
    gb = GrowthBook(
        user={
            "id": "1",
        },
    )

    resetFiredFlag()
    unsubscriber = gb.subscribe(flagSubscription)

    assert hasFired() is False

    exp = Experiment(
        key="my-test",
        variations=[0, 1],
    )

    # Should fire when user is put in an experiment
    gb.run(exp)
    assert hasFired() is True

    # Does not fire if nothing has changed
    resetFiredFlag()
    gb.run(exp)
    assert hasFired() is False

    # Does not fire after unsubscribed
    unsubscriber()
    gb.run(
        Experiment(
            key="other-test",
            variations=[0, 1],
        )
    )

    assert hasFired() is False

    gb.destroy()


def test_stores_assigned_variations_in_the_user():
    gb = GrowthBook(
        user={
            "id": "1",
        },
    )

    gb.run(Experiment(key="my-test", variations=[0, 1]))
    gb.run(Experiment(key="my-test-3", variations=[0, 1]))

    assigned = gb.getAllResults()
    assignedArr = []

    for e in assigned:
        assignedArr.append({
            "key": e,
            "variation": assigned[e]["result"].variationId
        })

    assert len(assignedArr) == 2
    assert assignedArr[0]["key"] == "my-test"
    assert assignedArr[0]["variation"] == 1
    assert assignedArr[1]["key"] == "my-test-3"
    assert assignedArr[1]["variation"] == 0

    gb.destroy()
