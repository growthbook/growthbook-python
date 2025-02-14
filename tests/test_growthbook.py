#!/usr/bin/env python

import json
import os
from growthbook import (
    FeatureRule,
    GrowthBook,
    Experiment,
    Feature,
    InMemoryStickyBucketService,
    decrypt,
    feature_repo,
    logger,
)

from growthbook.core import (
    getBucketRanges,
    gbhash,
    chooseVariation,
    paddedVersionString,
    getQueryStringOverride,
    inNamespace,
    getEqualWeights,
    evalCondition,
)

from time import time
import pytest

logger.setLevel("DEBUG")


def pytest_generate_tests(metafunc):
    folder = os.path.abspath(os.path.dirname(__file__))
    jsonfile = os.path.join(folder, "cases.json")
    with open(jsonfile) as file:
        data = json.load(file)

    for func, cases in data.items():
        key = func + "_data"

        if (func == "versionCompare"):
            for method, cases in cases.items():
                key = func + "_" + method + "_data"
                if (key in metafunc.fixturenames):
                    metafunc.parametrize(key, cases)
        elif key in metafunc.fixturenames:
            metafunc.parametrize(key, cases)


def test_hash(hash_data):
    seed, value, version, expected = hash_data
    assert gbhash(seed, value, version) == expected


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
    _, condition, attributes, expected, savedGroups = (evalCondition_data + [None]*5)[:5]
    assert evalCondition(attributes, condition, savedGroups) == expected


def test_decrypt(decrypt_data):
    _, encrypted, key, expected = decrypt_data
    try:
        assert (decrypt(encrypted, key)) == expected
    except Exception:
        assert (expected) is None


def test_feature(feature_data):
    _, ctx, key, expected = feature_data
    gb = GrowthBook(**ctx)
    res = gb.evalFeature(key)

    if "experiment" in expected:
        expected["experiment"] = Experiment(**expected["experiment"]).to_dict()

    actual = res.to_dict()

    # Set ruleId to empty string if missing to match test cases
    if "ruleId" not in actual:
        actual["ruleId"] = ""

    assert actual == expected
    gb.destroy()


def test_run(run_data):
    _, ctx, exp, value, inExperiment, hashUsed = run_data
    gb = GrowthBook(**ctx)

    res = gb.run(Experiment(**exp))
    assert res.value == value
    assert res.inExperiment == inExperiment
    assert res.hashUsed == hashUsed

    gb.destroy()


def test_stickyBucket(stickyBucket_data):
    _, ctx, initial_docs, key, expected_result, expected_docs = stickyBucket_data
    # Just use the interface directly, which passes and doesn't persist anywhere
    service = InMemoryStickyBucketService()

    for doc in initial_docs:
        service.save_assignments(doc)

    ctx['sticky_bucket_service'] = service

    if 'stickyBucketIdentifierAttributes' in ctx:
        ctx['sticky_bucket_identifier_attributes'] = ctx['stickyBucketIdentifierAttributes']
        ctx.pop('stickyBucketIdentifierAttributes')

    if 'stickyBucketAssignmentDocs' in ctx:
        service.docs = ctx['stickyBucketAssignmentDocs']
        ctx.pop('stickyBucketAssignmentDocs')

    gb = GrowthBook(**ctx)
    res = gb.eval_feature(key)
    
    if not res.experimentResult:
      assert None == expected_result
    else:
        assert res.experimentResult.to_dict() == expected_result

    # Ignore extra docs in service, just make sure each expected one matches
    for key, value in expected_docs.items():
        assert service.docs[key] == value

    service.destroy()
    gb.destroy()


def getTrackingMock(gb: GrowthBook):
    calls = []

    def track(experiment, result):
        return calls.append([experiment, result])

    gb._trackingCallback = track
    return lambda: calls


def test_tracking():
    gb = GrowthBook(attributes={"id": "1"})

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
    gb._attributes = {"id": "2"}
    res5 = gb.run(exp2)

    calls = getMockedCalls()
    assert len(calls) == 3
    assert calls[0] == [exp1, res1]
    assert calls[1] == [exp2, res4]
    assert calls[2] == [exp2, res5]

    gb.destroy()


def test_handles_weird_experiment_values():
    gb = GrowthBook(attributes={"id": "1"})

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
    gb = GrowthBook(attributes={"id": "6"})
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
        attributes={"id": "1"},
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
        attributes={"id": "123"},
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
    gb = GrowthBook(attributes={"id": "1", "company": "abc"})

    exp = Experiment(key="my-test", variations=[0, 1], hashAttribute="company")

    res = gb.run(exp)

    assert res.hashAttribute == "company"
    assert res.hashValue == "abc"

    gb.destroy()


def test_querystring_force_disabled_tracking():
    gb = GrowthBook(
        attributes={"id": "1"},
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
        attributes={"id": "1"},
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
        attributes={"id": "1"},
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
    gb = GrowthBook(attributes={"id": "1"})
    exp = Experiment(
        key="my-test",
        status="draft",
        variations=[0, 1],
    )

    res1 = gb.run(exp)
    gb._url = "http://example.com/?my-test=1"
    res2 = gb.run(exp)

    assert res1.inExperiment is False
    assert res1.hashUsed is False
    assert res1.value == 0
    assert res2.inExperiment is True
    assert res2.hashUsed is False
    assert res2.value == 1

    gb.destroy()


def test_ignores_stopped_experiments_unless_forced():
    gb = GrowthBook(attributes={"id": "1"})
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
    assert res2.inExperiment is True

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
        attributes={
            "id": "1",
        },
    )

    gb.run(Experiment(key="my-test", variations=[0, 1]))
    gb.run(Experiment(key="my-test-3", variations=[0, 1]))

    assigned = gb.getAllResults()
    assignedArr = []

    for e in assigned:
        assignedArr.append({"key": e, "variation": assigned[e]["result"].variationId})

    assert len(assignedArr) == 2
    assert assignedArr[0]["key"] == "my-test"
    assert assignedArr[0]["variation"] == 1
    assert assignedArr[1]["key"] == "my-test-3"
    assert assignedArr[1]["variation"] == 0

    gb.destroy()


def test_getters_setters():
    gb = GrowthBook()

    feat = Feature(defaultValue="yes", rules=[FeatureRule(force="no")])
    featuresInput = {"feature-1": feat.to_dict()}
    attributes = {"id": "123", "url": "/"}

    gb.setFeatures(featuresInput)
    gb.setAttributes(attributes)

    featuresOutput = {k: v.to_dict() for (k, v) in gb.getFeatures().items()}

    assert featuresOutput == featuresInput
    assert attributes == gb.getAttributes()

    newAttrs = {"url": "/hello"}
    gb.setAttributes(newAttrs)
    assert newAttrs == gb.getAttributes()

    gb.destroy()


def test_return_ruleid_when_evaluating_a_feature():
    gb = GrowthBook(
        features={"feature": {"defaultValue": 0, "rules": [{"force": 1, "id": "foo"}]}}
    )
    assert gb.eval_feature("feature").ruleId == "foo"
    gb.destroy()


def test_feature_methods():
    gb = GrowthBook(
        features={
            "featureOn": {"defaultValue": 12},
            "featureNone": {"defaultValue": None},
            "featureOff": {"defaultValue": 0},
        }
    )

    assert gb.isOn("featureOn") is True
    assert gb.isOff("featureOn") is False
    assert gb.getFeatureValue("featureOn", 15) == 12

    assert gb.isOn("featureOff") is False
    assert gb.isOff("featureOff") is True
    assert gb.getFeatureValue("featureOff", 10) == 0

    assert gb.isOn("featureNone") is False
    assert gb.isOff("featureNone") is True
    assert gb.getFeatureValue("featureNone", 10) == 10

    gb.destroy()


class MockHttpResp:
    def __init__(self, status: int, data: str) -> None:
        self.status = status
        self.data = data.encode("utf-8")


def test_feature_repository(mocker):
    m = mocker.patch.object(feature_repo, "_get")
    expected = {"features": {"feature": {"defaultValue": 5}}}
    m.return_value = MockHttpResp(200, json.dumps(expected))
    features = feature_repo.load_features("https://cdn.growthbook.io", "sdk-abc123")

    m.assert_called_once_with("https://cdn.growthbook.io/api/features/sdk-abc123")
    assert features == expected

    # Uses in-memory cache for the 2nd call
    features = feature_repo.load_features("https://cdn.growthbook.io", "sdk-abc123")
    assert m.call_count == 1
    assert features == expected

    # Does a new request if cache entry is expired
    feature_repo.cache.cache["https://cdn.growthbook.io::sdk-abc123"].expires = (
        time() - 10
    )
    features = feature_repo.load_features("https://cdn.growthbook.io", "sdk-abc123")
    assert m.call_count == 2
    assert features == expected

    feature_repo.clear_cache()


def test_feature_repository_error(mocker):
    m = mocker.patch.object(feature_repo, "_get")
    m.return_value = MockHttpResp(400, "400 Error")
    features = feature_repo.load_features("https://cdn.growthbook.io", "sdk-abc123")

    m.assert_called_once_with("https://cdn.growthbook.io/api/features/sdk-abc123")
    assert features is None

    # Does not cache errors
    features = feature_repo.load_features("https://cdn.growthbook.io", "sdk-abc123")
    assert m.call_count == 2
    assert features is None

    # Handles broken JSON response
    m.return_value = MockHttpResp(200, "{'corrupted':6('4")
    features = feature_repo.load_features("https://cdn.growthbook.io", "sdk-abc123")
    assert m.call_count == 3
    assert features is None

    feature_repo.clear_cache()


def test_feature_repository_encrypted(mocker):
    m = mocker.patch.object(feature_repo, "_get")
    m.return_value = MockHttpResp(
        200,
        json.dumps(
            {
                "features": {},
                "encryptedFeatures": "m5ylFM6ndyOJA2OPadubkw==.Uu7ViqgKEt/dWvCyhI46q088PkAEJbnXKf3KPZjf9IEQQ+A8fojNoxw4wIbPX3aj",
            }
        ),
    )
    features = feature_repo.load_features(
        "https://cdn.growthbook.io", "sdk-abc123", "Zvwv/+uhpFDznZ6SX28Yjg=="
    )

    m.assert_called_once_with("https://cdn.growthbook.io/api/features/sdk-abc123")
    assert features == {"features": {"feature": {"defaultValue": True}}}

    feature_repo.clear_cache()

    # Raises exception if missing decryption key
    with pytest.raises(Exception):
        feature_repo.load_features("https://cdn.growthbook.io", "sdk-abc123")


def test_load_features(mocker):
    m = mocker.patch.object(feature_repo, "_get")
    m.return_value = MockHttpResp(
        200, json.dumps({"features": {"feature": {"defaultValue": 5}}})
    )

    gb = GrowthBook(api_host="https://cdn.growthbook.io", client_key="sdk-abc123")

    assert m.call_count == 0

    gb.load_features()
    m.assert_called_once_with("https://cdn.growthbook.io/api/features/sdk-abc123")

    assert gb.get_features()["feature"].to_dict() == {"defaultValue": 5, "rules": []}

    feature_repo.clear_cache()
    gb.destroy()


def test_loose_unmarshalling(mocker):
    m = mocker.patch.object(feature_repo, "_get")
    m.return_value = MockHttpResp(200, json.dumps({
        "features": {
            "feature": {
                "defaultValue": 5,
                "rules": [
                    {
                        "condition": {"country": "US"},
                        "force": 3,
                        "hashVersion": 1,
                        "unknown": "foo"
                    },
                    {
                        "key": "my-exp",
                        "hashVersion": 2,
                        "variations": [0, 1],
                        "meta": [
                            {
                                "key": "control",
                                "unknown": "foo"
                            },
                            {
                                "key": "variation1",
                                "unknown": "foo"
                            }
                        ],
                        "filters": [
                            {
                                "seed": "abc123",
                                "ranges": [[0, 0.0001]],
                                "hashVersion": 2,
                                "attribute": "id",
                                "unknown": "foo"
                            }
                        ]
                    },
                    {
                        "unknownRuleType": "foo"
                    }
                ],
                "unknown": "foo"
            }
        },
        "unknown": "foo"
    }))

    gb = GrowthBook(api_host="https://cdn.growthbook.io", client_key="sdk-abc123")

    assert m.call_count == 0

    gb.load_features()
    m.assert_called_once_with("https://cdn.growthbook.io/api/features/sdk-abc123")

    assert gb.get_features()["feature"].to_dict() == {
        "defaultValue": 5,
        "rules": [
            {
                "condition": {"country": "US"},
                "force": 3,
                "hashVersion": 1
            },
            {
                "key": "my-exp",
                "hashVersion": 2,
                "variations": [0, 1],
                "meta": [
                    {
                        "key": "control",
                        "unknown": "foo"
                    },
                    {
                        "key": "variation1",
                        "unknown": "foo"
                    }
                ],
                "filters": [
                    {
                        "seed": "abc123",
                        "ranges": [[0, 0.0001]],
                        "hashVersion": 2,
                        "attribute": "id",
                        "unknown": "foo"
                    }
                ]
            },
            {
                "hashVersion": 1
            }
        ]
    }

    value = gb.get_feature_value("feature", -1)
    assert value == 5

    feature_repo.clear_cache()
    gb.destroy()


def test_sticky_bucket_service(mocker):
    # Start forcing everyone to variation1
    features = {
        "feature": {
            "defaultValue": 5,
            "rules": [{
                "key": "exp",
                "variations": [0, 1],
                "weights": [0, 1],
                "meta": [
                    {"key": "control"},
                    {"key": "variation1"}
                ]
            }]
        },
    }

    service = InMemoryStickyBucketService()
    gb = GrowthBook(
        sticky_bucket_service=service,
        attributes={
            "id": "1"
        },
        features=features
    )

    assert gb.get_feature_value("feature", -1) == 1
    assert service.get_assignments("id", "1") == {
        "attributeName": "id",
        "attributeValue": "1",
        "assignments": {
            "exp__0": "variation1"
        }
    }

    logger.debug("Change weights and ensure old user still gets variation")
    features["feature"]["rules"][0]["weights"] = [1, 0]
    gb.set_features(features)
    assert gb.get_feature_value("feature", -1) == 1

    logger.debug("New GrowthBook instance should also get variation")
    gb2 = GrowthBook(
        sticky_bucket_service=service,
        attributes={
            "id": "1"
        },
        features=features
    )
    assert gb2.get_feature_value("feature", -1) == 1
    gb2.destroy()

    logger.debug("New users should get control")
    gb.set_attributes({"id": "2"})
    assert gb.get_feature_value("feature", -1) == 0

    logger.debug("Bumping bucketVersion, should reset sticky buckets")
    gb.set_attributes({"id": "1"})
    features["feature"]["rules"][0]["bucketVersion"] = 1
    gb.set_features(features)
    assert gb.get_feature_value("feature", -1) == 0

    assert service.get_assignments("id", "1") == {
        "attributeName": "id",
        "attributeValue": "1",
        "assignments": {
            "exp__0": "variation1",
            "exp__1": "control"
        }
    }
    gb.destroy()
    service.destroy()
