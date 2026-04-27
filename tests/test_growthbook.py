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
    logger,FeatureRepository,
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
import asyncio
import pytest
from unittest.mock import MagicMock,AsyncMock,patch
from growthbook.growthbook import SSEClient

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
    res = gb.eval_feature(key)

    if "experiment" in expected:
        expected["experiment"] = Experiment(**expected["experiment"]).to_dict()

    actual = res.to_dict()

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

    def track(experiment, result, user_context):
        return calls.append([experiment, result, user_context])

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
    # validate experiment and result only
    assert calls[0][0] == exp1 and calls[0][1] == res1
    assert calls[1][0] == exp2 and calls[1][1] == res4
    assert calls[2][0] == exp2 and calls[2][1] == res5

    gb.destroy()


def test_feature_usage_callback():
    """Test that feature usage callback is called correctly"""
    calls = []
    
    def feature_usage_cb(key, result, user_context):
        calls.append([key, result, user_context])
    
    gb = GrowthBook(
        attributes={"id": "1"},
        on_feature_usage=feature_usage_cb,
        features={
            "feature-1": Feature(defaultValue=True),
            "feature-2": Feature(defaultValue=False),
            "feature-3": Feature(
                defaultValue="blue",
                rules=[
                    FeatureRule(force="red", condition={"id": "1"})
                ]
            ),
        }
    )
    
    # Test eval_feature
    result1 = gb.eval_feature("feature-1")
    assert len(calls) == 1
    assert calls[0][0] == "feature-1"
    assert calls[0][1].value is True
    assert calls[0][1].source == "defaultValue"
    assert calls[0][2].attributes == {"id": "1"}
    
    # Test is_on
    gb.is_on("feature-2")
    assert len(calls) == 2
    assert calls[1][0] == "feature-2"
    assert calls[1][1].value is False
    assert calls[1][2].attributes == {"id": "1"}
    
    # Test get_feature_value
    value = gb.get_feature_value("feature-3", "blue")
    assert len(calls) == 3
    assert calls[2][0] == "feature-3"
    assert calls[2][1].value == "red"
    assert value == "red"
    assert calls[2][2].attributes == {"id": "1"}
    
    # Test is_off
    gb.is_off("feature-1")
    assert len(calls) == 4
    assert calls[3][0] == "feature-1"
    assert calls[3][2].attributes == {"id": "1"}
    
    # Calling same feature multiple times should trigger callback each time
    gb.eval_feature("feature-1")
    gb.eval_feature("feature-1")
    assert len(calls) == 6
    
    gb.destroy()


def test_feature_usage_callback_error_handling():
    """Test that feature usage callback errors are handled gracefully"""
    
    def failing_callback(key, result, user_context):
        raise Exception("Callback error")
    
    gb = GrowthBook(
        attributes={"id": "1"},
        on_feature_usage=failing_callback,
        features={
            "feature-1": Feature(defaultValue=True),
        }
    )
    
    # Should not raise an error even if callback fails
    result = gb.eval_feature("feature-1")
    assert result.value is True
    
    # Should work with is_on as well
    assert gb.is_on("feature-1") is True
    
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


def test_skip_all_experiments_flag():
    """Test that skip_all_experiments flag prevents users from being put into experiments"""
    
    # Test with skip_all_experiments=True
    gb_skip = GrowthBook(
        attributes={"id": "1"},
        skip_all_experiments=True,
        features={
            "feature-with-experiment": Feature(
                defaultValue="control",
                rules=[
                    FeatureRule(
                        key="exp-123",
                        variations=["control", "variation"],
                        weights=[0.5, 0.5]
                    )
                ]
            )
        }
    )
    
    # User should NOT be in experiment due to skip_all_experiments flag
    result = gb_skip.eval_feature("feature-with-experiment")
    assert result.value == "control"  # Should get default value
    assert result.source == "defaultValue"
    assert result.experiment is None  # No experiment should be assigned
    assert result.experimentResult is None
    
    # Test running experiment directly
    exp = Experiment(key="direct-exp", variations=["a", "b"])
    exp_result = gb_skip.run(exp)
    assert exp_result.inExperiment is False
    assert exp_result.value == "a"  # Should get first variation (control)
    
    gb_skip.destroy()
    
    # Test with skip_all_experiments=False (default behavior)
    gb_normal = GrowthBook(
        attributes={"id": "1"},
        skip_all_experiments=False,  # explicit False
        features={
            "feature-with-experiment": Feature(
                defaultValue="control",
                rules=[
                    FeatureRule(
                        key="exp-123",
                        variations=["control", "variation"],
                        weights=[0.5, 0.5]
                    )
                ]
            )
        }
    )
    
    # User SHOULD be in experiment normally
    result_normal = gb_normal.eval_feature("feature-with-experiment")
    # With id="1", this user should be assigned a variation
    assert result_normal.value in ["control", "variation"]
    assert result_normal.source == "experiment"
    
    gb_normal.destroy()

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

    assigned = gb.get_all_results()
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

    gb.set_features(featuresInput)
    gb.set_attributes(attributes)

    featuresOutput = {k: v.to_dict() for (k, v) in gb.get_features().items()}

    assert featuresOutput == featuresInput
    assert attributes == gb.get_attributes()

    newAttrs = {"url": "/hello"}
    gb.set_attributes(newAttrs)
    assert newAttrs == gb.get_attributes()

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

    assert gb.is_on("featureOn") is True
    assert gb.is_off("featureOn") is False
    assert gb.get_feature_value("featureOn", 15) == 12

    assert gb.is_on("featureOff") is False
    assert gb.is_off("featureOff") is True
    assert gb.get_feature_value("featureOff", 10) == 0

    assert gb.is_on("featureNone") is False
    assert gb.is_off("featureNone") is True
    assert gb.get_feature_value("featureNone", 10) == 10

    gb.destroy()


class MockHttpResp:
    def __init__(self, status: int, data: str) -> None:
        self.status = status
        self.data = data.encode("utf-8")
        self.headers = {}  # Add headers attribute for ETag support


def test_feature_repository(mocker):
    m = mocker.patch.object(feature_repo, "_get")
    expected = {"features": {"feature": {"defaultValue": 5}}}
    m.return_value = MockHttpResp(200, json.dumps(expected))
    features = feature_repo.load_features("https://cdn.growthbook.io", "sdk-abc123")

    # Updated assertion to account for headers parameter
    assert m.call_count == 1
    call_args = m.call_args[0]
    assert call_args[0] == "https://cdn.growthbook.io/api/features/sdk-abc123"
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

    # Updated assertion to account for headers parameter
    assert m.call_count == 1
    call_args = m.call_args[0]
    assert call_args[0] == "https://cdn.growthbook.io/api/features/sdk-abc123"
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

    # Updated assertion to account for headers parameter
    assert m.call_count == 1
    call_args = m.call_args[0]
    assert call_args[0] == "https://cdn.growthbook.io/api/features/sdk-abc123"
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
    # Updated assertion to account for headers parameter
    assert m.call_count == 1
    call_args = m.call_args[0]
    assert call_args[0] == "https://cdn.growthbook.io/api/features/sdk-abc123"

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
    # Updated assertion to account for headers parameter
    assert m.call_count == 1
    call_args = m.call_args[0]
    assert call_args[0] == "https://cdn.growthbook.io/api/features/sdk-abc123"

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


def test_ttl_automatic_feature_refresh(mocker):
    """Test that GrowthBook instances automatically get updated features when cache expires during evaluation"""
    # Mock responses to simulate feature flag changes
    mock_responses = [
        {"features": {"test_feature": {"defaultValue": False}}, "savedGroups": {}},
        {"features": {"test_feature": {"defaultValue": True}}, "savedGroups": {}}
    ]
    
    call_count = 0
    def mock_fetch_features(api_host, client_key, decryption_key=""):
        nonlocal call_count
        response = mock_responses[min(call_count, len(mock_responses) - 1)]
        call_count += 1
        return response
    
    # Clear cache and mock the fetch method
    feature_repo.clear_cache()
    m = mocker.patch.object(feature_repo, '_fetch_features', side_effect=mock_fetch_features)
    
    # Create GrowthBook instance with short TTL
    gb = GrowthBook(
        api_host="https://cdn.growthbook.io",
        client_key="test-key",
        cache_ttl=1  # 1 second TTL for testing
    )
    
    try:
        # Initial evaluation - should trigger first load
        assert gb.is_on('test_feature') == False
        assert call_count == 1
        
        # Manually expire the cache by setting expiry time to past
        cache_key = "https://cdn.growthbook.io::test-key"
        if hasattr(feature_repo.cache, 'cache') and cache_key in feature_repo.cache.cache:
            feature_repo.cache.cache[cache_key].expires = time() - 10
        
        # Next evaluation should automatically refresh cache and update features
        assert gb.is_on('test_feature') == True
        assert call_count == 2
        
    finally:
        gb.destroy()
        feature_repo.clear_cache()


def test_multiple_instances_get_updated_on_cache_expiry(mocker):
    """Test that multiple GrowthBook instances all get updated when cache expires during evaluation"""
    mock_responses = [
        {"features": {"test_feature": {"defaultValue": "v1"}}, "savedGroups": {}},
        {"features": {"test_feature": {"defaultValue": "v2"}}, "savedGroups": {}}
    ]
    
    call_count = 0
    def mock_fetch_features(api_host, client_key, decryption_key=""):
        nonlocal call_count
        response = mock_responses[min(call_count, len(mock_responses) - 1)]
        call_count += 1
        return response
    
    feature_repo.clear_cache()
    m = mocker.patch.object(feature_repo, '_fetch_features', side_effect=mock_fetch_features)
    
    # Create multiple GrowthBook instances
    gb1 = GrowthBook(api_host="https://cdn.growthbook.io", client_key="test-key")
    gb2 = GrowthBook(api_host="https://cdn.growthbook.io", client_key="test-key")
    
    try:
        # Initial evaluation from first instance - should trigger first load
        assert gb1.get_feature_value('test_feature', 'default') == "v1"
        assert call_count == 1
        
        # Second instance should use cached value (no additional API call)
        assert gb2.get_feature_value('test_feature', 'default') == "v1"
        assert call_count == 1  # Still 1, used cache
        
        # Manually expire the cache
        cache_key = "https://cdn.growthbook.io::test-key"
        if hasattr(feature_repo.cache, 'cache') and cache_key in feature_repo.cache.cache:
            feature_repo.cache.cache[cache_key].expires = time() - 10
        
        # Next evaluation should automatically refresh and notify both instances via callbacks
        assert gb1.get_feature_value('test_feature', 'default') == "v2"
        assert call_count == 2
        
        # Second instance should also have the updated value due to callbacks
        assert gb2.get_feature_value('test_feature', 'default') == "v2"
        
    finally:
        gb1.destroy()
        gb2.destroy()
        feature_repo.clear_cache()


def test_stale_while_revalidate_basic_functionality(mocker):
    """Test basic stale-while-revalidate functionality"""
    # Mock responses - first call returns v1, subsequent calls return v2
    mock_responses = [
        {"features": {"test_feature": {"defaultValue": "v1"}}, "savedGroups": {}},
        {"features": {"test_feature": {"defaultValue": "v2"}}, "savedGroups": {}}
    ]
    
    call_count = 0
    def mock_fetch_features(api_host, client_key, decryption_key=""):
        nonlocal call_count
        response = mock_responses[min(call_count, len(mock_responses) - 1)]
        call_count += 1
        return response
    
    # Clear cache and mock the fetch method
    feature_repo.clear_cache()
    m = mocker.patch.object(feature_repo, '_fetch_features', side_effect=mock_fetch_features)
    
    # Create GrowthBook instance with stale-while-revalidate enabled and short refresh interval
    gb = GrowthBook(
        api_host="https://cdn.growthbook.io",
        client_key="test-key",
        cache_ttl=10,  # 10 second TTL
        stale_while_revalidate=True,
        stale_ttl=1  # 1 second refresh interval for testing
    )
    
    try:
        # Initial evaluation - should use initial loaded data
        assert gb.get_feature_value('test_feature', 'default') == "v1"
        assert call_count == 1  # Initial load
        
        # Wait for background refresh to happen
        import time as time_module
        time_module.sleep(1.5)  # Wait longer than refresh interval
        
        # Should have triggered background refresh
        assert call_count >= 2
        
        # Next evaluation should get updated data from background refresh
        assert gb.get_feature_value('test_feature', 'default') == "v2"
        
    finally:
        gb.destroy()
        feature_repo.clear_cache()


def test_stale_while_revalidate_starts_background_task(mocker):
    """Test that stale-while-revalidate starts background refresh task"""
    mock_response = {"features": {"test_feature": {"defaultValue": "fresh"}}, "savedGroups": {}}
    
    call_count = 0
    def mock_fetch_features(api_host, client_key, decryption_key=""):
        nonlocal call_count
        call_count += 1
        return mock_response
    
    # Clear cache and mock the fetch method
    feature_repo.clear_cache()
    m = mocker.patch.object(feature_repo, '_fetch_features', side_effect=mock_fetch_features)
    
    # Create GrowthBook instance with stale-while-revalidate enabled
    gb = GrowthBook(
        api_host="https://cdn.growthbook.io",
        client_key="test-key",
        stale_while_revalidate=True,
        stale_ttl=5
    )
    
    try:
        # Should have started background refresh task
        assert feature_repo._refresh_thread is not None
        assert feature_repo._refresh_thread.is_alive()
        
        # Initial evaluation should work
        assert gb.get_feature_value('test_feature', 'default') == "fresh"
        assert call_count == 1  # Initial load
        
    finally:
        gb.destroy()
        feature_repo.clear_cache()

def test_stale_while_revalidate_disabled_fallback(mocker):
    """Test that when stale_while_revalidate is disabled, it falls back to normal behavior"""
    mock_response = {"features": {"test_feature": {"defaultValue": "normal"}}, "savedGroups": {}}
    
    call_count = 0
    def mock_fetch_features(api_host, client_key, decryption_key=""):
        nonlocal call_count
        call_count += 1
        return mock_response
    
    # Clear cache and mock the fetch method
    feature_repo.clear_cache()
    m = mocker.patch.object(feature_repo, '_fetch_features', side_effect=mock_fetch_features)
    
    # Create GrowthBook instance with stale-while-revalidate disabled (default)
    gb = GrowthBook(
        api_host="https://cdn.growthbook.io",
        client_key="test-key",
        cache_ttl=1,  # Short TTL
        stale_while_revalidate=False  # Explicitly disabled
    )
    
    try:
        # Should NOT have started background refresh task
        assert feature_repo._refresh_thread is None
        
        # Initial evaluation
        assert gb.get_feature_value('test_feature', 'default') == "normal"
        assert call_count == 1
        
        # Manually expire the cache
        cache_key = "https://cdn.growthbook.io::test-key"
        if hasattr(feature_repo.cache, 'cache') and cache_key in feature_repo.cache.cache:
            feature_repo.cache.cache[cache_key].expires = time() - 10
        
        # Next evaluation should fetch synchronously (normal behavior)
        assert gb.get_feature_value('test_feature', 'default') == "normal"
        assert call_count == 2  # Should have fetched again
        
    finally:
        gb.destroy()
        feature_repo.clear_cache()


def test_stale_while_revalidate_cleanup(mocker):
    """Test that background refresh is properly cleaned up"""
    mock_response = {"features": {"test_feature": {"defaultValue": "test"}}, "savedGroups": {}}
    
    # Mock the fetch method
    feature_repo.clear_cache()
    m = mocker.patch.object(feature_repo, '_fetch_features', return_value=mock_response)
    
    # Create GrowthBook instance with stale-while-revalidate enabled
    gb = GrowthBook(
        api_host="https://cdn.growthbook.io",
        client_key="test-key",
        stale_while_revalidate=True
    )
    
    try:
        # Should have started background refresh task
        assert feature_repo._refresh_thread is not None
        assert feature_repo._refresh_thread.is_alive()
        
        # Destroy should clean up the background task
        gb.destroy()
        
        # Background task should be stopped
        assert feature_repo._refresh_thread is None or not feature_repo._refresh_thread.is_alive()
        
    finally:
        # Ensure cleanup even if test fails
        if feature_repo._refresh_thread:
            feature_repo.stop_background_refresh()
        feature_repo.clear_cache()

def make_repo() -> FeatureRepository:
    repo=FeatureRepository()
    repo.cache.clear()
    return repo

FEATURES_RESPONSE={
    "features":{"dark-mode":{"defaultValue":False}},
    "savedGroups":[{"id":"g1","values":["a"]}],
}

@pytest.mark.asyncio
async def test_repo_cache_miss_fetches_and_stores():
    """Cache miss → calls _fetch_features_async, stores result, returns it."""
    repo=make_repo()
    with patch.object(repo,"_fetch_features_async",new=AsyncMock(return_value=FEATURES_RESPONSE)):
        result=await repo.load_features_async("https://cdn.example.com","sdk-key")

    assert result==FEATURES_RESPONSE
    # Result must be in cache now
    assert repo.cache.get("https://cdn.example.com::sdk-key")==FEATURES_RESPONSE

@pytest.mark.asyncio
async def test_repo_cache_hit_skips_fetch():
    """Cache hit → _fetch_features_async is NOT called."""
    repo=make_repo()
    repo.cache.set("https://cdn.example.com::sdk-key",FEATURES_RESPONSE,ttl=600)

    with patch.object(repo,"_fetch_features_async",new=AsyncMock()) as mock_fetch:
        result=await repo.load_features_async("https://cdn.example.com","sdk-key")

    mock_fetch.assert_not_called()
    assert result==FEATURES_RESPONSE

@pytest.mark.asyncio
async def test_repo_fetch_returns_none_does_not_cache():
    """If fetch returns None (network error etc.) → nothing is cached, None returned."""
    repo=make_repo()
    with patch.object(repo,"_fetch_features_async",new=AsyncMock(return_value=None)):
        result=await repo.load_features_async("https://cdn.example.com","sdk-key")

    assert result is None
    assert repo.cache.get("https://cdn.example.com::sdk-key") is None

@pytest.mark.asyncio
async def test_repo_notifies_callbacks_on_fresh_fetch():
    """Feature-update callbacks are called when features are fetched fresh."""
    repo=make_repo()
    callback=MagicMock()
    repo.add_feature_update_callback(callback)

    with patch.object(repo,"_fetch_features_async",new=AsyncMock(return_value=FEATURES_RESPONSE)):
        await repo.load_features_async("https://cdn.example.com","sdk-key")

    callback.assert_called_once_with(FEATURES_RESPONSE)

@pytest.mark.asyncio
async def test_repo_no_callback_on_cache_hit():
    """Callbacks must NOT fire on a cache hit — features didn't change."""
    repo=make_repo()
    repo.cache.set("https://cdn.example.com::sdk-key",FEATURES_RESPONSE,ttl=600)
    callback=MagicMock()
    repo.add_feature_update_callback(callback)

    with patch.object(repo,"_fetch_features_async",new=AsyncMock(return_value=FEATURES_RESPONSE)):
        await repo.load_features_async("https://cdn.example.com","sdk-key")

    callback.assert_not_called()

@pytest.mark.asyncio
async def test_repo_ttl_passed_to_cache():
    """TTL argument is forwarded to cache.set."""
    repo=make_repo()
    with patch.object(repo,"_fetch_features_async",new=AsyncMock(return_value=FEATURES_RESPONSE)),\
            patch.object(repo.cache,"set") as mock_set:
        await repo.load_features_async("https://cdn.example.com","sdk-key",ttl=1234)

    mock_set.assert_called_once_with("https://cdn.example.com::sdk-key",FEATURES_RESPONSE,1234)

@pytest.mark.asyncio
async def test_repo_cache_key_combines_host_and_key():
    """Cache key = api_host + '::' + client_key."""
    repo=make_repo()
    with patch.object(repo,"_fetch_features_async",new=AsyncMock(return_value=FEATURES_RESPONSE)):
        await repo.load_features_async("https://host-a.io","key-1")
        await repo.load_features_async("https://host-b.io","key-2")

    assert repo.cache.get("https://host-a.io::key-1")==FEATURES_RESPONSE
    assert repo.cache.get("https://host-b.io::key-2")==FEATURES_RESPONSE

def make_gb(**kwargs) -> GrowthBook:
    defaults=dict(api_host="https://cdn.example.com",client_key="sdk-abc")
    return GrowthBook(**{**defaults,**kwargs})

@pytest.mark.asyncio
async def test_gb_raises_without_client_key():
    gb=GrowthBook()  # no client_key
    with pytest.raises(ValueError,match="client_key"):
        await gb.load_features_async()

@pytest.mark.asyncio
async def test_gb_sets_features_from_response():
    gb=make_gb()
    with patch("growthbook.growthbook.feature_repo.load_features_async",
               new=AsyncMock(return_value=FEATURES_RESPONSE)):
        await gb.load_features_async()

    assert "dark-mode" in gb.getFeatures()

@pytest.mark.asyncio
async def test_gb_sets_saved_groups_from_response():
    gb=make_gb()
    with patch("growthbook.growthbook.feature_repo.load_features_async",
               new=AsyncMock(return_value=FEATURES_RESPONSE)):
        await gb.load_features_async()

    assert gb._saved_groups==FEATURES_RESPONSE["savedGroups"]

@pytest.mark.asyncio
async def test_gb_handles_none_response_gracefully():
    """If repo returns None, GrowthBook should not crash or modify features."""
    gb=make_gb()
    gb.set_features({"existing":{"defaultValue":True}})

    with patch("growthbook.growthbook.feature_repo.load_features_async",
               new=AsyncMock(return_value=None)):
        await gb.load_features_async()  # must not raise

    assert "existing" in gb.getFeatures()  # unchanged

@pytest.mark.asyncio
async def test_gb_response_without_saved_groups():
    """Response may omit savedGroups — should not crash."""
    gb=make_gb()
    response={"features":{"flag":{"defaultValue":1}}}
    with patch("growthbook.growthbook.feature_repo.load_features_async",
               new=AsyncMock(return_value=response)):
        await gb.load_features_async()

    assert "flag" in gb.getFeatures()

@pytest.mark.asyncio
async def test_gb_passes_correct_args_to_repo():
    """GrowthBook forwards api_host, client_key, decryption_key, cache_ttl to repo."""
    gb=GrowthBook(
        api_host="https://my-host.io",
        client_key="sdk-xyz",
        decryption_key="secret",
        cache_ttl=120,
    )
    with patch("growthbook.growthbook.feature_repo.load_features_async",
               new=AsyncMock(return_value=None)) as mock_load:
        await gb.load_features_async()

    mock_load.assert_called_once_with("https://my-host.io","sdk-xyz","secret",120)

@pytest.mark.asyncio
async def test_fetch_features_async_returns_decoded():
    """_fetch_features_async returns data after decryption."""
    repo=make_repo()
    response={"features":{"flag":{"defaultValue":True}}}

    with patch.object(repo,"_fetch_and_decode_async",new=AsyncMock(return_value=response)):
        result=await repo._fetch_features_async("https://cdn.example.com","sdk-key")

    assert result==response
    assert result["features"]["flag"]["defaultValue"] is True

@pytest.mark.asyncio
async def test_fetch_features_async_returns_none_on_empty():
    """_fetch_features_async returns None if _fetch_and_decode_async returned None."""
    repo=make_repo()

    with patch.object(repo,"_fetch_and_decode_async",new=AsyncMock(return_value=None)):
        result=await repo._fetch_features_async("https://cdn.example.com","sdk-key")

    assert result is None

@pytest.mark.asyncio
async def test_fetch_features_async_decrypts_payload():
    """_fetch_features_async calls decrypt_response with decryption_key passed."""
    repo=make_repo()
    raw={"features":{"x":{"defaultValue":1}}}

    with patch.object(repo,"_fetch_and_decode_async",new=AsyncMock(return_value=raw)),\
         patch.object(repo,"decrypt_response",return_value=raw) as mock_decrypt:
        await repo._fetch_features_async("https://cdn.example.com","sdk-key","my-secret")

    mock_decrypt.assert_called_once_with(raw,"my-secret")

@pytest.mark.asyncio
async def test_fetch_and_decode_async_returns_json():
    """_fetch_and_decode_async returns parsed JSON at 200."""
    repo=make_repo()
    payload={"features":{"dark-mode":{"defaultValue":False}}}

    mock_response=AsyncMock()
    mock_response.status=200
    mock_response.json=AsyncMock(return_value=payload)
    mock_response.headers={}
    mock_response.__aenter__=AsyncMock(return_value=mock_response)
    mock_response.__aexit__=AsyncMock(return_value=False)

    mock_session=AsyncMock()
    mock_session.get=MagicMock(return_value=mock_response)
    mock_session.__aenter__=AsyncMock(return_value=mock_session)
    mock_session.__aexit__=AsyncMock(return_value=False)

    with patch("aiohttp.ClientSession",return_value=mock_session):
        result=await repo._fetch_and_decode_async("https://cdn.example.com","sdk-key")

    assert result==payload

@pytest.mark.asyncio
async def test_fetch_and_decode_async_returns_none_on_4xx():
    """_fetch_and_decode_async returns None when HTTP >= 400."""
    repo=make_repo()

    mock_response=AsyncMock()
    mock_response.status=403
    mock_response.headers={}
    mock_response.__aenter__=AsyncMock(return_value=mock_response)
    mock_response.__aexit__=AsyncMock(return_value=False)

    mock_session=AsyncMock()
    mock_session.get=MagicMock(return_value=mock_response)
    mock_session.__aenter__=AsyncMock(return_value=mock_session)
    mock_session.__aexit__=AsyncMock(return_value=False)

    with patch("aiohttp.ClientSession",return_value=mock_session):
        result=await repo._fetch_and_decode_async("https://cdn.example.com","sdk-key")

    assert result is None

@pytest.mark.asyncio
async def test_fetch_and_decode_async_uses_etag_cache():
    """At 304, it returns data from the etag cache without parsing the body."""
    repo=make_repo()
    cached_payload={"features":{"cached":{"defaultValue":True}}}
    url=repo._get_features_url("https://cdn.example.com","sdk-key")
    repo._etag_cache[url]=("etag-abc",cached_payload)

    mock_response=AsyncMock()
    mock_response.status=304
    mock_response.headers={}
    mock_response.__aenter__=AsyncMock(return_value=mock_response)
    mock_response.__aexit__=AsyncMock(return_value=False)

    mock_session=AsyncMock()
    mock_session.get=MagicMock(return_value=mock_response)
    mock_session.__aenter__=AsyncMock(return_value=mock_session)
    mock_session.__aexit__=AsyncMock(return_value=False)

    with patch("aiohttp.ClientSession",return_value=mock_session):
        result=await repo._fetch_and_decode_async("https://cdn.example.com","sdk-key")

    assert result==cached_payload

@pytest.mark.asyncio
async def test_fetch_and_decode_async_stores_new_etag():
    """The new ETag from the response is saved in _etag_cache."""
    repo=make_repo()
    payload={"features":{}}

    mock_response=AsyncMock()
    mock_response.status=200
    mock_response.json=AsyncMock(return_value=payload)
    mock_response.headers={"ETag":"new-etag-123"}
    mock_response.__aenter__=AsyncMock(return_value=mock_response)
    mock_response.__aexit__=AsyncMock(return_value=False)

    mock_session=AsyncMock()
    mock_session.get=MagicMock(return_value=mock_response)
    mock_session.__aenter__=AsyncMock(return_value=mock_session)
    mock_session.__aexit__=AsyncMock(return_value=False)

    with patch("aiohttp.ClientSession",return_value=mock_session):
        await repo._fetch_and_decode_async("https://cdn.example.com","sdk-key")

    url=repo._get_features_url("https://cdn.example.com","sdk-key")
    assert repo._etag_cache[url][0]=="new-etag-123"

@pytest.mark.asyncio
async def test_fetch_and_decode_async_returns_none_on_network_error():
    """_fetch_and_decode_async return None when network error occur."""
    import aiohttp
    repo=make_repo()

    with patch("aiohttp.ClientSession") as mock_cls:
        mock_session=AsyncMock()
        mock_session.__aenter__=AsyncMock(return_value=mock_session)
        mock_session.__aexit__=AsyncMock(return_value=False)
        mock_session.get=MagicMock(side_effect=aiohttp.ClientError("connection failed"))
        mock_cls.return_value=mock_session

        result=await repo._fetch_and_decode_async("https://cdn.example.com","sdk-key")

    assert result is None

def test_start_auto_refresh_creates_sse_client():
    """startAutoRefresh creates SSEClient and calls connect."""
    repo=make_repo()
    cb=MagicMock()

    with patch("growthbook.growthbook.SSEClient") as mock_sse_cls:
        mock_sse=MagicMock()
        mock_sse_cls.return_value=mock_sse
        repo.startAutoRefresh("https://cdn.example.com","sdk-key",cb)

    mock_sse_cls.assert_called_once_with(
        api_host="https://cdn.example.com",
        client_key="sdk-key",
        on_event=cb,
        timeout=30,
    )
    mock_sse.connect.assert_called_once()

def test_start_auto_refresh_reuses_existing_client():
    """startAutoRefresh does not create a new SSEClient if existed."""
    repo=make_repo()
    existing_client=MagicMock()
    repo.sse_client=existing_client

    with patch("growthbook.growthbook.SSEClient") as mock_sse_cls:
        repo.startAutoRefresh("https://cdn.example.com","sdk-key",MagicMock())

    mock_sse_cls.assert_not_called()
    existing_client.connect.assert_called_once()

def test_start_auto_refresh_raises_without_client_key():
    """startAutoRefresh throws ValueError if there is no client_key."""
    repo=make_repo()

    with pytest.raises(ValueError,match="client_key"):
        repo.startAutoRefresh("https://cdn.example.com","",MagicMock())

def make_gb_simple(**kwargs):
    defaults=dict(api_host="https://cdn.example.com", client_key="sdk-key")
    return GrowthBook(**{**defaults, **kwargs})

def test_dispatch_sse_event_features_calls_handler():
    """Event 'features' pass data into _features_event_handler."""
    gb=make_gb_simple()
    event_data={"type":"features","data":'{"features":{"flag":{"defaultValue":True}}}'}

    with patch.object(gb,"_features_event_handler") as mock_handler:
        gb._dispatch_sse_event(event_data)

    mock_handler.assert_called_once_with('{"features":{"flag":{"defaultValue":True}}}')

def test_dispatch_sse_event_features_updated_calls_load():
    """Event 'features-updated' calls load_features."""
    gb=make_gb_simple()
    event_data={"type":"features-updated","data":"{}"}

    with patch.object(gb,"load_features") as mock_load:
        gb._dispatch_sse_event(event_data)

    mock_load.assert_called_once()

def test_dispatch_sse_event_unknown_type_does_nothing():
    """An unknown event type does not call either load_features or _features_event_handler."""
    gb=make_gb_simple()
    event_data={"type":"ping","data":"{}"}

    with patch.object(gb,"load_features") as mock_load,\
         patch.object(gb,"_features_event_handler") as mock_handler:
        gb._dispatch_sse_event(event_data)

    mock_load.assert_not_called()
    mock_handler.assert_not_called()

def test_dispatch_sse_event_features_updates_gb_state():
    """After 'features' event, it updates features in GrowthBook."""
    gb=make_gb_simple()
    payload={"features":{"dark-mode":{"defaultValue":True}}}
    event_data={"type":"features","data":json.dumps(payload)}

    with patch.object(feature_repo,"decrypt_response",return_value=payload),\
         patch.object(feature_repo,"save_in_cache"):
        gb._dispatch_sse_event(event_data)

    assert "dark-mode" in gb.get_features()

def test_gb_start_auto_refresh_raises_without_client_key():
    """startAutoRefresh throws ValueError if there is no client_key."""
    gb=GrowthBook()

    with pytest.raises(ValueError,match="client_key"):
        gb.startAutoRefresh()

def test_gb_start_auto_refresh_delegates_to_feature_repo():
    """startAutoRefresh passes args into feature_repo.startAutoRefresh."""
    gb=GrowthBook(
        api_host="https://cdn.example.com",
        client_key="sdk-key",
        streaming_connection_timeout=60,
    )

    with patch.object(feature_repo,"startAutoRefresh") as mock_start:
        gb.startAutoRefresh()

    mock_start.assert_called_once_with(
        api_host="https://cdn.example.com",
        client_key="sdk-key",
        cb=gb._dispatch_sse_event,
        streaming_timeout=60,
    )

def test_gb_start_auto_refresh_passes_dispatch_as_callback():
    """Passed callback into feature_repo — it is _dispatch_sse_event."""
    gb=make_gb_simple()
    captured={}

    def fake_start(**kwargs):
        captured["cb"]=kwargs["cb"]

    with patch.object(feature_repo,"startAutoRefresh",side_effect=fake_start):
        gb.startAutoRefresh()

    assert captured["cb"].__func__ is gb._dispatch_sse_event.__func__
    assert captured["cb"].__self__ is gb

@pytest.mark.asyncio
async def test_gb_load_features_async_sets_features():
    """load_features_async updates features in GrowthBook via feature_repo."""
    gb=make_gb_simple()
    payload={"features":{"new-flag":{"defaultValue":42}},"savedGroups":[]}

    with patch("growthbook.growthbook.feature_repo.load_features_async",
               new=AsyncMock(return_value=payload)):
        await gb.load_features_async()

    assert "new-flag" in gb.get_features()
    assert gb.get_features()["new-flag"].defaultValue==42

@pytest.mark.asyncio
async def test_gb_load_features_async_sets_saved_groups():
    """load_features_async updates savedGroups."""
    gb=make_gb_simple()
    groups=[{"id":"g1","values":["a","b"]}]
    payload={"features":{},"savedGroups":groups}

    with patch("growthbook.growthbook.feature_repo.load_features_async",
               new=AsyncMock(return_value=payload)):
        await gb.load_features_async()

    assert gb._saved_groups==groups

@pytest.mark.asyncio
async def test_gb_load_features_async_none_does_not_reset_features():
    """If the repo returns None, the existing features are not reset."""
    gb=make_gb_simple()
    gb.set_features({"existing":{"defaultValue":True}})

    with patch("growthbook.growthbook.feature_repo.load_features_async",
               new=AsyncMock(return_value=None)):
        await gb.load_features_async()

    assert "existing" in gb.get_features()

@pytest.mark.asyncio
async def test_gb_load_features_async_raises_without_client_key():
    """load_features_async throws ValueError if there is no client_key."""
    gb=GrowthBook()

    with pytest.raises(ValueError,match="client_key"):
        await gb.load_features_async()

def make_sse_client(**kwargs):
    defaults=dict(api_host="https://cdn.example.com",client_key="sdk-key",on_event=MagicMock())
    return SSEClient(**{**defaults,**kwargs})

def make_fake_response(lines: list[str]):
    """Converts a list of strings into an async byte iterator, like aiohttp does."""
    async def content_iter():
        for line in lines:
            yield line.encode("utf-8")

    response=MagicMock()
    response.content=content_iter()
    return response

@pytest.mark.asyncio
async def test_process_response_fires_on_event():
    """A full SSE message calls on_event once."""
    on_event=MagicMock()
    client=make_sse_client(on_event=on_event)
    client.is_running=True

    response=make_fake_response([
        "event: features\n",
        "data: {\"features\":{}}\n",
        "\n",  # empty line = end of message
    ])

    await client._process_response(response)

    on_event.assert_called_once_with({"type":"features","data":'{"features":{}}'})

@pytest.mark.asyncio
async def test_process_response_no_event_without_empty_line():
    """A message without an empty line separator does not trigger an on_event."""
    on_event=MagicMock()
    client=make_sse_client(on_event=on_event)
    client.is_running=True

    response=make_fake_response([
        "event: features\n",
        "data: {}\n",
        # no empty line
    ])

    await client._process_response(response)

    on_event.assert_called_once()

@pytest.mark.asyncio
async def test_process_response_multiple_events():
    """Multiple messages in one thread."""
    on_event=MagicMock()
    client=make_sse_client(on_event=on_event)
    client.is_running=True

    response=make_fake_response([
        "event: features\n","data: first\n","\n",
        "event: ping\n","data: {}\n","\n",
    ])

    await client._process_response(response)

    assert on_event.call_count==2
    assert on_event.call_args_list[0][0][0]["data"]=="first"
    assert on_event.call_args_list[1][0][0]["type"]=="ping"

@pytest.mark.asyncio
async def test_process_response_stops_when_not_running():
    """If is_running become False — process stops."""
    on_event=MagicMock()
    client=make_sse_client(on_event=on_event)
    client.is_running=False

    response=make_fake_response([
        "event: features\n","data: {}\n","\n",
    ])

    await client._process_response(response)

    on_event.assert_not_called()

@pytest.mark.asyncio
async def test_process_response_handler_exception_does_not_crash():
    """Exception in on_event should not stip processing."""
    on_event=MagicMock(side_effect=RuntimeError("boom"))
    client=make_sse_client(on_event=on_event)
    client.is_running=True

    response=make_fake_response([
        "event: features\n","data: {}\n","\n",
        "event: ping\n","data: {}\n","\n",
    ])

    await client._process_response(
        response)
    assert on_event.call_count==2

@pytest.mark.asyncio
async def test_init_session_connects_and_processes():
    """_init_session calls GET and processes the response."""
    on_event=MagicMock()
    client=make_sse_client(on_event=on_event)
    client.is_running=True

    mock_response=MagicMock()
    mock_response.raise_for_status=MagicMock()

    # _process_response stops the loop, because is_running will be False
    async def fake_process(response):
        client.is_running=False

    with patch.object(client,"_process_response",side_effect=fake_process),\
            patch("aiohttp.ClientSession") as mock_session_cls:
        mock_session=AsyncMock()
        mock_session.__aenter__=AsyncMock(return_value=mock_session)
        mock_session.__aexit__=AsyncMock(return_value=False)
        mock_session.get=MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))
        mock_session_cls.return_value=mock_session

        await client._init_session()

@pytest.mark.asyncio
async def test_init_session_stops_on_4xx():
    """HTTP 4xx stops is_running (not connect)."""
    from aiohttp.client_exceptions import ClientResponseError

    client=make_sse_client(on_event=MagicMock())
    client.is_running=True

    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_session=AsyncMock()
        mock_session.__aenter__=AsyncMock(return_value=mock_session)
        mock_session.__aexit__=AsyncMock(return_value=False)

        error=ClientResponseError(request_info=MagicMock(),history=(),status=401)
        mock_response=MagicMock()
        mock_response.raise_for_status=MagicMock(side_effect=error)
        mock_session.get=MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))
        mock_session_cls.return_value=mock_session

        await client._init_session()

    assert not client.is_running

def test_connect_starts_thread():
    client=make_sse_client(on_event=MagicMock())

    with patch.object(client,
                      "_run_sse_channel"):
        client.connect()

    assert client.is_running
    assert client._sse_thread is not None

def test_connect_is_idempotent():
    """Calling connect() again does not start a new thread."""
    client=make_sse_client(on_event=MagicMock())
    client.is_running=True

    with patch("threading.Thread") as mock_thread:
        client.connect()

    mock_thread.assert_not_called()

def test_get_sse_url():
    client=make_sse_client(on_event=MagicMock())
    url=client._get_sse_url("https://cdn.growthbook.io","sdk-abc")
    assert url=="https://cdn.growthbook.io/sub/sdk-abc"

def test_get_sse_url_strips_trailing_slash():
    client=make_sse_client(on_event=MagicMock())
    url=client._get_sse_url("https://cdn.growthbook.io/","sdk-abc")
    assert url=="https://cdn.growthbook.io/sub/sdk-abc"

def test_custom_headers_merged():
    client=make_sse_client(on_event=MagicMock(),headers={"X-Custom":"value"})
    assert client.headers["X-Custom"]=="value"
    assert "Accept" in client.headers

@pytest.mark.asyncio
async def test_wait_for_reconnect_sleeps():
    """_wait_for_reconnect waits reconnect_delay s."""
    client=make_sse_client(on_event=MagicMock(),reconnect_delay=7)

    with patch("asyncio.sleep",new=AsyncMock()) as mock_sleep:
        await client._wait_for_reconnect()

    mock_sleep.assert_called_once_with(7)

@pytest.mark.asyncio
async def test_wait_for_reconnect_propagates_cancelled():
    """CanceledError is thrown."""
    client=make_sse_client(on_event=MagicMock())

    with patch("asyncio.sleep",new=AsyncMock(side_effect=asyncio.CancelledError)):
        with pytest.raises(asyncio.CancelledError):
            await client._wait_for_reconnect()

@pytest.mark.asyncio
async def test_stop_session_closes_open_session():
    """_stop_session closes opened session."""
    client=make_sse_client(on_event=MagicMock())
    mock_session=AsyncMock()
    mock_session.closed=False
    client._sse_session=mock_session
    client._loop=None

    await client._stop_session()

    mock_session.close.assert_called_once()

@pytest.mark.asyncio
async def test_stop_session_skips_already_closed():
    """_stop_session does not touch an already closed session."""
    client=make_sse_client(on_event=MagicMock())
    mock_session=AsyncMock()
    mock_session.closed=True
    client._sse_session=mock_session
    client._loop=None

    await client._stop_session()

    mock_session.close.assert_not_called()

@pytest.mark.asyncio
async def test_stop_session_no_session_does_not_crash():
    """_stop_session works when there is no session."""
    client=make_sse_client(on_event=MagicMock())
    client._sse_session=None
    client._loop=None

    await client._stop_session()

def test_disconnect_sets_is_running_false():
    """disconnect always sets is_running = False."""
    client=make_sse_client(on_event=MagicMock())
    client.is_running=True
    client._loop=None
    client._sse_thread=None

    client.disconnect()

    assert client.is_running is False

def test_disconnect_no_loop_no_thread_does_not_crash():
    """disconnect without loop and thread works without exceptions."""
    client=make_sse_client(on_event=MagicMock())
    client._loop=None
    client._sse_thread=None

    client.disconnect()

def test_disconnect_joins_thread():
    """disconnect calls join on the stream with the passed timeout."""
    client=make_sse_client(on_event=MagicMock())
    client._loop=None

    mock_thread=MagicMock()
    mock_thread.is_alive.return_value=False
    client._sse_thread=mock_thread

    client.disconnect(timeout=5)

    mock_thread.join.assert_called_once_with(timeout=5)

def test_disconnect_warns_if_thread_still_alive():
    """disconnect logs a warning if the flow has not ended before the timeout."""
    client=make_sse_client(on_event=MagicMock())
    client._loop=None

    mock_thread=MagicMock()
    mock_thread.is_alive.return_value=True
    client._sse_thread=mock_thread

    with patch("growthbook.growthbook.logger") as mock_logger:
        client.disconnect(timeout=1)

    warning_calls=[str(c) for c in mock_logger.warning.call_args_list]
    assert any("did not terminate" in w for w in warning_calls)

def test_disconnect_calls_stop_session_when_loop_running():
    """If the loop is running, disconnect sends _stop_session via run_coroutine_threadsafe."""
    client=make_sse_client(on_event=MagicMock())
    client._sse_thread=None

    mock_loop=MagicMock()
    mock_loop.is_running.return_value=True
    client._loop=mock_loop

    mock_future=MagicMock()
    mock_future.result=MagicMock(return_value=None)

    with patch("asyncio.run_coroutine_threadsafe",return_value=mock_future) as mock_rcts:
        client.disconnect(timeout=7)

    mock_rcts.assert_called_once()
    mock_future.result.assert_called_once_with(timeout=7)

def test_disconnect_force_stops_loop_on_future_exception():
    """If future.result threw an exception, the loop is forced to stop."""
    client=make_sse_client(on_event=MagicMock())
    client._sse_thread=None

    mock_loop=MagicMock()
    mock_loop.is_running.return_value=True
    client._loop=mock_loop

    mock_future=MagicMock()
    mock_future.result=MagicMock(side_effect=TimeoutError("timeout"))

    with patch("asyncio.run_coroutine_threadsafe",return_value=mock_future):
        client.disconnect()

    mock_loop.call_soon_threadsafe.assert_called_once_with(mock_loop.stop)

def test_disconnect_skips_stop_session_when_loop_not_running():
    """If the loop is not running, run_coroutine_threadsafe is not called."""
    client=make_sse_client(on_event=MagicMock())
    client._sse_thread=None

    mock_loop=MagicMock()
    mock_loop.is_running.return_value=False
    client._loop=mock_loop

    with patch("asyncio.run_coroutine_threadsafe") as mock_rcts:
        client.disconnect()

    mock_rcts.assert_not_called()
