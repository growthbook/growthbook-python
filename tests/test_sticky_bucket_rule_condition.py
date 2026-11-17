"""A sticky bucket must win over an experiment rule's condition, like the JS SDK.

_eval_feature checks rule.condition up front only for force rules. An
experiment rule's condition is checked in _run_experiment after the sticky
bucket lookup, so narrowing the condition mid-test keeps bucketed users.
"""

from growthbook import GrowthBook, InMemoryStickyBucketService

USER = {"id": "user-8", "country": "US"}


def _exp_rule(countries, **extra):
    rule = {
        "key": "checkout-exp",
        "variations": ["old", "new"],
        "meta": [{"key": "0"}, {"key": "1"}],
        "hashAttribute": "id",
        "hashVersion": 2,
        "condition": {"country": {"$in": countries}},
    }
    rule.update(extra)
    return rule


def _features(rules, **extra):
    features = {"checkout-redesign": {"defaultValue": "old", "rules": rules}}
    features.update(extra)
    return features


def _eval(features, sb=None, **kwargs):
    gb = GrowthBook(features=features, attributes=dict(USER), sticky_bucket_service=sb, **kwargs)
    try:
        return gb.eval_feature("checkout-redesign")
    finally:
        gb.destroy()


def _bucket_user(sb, rule_extra=None, features_extra=None):
    """Bucket USER into "new" while the condition still matches."""
    r = _eval(_features([_exp_rule(["US", "CA"], **(rule_extra or {}))], **(features_extra or {})), sb)
    assert (r.value, r.source) == ("new", "experiment")


def test_sticky_bucket_wins_over_experiment_rule_condition():
    sb = InMemoryStickyBucketService()
    _bucket_user(sb)

    r = _eval(_features([_exp_rule(["CA"])]), sb)
    assert (r.value, r.source) == ("new", "experiment")
    assert r.experimentResult.stickyBucketUsed is True


def test_sticky_bucket_wins_over_later_force_rule():
    sb = InMemoryStickyBucketService()
    _bucket_user(sb)

    r = _eval(_features([_exp_rule(["CA"]), {"force": "forced"}]), sb)
    assert (r.value, r.source) == ("new", "experiment")


def test_forced_variation_wins_over_experiment_rule_condition():
    r = _eval(_features([_exp_rule(["CA"])]), forced_variations={"checkout-exp": 1})
    assert (r.value, r.source) == ("new", "experiment")


def test_experiment_rule_condition_applies_without_sticky_bucket():
    sb = InMemoryStickyBucketService()
    r = _eval(_features([_exp_rule(["CA"]), {"force": "forced"}]), sb)
    assert (r.value, r.source) == ("forced", "force")

    r = _eval(_features([_exp_rule(["CA"])]))
    assert (r.value, r.source) == ("old", "defaultValue")


def test_force_rule_condition_still_applies():
    rules = [{"force": "forced", "condition": {"country": "CA"}}]
    r = _eval(_features(rules), InMemoryStickyBucketService())
    assert (r.value, r.source) == ("old", "defaultValue")


def test_filters_still_apply_with_sticky_bucket():
    def filters(ranges):
        return [{"seed": "f", "attribute": "id", "hashVersion": 2, "ranges": ranges}]

    sb = InMemoryStickyBucketService()
    _bucket_user(sb, rule_extra={"filters": filters([[0, 1]])})

    r = _eval(_features([_exp_rule(["CA"], filters=filters([[0, 0]]))]), sb)
    assert (r.value, r.source) == ("old", "defaultValue")


def test_prerequisites_still_apply_with_sticky_bucket():
    def parent(value):
        return {"parent-flag": {"defaultValue": value}}

    parent_conditions = [{"id": "parent-flag", "condition": {"value": True}}]
    sb = InMemoryStickyBucketService()
    _bucket_user(sb, rule_extra={"parentConditions": parent_conditions}, features_extra=parent(True))

    features = _features([_exp_rule(["CA"], parentConditions=parent_conditions)], **parent(False))
    r = _eval(features, sb)
    assert (r.value, r.source) == ("old", "defaultValue")
