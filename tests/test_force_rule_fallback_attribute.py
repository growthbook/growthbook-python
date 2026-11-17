"""A force rule's rollout uses fallbackAttribute only with sticky bucketing on, like the JS SDK."""

from growthbook import GrowthBook, InMemoryStickyBucketService


def _eval(rule_extra=None, sb=None):
    rule = {"force": "on", "coverage": 1.0, "hashAttribute": "id", "fallbackAttribute": "deviceId"}
    rule.update(rule_extra or {})
    gb = GrowthBook(
        features={"flag": {"defaultValue": "off", "rules": [rule]}},
        attributes={"deviceId": "dev-1"},
        sticky_bucket_service=sb,
    )
    try:
        r = gb.eval_feature("flag")
        return r.value, r.source
    finally:
        gb.destroy()


def test_force_rule_ignores_fallback_attribute_without_sticky_bucketing():
    assert _eval() == ("off", "defaultValue")


def test_force_rule_uses_fallback_attribute_with_sticky_bucketing():
    assert _eval(sb=InMemoryStickyBucketService()) == ("on", "force")


def test_force_rule_ignores_fallback_attribute_when_sticky_bucketing_disabled():
    rule_extra = {"disableStickyBucketing": True}
    assert _eval(rule_extra, sb=InMemoryStickyBucketService()) == ("off", "defaultValue")


def test_force_rule_uses_hash_attribute_when_present():
    gb = GrowthBook(
        features={"flag": {"defaultValue": "off", "rules": [
            {"force": "on", "coverage": 1.0, "hashAttribute": "id", "fallbackAttribute": "deviceId"}
        ]}},
        attributes={"id": "user-1"},
    )
    try:
        assert gb.eval_feature("flag").value == "on"
    finally:
        gb.destroy()
