"""Regression tests for memory-leak / redundant-reparse fixes."""
import gc
import weakref

from growthbook import GrowthBook
from growthbook.growthbook import feature_repo
from growthbook.growthbook_client import GrowthBookClient
from growthbook.common_types import Options, UserContext, Experiment, Result


def test_feature_update_callback_does_not_pin_instance():
    """Per-request GrowthBook(client_key=...) must be GC-able without destroy()."""
    feature_repo._feature_update_callbacks = []
    gb = GrowthBook(client_key="sdk-abc", features={"f": {"defaultValue": 1}})
    assert len(feature_repo._feature_update_callbacks) == 1
    inst_ref = weakref.ref(gb)
    del gb
    gc.collect()
    assert inst_ref() is None, "registered callback is pinning the GrowthBook instance"
    # Dead refs are pruned on next notify
    feature_repo._notify_feature_update_callbacks({"features": {}})
    assert len(feature_repo._feature_update_callbacks) == 0


def test_feature_update_callback_dedup_and_remove():
    feature_repo._feature_update_callbacks = []
    calls = []

    def cb(data):
        calls.append(data)

    feature_repo.add_feature_update_callback(cb)
    feature_repo.add_feature_update_callback(cb)
    assert len(feature_repo._feature_update_callbacks) == 1
    feature_repo._notify_feature_update_callbacks({"features": {"x": {}}})
    assert len(calls) == 1
    feature_repo.remove_feature_update_callback(cb)
    assert len(feature_repo._feature_update_callbacks) == 0


def test_load_features_skips_reparse_on_cache_hit(mocker):
    feature_repo.clear_cache()
    feature_repo._feature_update_callbacks = []
    payload = {"features": {"f": {"defaultValue": True}}, "savedGroups": {}}
    mocker.patch.object(feature_repo, "load_features", return_value=payload)
    gb = GrowthBook(client_key="sdk-abc")
    spy = mocker.spy(gb, "set_features")
    gb.load_features()
    gb.load_features()
    gb.load_features()
    assert spy.call_count == 1, "identity-equal payload should not trigger re-parse"
    assert gb.is_on("f") is True
    gb.destroy()


def test_destroy_does_not_mutate_caller_dicts():
    attrs = {"id": "u1"}
    gb = GrowthBook(features={"f": {"defaultValue": 1}}, attributes=attrs)
    gb.destroy()
    assert attrs == {"id": "u1"}, "destroy() must not .clear() caller-supplied dicts"


def test_growthbook_client_tracked_is_bounded():
    client = GrowthBookClient(Options(on_experiment_viewed=lambda e, r, u: None))
    client._tracked_max = 8
    exp = Experiment(key="exp", variations=[0, 1])
    uc = UserContext(attributes={})
    for i in range(40):
        res = Result(
            variationId=0,
            inExperiment=True,
            value=0,
            hashUsed=True,
            hashAttribute="id",
            hashValue=str(i),
            featureId="f",
        )
        client._track(exp, res, uc)
    assert len(client._tracked) <= client._tracked_max
