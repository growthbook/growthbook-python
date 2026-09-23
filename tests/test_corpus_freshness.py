"""Capability sub-suites must participate in the corpus drift check."""

import runpy
from pathlib import Path

import pytest


diff_corpora = runpy.run_path(
    str(Path(__file__).parent / "scripts" / "check_corpus_freshness.py")
)["_diff"]


@pytest.mark.parametrize("suite", ["evalCondition", "feature", "run"])
def test_nested_suites_report_missing_drift_and_local_cases(suite):
    upstream = {"savedGroupReferencesV2": {suite: [["missing", True], ["changed", True]]}}
    local = {"savedGroupReferencesV2": {suite: [["changed", False], ["local", True]]}}
    key = "savedGroupReferencesV2." + suite
    missing, _, extras, drift, _ = diff_corpora(upstream, local, {})
    assert missing[key] == ["missing"]
    assert drift[key] == ["changed"]
    assert extras[key] == ["local"]

    missing, skipped, _, drift, skipped_drift = diff_corpora(upstream, local, {
        "missing": {key: {"missing"}}, "drift": {key: {"changed"}},
    })
    assert missing[key] == drift[key] == []
    assert skipped[key] == ["missing"]
    assert skipped_drift[key] == ["changed"]


@pytest.mark.parametrize("capability", [None, {}, [], {"evalCondition": {}}])
def test_missing_or_malformed_capability_suite_cannot_hide_missing_cases(capability):
    upstream = {"savedGroupReferencesV2": {"evalCondition": [["new", True]]}}
    local = {"savedGroupReferencesV2": capability}
    missing, _, _, _, _ = diff_corpora(upstream, local, {})
    assert missing["savedGroupReferencesV2.evalCondition"] == ["new"]
