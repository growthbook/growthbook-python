"""Every section of the shared spec (tests/cases.json) is either run by a test or
excluded here with a reason, so a new section in a spec update can't be
skipped without anyone noticing.

Top-level sections reach their tests through the `<section>_data` fixtures in
test_growthbook.py's pytest_generate_tests, which silently ignores a section no
test asks for; nested ones (`parent.child`) are read by their own suites."""

import json
import os

# Sections the tests run, and the test that runs each one
RUN_SECTIONS = {
    "evalCondition": "test_growthbook.py::test_conditions",
    "hash": "test_growthbook.py::test_hash",
    "getBucketRange": "test_growthbook.py::test_get_bucket_range",
    "feature": "test_growthbook.py::test_feature",
    "contextualBandit": "test_growthbook.py::test_contextual_bandit",
    "run": "test_growthbook.py::test_run",
    "chooseVariation": "test_growthbook.py::test_choose_variation",
    "getQueryStringOverride": "test_growthbook.py::test_get_qs_override",
    "inNamespace": "test_growthbook.py::test_namespace",
    "getEqualWeights": "test_growthbook.py::test_equal_weights",
    "decrypt": "test_growthbook.py::test_decrypt",
    "stickyBucket": "test_growthbook.py::test_stickyBucket",
    "trackingCalls": "test_growthbook.py::test_tracking_calls",
    "pruneCondition": "test_growthbook.py::test_prune_condition",
    "reducePayload": "test_growthbook.py::test_reduce_payload",
    "savedGroupReferencesV2.evalCondition": "test_saved_groups.py",
    "savedGroupReferencesV2.feature": "test_saved_groups.py",
    "savedGroupReferencesV2.run": "test_saved_groups.py",
    "savedGroupReferencesRemote.evalCondition": "test_remote_saved_groups.py",
    "savedGroupReferencesRemote.feature": "test_remote_saved_groups.py",
}

# Sections the tests don't run, and why
EXCLUDED_SECTIONS = {
    "urlRedirect": "URL redirect experiments are a browser feature",
}


def spec_sections():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.json")
    with open(path) as file:
        cases = json.load(file)
    sections = set()
    for name, value in cases.items():
        if isinstance(value, list):
            sections.add(name)
        elif isinstance(value, dict):
            sections.update(f"{name}.{child}" for child in value)
    return sections


def test_every_spec_section_is_run_or_excluded():
    unaccounted = spec_sections() - RUN_SECTIONS.keys() - EXCLUDED_SECTIONS.keys()
    assert unaccounted == set(), "Run these spec sections or exclude them with a reason"


def test_listed_sections_exist_in_the_spec():
    unknown = (RUN_SECTIONS.keys() | EXCLUDED_SECTIONS.keys()) - spec_sections()
    assert unknown == set(), "These sections aren't in cases.json"


def test_no_section_is_both_run_and_excluded():
    assert RUN_SECTIONS.keys() & EXCLUDED_SECTIONS.keys() == set()


def test_every_excluded_section_has_a_reason():
    assert all(reason.strip() for reason in EXCLUDED_SECTIONS.values())
