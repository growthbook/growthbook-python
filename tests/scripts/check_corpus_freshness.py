#!/usr/bin/env python3
"""Check Python's tests/cases.json against the JS SDK's cases.json.

The two corpora are maintained by hand; `specVersion` is a label, not a
contract. This script diffs the corpora and makes drift visible.

Diff categories:

  - "missing" — JS has a case name Python doesn't.
                Fails CI unless the name is in skiplist["missing"][key].
  - "drift"   — Both sides have the case name, but the bodies differ
                (canonical-JSON SHA1 mismatch). Fails CI unless the name
                is in skiplist["drift"][key]. Catches the silent
                case-body update that pure name-matching misses.
  - "extra"   — Python has a case name JS doesn't. Reported as
                informational only — Python carries documented
                extensions plus locally-authored regressions. Never
                fails CI.

Source-of-truth URL is configurable via --js-source or env GB_JS_CASES_URL.
Defaults to the JS SDK's main-branch raw URL.

Exit codes:
  0 — no actionable findings (or all on skiplist)
  1 — at least one missing or drifted case isn't on the skiplist
  2 — fetch / parse / IO error (treated as build infra failure)

Run locally:
  python3 tests/scripts/check_corpus_freshness.py
  python3 tests/scripts/check_corpus_freshness.py --js-source /path/to/local/cases.json
  GB_JS_CASES_URL=https://... python3 tests/scripts/check_corpus_freshness.py

In CI, this runs on every push as a separate step in the build workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CASES = REPO_ROOT / "tests" / "cases.json"
SKIPLIST = REPO_ROOT / "tests" / "scripts" / "corpus_skiplist.json"

DEFAULT_JS_URL = (
    "https://raw.githubusercontent.com/growthbook/growthbook/main/"
    "packages/sdk-js/test/cases.json"
)

# Top-level keys to diff. Other keys in cases.json (specVersion, decrypt
# binary blobs, urlRedirect which Python doesn't yet wire) are skipped
# either because they're scalar metadata or because the divergence is
# tracked separately.
KEYS_TO_DIFF = (
    "evalCondition",
    "feature",
    "run",
    "hash",
    "getBucketRange",
    "chooseVariation",
    "getQueryStringOverride",
    "inNamespace",
    "getEqualWeights",
    "stickyBucket",
)


def _fetch_js_cases(source: str) -> dict:
    """Fetch JS cases.json from a URL or local path.

    Raises RuntimeError with a human-readable message on failure.
    """
    if source.startswith(("http://", "https://")):
        try:
            req = urllib.request.Request(
                source, headers={"User-Agent": "growthbook-python-corpus-check"}
            )
            with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(f"fetch failed: {source}: {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JS source did not return valid JSON: {e}") from e
    path = Path(source)
    if not path.is_file():
        raise RuntimeError(f"local source not found: {source}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"local source invalid JSON: {e}") from e


def _load_local_cases() -> dict:
    if not LOCAL_CASES.is_file():
        raise RuntimeError(f"local cases.json not found: {LOCAL_CASES}")
    return json.loads(LOCAL_CASES.read_text(encoding="utf-8"))


def _load_skiplist() -> Dict[str, Dict[str, Set[str]]]:
    """Load skiplist. File format:

        {
          "missing": { "<top_level_key>": ["case name", ...] },
          "drift":   { "<top_level_key>": ["case name", ...] }
        }

    `missing` — case names Python deliberately doesn't carry from JS.
    `drift`   — case names where Python deliberately diverges from JS's
                body (rare; reserved for cases that test a Python-only
                extension behavior).

    Extras (Python has, JS doesn't) are reported but never fail, so they
    don't need a skiplist entry. The file is optional.
    """
    if not SKIPLIST.is_file():
        return {"missing": {}, "drift": {}}
    try:
        data = json.loads(SKIPLIST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"skiplist invalid JSON: {e}") from e
    return {
        "missing": {k: set(v) for k, v in (data.get("missing") or {}).items()},
        "drift": {k: set(v) for k, v in (data.get("drift") or {}).items()},
    }


def _case_signatures_grouped(cases: list) -> Dict[str, List[str]]:
    """Return {case_name: [body_hash, body_hash, ...]} preserving order.

    Body = everything after the name (case[1:]), serialized via canonical
    JSON (sorted keys + compact separators) so logically-equal cases hash
    the same regardless of dict key insertion order or whitespace.

    Same-named cases keep every occurrence in the list because pytest's
    `pytest_generate_tests` parametrizes the FULL case list — so all
    duplicates run. Collapsing them to a single entry would silently hide
    body drift in any occurrence except the last (the original bug here).

    Drift detection compares per-name multisets, so the order of
    duplicates within a name doesn't matter; only the set of body hashes
    does.
    """
    out: Dict[str, List[str]] = {}
    for c in cases:
        if not (isinstance(c, list) and c and isinstance(c[0], str)):
            continue
        name = c[0]
        body = json.dumps(c[1:], sort_keys=True, separators=(",", ":"))
        h = hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]
        out.setdefault(name, []).append(h)
    return out


def _diff(
    js_cases: dict, py_cases: dict, skip: Dict[str, Set[str]]
) -> Tuple[
    Dict[str, List[str]],  # actionable_missing
    Dict[str, List[str]],  # skipped_missing
    Dict[str, List[str]],  # extras
    Dict[str, List[str]],  # actionable_drift (same name, different body)
    Dict[str, List[str]],  # skipped_drift
]:
    """Diff (name, body-hash) pairs.

    Returns five per-key dictionaries:
      * actionable_missing — JS has the name, Python doesn't (fails CI)
      * skipped_missing    — same, but the name is on the skiplist
      * extras             — Python has the name, JS doesn't (never fails)
      * actionable_drift   — same name, hash differs (fails CI)
      * skipped_drift      — same name, hash differs, name on drift-skiplist
    """
    actionable_missing: Dict[str, List[str]] = {}
    skipped_missing: Dict[str, List[str]] = {}
    extras: Dict[str, List[str]] = {}
    actionable_drift: Dict[str, List[str]] = {}
    skipped_drift: Dict[str, List[str]] = {}

    missing_skip = skip.get("missing", {})
    drift_skip = skip.get("drift", {})

    for key in KEYS_TO_DIFF:
        js_list = js_cases.get(key, [])
        py_list = py_cases.get(key, [])
        if not isinstance(js_list, list) or not isinstance(py_list, list):
            continue

        js_grouped = _case_signatures_grouped(js_list)
        py_grouped = _case_signatures_grouped(py_list)

        # Preserve JS file ordering for the missing/drift reports; iterate
        # the original list and dedupe so each unique name appears once.
        js_names_ordered: List[str] = []
        seen_js: Set[str] = set()
        for c in js_list:
            if isinstance(c, list) and c and isinstance(c[0], str) and c[0] not in seen_js:
                js_names_ordered.append(c[0])
                seen_js.add(c[0])

        py_names_ordered: List[str] = []
        seen_py: Set[str] = set()
        for c in py_list:
            if isinstance(c, list) and c and isinstance(c[0], str) and c[0] not in seen_py:
                py_names_ordered.append(c[0])
                seen_py.add(c[0])

        missing = [n for n in js_names_ordered if n not in py_grouped]
        extra = [n for n in py_names_ordered if n not in js_grouped]

        # Drift = same name on both sides, but the multiset of body hashes
        # differs. Catches single-occurrence drift AND drift where only
        # ONE of several duplicates changed (the original bug).
        drift = [
            n for n in js_names_ordered
            if n in py_grouped and Counter(js_grouped[n]) != Counter(py_grouped[n])
        ]

        key_missing_skip = missing_skip.get(key, set())
        key_drift_skip = drift_skip.get(key, set())

        actionable_missing[key] = [n for n in missing if n not in key_missing_skip]
        skipped_missing[key] = [n for n in missing if n in key_missing_skip]
        extras[key] = extra
        actionable_drift[key] = [n for n in drift if n not in key_drift_skip]
        skipped_drift[key] = [n for n in drift if n in key_drift_skip]

    return actionable_missing, skipped_missing, extras, actionable_drift, skipped_drift


def _spec_versions(js_cases: dict, py_cases: dict) -> Tuple[str, str]:
    return (
        str(js_cases.get("specVersion", "<unset>")),
        str(py_cases.get("specVersion", "<unset>")),
    )


def _format_report(
    js_spec: str,
    py_spec: str,
    actionable_missing: Dict[str, List[str]],
    skipped_missing: Dict[str, List[str]],
    extras: Dict[str, List[str]],
    actionable_drift: Dict[str, List[str]],
    skipped_drift: Dict[str, List[str]],
) -> str:
    lines = []
    lines.append("=== Corpus freshness check (Python vs JS SDK) ===")
    lines.append(f"  JS specVersion: {js_spec}")
    lines.append(f"  Python specVersion: {py_spec}")
    if js_spec != py_spec:
        lines.append(
            "  ⚠ specVersion mismatch — bump Python's value when you "
            "catch up to JS's."
        )
    lines.append("")

    n_missing = sum(len(v) for v in actionable_missing.values())
    n_skip_missing = sum(len(v) for v in skipped_missing.values())
    n_drift = sum(len(v) for v in actionable_drift.values())
    n_skip_drift = sum(len(v) for v in skipped_drift.values())
    n_extra = sum(len(v) for v in extras.values())

    n_fail = n_missing + n_drift
    if n_fail == 0:
        lines.append(
            f"OK: no missing/drifted cases "
            f"(skipped-missing: {n_skip_missing}, skipped-drift: {n_skip_drift}, extras: {n_extra})"
        )
    else:
        lines.append(
            f"DRIFT: {n_missing} missing + {n_drift} body-drift "
            f"(skipped: {n_skip_missing} missing, {n_skip_drift} drift; "
            f"{n_extra} Python extras)"
        )
    lines.append("")

    def _section(title: str, data: Dict[str, List[str]]) -> None:
        if not any(data.values()):
            return
        lines.append(f"--- {title} ---")
        for key, names in data.items():
            if not names:
                continue
            lines.append(f"  [{key}] ({len(names)})")
            for n in names:
                lines.append(f"    - {n}")
        lines.append("")

    _section("Missing in Python (FAILS CI)", actionable_missing)
    _section("Body-drift: same name, different case body (FAILS CI)", actionable_drift)
    _section("Missing in Python — skipped via corpus_skiplist.json", skipped_missing)
    _section("Body-drift — skipped via corpus_skiplist.json", skipped_drift)
    _section("Extra in Python (informational; never fails)", extras)
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--js-source",
        default=os.environ.get("GB_JS_CASES_URL", DEFAULT_JS_URL),
        help="URL or local path to JS cases.json (default: JS SDK main branch)",
    )
    parser.add_argument(
        "--json", action="store_true", help="output machine-readable JSON instead of text"
    )
    args = parser.parse_args(argv)

    try:
        js_cases = _fetch_js_cases(args.js_source)
        py_cases = _load_local_cases()
        skip = _load_skiplist()
    except RuntimeError as e:
        print(f"corpus check infra error: {e}", file=sys.stderr)
        return 2

    actionable_missing, skipped_missing, extras, actionable_drift, skipped_drift = _diff(
        js_cases, py_cases, skip
    )
    js_spec, py_spec = _spec_versions(js_cases, py_cases)

    if args.json:
        print(
            json.dumps(
                {
                    "js_specVersion": js_spec,
                    "py_specVersion": py_spec,
                    "missing_actionable": actionable_missing,
                    "missing_skipped": skipped_missing,
                    "drift_actionable": actionable_drift,
                    "drift_skipped": skipped_drift,
                    "extras": extras,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            _format_report(
                js_spec, py_spec, actionable_missing, skipped_missing,
                extras, actionable_drift, skipped_drift,
            )
        )

    fail = any(actionable_missing.values()) or any(actionable_drift.values())
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
