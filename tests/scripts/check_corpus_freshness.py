#!/usr/bin/env python3
"""Check Python's tests/cases.json against the JS SDK's cases.json.

The two corpora are maintained by hand; `specVersion` is a label, not a
contract. This script makes drift visible:

  - "missing"  — case present in JS, absent in Python. Treated as an
                 error (fail CI) so corpus catch-up is an active decision
                 rather than a silent omission. The only escape is the
                 skiplist (see corpus_skiplist.json) for cases Python
                 deliberately can't or shouldn't carry.
  - "extra"    — case present in Python, absent in JS. Reported as
                 informational only — Python carries documented
                 extensions (e.g., $notRegex regression cases) plus
                 locally-authored regressions. Extras NEVER fail CI.

Source-of-truth URL is configurable via --js-source or env GB_JS_CASES_URL.
Defaults to the JS SDK's main-branch raw URL.

Exit codes:
  0 — no drift, or all drift is in the skiplist
  1 — at least one case missing from Python that isn't in the skiplist
  2 — fetch / parse / IO error (treated as build infra failure, not drift)

Run locally:
  python3 tests/scripts/check_corpus_freshness.py
  python3 tests/scripts/check_corpus_freshness.py --js-source /path/to/local/cases.json
  GB_JS_CASES_URL=https://... python3 tests/scripts/check_corpus_freshness.py

In CI, this runs on every push as a separate step in the build workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
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


def _load_skiplist() -> Dict[str, Set[str]]:
    """Load skiplist. File format:

        {
          "missing": {
            "<top_level_key>": ["case name 1", "case name 2"]
          }
        }

    `missing` entries are case names Python deliberately doesn't carry —
    drift that won't fail CI. `extra` is reported but never fails so it's
    not configured here. The file is optional.
    """
    if not SKIPLIST.is_file():
        return {}
    try:
        data = json.loads(SKIPLIST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"skiplist invalid JSON: {e}") from e
    missing = data.get("missing", {}) or {}
    return {k: set(v) for k, v in missing.items()}


def _case_names(cases: list) -> List[str]:
    """First element of each case is the human-readable name."""
    out = []
    for c in cases:
        if isinstance(c, list) and c and isinstance(c[0], str):
            out.append(c[0])
    return out


def _diff(
    js_cases: dict, py_cases: dict, skip: Dict[str, Set[str]]
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """Return (actionable_missing, skipped_missing, extras) per key."""
    actionable: Dict[str, List[str]] = {}
    skipped: Dict[str, List[str]] = {}
    extras: Dict[str, List[str]] = {}

    for key in KEYS_TO_DIFF:
        js_list = js_cases.get(key, [])
        py_list = py_cases.get(key, [])
        if not isinstance(js_list, list) or not isinstance(py_list, list):
            continue
        js_names = _case_names(js_list)
        py_names_set = set(_case_names(py_list))
        js_names_set = set(js_names)

        # Order missing by JS's order so the report reads naturally.
        missing = [n for n in js_names if n not in py_names_set]
        extra = [n for n in _case_names(py_list) if n not in js_names_set]

        key_skip = skip.get(key, set())
        actionable[key] = [n for n in missing if n not in key_skip]
        skipped[key] = [n for n in missing if n in key_skip]
        extras[key] = extra

    return actionable, skipped, extras


def _spec_versions(js_cases: dict, py_cases: dict) -> Tuple[str, str]:
    return (
        str(js_cases.get("specVersion", "<unset>")),
        str(py_cases.get("specVersion", "<unset>")),
    )


def _format_report(
    js_spec: str,
    py_spec: str,
    actionable: Dict[str, List[str]],
    skipped: Dict[str, List[str]],
    extras: Dict[str, List[str]],
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

    total_actionable = sum(len(v) for v in actionable.values())
    total_skipped = sum(len(v) for v in skipped.values())
    total_extra = sum(len(v) for v in extras.values())

    if total_actionable == 0:
        lines.append(f"OK: no missing cases (skipped: {total_skipped}, extras: {total_extra})")
    else:
        lines.append(
            f"DRIFT: {total_actionable} case(s) in JS but missing from Python "
            f"(plus {total_skipped} on skiplist, {total_extra} Python extras)"
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

    _section("Missing in Python (FAILS CI)", actionable)
    _section("Missing in Python but skipped via corpus_skiplist.json", skipped)
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

    actionable, skipped, extras = _diff(js_cases, py_cases, skip)
    js_spec, py_spec = _spec_versions(js_cases, py_cases)

    if args.json:
        print(
            json.dumps(
                {
                    "js_specVersion": js_spec,
                    "py_specVersion": py_spec,
                    "missing_actionable": actionable,
                    "missing_skipped": skipped,
                    "extras": extras,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(_format_report(js_spec, py_spec, actionable, skipped, extras))

    return 1 if any(actionable.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
