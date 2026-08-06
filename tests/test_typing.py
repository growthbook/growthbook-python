"""Typing regression tests: the public API's checker-time guarantees.

tests/typing/good_usage.py must type-check clean, and every line in
tests/typing/bad_usage.py tagged with an expect-error comment must produce an
error (with no errors on untagged lines). This pins the promises the SDK makes
to IDE/agent users: fallback-driven inference, experiment result inference,
callback signature checking, kwargs typo detection, and the public API surface.

Always runs under mypy. Also runs under pyright when a `pyright` executable is
available (the CI pyright job installs one; locally `npm i -g pyright`).
"""

import json
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, Set

import pytest

from growthbook.codegen import generate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPING_DIR = os.path.join(REPO_ROOT, "tests", "typing")
GOOD = os.path.join(TYPING_DIR, "good_usage.py")
BAD = os.path.join(TYPING_DIR, "bad_usage.py")

PYRIGHT = shutil.which("pyright")

EXPECT_TAG = "# expect-error"


def tagged_lines(path: str) -> Set[int]:
    with open(path, encoding="utf-8") as f:
        return {i for i, line in enumerate(f, 1) if EXPECT_TAG in line}


def mypy_error_lines(*paths: str, cwd: str = REPO_ROOT) -> Dict[str, Set[int]]:
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", *paths],
        capture_output=True,
        text=True,
        cwd=cwd,
        env={**os.environ, "MYPYPATH": REPO_ROOT},
    )
    errors: Dict[str, Set[int]] = {}
    for m in re.finditer(r"^(.+?):(\d+): error:", result.stdout, re.M):
        errors.setdefault(os.path.basename(m.group(1)), set()).add(int(m.group(2)))
    return errors


def pyright_error_lines(*paths: str) -> Dict[str, Set[int]]:
    assert PYRIGHT is not None
    result = subprocess.run(
        [PYRIGHT, "--outputjson", *paths],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    errors: Dict[str, Set[int]] = {}
    for diag in json.loads(result.stdout)["generalDiagnostics"]:
        if diag["severity"] == "error":
            name = os.path.basename(diag["file"])
            errors.setdefault(name, set()).add(diag["range"]["start"]["line"] + 1)
    return errors


CHECKERS = ["mypy"] + (["pyright"] if PYRIGHT else [])


def run_checker(checker: str, *paths: str) -> Dict[str, Set[int]]:
    return mypy_error_lines(*paths) if checker == "mypy" else pyright_error_lines(*paths)


@pytest.mark.parametrize("checker", CHECKERS)
class TestPublicAPITyping:
    def test_good_usage_is_clean_and_bad_usage_errors_exactly_on_tags(self, checker):
        errors = run_checker(checker, GOOD, BAD)
        assert errors.get("good_usage.py", set()) == set(), (
            f"{checker}: good_usage.py must be error-free"
        )
        expected = tagged_lines(BAD)
        actual = errors.get("bad_usage.py", set())
        assert actual == expected, (
            f"{checker}: bad_usage.py errors {sorted(actual)} != tagged lines {sorted(expected)}"
        )


class TestGeneratedClientTyping:
    """The generated typed client's strictness, verified under mypy."""

    @pytest.fixture()
    def generated_dir(self, tmp_path):
        with open(
            os.path.join(REPO_ROOT, "tests", "codegen", "sample_features.json"),
            encoding="utf-8",
        ) as f:
            payload = json.load(f)
        (tmp_path / "growthbook_features.py").write_text(generate(payload), encoding="utf-8")
        return tmp_path

    def test_generated_module_checks_clean(self, generated_dir):
        errors = mypy_error_lines(str(generated_dir / "growthbook_features.py"))
        assert errors == {}

    def test_typed_client_strictness(self, generated_dir):
        snippet = generated_dir / "typed_usage.py"
        snippet.write_text(
            "from growthbook_features import TypedGrowthBook\n"
            "\n"
            "tgb = TypedGrowthBook()\n"
            "ok: str = tgb.get_feature_value('banner_text', 'hi')\n"
            "ok2: bool = tgb.is_on('dark_mode')\n"
            "tgb.is_on('buton_color')  " + EXPECT_TAG + "\n"
            "tgb.get_feature_value('max_items', '12')  " + EXPECT_TAG + "\n"
            "bad: str = tgb.get_feature_value('donut_price', 1.0)  " + EXPECT_TAG + "\n",
            encoding="utf-8",
        )
        errors = mypy_error_lines(str(snippet))
        assert errors.get("typed_usage.py", set()) == tagged_lines(str(snippet))
