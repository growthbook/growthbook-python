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


class TestRuntimeIntrospection:
    """Annotations must stay resolvable at runtime, not just for checkers."""

    def test_options_type_hints_resolve(self):
        # Regression: PluginLike used to be TYPE_CHECKING-only, so dataclass
        # introspection of Options (pydantic, dacite, ...) raised NameError.
        import typing

        from growthbook import Options

        hints = typing.get_type_hints(Options)
        assert "tracking_plugins" in hints

    def test_plugin_types_importable_from_root(self):
        from growthbook import GrowthBookPlugin, PluginLike  # noqa: F401


SINGLE_FEATURE_PAYLOAD = {"features": {"only_one": {"defaultValue": True}}}


def _multi_feature_payload():
    with open(
        os.path.join(REPO_ROOT, "tests", "codegen", "sample_features.json"),
        encoding="utf-8",
    ) as f:
        return json.load(f)


class TestGeneratedClientTyping:
    """The generated typed client's strictness, verified under mypy."""

    @pytest.fixture()
    def generated_dir(self, tmp_path):
        (tmp_path / "growthbook_features.py").write_text(
            generate(_multi_feature_payload()), encoding="utf-8"
        )
        return tmp_path

    @pytest.mark.parametrize("checker", CHECKERS)
    @pytest.mark.parametrize(
        "payload_name", ["multi_feature", "single_feature"]
    )
    def test_generated_module_checks_clean(self, tmp_path, checker, payload_name):
        # A one-feature payload is the common small-project case and uses the
        # plain-method (non-overload) code path — both must check clean.
        payload = (
            SINGLE_FEATURE_PAYLOAD
            if payload_name == "single_feature"
            else _multi_feature_payload()
        )
        path = tmp_path / "growthbook_features.py"
        path.write_text(generate(payload), encoding="utf-8")
        errors = run_checker(checker, str(path))
        assert errors == {}, f"{checker}/{payload_name}: {errors}"

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

    def test_typed_async_client_strictness(self, generated_dir):
        # Also guards the generated `# type: ignore[override]` lines against
        # silent signature drift: if TypedGrowthBookClient's overloads stop
        # matching the real async signature, these lines change verdicts.
        snippet = generated_dir / "typed_async_usage.py"
        snippet.write_text(
            "from growthbook import UserContext\n"
            "from growthbook_features import TypedGrowthBookClient\n"
            "\n"
            "tgb = TypedGrowthBookClient()\n"
            "ctx = UserContext(attributes={'id': '1'})\n"
            "\n"
            "async def _usage() -> None:\n"
            "    ok: str = await tgb.get_feature_value('banner_text', 'hi', ctx)\n"
            "    ok2: bool = await tgb.is_on('dark_mode', ctx)\n"
            "    await tgb.is_on('buton_color', ctx)  " + EXPECT_TAG + "\n"
            "    await tgb.get_feature_value('max_items', '12', ctx)  " + EXPECT_TAG + "\n"
            "    bad: str = await tgb.get_feature_value('banner_text', 'hi', ctx)  # ok line\n"
            "    await tgb.get_feature_value('banner_text', 'hi')  " + EXPECT_TAG + "\n",
            encoding="utf-8",
        )
        errors = mypy_error_lines(str(snippet))
        assert errors.get("typed_async_usage.py", set()) == tagged_lines(str(snippet))

    def test_single_feature_client_strictness(self, tmp_path):
        """The non-overload code path must enforce the same guarantees."""
        (tmp_path / "growthbook_features.py").write_text(
            generate(SINGLE_FEATURE_PAYLOAD), encoding="utf-8"
        )
        snippet = tmp_path / "typed_usage.py"
        snippet.write_text(
            "from growthbook_features import TypedGrowthBook\n"
            "\n"
            "tgb = TypedGrowthBook()\n"
            "ok: bool = tgb.get_feature_value('only_one', True)\n"
            "tgb.is_on('typo_key')  " + EXPECT_TAG + "\n"
            "tgb.get_feature_value('only_one', 'wrong')  " + EXPECT_TAG + "\n",
            encoding="utf-8",
        )
        errors = mypy_error_lines(str(snippet))
        assert errors.get("typed_usage.py", set()) == tagged_lines(str(snippet))
