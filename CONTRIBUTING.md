# Contributing Guide

We welcome all contributions!

This repo is the official GrowthBook SDK for Python — a client library for
evaluating feature flags and running experiments, with synchronous and async
clients.

## Requirements

- **Python 3.9+**, as declared in `pyproject.toml` and `setup.py`. CI tests
  3.9–3.13; the examples below use 3.13. Keep library code compatible with 3.9.
- **Make** for the convenience commands below. You can also invoke `pytest`,
  `mypy`, and `flake8` directly from the activated environment.
- **Node.js and npm** only for the optional local Pyright checks.

`uv` is optional. There is no dependency lockfile; CI installs from
`requirements.txt` and `requirements_dev.txt`, then installs the package.

## Getting started

Fork the repo, or clone directly if you have write access:

```bash
git clone git@github.com:growthbook/growthbook-python.git
cd growthbook-python
```

Create a virtual environment and install the package in editable mode with
the development dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -r requirements.txt -r requirements_dev.txt
make test
make type-check
```

If you use `uv`, the equivalent setup is:

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -e . -r requirements.txt -r requirements_dev.txt
```

Activate `.venv` in each new terminal with `source .venv/bin/activate`. Configure
your editor to use `.venv/bin/python`. Changes to `growthbook/` are available
immediately without reinstalling the package.

The unit tests do not need GrowthBook credentials or a running server. Get a
passing test and type-check baseline before changing code.

## Writing code

### Layout

| Path | Contents |
| --- | --- |
| `growthbook/core.py` | Shared feature and experiment evaluation, targeting conditions, hashing, and bucketing |
| `growthbook/growthbook.py` | The sync `GrowthBook` client, feature repository, caching, SSE, and decryption |
| `growthbook/growthbook_client.py` | The async `GrowthBookClient`, shared feature repository, refreshes, and per-user evaluation |
| `growthbook/common_types.py` | Configuration, user and evaluation contexts, results, callback protocols, and sticky bucket interfaces |
| `growthbook/plugins/` | Tracking and request-context plugins |
| `growthbook/codegen.py` | Generator for typed clients built from feature definitions |
| `growthbook/__init__.py`, `growthbook/py.typed` | Public exports, package version, and the typed-package marker |
| `tests/test_*.py`, `tests/conftest.py` | Tests and shared fixtures |
| `tests/cases.json` | Cross-SDK conformance corpus, including Python-local cases |
| `tests/typing/`, `tests/codegen/` | Typing regression cases and code-generation fixtures |
| `tests/scripts/` | Corpus checks, benchmarks, and integration verification scripts |

### Public API and typing

`growthbook/__init__.py` lists the package's public exports in `__all__`.
Callers also use the documented client methods and configuration types. Discuss
changes to signatures, defaults, or return types in an issue before implementing
them; type-checking behavior is part of the API too.

The package ships inline type annotations through `py.typed`. Add annotations
to new library functions and keep public types importable and resolvable at
runtime. When changing typing behavior, update the positive and negative cases
in `tests/typing/` and run both MyPy and Pyright — see
[Type checking](#type-checking).

The Python 3.9 floor applies even if your local interpreter is newer. Use
`typing_extensions` where a newer typing feature needs a backport. Changes to
the minimum Python version should be discussed explicitly.

### SDK parity

GrowthBook SDKs should make identical decisions given identical inputs. The
[JavaScript SDK](https://github.com/growthbook/growthbook/tree/main/packages/sdk-js)
is the reference implementation. When changing evaluation, compare the
corresponding JS behavior, especially hashing, missing values, numeric types,
and targeting comparisons where Python semantics differ.

Describe deliberate divergences in the PR. A change to which users match a
targeting rule can change live audiences when an application upgrades.

### Sync, async, and shared state

`GrowthBook` stores user attributes on the instance; web applications normally
create one per request and call `destroy()` afterward. `GrowthBookClient` is
reused across requests and receives a separate `UserContext` for each user.
Call `await client.close()` at shutdown, or use its async context manager.

Both clients use `core.py`, but have separate loading and lifecycle code.
Cover both clients when changing shared behavior. In async code, avoid blocking
the event loop and test cancellation and cleanup of any background work you add.

Feature repositories and caches can be shared. In particular,
`EnhancedFeatureRepository` instances are keyed by class, API host, and client
key. Keep per-user state isolated, and test with multiple users and clients when
changing cache keys, tracking, or sticky bucketing. The fixtures in
`tests/conftest.py` clean up async tasks and repository singletons between tests;
they do not replace testing the client's own cleanup.

Use the existing `logging` loggers for diagnostics. Library code should not
configure the application's root logger or print to stdout.

## Testing

```bash
make test                                   # full suite (pytest)
pytest tests/test_remote_eval.py             # one file
pytest tests/test_growthbook.py::test_hash    # one parametrized test
pytest -k sticky -x                          # matching tests, stop on first failure
make coverage                               # terminal report and HTML, opens a browser
```

For coverage without opening a browser:

```bash
coverage run --source growthbook -m pytest
coverage report -m
coverage html
```

Tests use `pytest`, `pytest-mock`, and `pytest-asyncio`. Asyncio runs in strict
mode: mark async tests with `@pytest.mark.asyncio` and use
`@pytest_asyncio.fixture` for async fixtures. Mock external HTTP calls in unit
tests, and close clients and sessions created by the test.

A bug fix should include a regression test that fails before the fix. For
generator changes, review `tests/codegen/expected_output.py` alongside the
generator and run both `tests/test_codegen.py` and `tests/test_typing.py`.

### Conformance corpus

`tests/cases.json` contains cases mirrored from the
[JS corpus](https://github.com/growthbook/growthbook/blob/main/packages/sdk-js/test/cases.json)
plus Python-local extensions and regressions. `tests/test_growthbook.py` loads
the corpus into parametrized tests. Investigate a failing shared case before
changing its expected result.

CI compares the corpus with live JS `main` on every push:

```bash
python tests/scripts/check_corpus_freshness.py
```

The command needs network access. To compare with a local JS checkout instead:

```bash
python tests/scripts/check_corpus_freshness.py \
  --js-source /path/to/growthbook/packages/sdk-js/test/cases.json
```

Missing cases and changed bodies of shared cases fail the check unless listed
in `tests/scripts/corpus_skiplist.json`. Python-only cases are reported but do
not fail it. Add a reason under `_reasons` for each intentional skip. The check
compares selected corpus sections; it does not replace running the tests.

Preserve Python-local cases when syncing from upstream. For a parity bug that
affects other SDKs, propose a regression case in the JS corpus as well.

### Benchmarks and integration checks

Run benchmarks from the repository root with the virtual environment active:

```bash
python tests/scripts/benchmark_eval_overhead.py 100000
python tests/scripts/benchmark_async_client.py
```

The first measures sync evaluation overhead; the second measures async
throughput, latency, and event-loop lag with simulated sticky bucket services.
Neither needs external services. Include before/after results in the PR when
changing evaluation performance or concurrency behavior, using the same Python
version, machine, and workload for both runs.

For remote-evaluation checks against an in-process test proxy:

```bash
python tests/scripts/verify_remote_eval.py
```

This starts a localhost server. Real-proxy checks are separate and need Docker
and GrowthBook credentials; see [the verification scripts](tests/scripts/README.md)
for setup and teardown.

## Code quality

### Type checking

MyPy is pinned in `requirements_dev.txt` and configured in `pyproject.toml`:

```bash
make type-check
pytest tests/test_typing.py -q
```

The typing regression suite checks valid usage and verifies that invalid calls
fail on the lines marked `# expect-error`. It always runs MyPy and also runs
Pyright when a `pyright` executable is on `PATH`.

To reproduce CI's Pyright checks locally, install the version pinned in
`.github/workflows/main.yml`:

```bash
npm install -g pyright@1.1.403
pyright
pytest tests/test_typing.py -q
```

Keep the virtual environment active so Pyright can resolve the installed
dependencies. CI runs Pyright on Python 3.11; both checker configurations target
Python 3.9 language compatibility.

### Linting and formatting

CI's blocking lint check selects syntax and correctness errors. Exclude the
local virtual environment when running it from the repository root:

```bash
flake8 . --extend-exclude=.venv --count --select=E9,F63,F7,F82 --show-source --statistics
```

CI also reports broader style warnings without failing the build:

```bash
flake8 . --extend-exclude=.venv --count --exit-zero --max-complexity=25 --max-line-length=127 --statistics
```

The existing Make target is different:

```bash
make lint
```

It checks only `growthbook/growthbook.py`, uses a 150-character line limit, and
fails on existing style and unused-import issues. A failure there does not by
itself mean the environment is broken. Check whether a finding is introduced
by your change.

There is no automatic formatter configured. Follow `.editorconfig` and the
surrounding code, use docstrings for declarations that need documentation, and
keep formatting changes limited to the code you touch.

## Opening pull requests

Open an issue first for API changes, new configuration, or behavior changes so
the design can be discussed. For a typo or an obvious fix, open a PR directly.

1. Branch from `main` and keep the diff focused.
2. Add regression coverage and update `README.md` for changes to documented
   behavior or usage.
3. Run `make test`, `make type-check`, and the blocking lint command above.
   Run Pyright for typing changes, corpus freshness for evaluation or fixture
   changes, and benchmarks for performance-sensitive changes.
4. Use descriptive commit subjects with conventional-commit prefixes such as
   `fix:`, `feat:`, `docs:`, or `test:`. Release Please uses commit history to
   prepare releases; make the eventual merge or squash subject describe the
   change accurately.
5. Open the PR against `main`. Explain the problem, the resulting behavior,
   and how you tested it. Call out compatibility and SDK parity implications.

The build workflow (`.github/workflows/main.yml`) runs on pushes. It tests and
type-checks on Python 3.9–3.13, runs the two lint passes, checks Pyright and its
typing regressions on Python 3.11, and checks corpus freshness against JS
`main`. A local test pass on one interpreter does not cover the full matrix.

Push follow-up commits rather than force-pushing where possible so review
comments stay attached to the code they discuss.

## Releasing

Maintainers handle releases through `.github/workflows/release-please.yml`:

1. Pushes to `main` run Release Please, which prepares a release PR with version
   and changelog updates.
2. Merging the release PR lets Release Please create the release and tag.
3. When a release is created, the publishing job runs tests on Python 3.11,
   builds the source distribution and wheel with `python -m build`, and uploads
   them to PyPI.

`setup.py` reads the package version from `growthbook/__init__.py`.
`release-please-config.json` and `.release-please-manifest.json` configure and
track releases. Contributors normally do not need to edit version numbers or
`CHANGELOG.md`; include enough detail in the PR for a useful changelog entry.

`make dist`, `make release`, and the bumpversion files are legacy tooling.
In particular, `make release` uploads directly to PyPI; it is not a local
verification command or the normal release process.

## Getting help

- [Slack community](https://slack.growthbook.io?ref=contributing)
- [Open an issue](https://github.com/growthbook/growthbook-python/issues)
- [Python SDK docs](https://docs.growthbook.io/lib/python)

Found a security vulnerability? Email security@growthbook.io instead of filing
a public issue. See the [security policy](https://github.com/growthbook/growthbook/blob/main/SECURITY.md).

Contributors are expected to follow the [Code of Conduct](https://github.com/growthbook/growthbook/blob/main/CODE_OF_CONDUCT.md).
