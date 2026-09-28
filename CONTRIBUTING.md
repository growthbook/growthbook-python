# Contributing

## Local setup

Python 3.9+ is supported; CI tests 3.9–3.13. From a cloned checkout:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -r requirements.txt -r requirements_dev.txt
```

Alternatively, use `uv venv --python 3.13 .venv` and `uv pip install` with the
same dependencies. Configure your editor to use `.venv/bin/python`. Unit tests
need no GrowthBook credentials or running server.

## Checks

With the virtual environment active, run from the repository root:

```bash
pytest -q
mypy
flake8 . --extend-exclude=.venv --count --select=E9,F63,F7,F82 --show-source --statistics
```

For typing changes, also run Pyright with the version pinned in CI:

```bash
npm install -g pyright@1.1.403
pyright
pytest tests/test_typing.py -q
```

`make test` and `make type-check` wrap pytest and MyPy. `make coverage` produces
an HTML report. `make lint` uses older rules and fails on existing style issues;
the flake8 command above matches CI's blocking check.

## Working on the SDK

- `growthbook/core.py` contains evaluation shared by the sync `GrowthBook`
  (`growthbook.py`) and async `GrowthBookClient` (`growthbook_client.py`). Test
  both clients, including loading and refresh when payload handling changes.
- Keep library code compatible with Python 3.9 and add type annotations.
  Public API changes should include typing regressions in `tests/typing/`.
- Mock external HTTP calls and close clients created in tests. Async tests use
  `@pytest.mark.asyncio`; async fixtures use `@pytest_asyncio.fixture`.
- Follow the [JavaScript reference SDK](https://github.com/growthbook/growthbook/tree/main/packages/sdk-js)
  for evaluation behavior. Preserve Python-local cases when syncing
  `tests/cases.json` with the shared corpus.

For evaluation or corpus changes, check parity with JS `main` (requires network):

```bash
python tests/scripts/check_corpus_freshness.py
```

Use `--js-source /path/to/cases.json` for a local reference. Missing or changed
shared cases fail unless listed with a reason in `tests/scripts/corpus_skiplist.json`.

For performance changes, compare both checkouts using the same interpreter and
machine. Benchmark commands and integration checks are in
[tests/scripts/README.md](tests/scripts/README.md).

## Pull requests

Discuss public API changes in an issue first. Branch from `main`, include
regression tests and relevant usage docs, and run the checks above. Describe
behavior changes, compatibility concerns, and validation in the PR.

Use conventional commit subjects (`feat:`, `fix:`, etc.); Release Please uses
them to determine versions and changelog entries. Keep the merge or squash
subject consistent with the change.

## Releases

The [Release Please workflow](.github/workflows/release-please.yml) creates a
release PR after changes reach `main`. Merging that PR creates the tag and
GitHub release, then tests, builds, and publishes the package to PyPI.

`setup.py` reads the version from `growthbook/__init__.py`. Contributors normally
leave version and changelog updates to Release Please. The legacy `make release`
command uploads directly to PyPI; it is not a verification command.

## Help

- [SDK docs](https://docs.growthbook.io/lib/python)
- [Issues](https://github.com/growthbook/growthbook-python/issues)
- [Slack](https://slack.growthbook.io?ref=contributing)

Report vulnerabilities to security@growthbook.io. Follow the
[Code of Conduct](https://github.com/growthbook/growthbook/blob/main/CODE_OF_CONDUCT.md).
