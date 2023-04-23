# Contributing Guide

We welcome all contributions!

## Type checking

We use MyPy for type checking.

```bash
mypy growthbook.py --implicit-optional
```

## Tests

Run the test suite with `pytest`

## Linting

Lint the code with flake8 (note: we only use a subset of linting rules)

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Releasing

1. Bump the version in `setup.cfg` and `setup.py`
2. Merge code to `main`
3. Create a new release on GitHub with your version as the tag (e.g. `v0.2.0`)
4. Run `make dist` to create a distribution file
5. Run `make release` to upload to pypi