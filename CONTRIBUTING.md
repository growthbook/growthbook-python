# Contributing Guide

We welcome all contributions!

## Type checking

We use MyPy for type checking.

```bash
make type-check
```

## Tests

Run the test suite with

```bash
make test
```

## Linting

Lint the code with flake8

```bash
make lint
```

## Releasing

1. Bump the version in `setup.cfg` and `setup.py`
2. Merge code to `main`
3. Create a new release on GitHub with your version as the tag (e.g. `v0.2.0`)
4. Run `make dist` to create a distribution file
5. Run `make release` to upload to pypi
