# Contributing to rctd-py

Thank you for your interest in contributing to `rctd-py`! We welcome bug reports, feature requests, and pull requests from everyone.

## Local Development Setup

We manage our dependencies and environments using [uv](https://docs.astral.sh/uv/), a lightning-fast Rust-based package manager.

1. **Clone the repository:**
```bash
git clone https://github.com/p-gueguen/rctd-py.git
cd rctd-py
```

2. **Install editable mode with development dependencies:**
Using `uv`, create a `.venv` first, then install the package from source in an editable path with the development dependencies:
```bash
uv venv
uv pip install -e ".[dev]"
```

3. **Install pre-commit hooks (recommended):**
We use [pre-commit](https://pre-commit.com/) to mirror the CI lint/format checks locally so that `ruff format --check` failures are caught at commit time, not at PR review:
```bash
pre-commit install   # pre-commit is part of the dev extra
```
Once installed, `ruff format` and `ruff check --fix` run automatically on every commit against the Python files in `src/`, `tests/` and `docs/`. To run them manually across the whole tree (e.g. after rebasing): `pre-commit run --all-files`.

## Running Tests

`rctd-py` relies on `pytest` to check mathematical correctness and agreement with the R `spacexr` reference implementation within set tolerances.

```bash
# Run the test suite (performance tests are excluded by default; add -m performance to run them)
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ -v --cov=rctd --cov-report=term-missing
```

## Linting & Formatting

We use `ruff`, an extremely fast Python linter and formatter, to ensure all files look unified and follow PEP8 best practices. We have a Github Action CI workflow that checks this on every PR!

```bash
# Automatically format all python files
uv run ruff format src/ tests/ docs/

# Automatically fix any linting errors like unused imports
uv run ruff check src/ tests/ docs/ --fix
```

## Submitting Pull Requests

Please make sure you create a new feature branch from `main`, push to your branch, and create a PR detailing exactly what your changes do. We will review it shortly.
