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
Using `uv`, this will automatically create a `.venv`, download strict dependencies, and install the package from source in an editable path:
```bash
uv pip install -e ".[dev]"
```

3. **Install pre-commit hooks (recommended):**
We use [pre-commit](https://pre-commit.com/) to mirror the CI lint/format checks locally so that `ruff format --check` failures are caught at commit time, not at PR review:
```bash
uv pip install pre-commit
pre-commit install
```
Once installed, `ruff format` and `ruff check --fix` run automatically on every commit against `src/` and `tests/`. To run them manually across the whole tree (e.g. after rebasing): `pre-commit run --all-files`.

## Running Tests

`rctd-py` heavily relies on `pytest` to guarantee mathematical correctness and exact equivalency against the R `spacexr` reference implementation.

```bash
# Run the full test suite
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ -v --cov=rctd --cov-report=term-missing
```

## Linting & Formatting

We use `ruff`, an extremely fast Python linter and formatter, to ensure all files look unified and follow PEP8 best practices. We have a Github Action CI workflow that checks this on every PR!

```bash
# Automatically format all python files
uv run ruff format src/ tests/

# Automatically fix any linting errors like unused imports
uv run ruff check src/ tests/ --fix
```

## Submitting Pull Requests

Please make sure you create a new feature branch from `main`, push to your branch, and create a PR detailing exactly what your changes do. We will review it shortly.
