# rctd-py Local Development

## GitHub CLI

`gh` is needed for issue/PR/release management against this repo. Install
via `conda install -c conda-forge gh` (or your distro package manager) and
authenticate once with `gh auth login`. If you keep `gh` in a dedicated
conda env, remember to activate it before running release / CI commands.

## Pre-commit hooks

`.pre-commit-config.yaml` mirrors the CI ruff checks, and `.git/hooks/pre-commit` is
already generated in this clone. The hook calls the venv's Python, so if `pre-commit`
itself is not installed there **every commit fails** with the cryptic
`No module named pre_commit` and nothing else — never `--no-verify` past it, that also
skips the ruff gate CI will then fail on.

`pre-commit` is in the `dev` extra for this reason. Do **not** rely on
`uv pip install pre-commit`: any later `uv run` re-syncs the venv from the lockfile and
prunes it again, so the hook breaks a second time a few commits later. Verify with
`uv run pre-commit run --all-files`. After a fresh clone also run `pre-commit install`,
per the comment at the top of the config.

## Architecture

GPU-accelerated Python reimplementation of spacexr RCTD for spatial transcriptomics deconvolution.

### Key source files

| File | Purpose |
|------|---------|
| `src/rctd/_irwls.py` | Core IRWLS solver — hot path, ~60% of runtime |
| `src/rctd/_likelihood.py` | Poisson-Lognormal likelihood with cubic spline interpolation (`calc_q_all`) |
| `src/rctd/_simplex.py` | Simplex projection for weight constraints |
| `src/rctd/_full.py` | Full-mode pipeline (all cell types per pixel) |
| `src/rctd/_doublet.py` | Doublet-mode pipeline (top 2 cell types per pixel) |
| `src/rctd/_multi.py` | Multi-mode pipeline (variable number of types) |
| `src/rctd/_sigma.py` | Sigma estimation (noise parameter) |
| `src/rctd/_rctd.py` | Top-level `run_rctd()` entry point |
| `src/rctd/cli.py` | Click CLI: `rctd info`, `rctd validate`, `rctd run` |
| `src/rctd/_types.py` | Config, result types, `auto_batch_size`, `resolve_device` |

### Solver pipeline

1. **Sigma estimation** (`_sigma.py`): Find optimal noise parameter
2. **Full mode** (`_full.py`): Unconstrained IRWLS → all K cell type weights per pixel
3. **Doublet/Multi mode** (`_doublet.py`, `_multi.py`): Select top types → constrained IRWLS

The IRWLS solver (`solve_irwls_batch_shared`) is the innermost hot loop:
- Iteratively solves weighted least squares with Poisson-Lognormal likelihood
- Each iteration: predict → derivatives → Hessian → PSD projection → box-QP → update
- Active pixel compaction skips converged pixels

### GPU optimization layers

Two optimization layers accelerate the hot path beyond basic `torch.compile`:

1. **Analytical K=2 fast paths** (`_irwls.py`): For doublet-mode pairwise fits (K=2), closed-form solutions replace iterative solvers:
   - `_psd_2x2`: Analytical eigendecomposition (avoids cuSOLVER)
   - `_solve_box_qp_2`: Cramer's rule + clamping (avoids 50-sweep Gauss-Seidel)
   - Dispatched automatically by `_psd_batch` and `_solve_box_qp_batch` when K==2

2. **Auto batch sizing** (`_types.py:auto_batch_size`): Calculates optimal GPU batch size from available VRAM using per-pixel memory footprint estimation. Used by `run_rctd(batch_size="auto")` (default).

**Why no custom Triton kernel**: A hand-written Triton kernel for `calc_q_all` was attempted but causes float64 precision divergence in the grid index computation (`sqrt → floor → m`). Even with dtype-preserving casts, Triton's `tl.math.sqrt` produces different rounding than PyTorch's `torch.sqrt` at float64 boundaries, shifting spline table lookups and changing results. `torch.compile` already generates fused Inductor/Triton kernels from `_calc_q_all_impl` that are both fast and hash-identical to the eager implementation.

## CLI

Entry point `rctd` is registered in `pyproject.toml` via `[project.scripts]`. Three subcommands:

- `rctd info` — environment info (versions, GPU detection), `--json` for machine-readable
- `rctd validate` — pre-flight checks on inputs (fast, no GPU)
- `rctd run` — full deconvolution pipeline, writes annotated h5ad output

Both `validate` and `run` read their two positional arguments through
`_read_adata()` (`cli.py`), which accepts a `.h5ad` file, an AnnData Zarr store,
or a SpatialData Zarr store (a directory with a `tables/` group; its single
table is used, or the one named by `--table` when there are several - spatial
argument only, and an error anywhere it cannot apply). It also casts pandas nullable string dtypes back to object —
SpatialData writes `obs`/`var` names as `nullable-string-array`, which anndata
reads as a `StringArray` and then refuses to write back to `.h5ad`, so without
that cast `run` dies at the final write with the compute already done.

The `run` command uses `RCTD` class directly (not `run_rctd()`), manages its own data loading, and writes results back into a copy of the spatial AnnData with `_write_results_to_adata()`. Progress goes to stderr when `--json` is set.

Tests in `tests/test_cli.py` — slow tests (marked `@pytest.mark.slow`) run actual deconvolution on synthetic data via `conftest._make_synthetic_reference` / `_make_synthetic_spatial`.

## Q-matrices

`q_matrices.npz` (404 MB: `X_vals` plus `Q_10`..`Q_200`, one 1003x439 float64 table per
sigma) is excluded from the wheel and sdist in `pyproject.toml` and lives as a v0.1.1
release asset. It cannot be vendored: it does not compress at all (`savez_compressed`
gives back the same 404 MB) and PyPI caps one file at 100 MB. `choose_sigma` searches the
whole sigma grid, so a run needs all 126 tables, not one.

Lookup order in `load_cached_q_matrices()`: explicit `data_dir` -> `$RCTD_Q_MATRICES` ->
package `data/` -> `~/.cache/rctd` -> download. Set `RCTD_Q_MATRICES` on a cluster so one
read-only copy serves everyone. CI provisions it with `gh release download` into
`~/.cache/rctd` behind an `actions/cache` key, so the test job never pulls 404 MB twice.

Note `RCTD(sigma_override=...)` still loads all 404 MB and splines all 126 tables before
discarding everything but one - unnecessary, but not currently optimised.

## Testing

```bash
uv run pytest tests/ -v
```

Tests use `torch.compile(dynamic=True)` which has a ~60s JIT warmup on first run.

### Test markers

- No marker: fast unit tests (~90s total including JIT warmup)
- `@pytest.mark.slow`: CLI integration tests that run full RCTD pipeline (~30s each)
- `@pytest.mark.performance`: benchmarking tests (excluded by default via `addopts`)

### CLI test notes

- `CliRunner` mixes stderr into stdout by default (`mix_stderr` was removed in Click 8.2+)
- When testing `--json` output, extract the JSON block from mixed output (`output.index("{")` to `output.rindex("}") + 1`) rather than parsing `result.output` directly
- CLI tests import synthetic data helpers via `from conftest import ...` (not `from tests.conftest` — `tests/` is not a package)

### Known tolerance notes

- `test_batch_matches_single` uses `atol=5e-5` — batch vs single-pixel IRWLS can differ slightly due to floating-point convergence order

### Compile / JIT dispatch

Before touching `_solve_box_qp_batch` or `calc_q_all` dispatch, read `.claude/docs/compile-jit.md` (why `compile=False` takes the TorchScript path, `_USE_COMPILE` states).

## R spacexr Concordance

**The remaining R/py gap is sigma, and R does not reproduce itself.** Both sides pick sigma from a
random pixel sample and spacexr sets no seed: two R runs that land on different sigmas agree on
~99.3% of spot classes, the same level as rctd-py vs R; at the same sigma the calls match. Compare
like for like with `run_rctd(..., sigma_override=round(100 * rctd@internal_vars$sigma))`.

**Measure `first_type` too, not only dominant type or spot class.** The 2026-10 audit found four
per-pixel differences on `main` that both of those metrics hid: singlets relabelled to the best
singlet-score type, cell types sorted by code point instead of R's factor order, `N_fit` 100 vs
R's 1000, and an inexact 50-sweep QP. Fixes and a pixel-level parity test (R's own per-pixel
inputs in, identical calls out) are on branch `fix/r-parity`.

Comparison hygiene:

1. **Same reference cells**: a different reference (full Seurat object vs pre-downsampled spacexr
   Reference) drops agreement to ~95%.
2. **Weight normalization**: spacexr stores `normalize_weights()` output (sum=1, clipped [0,1]);
   rctd-py stores raw full-mode weights in `obsm["rctd_weights"]`. Normalize before comparing.
3. **Cell type naming**: R spacexr/zarr may normalize names (e.g. `L2/3 IT` -> `L2_3 IT`).
4. `_longdouble_sum` in `_normalize.py` uses 80-bit sums for `bulk_Y`/`bulk_nUMI`, matching R's
   long double accumulation on x86-64.

## Tutorial

The tutorial is a marimo notebook at `examples/tutorial.py`. The docs ship it as an executed
notebook, `docs/tutorials/slideseq_cerebellum.ipynb`; regenerate it on a GPU host with
`python docs/tutorials/export.py` (see its docstring).

Before editing `examples/tutorial.py` or a tutorial claim, read `.claude/docs/tutorial.md` (marimo export gotchas, doc-vs-code drift audits, spot-class encoding).

## CI

CI runs `ruff check src/ tests/` AND `ruff format --check src/ tests/`. Both must pass. Always run both locally before pushing:

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

CI failing on a PyTorch API change or a runner g++ ICE: `.claude/docs/ci-notes.md`.

### Publish workflow runs on tag, not on push

`publish.yml` is gated on tag pushes; `ci.yml` is gated on push to main
and PRs. A release tag therefore does not run the test workflow on the
exact tagged commit. Test breakages introduced in release commits will
only surface on the **first PR after the tag** (this happened with PR
#17 surfacing the stale `test_compile_false_uses_eager` test).

Before tagging, manually run the full matrix on the release branch.
`publish.yml` already auto-creates the GitHub Release entity from
`CHANGELOG.md` (PR #16 / commit `b80ae88`).

## Answering issues

Recurring user confusions (full-mode `result.weights` in doublet/multi results, "R weights are on the simplex"): `.claude/docs/issue-patterns.md`.

## R spacexr source

Where each spacexr function lives and how to read the installed R source: `.claude/docs/r-spacexr-source.md`.

## Benchmarking

Benchmark scripts, the autoresearch loop, profiling hot spots: `.claude/docs/benchmarking.md`.
