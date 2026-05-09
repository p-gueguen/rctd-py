# rctd-py Local Development

## GitHub CLI

`gh` is needed for issue/PR/release management against this repo. Install
via `conda install -c conda-forge gh` (or your distro package manager) and
authenticate once with `gh auth login`. If you keep `gh` in a dedicated
conda env, remember to activate it before running release / CI commands.

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
- `rctd validate` — pre-flight checks on h5ad inputs (fast, no GPU)
- `rctd run` — full deconvolution pipeline, writes annotated h5ad output

The `run` command uses `RCTD` class directly (not `run_rctd()`), manages its own data loading, and writes results back into a copy of the spatial AnnData with `_write_results_to_adata()`. Progress goes to stderr when `--json` is set.

Tests in `tests/test_cli.py` — slow tests (marked `@pytest.mark.slow`) run actual deconvolution on synthetic data via `conftest._make_synthetic_reference` / `_make_synthetic_spatial`.

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

### Compile / JIT dispatch dynamics

`_solve_box_qp_batch` (`_irwls.py:466-515`) has three paths:

| `_USE_COMPILE` state | Path | Has fallback? |
|---|---|---|
| `False` (or `_solve_box_qp_batch_compiled is None`) | `_solve_box_qp_batch_adaptive_jit` (TorchScript JIT) | n/a |
| `None` (first call) | try compiled, except RuntimeError → fall back to eager | yes |
| `True` (after first success) | `_solve_box_qp_batch_compiled` directly | **no** |

**Footgun:** the lazy try/except only protects the *first* call. If a later
recompile fails (different shape, transient toolchain breakage), there is no
automatic recovery. For environments with unreliable torch.compile, set
`compile=False` explicitly via `RCTDConfig`.

**v0.3.2 behavior change:** the `compile=False` path used to return
`_solve_box_qp_batch_impl` (eager Python loop). It now returns
`_solve_box_qp_batch_adaptive_jit`. The two paths converge to ~2e-4
numerical agreement (different op ordering). When writing tests that
assert equivalence, compare against `_solve_box_qp_batch_adaptive_jit`,
not `_solve_box_qp_batch_impl`. See commit `c068109`.

**`calc_q_all`** (`_likelihood.py:320-353`) has the same lazy fallback
pattern but its `compile=False` path still routes to the eager
`_calc_q_all_impl` — not changed in v0.3.2.

## R spacexr Concordance

### Current: 99.8% dominant type agreement (matched reference)

The per-pixel IRWLS solver is **bit-identical** to R spacexr given the same `norm_profiles` and `sigma`. The remaining ~0.2% gap comes from `fit_bulk()` platform effect estimation producing slightly different normalized profiles.

### Key findings (2026-03-21, comparison against an internal R spacexr Xenium pipeline)

1. **Reference preparation matters**: Using a different reference (e.g. full Seurat object vs pre-downsampled spacexr Reference) drops agreement to ~95%. Always use the SAME reference cells for fair comparisons.
2. **Weight normalization**: spacexr stores `normalize_weights()` output (sum=1, clipped [0,1]). rctd-py stores raw full-mode weights in `obsm["rctd_weights"]`. **Always normalize rctd-py weights before comparing.**
3. **`_longdouble_sum` in `_normalize.py`**: Uses 80-bit extended precision for `bulk_Y`/`bulk_nUMI` sums, matching R's long double accumulation on x86-64. Zero performance impact.
4. **Cell type naming**: R spacexr/zarr may normalize names (e.g. `L2/3 IT` → `L2_3 IT`). Always normalize names before comparing.

### Concordance testing

See `docs/plans/100-percent-matching-roadmap.md` for the detailed investigation
plan and the test fixtures in `tests/fixtures/vignette/` for the canonical
R-vs-Python concordance comparison.

## Tutorial

The tutorial is a marimo notebook at `examples/tutorial.py`, exported to `examples/tutorial.html`.

### Marimo gotchas

**Figures in static HTML export**: `plt.show()` does NOT produce capturable output in marimo's static export. The figure must be the **last expression** in the cell (like a return value), and it must use a **non-underscore name** (underscore-prefixed variables are cell-private in marimo and won't be rendered):

```python
# WRONG — no output in static HTML
_fig, _ax = plt.subplots()
_ax.plot(x, y)
plt.show()
return

# CORRECT — figure rendered in static HTML
fig_plot, ax_plot = plt.subplots()
ax_plot.plot(x, y)
fig_plot  # last expression, non-underscore name
```

**Underscore-prefixed names are cell-private**: Functions/variables starting with `_` (e.g., `_fig`, `_detect_ct_col`) are NOT exported from a cell to other cells or to static HTML output.

**Deprecated matplotlib API**: Use `plt.colormaps.get_cmap('tab20').resampled(n)` instead of `plt.cm.get_cmap('tab20', n)` (deprecated in matplotlib 3.7+).

### Rendering

```bash
uv run marimo export html examples/tutorial.py -o examples/tutorial.html --no-include-code
```

`examples/tutorial.html` is gitignored (local artifact). The tracked,
GitHub-Pages-served copy is `docs/tutorial.html`. After re-rendering,
copy `examples/tutorial.html` → `docs/tutorial.html`. Re-render takes
~13 min (sigma estimation runs 3× because the tutorial calls `run_rctd`
for full / doublet / multi separately, each from scratch).

### Doc-vs-code drift audits

Tutorial doc strings have drifted from code multiple times (e.g. v0.3.x
shipped with three errors in `examples/tutorial.py`: a "weights sum to 1"
claim that was never true under `constrain=FALSE`, a `N_fit` default of
1000 vs actual 100, and a `doublet_mode_alpha` parameter that has never
existed in any version — `git log --all -S doublet_mode_alpha` shows it
only in tutorial commits).

After non-trivial changes to `RCTDConfig` defaults, mode behavior, or
weight normalization, run a verification script that executes the
tutorial workflow end-to-end and prints the empirical answer next to the
documented claim. Reference fix: commit `a05067f` (source) + `428da09`
(re-rendered HTML). Cross-check parameter names against `RCTDConfig` in
`_types.py` to catch hallucinated names early.

**Spot class encoding** is 0-indexed (unlike R spacexr which is 1-indexed):
```
0 = reject, 1 = singlet, 2 = doublet_certain, 3 = doublet_uncertain
```
Use `SPOT_CLASS_NAMES` from `rctd._types` (exported in public API) — never hardcode the mapping.

## CI

CI runs `ruff check src/ tests/` AND `ruff format --check src/ tests/`. Both must pass. Always run both locally before pushing:

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

### PyTorch API compatibility

PyTorch renamed `cuda.get_device_properties().total_mem` → `total_memory` in 2.10+. Use `getattr` fallback pattern (see `cli.py:59`). CI runs an older PyTorch than the GPU nodes — test both paths.

### GitHub Actions runner toolchain ICEs

The ubuntu-24.04 runner image periodically rolls forward to a g++ that
ICEs on Inductor's `vec512.h` C++ codegen:

```
internal compiler error: in gimple_duplicate_bb, at tree-cfg.cc:6545
```

This surfaces in `Run slow integration + R concordance tests` and is a
GCC bug we don't control. **Workaround:** pin slow-test fixtures that
exercise the box-QP solver to `compile=False` so they take the
TorchScript JIT path (no g++ involved). Reference: `tests/test_concordance.py`
fixture comment + commit `b2a206f`. Don't try to "fix" this by upgrading
actions — it's intermittent and tied to the runner image version.

### Publish workflow runs on tag, not on push

`publish.yml` is gated on tag pushes; `ci.yml` is gated on push to main
and PRs. A release tag therefore does not run the test workflow on the
exact tagged commit. Test breakages introduced in release commits will
only surface on the **first PR after the tag** (this happened with PR
#17 surfacing the stale `test_compile_false_uses_eager` test).

Before tagging, manually run the full matrix on the release branch.
`publish.yml` already auto-creates the GitHub Release entity from
`CHANGELOG.md` (PR #16 / commit `b80ae88`).

## Common User Confusion (issue tracker patterns)

### `result.weights` on `MultiResult` / `DoubletResult` is the **full-mode** matrix

External users (e.g. issue #18) consistently expect `MultiResult.weights`
to be the multi-mode answer. It's not — it's the full-mode candidate
matrix used internally for type selection (`_multi.py:173, 348`),
unnormalized.

The actual multi-mode answer is `sub_weights[n, :n_types[n]]` paired
with `cell_type_indices[n, :n_types[n]]` and `n_types[n]` — these are
normalized to sum=1. Same shape for doublet: `weights_doublet` is the
(N, 2) normalized two-type result; `weights` is the unnormalized
full-mode candidate matrix.

This is preserved for R spacexr compatibility but it's a footgun. When
extending the README "Output schema" table or doublet/multi tutorial
sections, call this out explicitly.

### "R weights are on the simplex" misconception

R `myRCTD@results$weights` is **also** unnormalized — same
`constrain=FALSE` NNLS output as ours. The R vignette explicitly calls
`normalize_weights(results$weights)` before stuffing into the Seurat
Assay. Users reading R Section 6 (about the Assay) easily miss the
normalize step earlier and assume the raw `results$weights` is on the
simplex.

When responding to such questions, link to the R source line that does
the normalize — it's the most convincing evidence. See R spacexr
reference table below.

## R spacexr Source Reference

For citations and cross-checks (avoid re-grepping each time):

| Topic | R location |
|-------|------------|
| IRWLS damping (alpha=0.3 → our `step_size`) | [`R/IRWLS.R:95`](https://github.com/dmcable/spacexr/blob/master/R/IRWLS.R#L95) |
| Multi candidate threshold (0.01 → our `WEIGHT_THRESHOLD`) | [`R/RCTD_helper.R:76, 153`](https://github.com/dmcable/spacexr/blob/master/R/RCTD_helper.R) |
| `normalize_weights` definition | [`R/postProcessing.R:135-137`](https://github.com/dmcable/spacexr/blob/master/R/postProcessing.R#L135-L137) |
| Tutorial Assay construction (the `# normalize the cell type proportions to sum to 1.` line) | [`vignettes/spatial-transcriptomics.Rmd:135-143`](https://github.com/dmcable/spacexr/blob/master/vignettes/spatial-transcriptomics.Rmd#L135-L143) |
| `solveOLS` / `solveIRWLS.weights` | `R/IRWLS.R:1-50` |
| `process_bead_doublet` / `check_pairs_type` | `R/RCTD_helper.R:44-168` |

## Benchmarking

GPU benchmarks are in `benchmarks/`. Submit via SLURM:

```bash
sbatch benchmarks/sbatch_bench_compare.sh   # baseline vs optimized comparison
sbatch benchmarks/sbatch_bench_gpu.sh        # optimized-only timing
```

## Autoresearch

Autonomous optimization framework inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The agent modifies solver code, benchmarks, keeps/discards based on speed + correctness.

### How to run

1. Read `benchmarks/program.md` for full instructions
2. Create branch: `git checkout -b autoresearch/<tag>`
3. Establish baseline: run benchmark as-is
4. Loop: modify → commit → benchmark → keep/discard

### Key rules

- **Metric**: `elapsed_s` (lower is better)
- **Correctness**: `weights_hash` must match baseline — if it changes, discard
- **Files to modify**: `src/rctd/_irwls.py`, `src/rctd/_likelihood.py`, `src/rctd/_simplex.py`
- **Read-only**: `benchmarks/bench_gpu.py`, `tests/`
- **Log results** to `results.tsv` (tab-separated)
- **GPU partition**: target an L40S or Blackwell (sm_100+) node via your scheduler (e.g. `--partition=GPU --gres=gpu:1` for SLURM clusters that expose GPUs as a generic resource)

### Profiling hot spots

Pre-optimization CPU profiling (2k pixels): `calc_q_all` 36%, QP solver 27%, eigh (PSD) 14%, bmm 9%.

Post-optimization: `calc_q_all` is now a fused Triton kernel (single launch), K=2 PSD/QP are analytical (zero cuSOLVER calls in doublet mode). Run `torch.profiler` to identify remaining GPU-specific bottlenecks.

### Benchmarking tips

- **GPU benchmarks must pin to a specific node** via `--nodelist=<your-node>` for reproducible comparisons (different GPU SKUs and even different nodes of the same SKU produce different timings)
- **Warmup is mandatory**: torch.compile and Triton JIT have ~60s first-call overhead. Run a warmup dataset before the timed benchmarks
- **A/B comparisons**: Use `benchmarks/SBATCH_compare_optimizations.sh` — checks out baseline vs optimized source files, runs both on the same GPU node back-to-back
- **Spatial data loading**: Use `sc.read_10x_h5()` (scanpy), NOT `anndata.read_h5()` which doesn't exist
