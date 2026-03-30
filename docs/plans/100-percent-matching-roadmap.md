# Roadmap: 100% Concordance with R spacexr

**Date:** 2026-03-05 (updated 2026-03-21)
**Current concordance:** 99.77% dominant type agreement (Xenium Mouse Brain, 135,529 pixels, K=22, doublet mode, matched reference)
**With R's intermediates injected:** 100.00% (0 disagreeing pixels)

---

## Investigation Summary (2026-03-05)

### What Was Tested

| Hypothesis | Result | Verdict |
|------------|--------|---------|
| QP solver divergence (Gauss-Seidel vs Goldfarb-Idnani) | Both produce identical results | **NOT the cause** |
| Spline coefficient precision (np.linalg.inv vs solve vs tridiag) | Zero difference in likelihood values | **NOT the cause** |
| Per-pixel IRWLS algorithm | 100% match when given same inputs | **NOT the cause** |
| Platform effects (norm_profiles + sigma) | 100% match when R's values injected | **THE cause** |

### Root Cause: Platform Effect Estimation

The entire 0.27% gap comes from `fit_bulk()` producing slightly different normalized reference profiles than R's `fitBulk()`, which cascades into a different sigma estimate:

| Parameter | Python | R |
|-----------|--------|---|
| sigma (full dataset) | 63 | 62 |
| sigma (1000-cell subset) | 58 | 62 |
| gene_list_bulk | 243 | 243 |
| gene_list_reg | 206 | 206 |

The gene lists match exactly. The difference is in the Gaussian IRWLS (`bulk_mode=True`) used for platform normalization, where floating-point accumulation in log/division operations over ~14k cells produces slightly different bulk proportions. This causes a small norm_profiles shift, which shifts sigma estimation by 1 unit, which shifts all per-pixel likelihood values.

### Key Proof

**Experiment:** Inject R's exact `norm_profiles` (206×45) and `sigma=62` into Python, then run the full doublet pipeline.

| Approach | Dominant Type Agreement | Disagreeing Pixels | MAE |
|----------|-------------------------|--------------------|-----|
| Python platform effects (full dataset) | 99.73% | 37/13,936 | 0.000084 |
| R's norm_profiles + sigma injected | **100.00%** | **0/13,936** | **0.000000** |
| Python platform effects (1000-cell subset) | 95.60% | 44/1,000 | 0.001342 |
| R's intermediates (1000-cell subset) | **100.00%** | **0/1,000** | **0.000000** |

The per-pixel IRWLS produces **bit-identical** weights to R spacexr given the same norm_profiles and sigma.

---

## External Validation (2026-03-21)

### XeniumSeurat (SUSHI) Comparison — p41033

Compared rctd-py output against spacexr results from the FGCZ XeniumSeurat SUSHI pipeline on Xenium Mouse Brain data (Cas1_Region1, 141,668 cells, 22 cell types from Allen cortex reference).

#### Key Finding: Reference Preparation Matters

The XeniumSeurat app used `allen_cortex_rctd.rds` (pre-built spacexr Reference, 6,675 downsampled cells). When rctd-py used the full `allen_cortex.rds` (14,249 cells, auto-subsampled to ~10,000), agreement was lower than expected due to different reference cell composition.

| Metric | Mismatched ref (14k→10k cells) | Matched ref (6,675 cells) |
|--------|-------------------------------|---------------------------|
| Spot class agreement | 90.3% | **99.0%** |
| Dominant type (all classified) | 95.7% | **99.8%** |
| Singlet-singlet agreement | 98.3% | **99.9%** |
| Per-type weight r (mean) | — | **0.9999** |
| Per-pixel weight r (median) | 0.997 | **0.9999** |

#### Important: Weight Normalization

spacexr stores **normalized** weights (`normalize_weights()` → sum=1, clipped to [0,1]).
rctd-py stores **raw** full-mode weights (`obsm["rctd_weights"]` → sum < 1).

When comparing weights, **always normalize rctd-py weights first** to avoid artificially low correlations (~0.96 unnormalized vs ~0.999 normalized).

#### Report

Full comparison report: `p41033/Analyses_Paul/rctd_comparison/rctd_comparison_Cas1_Region1.html`

---

## Current Status

### Implemented
- `quadprog` optional dependency with `solver` config parameter (`"auto"`, `"quadprog"`, `"gauss_seidel"`)
- `sigma_override` parameter in `run_rctd()` and `RCTD.fit_platform_effects()` for injecting R's sigma
- **`longdouble` accumulation in `fit_bulk()`** — uses 80-bit extended precision for the critical `bulk_Y` and `bulk_nUMI` sums, matching R's long double accumulation on x86-64 (zero performance impact)
- All 55 tests passing

### The Concordance Gap

The remaining ~0.2% disagreeing pixels are **close-margin** cases where two cell types have weights within ~0.01-0.05 of each other. The platform effect shift causes the dominant type to flip for these borderline pixels.

The `longdouble` fix in `_normalize.py` eliminates one source of divergence (the N-pixel sum accumulation order). However, the IRWLS solver itself uses torch operations whose accumulation order still depends on BLAS. For the small IRWLS in fit_bulk (G=153, K=22), this residual difference is typically below the sigma quantization threshold.

---

## Remaining Improvement Options

### Option A: Match R's fit_bulk precision — PARTIALLY IMPLEMENTED

The `_longdouble_sum()` helper (added 2026-03-21) uses `np.longdouble` for the critical column sums in `fit_bulk()`. On x86-64, this gives 80-bit extended precision (18 significant digits), matching R's internal `sum()`.

**What's still different:** The IRWLS iterations within `fit_bulk()` use torch matmul/log operations which may accumulate differently from R. Since this operates on small tensors (G=153, K=22), the residual error is typically < 1 ULP.

**To fully close the gap:** The remaining IRWLS operations in `fit_bulk()` could also be promoted to longdouble. This would require converting the entire solve_irwls loop to numpy/longdouble (since PyTorch doesn't support float128). Since fit_bulk runs only once and on small tensors, the performance impact would be negligible (~10ms).

### Option B: Export R's norm_profiles as reference (Easy, 100% match)

For users who need exact R concordance:
1. Run R spacexr once to get norm_profiles and sigma
2. Export as CSV/NPZ
3. Pass to Python via `sigma_override` + a new `norm_profiles_override` parameter
4. This achieves 100.00% concordance

**Expected improvement:** 100.00% guaranteed
**Effort:** Small (add `norm_profiles_override` parameter)

### Option C: Accept 99.8% as sufficient (No effort)

The 0.2% gap affects only borderline pixels where two cell types are within ~0.01 weight of each other. These are genuinely ambiguous assignments that could go either way. The practical impact on downstream analysis is negligible.

---

## Invalidated Hypotheses

### 1. QP Solver Divergence — INVALIDATED

The original hypothesis was that Gauss-Seidel coordinate descent diverges from R's Goldfarb-Idnani active-set algorithm on ill-conditioned problems. Testing showed:
- Both solvers produce identical weights on the Region 1 dataset
- The Gauss-Seidel already converges to the same optimum
- Even for K=45 (cond(D) >> 1e6), the box constraints + simplex projection make the algorithms equivalent

### 2. Spline Coefficient Precision — INVALIDATED

The original hypothesis was that `np.linalg.inv(M)` for the 437×437 tridiagonal matrix differs from R's implementation. Testing showed:
- Condition number of M: 1.6e6 (moderate)
- Max absolute difference between inv/solve/tridiag: ~1.5e-8
- Impact on likelihood values: **exactly zero** (d0 diff = 0.00e+00)

### 3. Doublet Pair Selection Ties — INVALIDATED (indirectly)

Since the per-pixel IRWLS is bit-identical with R's inputs, tie-breaking cannot diverge when given the same upstream values.

### 4. Reference Mismatch as Solver Bug — INVALIDATED (2026-03-21)

Initial comparison with XeniumSeurat showed ~95.7% agreement. This was NOT a solver issue — it was caused by using a different reference (14k raw cells vs 6,675 pre-downsampled). With the matched reference: 99.8% agreement.

---

## Non-Goals

- **Matching R's performance characteristics:** We intentionally use GPU batching which changes memory access patterns. Numerical equivalence is the goal, not identical execution traces.
- **Supporting R's legacy quirks:** Some R spacexr behaviors (e.g., `norm(as.matrix(x))` defaulting to L1 instead of L2) are already matched. We won't match undocumented R behaviors that are clearly bugs.
