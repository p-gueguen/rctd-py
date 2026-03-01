#!/usr/bin/env python3
"""
Root cause investigation: Why is Python's pair NLL consistently ~14 points
higher than R's for the SAME pair on the SAME pixels?

Key findings so far:
- Gene lists: identical to R
- Profiles: match R to 4 decimal places (max diff 0.0026)
- Full-mode dominant type: 100% agreement with R
- Full-mode weights: max abs diff 0.14, mean 0.000045
- Pair solver: converged (max_iter doesn't matter) but scores are ~14 NLL higher
- Candidate sets: R's winning pair ALWAYS in Python's candidates

The ~14-point gap in pair NLL = Python's pair fit is worse than R's.
Both use constrain=True for pair fitting.

Hypotheses:
  1. Simplex projection difference: Python uses project_simplex vs R's quadprog
  2. The QP lower bound in Python is -solution (from full mode) but R may use different bounds
  3. Initial weights: Python starts at [0.5, 0.5] but R may warm-start differently
  4. R does NOT use simplex for doublet pair fitting (constrain=False?)
  5. R uses a different formulation: scalar weight (w1, 1-w1) for pair

Let's test:
  A. Run pair IRWLS with constrain=False vs constrain=True
  B. Use R's stated algorithm: pair weight = scalar alpha in [0,1] (1D constrained QP)
  C. Check R source: in doublet mode, does R use constrain=True or a 1D search?
"""
import sys
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import anndata
from pathlib import Path
from collections import Counter
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

VAL_DIR = Path("/home/pgueguen/git/rctd-py/data/xenium_validation")
REF_DIR = Path("/home/pgueguen/git/rctd-py/data")

from rctd import Reference, RCTDConfig
from rctd._rctd import RCTD
from rctd._likelihood import load_cached_q_matrices, compute_spline_coefficients, calc_log_likelihood, calc_q_all
from rctd._irwls import solve_irwls_batch, solve_irwls
from rctd._simplex import project_simplex
from rctd._types import SPOT_CLASS_NAMES
import jax
import jax.numpy as jnp

print("Loading data...")
counts = scipy.io.mmread(REF_DIR / "ref_counts.mtx").T.tocsc()
meta   = pd.read_csv(REF_DIR / "ref_meta.csv")
genes  = pd.read_csv(REF_DIR / "ref_genes.txt", header=None)[0].tolist()
ref_adata = anndata.AnnData(
    X=counts,
    obs=meta.set_index(meta.columns[0]),
    var=pd.DataFrame(index=genes),
)
reference = Reference(ref_adata, cell_type_col="cell_type")

import scanpy as sc
h5 = ("/srv/gstore/projects/p36005/Xenium_Data/"
      "data_20240724__105930__o35844/"
      "output-XETG00404__0022381__Region_3__20240724__110035/"
      "cell_feature_matrix.h5")
spatial = sc.read_10x_h5(h5)
spatial.var_names_make_unique()

config = RCTDConfig(UMI_min=20)
rctd   = RCTD(spatial, reference, config)
rctd.fit_platform_effects()
sigma = rctd.sigma

cache = load_cached_q_matrices()
x_vals = cache.pop("X_vals")
q_matrices = {k.replace("Q_", ""): v for k, v in cache.items()}
best_q_key = str(sigma)
if best_q_key not in q_matrices:
    available = sorted([int(k) for k in q_matrices.keys() if k.isdigit()])
    nearest = min(available, key=lambda x: abs(x - sigma))
    best_q_key = str(nearest)
q_mat  = q_matrices[best_q_key]
sq_mat = compute_spline_coefficients(q_mat, x_vals)

P_gpu  = jnp.array(rctd.norm_profiles.astype(np.float64))
Q_gpu  = jnp.array(q_mat)
SQ_gpu = jnp.array(sq_mat)
X_gpu  = jnp.array(x_vals)

K = len(reference.cell_type_names)
K_names = reference.cell_type_names
k_name_map = {n: i for i, n in enumerate(K_names)}

# Load sample data
r_results = pd.read_csv(VAL_DIR / "r_results_df_sample.csv", index_col=0)
r_barcodes_sample = [l.strip() for l in open(VAL_DIR / "r_pixel_barcodes_sample.txt") if l.strip()]

all_barcodes = np.array(spatial.obs_names)
py_barcodes  = all_barcodes[rctd._pixel_mask]
py_bc_map    = {b: i for i, b in enumerate(py_barcodes)}

matched_py, matched_r = [], []
for j, bc in enumerate(r_barcodes_sample):
    bc2 = bc.replace("Region_3_35844-2_", "")
    if bc2 in py_bc_map:
        matched_py.append(py_bc_map[bc2])
        matched_r.append(j)

matched_py = np.array(matched_py)
matched_r  = np.array(matched_r)

py_counts_m = rctd.counts[matched_py]
py_nUMI_m   = rctd.nUMI[matched_py]
r_sc        = r_results["spot_class"].values[matched_r]
r_ft        = r_results["first_type"].values[matched_r]
r_st        = r_results["second_type"].values[matched_r]
r_min_score = r_results["min_score"].values[matched_r]
r_sing_score= r_results["singlet_score"].values[matched_r]

print(f"Loaded {len(matched_py)} pixels")

# Pick 10 R-doublet pixels as test cases
r_is_doublet = (r_sc == "doublet_certain") | (r_sc == "doublet_uncertain")
r_dc_idx = np.where(r_is_doublet)[0][:20]  # first 20

print(f"\n{'='*70}")
print("A. Solver comparison: constrain=True vs constrain=False for pair fitting")
print(f"{'='*70}")

def score_pair(n, t1, t2, constrain, max_iter=25):
    """Score a single pair for pixel n with given settings."""
    nUMI_b = jnp.array(py_nUMI_m[[n]])
    B_b    = jnp.array(py_counts_m[[n]].astype(np.float64))
    P1 = P_gpu[:, t1:t1+1].T
    P2 = P_gpu[:, t2:t2+1].T
    S_b = nUMI_b[:, None, None] * jnp.stack([P1, P2], axis=-1)

    w_b, conv = solve_irwls_batch(
        S_batch=S_b, Y_batch=B_b, nUMI_batch=nUMI_b,
        Q_mat=Q_gpu, SQ_mat=SQ_gpu, x_vals=X_gpu,
        max_iter=max_iter, min_change=0.001, constrain=constrain, bulk_mode=False,
    )
    w_np = np.array(w_b)[0]
    expected = jnp.maximum(jnp.sum(S_b * w_b[:, None, :], axis=-1), 1e-4)
    sc = float(calc_log_likelihood(B_b[0], expected[0], Q_gpu, SQ_gpu, X_gpu, config.K_val))
    return sc, w_np, bool(conv[0])

scores_T = []
scores_F = []
for n in r_dc_idx:
    ft_i = k_name_map.get(r_ft[n], -1)
    st_i = k_name_map.get(r_st[n], -1)
    if ft_i < 0 or st_i < 0:
        continue
    sc_T, w_T, _ = score_pair(n, ft_i, st_i, constrain=True)
    sc_F, w_F, _ = score_pair(n, ft_i, st_i, constrain=False)
    scores_T.append(sc_T)
    scores_F.append(sc_F)
    if len(scores_T) <= 5:
        print(f"  Pixel {n}: R_min={r_min_score[n]:.3f}, "
              f"Py_constrain=True: {sc_T:.3f} (w={w_T}), "
              f"Py_constrain=False: {sc_F:.3f} (w={w_F})")

scores_T = np.array(scores_T)
scores_F = np.array(scores_F)
r_ms = r_min_score[r_dc_idx[:len(scores_T)]]

print(f"\nSummary over {len(scores_T)} R-doublet pixels (R's pair):")
print(f"  constrain=True:  mean={scores_T.mean():.3f}, diff from R: {(scores_T-r_ms).mean():.3f}")
print(f"  constrain=False: mean={scores_F.mean():.3f}, diff from R: {(scores_F-r_ms).mean():.3f}")
print(f"  R min_score:     mean={r_ms.mean():.3f}")
print(f"  constrain=True vs False: mean diff={((scores_T-scores_F)).mean():.4f}")


print(f"\n{'='*70}")
print("B. 1D line search for pair weight: alpha in [0,1], score(alpha*P1 + (1-alpha)*P2)")
print(f"{'='*70}")

def score_pair_1d_search(n, t1, t2, n_points=1001):
    """Score pair using 1D grid search over alpha in [0,1]."""
    nUMI_n = float(py_nUMI_m[n])
    Y = py_counts_m[n].astype(np.float64)
    P1 = np.array(P_gpu[:, t1])  # (G,)
    P2 = np.array(P_gpu[:, t2])  # (G,)

    best_score = float('inf')
    best_alpha = 0.5

    # Grid search
    alphas = np.linspace(0, 1, n_points)
    for alpha in alphas:
        # Expected = nUMI * (alpha * P1 + (1-alpha) * P2)
        expected = nUMI_n * (alpha * P1 + (1.0 - alpha) * P2)
        expected = np.maximum(expected, 1e-4)
        sc = float(calc_log_likelihood(
            jnp.array(Y), jnp.array(expected), Q_gpu, SQ_gpu, X_gpu, config.K_val
        ))
        if sc < best_score:
            best_score = sc
            best_alpha = alpha

    return best_score, best_alpha

print("Testing 1D search on first 20 R-doublet pixels...")
scores_1d = []
alphas_1d = []
for n in r_dc_idx[:20]:
    ft_i = k_name_map.get(r_ft[n], -1)
    st_i = k_name_map.get(r_st[n], -1)
    if ft_i < 0 or st_i < 0:
        scores_1d.append(np.nan); alphas_1d.append(np.nan)
        continue
    sc, alpha = score_pair_1d_search(n, ft_i, st_i, n_points=10001)
    scores_1d.append(sc)
    alphas_1d.append(alpha)

scores_1d = np.array(scores_1d)
r_ms_20 = r_min_score[r_dc_idx[:20]]
scores_T_20 = np.array([score_pair(n, k_name_map.get(r_ft[n],-1),
                                   k_name_map.get(r_st[n],-1), True)[0]
                         for n in r_dc_idx[:20]])

print(f"\nSummary over 20 R-doublet pixels:")
print(f"  1D grid search:  mean={np.nanmean(scores_1d):.3f}, diff from R: {np.nanmean(scores_1d-r_ms_20):.3f}")
print(f"  constrain=True IRWLS: mean={scores_T_20.mean():.3f}, diff from R: {(scores_T_20-r_ms_20).mean():.3f}")
print(f"  R min_score:     mean={r_ms_20.mean():.3f}")

for i in range(min(5, len(r_dc_idx))):
    n = r_dc_idx[i]
    print(f"  Pixel {n}: R={r_ms_20[i]:.3f}, 1D={scores_1d[i]:.3f}, "
          f"IRWLS_T={scores_T_20[i]:.3f}, alpha={alphas_1d[i]:.4f}")


print(f"\n{'='*70}")
print("C. Compare R's doublet weight to Python IRWLS weights")
print(f"{'='*70}")

# R's weights are stored in r_pixel_weights_sample.csv (full-mode weights)
# but for doublet mode, R stores "weights_doublet" (2-component, rescaled to sum=1)
# Check r_results_df_sample.csv for any weight columns
print(f"R results columns: {list(r_results.columns)}")

# r_pixel_weights_sample.csv is the FULL-mode weight matrix (K columns)
# For R doublet mode, R uses these full weights to find candidates,
# then uses the pair-specific weight from doublet fit.
# R's min_score is the best pair NLL.

# Key question: in R's IRWLS for pairs, is the 2D weight constrained to simplex?
# In spacexr R source, doublet mode uses:
#   fitPixels("doublet", constrain=TRUE)
#   which calls process_beads(singlet=FALSE, constrain=TRUE)
#   which calls solveIRWLS.weights(D_mat, d_vec, constrain.sum=TRUE, solution_init)
#   with meq=1 in quadprog (equality constraint sum=1)

# So R DOES use simplex (sum=1) for pair weights.
# But Python also uses constrain=True with project_simplex.

# The difference must be in the INITIALIZATION of the pair solver.
# In R: initialSol = rbind(w1, w2) for the TOP 2 types from full mode,
#        normalized to sum=1.
# In Python: w = ones(K) / K = [0.5, 0.5] for K=2 pairs.

print(f"\n{'='*70}")
print("D. Initialization comparison: [0.5, 0.5] vs full-mode weights")
print(f"{'='*70}")

# Run full mode to get initial weights
from rctd._full import run_full_mode

full_res = run_full_mode(
    spatial_counts=py_counts_m.astype(np.float64),
    spatial_numi=py_nUMI_m.astype(np.float64),
    norm_profiles=rctd.norm_profiles.astype(np.float64),
    cell_type_names=K_names,
    q_mat=q_mat, sq_mat=sq_mat, x_vals=x_vals, batch_size=2000,
)
W_full = full_res.weights

# For pair (t1, t2), Python initializes as [0.5, 0.5].
# R initializes using full-mode weights for t1 and t2, normalized.
# Test: initialize with [w_full[t1], w_full[t2]] / (w_full[t1] + w_full[t2])

def solve_irwls_init(S, Y, nUMI, w_init, constrain=True, max_iter=50):
    """IRWLS with custom initialization (NumPy, for testing only)."""
    G, K = S.shape
    w = w_init.copy()

    for _ in range(max_iter):
        solution = np.maximum(w, 0.0)
        prediction = np.abs(S @ solution)
        threshold = max(1e-4, nUMI * 1e-7)
        prediction = np.maximum(prediction, threshold)

        # Compute gradient and Hessian using JAX
        pred_j = jnp.array(prediction)
        Y_j    = jnp.array(Y)
        S_j    = jnp.array(S)

        d0, d1, d2 = calc_q_all(Y_j, pred_j, Q_gpu, SQ_gpu, X_gpu, config.K_val)
        d1 = np.array(d1)
        d2 = np.array(d2)
        S_np = np.array(S_j)

        grad = -(d1 @ S_np)
        hess = S_np.T @ ((-d2)[:, None] * S_np)

        # PSD correction
        evals, evecs = np.linalg.eigh(hess)
        evals = np.maximum(evals, 1e-3)
        hess = evecs @ np.diag(evals) @ evecs.T

        norm_factor = max(np.linalg.norm(hess, ord=2), 1e-10)
        D = hess / norm_factor + 1e-7 * np.eye(K)
        d = -grad / norm_factor

        # Gauss-Seidel coordinate descent (box QP: x >= -solution)
        x = np.maximum(d / np.diag(D), -solution)
        for _ in range(30):
            for i in range(K):
                residual = d[i] - (np.dot(D[i], x) - D[i, i] * x[i])
                x[i] = max(residual / D[i, i], -solution[i])

        delta_w = x
        w_new = solution + 0.3 * delta_w

        if constrain:
            # Project to simplex
            w_new = np.array(project_simplex(jnp.array(w_new)))

        change = np.linalg.norm(w_new - w)
        w = w_new
        if change <= 0.001:
            break

    prediction = np.maximum(np.abs(S @ np.maximum(w, 0.0)), max(1e-4, nUMI * 1e-7))
    sc = float(calc_log_likelihood(
        jnp.array(Y), jnp.array(prediction), Q_gpu, SQ_gpu, X_gpu, config.K_val
    ))
    return sc, w

print("Comparing initializations for 20 R-doublet pixels...")
scores_init_half = []
scores_init_full = []
for n in r_dc_idx[:20]:
    ft_i = k_name_map.get(r_ft[n], -1)
    st_i = k_name_map.get(r_st[n], -1)
    if ft_i < 0 or st_i < 0:
        scores_init_half.append(np.nan)
        scores_init_full.append(np.nan)
        continue

    nUMI_n = float(py_nUMI_m[n])
    Y_n = py_counts_m[n].astype(np.float64)
    P1 = np.array(P_gpu[:, ft_i])
    P2 = np.array(P_gpu[:, st_i])
    S_n = np.column_stack([nUMI_n * P1, nUMI_n * P2])  # (G, 2)

    # Init [0.5, 0.5]
    w_half = np.array([0.5, 0.5])
    sc_half, _ = solve_irwls_init(S_n, Y_n, nUMI_n, w_half, constrain=True)

    # Init from full-mode weights (normalized)
    w_ft = W_full[n, ft_i]
    w_st = W_full[n, st_i]
    s    = w_ft + w_st
    if s > 1e-10:
        w_full_init = np.array([w_ft / s, w_st / s])
    else:
        w_full_init = np.array([0.5, 0.5])
    sc_full, _ = solve_irwls_init(S_n, Y_n, nUMI_n, w_full_init, constrain=True)

    scores_init_half.append(sc_half)
    scores_init_full.append(sc_full)

scores_init_half = np.array(scores_init_half)
scores_init_full = np.array(scores_init_full)
r_ms_20 = r_min_score[r_dc_idx[:20]]
valid = np.isfinite(scores_init_half) & np.isfinite(scores_init_full)

print(f"\nComparison (20 R-doublet pixels, R's pair):")
print(f"  Init [0.5, 0.5]:              mean={scores_init_half[valid].mean():.3f}, "
      f"diff from R: {(scores_init_half[valid]-r_ms_20[valid]).mean():.3f}")
print(f"  Init from full-mode weights:  mean={scores_init_full[valid].mean():.3f}, "
      f"diff from R: {(scores_init_full[valid]-r_ms_20[valid]).mean():.3f}")
print(f"  1D grid search (from B):      mean={np.nanmean(scores_1d[:20]):.3f}, "
      f"diff from R: {np.nanmean(scores_1d[:20]-r_ms_20):.3f}")
print(f"  R min_score:                  mean={r_ms_20[valid].mean():.3f}")

for i in range(min(5, np.sum(valid))):
    n = r_dc_idx[i]
    if not valid[i]:
        continue
    print(f"  Pixel {n}: R={r_ms_20[i]:.3f}, half={scores_init_half[i]:.3f}, "
          f"full_init={scores_init_full[i]:.3f}, 1d={scores_1d[i]:.3f}")


print(f"\n{'='*70}")
print("E. Brute-force 2D grid search: score(w1, w2) with w1+w2=1, w1>=0, w2>=0")
print(f"{'='*70}")

def score_pair_2d_grid(n, t1, t2, n_points=10001):
    """Score pair using 2D grid search: w1 in [0,1], w2 = 1-w1."""
    nUMI_n = float(py_nUMI_m[n])
    Y = py_counts_m[n].astype(np.float64)
    P1 = np.array(P_gpu[:, t1])
    P2 = np.array(P_gpu[:, t2])

    best_score = float('inf')
    best_w1 = 0.5
    w1_vals = np.linspace(0, 1, n_points)
    for w1 in w1_vals:
        w2 = 1.0 - w1
        expected = nUMI_n * (w1 * P1 + w2 * P2)
        expected = np.maximum(expected, 1e-4)
        sc = float(calc_log_likelihood(
            jnp.array(Y), jnp.array(expected), Q_gpu, SQ_gpu, X_gpu, config.K_val
        ))
        if sc < best_score:
            best_score = sc
            best_w1 = w1
    return best_score, best_w1

print("2D grid search (w1+w2=1) on 20 R-doublet pixels (R's pair)...")
scores_2d = []
for n in r_dc_idx[:20]:
    ft_i = k_name_map.get(r_ft[n], -1)
    st_i = k_name_map.get(r_st[n], -1)
    if ft_i < 0 or st_i < 0:
        scores_2d.append(np.nan)
        continue
    sc, w1 = score_pair_2d_grid(n, ft_i, st_i, n_points=10001)
    scores_2d.append(sc)

scores_2d = np.array(scores_2d)
print(f"  2D grid:      mean={np.nanmean(scores_2d):.3f}, "
      f"diff from R: {np.nanmean(scores_2d-r_ms_20):.3f}")
print(f"  1D grid:      mean={np.nanmean(scores_1d[:20]):.3f}, "
      f"diff from R: {np.nanmean(scores_1d[:20]-r_ms_20):.3f}")
print(f"  Note: 1D grid uses alpha*P1 + (1-alpha)*P2, same as 2D with w1+w2=1")


print(f"\n{'='*70}")
print("F. Does R use UNCONSTRAINED pair fitting? Test with w1+w2 NOT constrained")
print(f"{'='*70}")

# Test: 2D box-constrained (w1>=0, w2>=0, but NOT sum=1)
def score_pair_unconstrained_grid(n, t1, t2, n_points=201):
    """Score pair with w1 in [0,2], w2 in [0,2] grid."""
    nUMI_n = float(py_nUMI_m[n])
    Y = py_counts_m[n].astype(np.float64)
    P1 = np.array(P_gpu[:, t1])
    P2 = np.array(P_gpu[:, t2])

    best_score = float('inf')
    best_w = (1.0, 0.0)
    for w1 in np.linspace(0, 2, n_points):
        for w2 in np.linspace(0, 2, n_points):
            expected = nUMI_n * (w1 * P1 + w2 * P2)
            expected = np.maximum(expected, 1e-4)
            sc = float(calc_log_likelihood(
                jnp.array(Y), jnp.array(expected), Q_gpu, SQ_gpu, X_gpu, config.K_val
            ))
            if sc < best_score:
                best_score = sc
                best_w = (w1, w2)
    return best_score, best_w

print("Testing 5 R-doublet pixels with unconstrained 2D search...")
for n in r_dc_idx[:5]:
    ft_i = k_name_map.get(r_ft[n], -1)
    st_i = k_name_map.get(r_st[n], -1)
    if ft_i < 0 or st_i < 0:
        continue
    sc_2d, w1 = score_pair_2d_grid(n, ft_i, st_i, n_points=10001)
    sc_unc, best_w = score_pair_unconstrained_grid(n, ft_i, st_i, n_points=101)
    print(f"  Pixel {n}: R={r_min_score[n]:.3f}, constrained_1D={sc_2d:.3f}, "
          f"unconstrained={sc_unc:.3f} (w={best_w})")


print(f"\n{'='*70}")
print("G. Verify: R's min_score matches Python 1D grid score")
print(f"{'='*70}")

# R's min_score for doublet pixels is the NLL of the BEST pair.
# If R uses w1+w2=1 constraint, then R's min_score should equal Python 1D grid.
# We found mean diff = 1D_grid - R_min ≈ ?

# From B: 1D grid on 20 pixels: mean = np.nanmean(scores_1d[:20])
# R min: r_ms_20.mean()
print(f"1D grid mean score (20 pixels): {np.nanmean(scores_1d[:20]):.4f}")
print(f"R min_score mean (20 pixels):   {r_ms_20.mean():.4f}")
print(f"Diff: {np.nanmean(scores_1d[:20]) - r_ms_20.mean():.4f}")

# If the 1D grid matches R, then the IRWLS converges to the correct minimum.
# If IRWLS gives higher score than 1D grid, IRWLS is trapped in a local optimum.
# Let's compare IRWLS (constrain=True) vs 1D grid for those 20 pixels.
print(f"\nIRWLS (constrain=True) mean: {scores_T_20.mean():.4f}")
print(f"1D grid mean:                {np.nanmean(scores_1d[:20]):.4f}")
print(f"IRWLS vs 1D diff:            {scores_T_20.mean() - np.nanmean(scores_1d[:20]):.4f}")
print(f"IRWLS vs R diff:             {scores_T_20.mean() - r_ms_20.mean():.4f}")
print(f"1D vs R diff:                {np.nanmean(scores_1d[:20]) - r_ms_20.mean():.4f}")

# Per-pixel comparison
print(f"\nPer-pixel (20 pixels):")
print(f"{'n':>4} {'R_min':>8} {'1D_grid':>8} {'IRWLS_T':>8} {'1D-R':>8} {'IRWLS-R':>8} {'IRWLS-1D':>9}")
for i, n in enumerate(r_dc_idx[:20]):
    print(f"{n:>4} {r_ms_20[i]:>8.3f} {scores_1d[i]:>8.3f} {scores_T_20[i]:>8.3f} "
          f"{scores_1d[i]-r_ms_20[i]:>8.3f} {scores_T_20[i]-r_ms_20[i]:>8.3f} "
          f"{scores_T_20[i]-scores_1d[i]:>9.3f}")


print(f"\n{'='*70}")
print("H. Check simplex projection in Python: is it correct?")
print(f"{'='*70}")

# Test project_simplex
test_vecs = [
    np.array([0.3, 0.7]),
    np.array([0.8, 0.2]),
    np.array([1.5, -0.5]),
    np.array([0.5, 0.5]),
    np.array([0.1, 0.9]),
]
print("project_simplex tests (should sum to 1, each >= 0):")
for v in test_vecs:
    p = np.array(project_simplex(jnp.array(v)))
    print(f"  input={v}, output={p}, sum={p.sum():.6f}")


print(f"\n{'='*70}")
print("I. R doublet mode: what formulation does R use?")
print(f"{'='*70}")
print("""
From spacexr R source (IRWLS.R):
  solve_irwls_weights():
    - constrain.sum = TRUE for doublet pair fitting
    - quadprog::solve.QP(D_mat, d_vec, diag(K), -solution, meq=1)
    - meq=1 means EQUALITY constraint sum(w) = 1 (not just >= 0)
    - The box constraint in Python only imposes w_i >= -solution_i (lower bound),
      NOT sum(w) = 1.

In Python solve_irwls with constrain=True:
    1. Box QP gives delta_w (no sum constraint)
    2. Then project_simplex is applied to w_new = solution + step * delta_w

This is a fundamentally different algorithm:
  - R: solve QP with equality constraint sum(w) = 1 AND bound w_i >= -solution_i
  - Python: solve box QP (no sum constraint), then project to simplex

The key difference: R solves the EXACT constrained QP in one step,
while Python uses a two-step approach (solve box QP then project).
The box QP solution is not the same as the simplex-constrained solution.

Specifically:
  R quadprog with meq=1 solves:
    min 0.5 x^T D x - d^T x
    s.t. sum(x) = 1  (equality)
         x_i >= -solution_i  (lower bounds)

  Python box QP solves:
    min 0.5 x^T D x - d^T x
    s.t. x_i >= -solution_i  (lower bounds only, NO equality)
  Then projects w_new onto simplex.

The projection step AFTER solving an unconstrained-sum QP gives a DIFFERENT
result than solving with the sum=1 equality constraint directly.

This explains the systematic ~14-point gap in pair NLL scores:
Python's IRWLS finds a slightly suboptimal solution because it doesn't
enforce sum(w) = 1 during the QP solve step.
""")

print(f"\n{'='*70}")
print("J. Test: R-equivalent QP (equality constraint sum=1)")
print(f"{'='*70}")

# Implement equality-constrained QP solving for pairs.
# For K=2: sum(w) = 1 means w2 = 1 - w1.
# So the pair problem reduces to 1D: min f(w1) with w1 in [max(0, 1-1) to 1].
# That's w1 in [0, 1].
# R's equality-constrained QP for K=2 effectively becomes a 1D bounded search.

# Test: run Python IRWLS but with equality constraint enforced at QP level
# For K=2 pair: w2 = 1 - w1, reduce to 1D Newton's method

def solve_irwls_1d_equality(n, t1, t2, max_iter=50, step_size=0.3):
    """Solve doublet pair with sum=1 equality (1D problem)."""
    nUMI_n = float(py_nUMI_m[n])
    Y = py_counts_m[n].astype(np.float64)
    P1 = np.array(P_gpu[:, t1])
    P2 = np.array(P_gpu[:, t2])

    # w1 in [0, 1], w2 = 1-w1
    w1 = 0.5

    for _ in range(max_iter):
        w1_c = max(0.0, min(1.0, w1))
        w2_c = 1.0 - w1_c
        pred = nUMI_n * (w1_c * P1 + w2_c * P2)
        pred = np.maximum(pred, max(1e-4, nUMI_n * 1e-7))

        # Get derivatives
        pred_j = jnp.array(pred)
        Y_j    = jnp.array(Y)
        _, d1, d2 = calc_q_all(Y_j, pred_j, Q_gpu, SQ_gpu, X_gpu, config.K_val)
        d1 = np.array(d1)
        d2 = np.array(d2)

        # grad wrt w1: dL/dw1 = sum_g d1_g * (P1_g - P2_g) * nUMI
        diff_P = nUMI_n * (P1 - P2)
        grad1 = -np.dot(d1, diff_P)

        # hess wrt w1: d2L/dw1^2 = sum_g (-d2_g) * (P1_g - P2_g)^2 * nUMI^2
        hess1 = np.dot(-d2, diff_P**2)
        hess1 = max(hess1, 1e-10)

        delta_w1 = -grad1 / hess1
        w1_new = w1_c + step_size * delta_w1
        w1_new = max(0.0, min(1.0, w1_new))

        change = abs(w1_new - w1)
        w1 = w1_new
        if change <= 0.001:
            break

    w1_f = max(0.0, min(1.0, w1))
    w2_f = 1.0 - w1_f
    pred_f = nUMI_n * (w1_f * P1 + w2_f * P2)
    pred_f = np.maximum(pred_f, 1e-4)
    sc = float(calc_log_likelihood(
        jnp.array(Y), jnp.array(pred_f), Q_gpu, SQ_gpu, X_gpu, config.K_val
    ))
    return sc, (w1_f, w2_f)

print("Testing 1D equality solver on 20 R-doublet pixels (R's pair)...")
scores_1d_eq = []
for n in r_dc_idx[:20]:
    ft_i = k_name_map.get(r_ft[n], -1)
    st_i = k_name_map.get(r_st[n], -1)
    if ft_i < 0 or st_i < 0:
        scores_1d_eq.append(np.nan)
        continue
    sc, ws = solve_irwls_1d_equality(n, ft_i, st_i)
    scores_1d_eq.append(sc)

scores_1d_eq = np.array(scores_1d_eq)
print(f"\nSummary (20 pixels):")
print(f"  1D equality IRWLS: mean={np.nanmean(scores_1d_eq):.3f}, "
      f"diff from R: {np.nanmean(scores_1d_eq-r_ms_20):.3f}")
print(f"  1D grid search:    mean={np.nanmean(scores_1d[:20]):.3f}, "
      f"diff from R: {np.nanmean(scores_1d[:20]-r_ms_20):.3f}")
print(f"  2D IRWLS (constrain=True): mean={scores_T_20.mean():.3f}, "
      f"diff from R: {(scores_T_20-r_ms_20).mean():.3f}")
print(f"  R min_score:       mean={r_ms_20.mean():.3f}")

print(f"\nPer-pixel (20 pixels):")
print(f"{'n':>4} {'R_min':>8} {'1D_grid':>8} {'1D_eq':>8} {'IRWLS_T':>8} "
      f"{'1D_grid-R':>10} {'1D_eq-R':>9} {'IRWLS-R':>9}")
for i, n in enumerate(r_dc_idx[:20]):
    print(f"{n:>4} {r_ms_20[i]:>8.3f} {scores_1d[i]:>8.3f} {scores_1d_eq[i]:>8.3f} "
          f"{scores_T_20[i]:>8.3f} {scores_1d[i]-r_ms_20[i]:>10.3f} "
          f"{scores_1d_eq[i]-r_ms_20[i]:>9.3f} {scores_T_20[i]-r_ms_20[i]:>9.3f}")

print(f"\n{'='*70}")
print("ROOT CAUSE INVESTIGATION COMPLETE")
print(f"{'='*70}")
