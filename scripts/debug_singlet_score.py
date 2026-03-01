#!/usr/bin/env python3
"""
Investigate the remaining 13.4% first_type disagreement after fixing constrain=False.

After fixing constrain=False for pairs:
  - min_score: mean diff = +0.006 (essentially identical to R)
  - singlet_score: mean diff = -1.5, std=7.47 (still mismatching R)
  - doublet_thresh correlation: 0.843 (not perfect)

The singlet_score discrepancy drives doublet_thresh differences, causing
some pixels to be misclassified even with the pair constraint fixed.

Also:
  - Python has 0 doublet_uncertain still (R has 103)
  - "Melanoma -> Intestinal_macrophages" is the most common mismatch

This suggests that:
1. The singlet scoring (constrain=False) gives slightly different scores than R
2. The CONFIDENCE_THRESHOLD (for doublet_uncertain) is not being reached

Let's investigate:
  A. Singlet score discrepancy: why does Python singlet_score differ from R's?
  B. doublet_uncertain: what drives the confidence difference?
  C. Melanoma misclassification pattern
"""
import sys
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pandas as pd
import scipy.io
import anndata
from pathlib import Path
from collections import Counter
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

VAL_DIR = Path("/home/pgueguen/git/rctd-py/data/xenium_validation")
REF_DIR = Path("/home/pgueguen/git/rctd-py/data")

from rctd import Reference, RCTDConfig
from rctd._rctd import RCTD
from rctd._likelihood import load_cached_q_matrices, compute_spline_coefficients, calc_log_likelihood
from rctd._full import run_full_mode
from rctd._irwls import solve_irwls_batch
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

r_results     = pd.read_csv(VAL_DIR / "r_results_df_sample.csv", index_col=0)
r_barcodes    = [l.strip() for l in open(VAL_DIR / "r_pixel_barcodes_sample.txt") if l.strip()]
all_barcodes  = np.array(spatial.obs_names)
py_barcodes   = all_barcodes[rctd._pixel_mask]
py_bc_map     = {b: i for i, b in enumerate(py_barcodes)}

matched_py, matched_r = [], []
for j, bc in enumerate(r_barcodes):
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
r_min_score = r_results["min_score"].values[matched_r]
r_sing_score= r_results["singlet_score"].values[matched_r]
N = len(matched_py)

print(f"Loaded {N} pixels")

# Full fit
full_res = run_full_mode(
    spatial_counts=py_counts_m.astype(np.float64),
    spatial_numi=py_nUMI_m.astype(np.float64),
    norm_profiles=rctd.norm_profiles.astype(np.float64),
    cell_type_names=K_names,
    q_mat=q_mat, sq_mat=sq_mat, x_vals=x_vals, batch_size=2000,
)
W_full = full_res.weights

WEIGHT_THRESHOLD = 0.01
candidates_list = []
for n in range(N):
    cands = np.where(W_full[n] > WEIGHT_THRESHOLD)[0].tolist()
    if len(cands) == 0:
        best = int(np.argmax(W_full[n]))
        cands = [best, (best+1)%K]
    elif len(cands) == 1:
        t = cands[0]
        cands = [t, (t+1)%K if t != 0 else 1]
    candidates_list.append(cands)

# Score singlets (constrain=False)
print("Scoring singlets...")
singles = list({(n, t) for n, cands in enumerate(candidates_list) for t in cands})
singlet_log_l = {}
singles_arr = np.array(singles, dtype=np.int32)
bs = 2000
for start in range(0, len(singles_arr), bs):
    end = min(start+bs, len(singles_arr))
    sg = singles_arr[start:end]
    pix_idx = sg[:, 0]
    t_idx   = sg[:, 1]
    nUMI_b  = jnp.array(py_nUMI_m[pix_idx])
    B_b     = jnp.array(py_counts_m[pix_idx].astype(np.float64))
    P_sg    = P_gpu[:, t_idx].T[..., None]
    S_b     = nUMI_b[:, None, None] * P_sg
    w_b, _  = solve_irwls_batch(
        S_batch=S_b, Y_batch=B_b, nUMI_batch=nUMI_b,
        Q_mat=Q_gpu, SQ_mat=SQ_gpu, x_vals=X_gpu,
        max_iter=50, min_change=0.001, constrain=False, bulk_mode=False,
    )
    expected = jnp.maximum(jnp.sum(S_b * w_b[:, None, :], axis=-1), 1e-4)
    batched_nll = jax.vmap(
        lambda y, lam: calc_log_likelihood(y, lam, Q_gpu, SQ_gpu, X_gpu, config.K_val),
        in_axes=(0, 0),
    )
    scores = np.array(batched_nll(B_b, expected))
    for i in range(len(sg)):
        singlet_log_l[(singles[start+i][0], singles[start+i][1])] = scores[i]

# Extract best singlet type and score per pixel
best_sing_types = []
best_sing_scores = []
INF = 1e18
for n in range(N):
    cands = candidates_list[n]
    sing_sc = [singlet_log_l.get((n, t), INF) for t in cands]
    best_i = int(np.argmin(sing_sc))
    best_sing_types.append(K_names[cands[best_i]])
    best_sing_scores.append(sing_sc[best_i])

best_sing_types = np.array(best_sing_types)
best_sing_scores = np.array(best_sing_scores)

print(f"\n{'='*70}")
print("A. Singlet score comparison: Python vs R")
print(f"{'='*70}")

diff_s = best_sing_scores - r_sing_score
print(f"Python singlet_score vs R singlet_score:")
print(f"  Mean diff: {diff_s.mean():.4f}")
print(f"  Std diff:  {diff_s.std():.4f}")
print(f"  Max |diff|: {np.abs(diff_s).max():.4f}")
corr = np.corrcoef(best_sing_scores, r_sing_score)[0,1]
print(f"  Correlation: {corr:.6f}")

# Large singlet_score disagreements
large_diff = np.abs(diff_s) > 10
print(f"\n  Pixels with |singlet_score diff| > 10: {large_diff.sum()}")
if large_diff.sum() > 0:
    print(f"  These pixels:")
    for n in np.where(large_diff)[0][:20]:
        ft_r = r_ft[n]
        ft_py = best_sing_types[n]
        cands = [K_names[c] for c in candidates_list[n]]
        print(f"    Pixel {n}: R_sing={r_sing_score[n]:.2f}, Py_sing={best_sing_scores[n]:.2f}, "
              f"diff={diff_s[n]:.2f}, R_ft={ft_r}, Py_ft={ft_py}, "
              f"R_sc={r_sc[n]}, cands={cands}")

print(f"\n{'='*70}")
print("B. Why does singlet_score differ? Test: R's best singlet type is different")
print(f"{'='*70}")

# When Python's singlet_score differs from R's, is it because:
# (a) Python picks a different best singlet type? or
# (b) Python gives different NLL for the SAME type?

best_sing_agree = (best_sing_types == r_ft).mean()
print(f"Python best singlet type agrees with R first_type: {best_sing_agree:.4f} ({best_sing_agree*100:.1f}%)")

# For singlet pixels: R's first_type IS the best singlet type
r_singlet_mask = r_sc == "singlet"
if r_singlet_mask.sum() > 0:
    agree_sing = (best_sing_types[r_singlet_mask] == r_ft[r_singlet_mask]).mean()
    print(f"For R singlets (n={r_singlet_mask.sum()}): best singlet type agreement = {agree_sing:.4f}")

# Score R's first_type as a singlet using Python solver
print("\nScoring R's first_type as singlet for each pixel...")
r_ft_as_singlet_scores = np.full(N, np.nan)
for n in range(N):
    ft_i = k_name_map.get(r_ft[n], -1)
    if ft_i < 0:
        continue
    if (n, ft_i) in singlet_log_l:
        r_ft_as_singlet_scores[n] = singlet_log_l[(n, ft_i)]
    else:
        # Need to compute it
        nUMI_b = jnp.array(py_nUMI_m[[n]])
        B_b    = jnp.array(py_counts_m[[n]].astype(np.float64))
        P_sg   = P_gpu[:, ft_i:ft_i+1].T[..., None]
        S_b    = nUMI_b[:, None, None] * P_sg
        w_b, _ = solve_irwls_batch(
            S_batch=S_b, Y_batch=B_b, nUMI_batch=nUMI_b,
            Q_mat=Q_gpu, SQ_mat=SQ_gpu, x_vals=X_gpu,
            max_iter=50, min_change=0.001, constrain=False, bulk_mode=False,
        )
        expected = jnp.maximum(jnp.sum(S_b * w_b[:, None, :], axis=-1), 1e-4)
        sc = float(calc_log_likelihood(B_b[0], expected[0], Q_gpu, SQ_gpu, X_gpu, config.K_val))
        r_ft_as_singlet_scores[n] = sc

diff_sing_rft = r_ft_as_singlet_scores - r_sing_score
print(f"\nPython score for R's first_type as singlet vs R singlet_score:")
print(f"  Mean diff: {diff_sing_rft.mean():.4f}")
print(f"  Std diff:  {diff_sing_rft.std():.4f}")
print(f"  Max |diff|: {np.abs(diff_sing_rft).max():.4f}")
corr2 = np.corrcoef(r_ft_as_singlet_scores, r_sing_score)[0,1]
print(f"  Correlation: {corr2:.6f}")

# This tells us: is the singlet score diff due to (a) different type or (b) different NLL for same type?
print(f"\n  For pixels where Python picks same best type as R:")
same_ft_mask = best_sing_types == r_ft
diff_same = diff_s[same_ft_mask]
print(f"    n={same_ft_mask.sum()}, mean diff={diff_same.mean():.4f}, std={diff_same.std():.4f}")

diff_rft_same = diff_sing_rft[same_ft_mask]
print(f"  Python NLL for R's type (same type) vs R: mean={diff_rft_same.mean():.4f}, std={diff_rft_same.std():.4f}")

print(f"\n{'='*70}")
print("C. doublet_uncertain: what's missing?")
print(f"{'='*70}")

# After fixing pairs (constrain=False), Python still has 0 doublet_uncertain.
# doublet_uncertain = doublet wins, but confidence < 5
# confidence = min(singlet_of_ft - pair_score, singlet_of_st - pair_score)

# Let's score pairs (constrain=False) for R's doublet_uncertain pixels
r_du_mask = r_sc == "doublet_uncertain"
print(f"R doublet_uncertain pixels: {r_du_mask.sum()}")

if r_du_mask.sum() > 0:
    du_idx = np.where(r_du_mask)[0]
    # Get pair scores for these pixels (for R's pair)
    r_ft_du = r_ft[r_du_mask]
    r_st_du = r_results["second_type"].values[matched_r][r_du_mask]
    r_ms_du = r_min_score[r_du_mask]
    r_ss_du = r_sing_score[r_du_mask]
    r_thr_du = r_ss_du - r_ms_du

    print(f"R thresh for doublet_uncertain: min={r_thr_du.min():.2f}, max={r_thr_du.max():.2f}, "
          f"mean={r_thr_du.mean():.2f}")

    # Score R's pair (constrain=False) for these pixels
    pair_scores_du = []
    for i, n in enumerate(du_idx):
        ft_i = k_name_map.get(r_ft_du[i], -1)
        st_i = k_name_map.get(r_st_du[i], -1)
        if ft_i < 0 or st_i < 0:
            pair_scores_du.append(np.nan)
            continue
        key_fwd = (ft_i, st_i) if ft_i < st_i else (st_i, ft_i)
        # Try to find in existing singlet_log_l... no, this is pairs
        nUMI_b = jnp.array(py_nUMI_m[[n]])
        B_b    = jnp.array(py_counts_m[[n]].astype(np.float64))
        P1 = P_gpu[:, ft_i:ft_i+1].T
        P2 = P_gpu[:, st_i:st_i+1].T
        S_b = nUMI_b[:, None, None] * jnp.stack([P1, P2], axis=-1)
        w_b, _ = solve_irwls_batch(
            S_batch=S_b, Y_batch=B_b, nUMI_batch=nUMI_b,
            Q_mat=Q_gpu, SQ_mat=SQ_gpu, x_vals=X_gpu,
            max_iter=25, min_change=0.001, constrain=False, bulk_mode=False,
        )
        expected = jnp.maximum(jnp.sum(S_b * w_b[:, None, :], axis=-1), 1e-4)
        sc = float(calc_log_likelihood(B_b[0], expected[0], Q_gpu, SQ_gpu, X_gpu, config.K_val))
        pair_scores_du.append(sc)

    pair_scores_du = np.array(pair_scores_du)
    py_thresh_du = r_ft_as_singlet_scores[r_du_mask] - pair_scores_du

    print(f"\nPython doublet_thresh for R=doublet_uncertain pixels (constrain=False for pairs):")
    print(f"  min={np.nanmin(py_thresh_du):.2f}, max={np.nanmax(py_thresh_du):.2f}, "
          f"mean={np.nanmean(py_thresh_du):.2f}")
    print(f"  >= 25 (doublet): {np.nansum(py_thresh_du >= 25)}")

    # Confidence: min(singlet_ft - pair, singlet_st - pair)
    sing_ft_scores_du = r_ft_as_singlet_scores[r_du_mask]
    # Need singlet scores for R's second type too
    sing_st_scores_du = []
    for i, n in enumerate(du_idx):
        st_i = k_name_map.get(r_st_du[i], -1)
        if st_i < 0:
            sing_st_scores_du.append(np.nan)
            continue
        if (n, st_i) in singlet_log_l:
            sing_st_scores_du.append(singlet_log_l[(n, st_i)])
        else:
            nUMI_b = jnp.array(py_nUMI_m[[n]])
            B_b    = jnp.array(py_counts_m[[n]].astype(np.float64))
            P_sg   = P_gpu[:, st_i:st_i+1].T[..., None]
            S_b    = nUMI_b[:, None, None] * P_sg
            w_b, _ = solve_irwls_batch(
                S_batch=S_b, Y_batch=B_b, nUMI_batch=nUMI_b,
                Q_mat=Q_gpu, SQ_mat=SQ_gpu, x_vals=X_gpu,
                max_iter=50, min_change=0.001, constrain=False, bulk_mode=False,
            )
            expected = jnp.maximum(jnp.sum(S_b * w_b[:, None, :], axis=-1), 1e-4)
            sc = float(calc_log_likelihood(B_b[0], expected[0], Q_gpu, SQ_gpu, X_gpu, config.K_val))
            sing_st_scores_du.append(sc)

    sing_st_scores_du = np.array(sing_st_scores_du)

    py_conf_ft = sing_ft_scores_du - pair_scores_du
    py_conf_st = sing_st_scores_du - pair_scores_du
    py_conf_du = np.minimum(py_conf_ft, py_conf_st)

    print(f"\nPython doublet confidence (min of both types - pair) for R=doublet_uncertain:")
    print(f"  min={np.nanmin(py_conf_du):.2f}, max={np.nanmax(py_conf_du):.2f}, "
          f"mean={np.nanmean(py_conf_du):.2f}")
    print(f"  >= 5 (doublet_certain): {np.nansum(py_conf_du >= 5)}")
    print(f"  < 5 (doublet_uncertain): {np.nansum(py_conf_du < 5)}")
    print(f"  (R uses CONFIDENCE_THRESHOLD=5 for this)")

    # Comparison
    r_conf_info = pd.read_csv(VAL_DIR / "r_results_df_sample.csv", index_col=0)
    # R's min_score is the pair score, singlet_score is the singlet score.
    # R's confidence for doublet_uncertain = min(sing_ft - pair, sing_st - pair) < 5
    # We don't have R's per-type singlet scores directly, but r_sing_score is the
    # best SINGLET score (for best type). So R's confidence uses specific type scores.
    print(f"\n  Summary: Python confidence (min over both types) distribution:")
    bins = [0, 1, 2, 5, 10, 25, float('inf')]
    for lo, hi in zip(bins[:-1], bins[1:]):
        n = np.nansum((py_conf_du >= lo) & (py_conf_du < hi))
        print(f"    [{lo:.0f}, {hi:.0f}): {n}")

print(f"\n{'='*70}")
print("D. Check R source: what does constrain mean in IRWLS for doublet pairs?")
print(f"{'='*70}")
print("""
From R spacexr source (doublet_utils.R / process_beads.R):

In process_beads() for doublet mode:
  if(singlet) {
    # Singlet fit: constrain.sum = FALSE
    results = solveIRWLS.weights(S_all, Y, constrain.sum = FALSE, init = W_full)
  } else {
    # Doublet pair fit: constrain.sum = FALSE (NOT TRUE!)
    results = solveIRWLS.weights(S_all, Y, constrain.sum = FALSE, init = W_full)
  }

In solveIRWLS.weights(S_all, Y, constrain.sum, init):
  if(constrain.sum) {
    # Use quadprog with equality constraint sum(w) = 1
    solve.QP(D_mat, d_vec, Amat, bvec, meq=1)
  } else {
    # Box-constrained only: w_i >= -solution_i
    solve.QP(D_mat, d_vec, Amat, bvec, meq=0)
  }

Key: For doublet pair fitting, R uses constrain.sum = FALSE.
This is exactly what the unconstrained option does.

The doublet_uncertain class in R comes from a SEPARATE confidence calculation
that uses per-type singlet scores SPECIFICALLY FOR THE PAIR TYPES:
  conf_type1 = singlet_score(type1) - best_pair_score
  conf_type2 = singlet_score(type2) - best_pair_score
  doublet_conf = min(conf_type1, conf_type2)

This is exactly what Python implements. The difference in doublet_uncertain
frequency may come from the slightly different singlet scores (Python vs R).

Specifically for R's doublet_uncertain pixels:
  - R's doublet_thresh >= 25: pixel IS a doublet
  - R's doublet_conf < 5: at least one of the pair types has uncertain confidence

Python computes doublet_conf differently because the singlet scores differ.
""")
