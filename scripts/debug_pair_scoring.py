#!/usr/bin/env python3
"""
Investigate why Python's min_score (best pair NLL) is consistently higher
than R's min_score on the same pixels.

Key finding from previous investigation:
  - singlet_score (Python vs R): corr=0.982, mean diff=-1.5  (close)
  - min_score (Python vs R): corr=0.982, mean diff=+5.8  (Python HIGHER = worse pair fit)
  - doublet_thresh (Python vs R): corr=0.652, mean diff=-7.3  (Python SMALLER by ~7 NLL units)

For R doublet pixels: Python's m_score is systematically higher -> threshold too small -> classified as singlet.

Hypotheses:
  1. Python constrained pair solver (simplex) vs R?
  2. Candidate set differences: Python explores fewer/different pairs?
  3. R uses ALL pairwise combinations including types NOT in Python's candidate set?
  4. R's singlet scoring uses constrain=True (simplex) in doublet mode?
  5. Python uses max_iter=25 for pairs vs R's default?
  6. Python uses step_size=0.3 vs R?

We'll test:
  A. For pixels where Python=singlet but R=doublet: what pair does R use?
     Is that pair in Python's candidate set?
  B. For a specific pixel, what is the exact NLL of R's winning pair under
     Python's solver (to isolate solver vs candidate set)?
  C. Check if R evaluates ALL K*(K-1)/2 pairs instead of just candidates.
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
from rctd._likelihood import load_cached_q_matrices, compute_spline_coefficients, calc_log_likelihood
from rctd._full import run_full_mode
from rctd._irwls import solve_irwls_batch
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
r_full_weights = pd.read_csv(VAL_DIR / "r_pixel_weights_sample.csv", index_col=0)

all_barcodes = np.array(spatial.obs_names)
py_barcodes  = all_barcodes[rctd._pixel_mask]
py_bc_map    = {b: i for i, b in enumerate(py_barcodes)}

r_barcodes_sample = [l.strip() for l in open(VAL_DIR / "r_pixel_barcodes_sample.txt") if l.strip()]
matched_py, matched_r = [], []
for j, bc in enumerate(r_barcodes_sample):
    bc2 = bc.replace("Region_3_35844-2_", "")
    if bc2 in py_bc_map:
        matched_py.append(py_bc_map[bc2])
        matched_r.append(j)

matched_py = np.array(matched_py)
matched_r  = np.array(matched_r)
print(f"Matched {len(matched_py)} pixels")

py_counts_m = rctd.counts[matched_py]
py_nUMI_m   = rctd.nUMI[matched_py]
r_sc        = r_results["spot_class"].values[matched_r]
r_ft        = r_results["first_type"].values[matched_r]
r_st        = r_results["second_type"].values[matched_r]
r_min_score = r_results["min_score"].values[matched_r]
r_sing_score= r_results["singlet_score"].values[matched_r]

# R full-mode weights
r_fw_types = list(r_full_weights.columns)
r_fw_vals  = r_full_weights.values[matched_r]

# ─────────────────────────────────────────────────────────────────────────────
# A. For R=doublet pixels: is R's winning pair always in Python's candidates?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("A. R's winning pair vs Python candidate set")
print("="*70)

# Run Python full mode to get candidate set
print("Running Python full mode for candidates...")
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
for n in range(len(matched_py)):
    cands = np.where(W_full[n] > WEIGHT_THRESHOLD)[0].tolist()
    if len(cands) == 0:
        best = int(np.argmax(W_full[n]))
        cands = [best, (best+1)%K]
    elif len(cands) == 1:
        t = cands[0]
        cands = [t, (t+1)%K if t != 0 else 1]
    candidates_list.append(cands)

# R is_doublet pixels
r_is_doublet = (r_sc == "doublet_certain") | (r_sc == "doublet_uncertain")
r_doublet_idx = np.where(r_is_doublet)[0]
print(f"\nR doublet pixels: {r_is_doublet.sum()}")

r_pair_in_py_cands = 0
r_pair_partial = 0  # at least one type in
r_pair_neither = 0
for n in r_doublet_idx:
    ft_i = k_name_map.get(r_ft[n], -1)
    st_i = k_name_map.get(r_st[n], -1)
    cands = candidates_list[n]
    ft_in = ft_i in cands
    st_in = st_i in cands
    if ft_in and st_in:
        r_pair_in_py_cands += 1
    elif ft_in or st_in:
        r_pair_partial += 1
    else:
        r_pair_neither += 1

print(f"R winning pair fully in Python candidates: {r_pair_in_py_cands}/{r_is_doublet.sum()}")
print(f"R winning pair partially in Python candidates: {r_pair_partial}")
print(f"R winning pair not in Python candidates at all: {r_pair_neither}")

# More detailed: for pixels where pair IS in Python candidates, compare scores
print("\nFor pixels where R pair IS in Python candidates:")
print("Scoring R's exact pair with Python solver...")

r_pair_in_idx = []
r_pair_data = []
for n in r_doublet_idx:
    ft_i = k_name_map.get(r_ft[n], -1)
    st_i = k_name_map.get(r_st[n], -1)
    if ft_i in candidates_list[n] and st_i in candidates_list[n]:
        r_pair_in_idx.append(n)
        r_pair_data.append((n, ft_i, st_i))

print(f"  {len(r_pair_data)} pixels where R's pair is in Python candidates")

if r_pair_data:
    # Score R's exact pair with Python solver
    bs = 500
    r_pair_scores_py = np.zeros(len(r_pair_data))
    r_pair_weights_py = np.zeros((len(r_pair_data), 2))

    for start in range(0, len(r_pair_data), bs):
        end = min(start+bs, len(r_pair_data))
        batch = r_pair_data[start:end]
        pix = np.array([x[0] for x in batch])
        t1  = np.array([x[1] for x in batch])
        t2  = np.array([x[2] for x in batch])

        nUMI_b = jnp.array(py_nUMI_m[pix])
        B_b    = jnp.array(py_counts_m[pix].astype(np.float64))
        P1 = P_gpu[:, t1].T
        P2 = P_gpu[:, t2].T
        S_b = nUMI_b[:, None, None] * jnp.stack([P1, P2], axis=-1)

        w_b, _ = solve_irwls_batch(
            S_batch=S_b, Y_batch=B_b, nUMI_batch=nUMI_b,
            Q_mat=Q_gpu, SQ_mat=SQ_gpu, x_vals=X_gpu,
            max_iter=25, min_change=0.001, constrain=True, bulk_mode=False,
        )
        expected = jnp.maximum(jnp.sum(S_b * w_b[:, None, :], axis=-1), 1e-4)

        batched_nll = jax.vmap(
            lambda y, lam: calc_log_likelihood(y, lam, Q_gpu, SQ_gpu, X_gpu, config.K_val),
            in_axes=(0, 0),
        )
        scores = np.array(batched_nll(B_b, expected))
        r_pair_scores_py[start:end] = scores
        r_pair_weights_py[start:end] = np.array(w_b)

    # Compare Python's score for R's pair vs R's min_score
    r_min_in = r_min_score[np.array(r_pair_in_idx)]
    py_score_for_r_pair = r_pair_scores_py
    diff_pair = py_score_for_r_pair - r_min_in

    print(f"\n  Score for R's pair under Python solver vs R's min_score:")
    print(f"  Python score for R's pair: min={py_score_for_r_pair.min():.2f}, "
          f"max={py_score_for_r_pair.max():.2f}, mean={py_score_for_r_pair.mean():.2f}")
    print(f"  R's min_score:             min={r_min_in.min():.2f}, "
          f"max={r_min_in.max():.2f}, mean={r_min_in.mean():.2f}")
    print(f"  Difference (Python - R): min={diff_pair.min():.4f}, "
          f"max={diff_pair.max():.4f}, mean={diff_pair.mean():.4f}, "
          f"median={np.median(diff_pair):.4f}")

    # What fraction of these pixels would classify as doublet if Python
    # used the Python-scored version of R's pair?
    r_sing_in = r_sing_score[np.array(r_pair_in_idx)]
    py_thresh_with_r_pair = r_sing_in - py_score_for_r_pair
    py_thresh_python_pair = r_sing_score[np.array(r_pair_in_idx)] - r_min_score[np.array(r_pair_in_idx)]
    # oops, let's use Python singlet scores...
    # Actually we need Python singlet scores for these pixels too.
    # Use R's singlet_score as proxy for now (they correlate 0.982)
    print(f"\n  Using R's singlet_score as proxy:")
    print(f"  Python doublet_thresh (R singlet_score - Python pair score):")
    print(f"    min={py_thresh_with_r_pair.min():.2f}, max={py_thresh_with_r_pair.max():.2f}, "
          f"mean={py_thresh_with_r_pair.mean():.2f}")
    print(f"  Would classify as doublet (>= 25): "
          f"{(py_thresh_with_r_pair >= 25).sum()}/{len(py_thresh_with_r_pair)}")


# ─────────────────────────────────────────────────────────────────────────────
# B. Test with more IRWLS iterations for pair fitting
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("B. Impact of max_iter on pair scoring (25 vs 50 vs 100)")
print("="*70)

# Take 50 R-doublet pixels for testing
test_idx = r_doublet_idx[:50]

def score_pairs_with_iter(n_iter, test_indices, use_r_pairs=True):
    scores = []
    for n in test_indices:
        ft_i = k_name_map.get(r_ft[n], -1) if use_r_pairs else None
        st_i = k_name_map.get(r_st[n], -1) if use_r_pairs else None
        if ft_i is None or st_i is None or ft_i < 0 or st_i < 0:
            scores.append(np.nan)
            continue
        nUMI_b = jnp.array(py_nUMI_m[[n]])
        B_b    = jnp.array(py_counts_m[[n]].astype(np.float64))
        P1 = P_gpu[:, ft_i:ft_i+1].T
        P2 = P_gpu[:, st_i:st_i+1].T
        S_b = nUMI_b[:, None, None] * jnp.stack([P1, P2], axis=-1)

        w_b, _ = solve_irwls_batch(
            S_batch=S_b, Y_batch=B_b, nUMI_batch=nUMI_b,
            Q_mat=Q_gpu, SQ_mat=SQ_gpu, x_vals=X_gpu,
            max_iter=n_iter, min_change=0.001, constrain=True, bulk_mode=False,
        )
        expected = jnp.maximum(jnp.sum(S_b * w_b[:, None, :], axis=-1), 1e-4)
        sc = float(calc_log_likelihood(B_b[0], expected[0], Q_gpu, SQ_gpu, X_gpu, config.K_val))
        scores.append(sc)
    return np.array(scores)

print("Scoring R's pairs for 50 doublet pixels with different max_iter:")
sc25  = score_pairs_with_iter(25,  test_idx)
sc50  = score_pairs_with_iter(50,  test_idx)
sc100 = score_pairs_with_iter(100, test_idx)

r_min_test = r_min_score[test_idx]

valid = np.isfinite(sc25) & np.isfinite(sc50) & np.isfinite(sc100)
print(f"  max_iter=25:  mean score={sc25[valid].mean():.4f}")
print(f"  max_iter=50:  mean score={sc50[valid].mean():.4f}")
print(f"  max_iter=100: mean score={sc100[valid].mean():.4f}")
print(f"  R min_score:  mean={r_min_test[valid].mean():.4f}")
print(f"  Diff 25 vs R:  {(sc25[valid]-r_min_test[valid]).mean():.4f}")
print(f"  Diff 50 vs R:  {(sc50[valid]-r_min_test[valid]).mean():.4f}")
print(f"  Diff 100 vs R: {(sc100[valid]-r_min_test[valid]).mean():.4f}")
print(f"  Score change 25->50:  mean={((sc25-sc50)[valid]).mean():.4f}")
print(f"  Score change 50->100: mean={((sc50-sc100)[valid]).mean():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# C. Test with ALL pairs (not just candidates)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("C. Score of ALL K*(K-1)/2 pairs for select pixels")
print("="*70)

# Take 5 pixels where R=doublet_certain but Python=singlet
r_dc_mask = r_sc == "doublet_certain"
# Get Python's classification (need to run doublet mode)
# Use threshold approach: pixels where Python's best pair gives higher score
# For simplicity, take first 5 R=doublet_certain pixels
r_dc_idx = np.where(r_dc_mask)[0][:5]

print(f"\nFor 5 R=doublet_certain pixels, scoring ALL {K*(K-1)//2} pairs:")
for n in r_dc_idx:
    bc = r_barcodes_sample[matched_r[n]]
    r_ft_n = r_ft[n]
    r_st_n = r_st[n]
    r_ms_n = r_min_score[n]
    r_ss_n = r_sing_score[n]

    nUMI_b = jnp.array(py_nUMI_m[[n]])
    B_b    = jnp.array(py_counts_m[[n]].astype(np.float64))

    best_py_score = float('inf')
    best_py_t1 = best_py_t2 = 0

    # Score all pairs
    all_pair_scores = {}
    for t1, t2 in combinations(range(K), 2):
        P1 = P_gpu[:, t1:t1+1].T
        P2 = P_gpu[:, t2:t2+1].T
        S_b = nUMI_b[:, None, None] * jnp.stack([P1, P2], axis=-1)
        w_b, _ = solve_irwls_batch(
            S_batch=S_b, Y_batch=B_b, nUMI_batch=nUMI_b,
            Q_mat=Q_gpu, SQ_mat=SQ_gpu, x_vals=X_gpu,
            max_iter=25, min_change=0.001, constrain=True, bulk_mode=False,
        )
        expected = jnp.maximum(jnp.sum(S_b * w_b[:, None, :], axis=-1), 1e-4)
        sc = float(calc_log_likelihood(B_b[0], expected[0], Q_gpu, SQ_gpu, X_gpu, config.K_val))
        all_pair_scores[(t1, t2)] = sc
        if sc < best_py_score:
            best_py_score = sc
            best_py_t1, best_py_t2 = t1, t2

    # Get score for R's pair
    r_ft_i = k_name_map.get(r_ft_n, -1)
    r_st_i = k_name_map.get(r_st_n, -1)
    if r_ft_i >= 0 and r_st_i >= 0:
        key = (min(r_ft_i, r_st_i), max(r_ft_i, r_st_i))
        r_pair_py_score = all_pair_scores.get(key, float('inf'))
    else:
        r_pair_py_score = float('inf')

    # Cand pair score (what Python would use in doublet mode)
    cands = candidates_list[n]
    best_cand_score = float('inf')
    best_ct1, best_ct2 = -1, -1
    for t1, t2 in combinations(cands, 2):
        key = (min(t1,t2), max(t1,t2))
        sc = all_pair_scores.get(key, float('inf'))
        if sc < best_cand_score:
            best_cand_score = sc
            best_ct1, best_ct2 = t1, t2

    print(f"\n  Pixel {n} (barcode={bc}, nUMI={py_nUMI_m[n]:.0f}):")
    print(f"    R: class={r_sc[n]}, first={r_ft_n}, second={r_st_n}")
    print(f"    R: min_score={r_ms_n:.2f}, singlet_score={r_ss_n:.2f}, thresh={r_ss_n-r_ms_n:.2f}")
    print(f"    Python candidates: {[K_names[c] for c in cands]}")
    print(f"    Python best pair (ALL K^2): {K_names[best_py_t1]}+{K_names[best_py_t2]} "
          f"(score={best_py_score:.2f})")
    print(f"    Python best pair (candidates only): "
          f"{K_names[best_ct1]}+{K_names[best_ct2]} (score={best_cand_score:.2f})")
    print(f"    Python score for R's pair: {r_pair_py_score:.2f}")
    print(f"    R's min_score (pair NLL):  {r_ms_n:.2f}")
    print(f"    Diff (Python all-pairs vs R): {best_py_score - r_ms_n:.2f}")
    print(f"    Diff (Python cands vs R):    {best_cand_score - r_ms_n:.2f}")

    # Show top-5 pairs by Python score
    top5 = sorted(all_pair_scores.items(), key=lambda x: x[1])[:5]
    print(f"    Top-5 pairs by Python NLL:")
    for (t1, t2), sc in top5:
        print(f"      {K_names[t1]:<35} + {K_names[t2]:<35}: {sc:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# D. R's full-mode weights: compare to Python's for the doublet pixels
#    This tells us if R's full-mode fit gives different candidates
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("D. Full-mode weight comparison for R-doublet pixels")
print("="*70)

# r_full_weights has R's full-mode weights for the 1000 sample pixels
# Compare to Python's W_full for the same pixels
r_fw_aligned = np.zeros((len(matched_py), K))
for ti, ct in enumerate(K_names):
    if ct in r_fw_types:
        ri = r_fw_types.index(ct)
        r_fw_aligned[:, ti] = r_fw_vals[:, ri]

# For R doublet pixels
r_fw_doublet = r_fw_aligned[r_doublet_idx]
py_fw_doublet = W_full[r_doublet_idx]

# Are R's winning types the highest-weight types in R's full mode?
r_ft_idx = np.array([k_name_map.get(r_ft[n], -1) for n in r_doublet_idx])
r_st_idx = np.array([k_name_map.get(r_st[n], -1) for n in r_doublet_idx])

# Rank of R's first type in R's full-mode weights
r_ft_rank_in_r = np.array([
    int(np.sum(r_fw_doublet[i] > r_fw_doublet[i, r_ft_idx[i]])) + 1
    if r_ft_idx[i] >= 0 else 999
    for i in range(len(r_doublet_idx))
])
r_ft_rank_in_py = np.array([
    int(np.sum(py_fw_doublet[i] > py_fw_doublet[i, r_ft_idx[i]])) + 1
    if r_ft_idx[i] >= 0 else 999
    for i in range(len(r_doublet_idx))
])
r_st_rank_in_r = np.array([
    int(np.sum(r_fw_doublet[i] > r_fw_doublet[i, r_st_idx[i]])) + 1
    if r_st_idx[i] >= 0 else 999
    for i in range(len(r_doublet_idx))
])
r_st_rank_in_py = np.array([
    int(np.sum(py_fw_doublet[i] > py_fw_doublet[i, r_st_idx[i]])) + 1
    if r_st_idx[i] >= 0 else 999
    for i in range(len(r_doublet_idx))
])

print(f"\nRank of R's first_type in full-mode weights:")
print(f"  In R's weights: mean={r_ft_rank_in_r.mean():.2f}, "
      f"rank=1: {(r_ft_rank_in_r==1).sum()}, rank<=2: {(r_ft_rank_in_r<=2).sum()}")
print(f"  In Python's weights: mean={r_ft_rank_in_py.mean():.2f}, "
      f"rank=1: {(r_ft_rank_in_py==1).sum()}, rank<=2: {(r_ft_rank_in_py<=2).sum()}")

print(f"\nRank of R's second_type in full-mode weights:")
print(f"  In R's weights: mean={r_st_rank_in_r.mean():.2f}, "
      f"rank=1: {(r_st_rank_in_r==1).sum()}, rank<=2: {(r_st_rank_in_r<=2).sum()}")
print(f"  In Python's weights: mean={r_st_rank_in_py.mean():.2f}, "
      f"rank=1: {(r_st_rank_in_py==1).sum()}, rank<=2: {(r_st_rank_in_py<=2).sum()}")

print(f"\nFor R doublet pixels: R's first_type weight in Python full mode (W_full):")
r_ft_w_in_py = np.array([py_fw_doublet[i, r_ft_idx[i]] if r_ft_idx[i] >= 0 else np.nan
                          for i in range(len(r_doublet_idx))])
print(f"  min={np.nanmin(r_ft_w_in_py):.4f}, max={np.nanmax(r_ft_w_in_py):.4f}, "
      f"mean={np.nanmean(r_ft_w_in_py):.4f}")
print(f"  > 0.01 (candidate threshold): {(r_ft_w_in_py > 0.01).sum()}")
print(f"  > 0.0 (positive): {(r_ft_w_in_py > 0.0).sum()}")

r_st_w_in_py = np.array([py_fw_doublet[i, r_st_idx[i]] if r_st_idx[i] >= 0 else np.nan
                          for i in range(len(r_doublet_idx))])
print(f"\nFor R doublet pixels: R's second_type weight in Python full mode:")
print(f"  min={np.nanmin(r_st_w_in_py):.4f}, max={np.nanmax(r_st_w_in_py):.4f}, "
      f"mean={np.nanmean(r_st_w_in_py):.4f}")
print(f"  > 0.01 (candidate threshold): {(r_st_w_in_py > 0.01).sum()}")

# Are the types missing from Python candidates?
r_ft_not_in_cands = 0
r_st_not_in_cands = 0
for i, n in enumerate(r_doublet_idx):
    if r_ft_idx[i] >= 0 and r_ft_idx[i] not in candidates_list[n]:
        r_ft_not_in_cands += 1
    if r_st_idx[i] >= 0 and r_st_idx[i] not in candidates_list[n]:
        r_st_not_in_cands += 1

print(f"\nR's first_type not in Python candidates: {r_ft_not_in_cands}/{len(r_doublet_idx)}")
print(f"R's second_type not in Python candidates: {r_st_not_in_cands}/{len(r_doublet_idx)}")

# ─────────────────────────────────────────────────────────────────────────────
# E. Compare Python vs R full-mode weights at the pixel level
#    Focus on the WEIGHT_THRESHOLD = 0.01 boundary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("E. Full-mode weight detailed comparison (all 1000 pixels)")
print("="*70)

# Per-type comparison
print(f"\nR full-mode weights: shape {r_fw_aligned.shape}")
print(f"Python full-mode weights: shape {W_full.shape}")

# Are Python weights close to R weights?
diff_fw = W_full - r_fw_aligned
abs_diff_fw = np.abs(diff_fw)
print(f"Element-wise absolute difference:")
print(f"  max={abs_diff_fw.max():.4f}, mean={abs_diff_fw.mean():.6f}, "
      f"median={np.median(abs_diff_fw):.6f}")

# Dominant type in full mode
r_dom_full = np.argmax(r_fw_aligned, axis=1)
py_dom_full = np.argmax(W_full, axis=1)
dom_agree_full = (r_dom_full == py_dom_full).mean()
print(f"\nFull-mode dominant type agreement: {dom_agree_full:.4f} ({dom_agree_full*100:.1f}%)")

# Number of types above WEIGHT_THRESHOLD
n_cands_r  = (r_fw_aligned > WEIGHT_THRESHOLD).sum(axis=1)
n_cands_py = (W_full > WEIGHT_THRESHOLD).sum(axis=1)
print(f"\nNumber of types above {WEIGHT_THRESHOLD} threshold:")
print(f"  R: mean={n_cands_r.mean():.2f}, median={np.median(n_cands_r):.2f}")
print(f"  Python: mean={n_cands_py.mean():.2f}, median={np.median(n_cands_py):.2f}")

# For R doublet pixels:
print(f"\nFor R doublet pixels ({r_is_doublet.sum()} pixels):")
print(f"  R n_cands: mean={n_cands_r[r_doublet_idx].mean():.2f}, "
      f"median={np.median(n_cands_r[r_doublet_idx]):.2f}")
print(f"  Python n_cands: mean={n_cands_py[r_doublet_idx].mean():.2f}, "
      f"median={np.median(n_cands_py[r_doublet_idx]):.2f}")

# Check: does R's full mode weight for second type (st) exceed 0.01?
r_st_w_in_r = np.array([r_fw_aligned[n, r_st_idx[i]] if r_st_idx[i] >= 0 else np.nan
                          for i, n in enumerate(r_doublet_idx)])
print(f"\nR's second_type weight in R's full mode (doublet pixels):")
print(f"  min={np.nanmin(r_st_w_in_r):.4f}, max={np.nanmax(r_st_w_in_r):.4f}, "
      f"mean={np.nanmean(r_st_w_in_r):.4f}")
print(f"  > 0.01: {(r_st_w_in_r > 0.01).sum()}")
print(f"  <= 0.01 but > 0.0: {((r_st_w_in_r <= 0.01) & (r_st_w_in_r > 0.0)).sum()}")
print(f"  <= 0.0: {(r_st_w_in_r <= 0.0).sum()}")

print("\n" + "="*70)
print("PAIR SCORING INVESTIGATION COMPLETE")
print("="*70)
