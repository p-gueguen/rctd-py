#!/usr/bin/env python3
"""
Deep dive into doublet classification disagreement.
Focus:
  A. Why does Python produce zero doublet_uncertain?
  B. Score distributions: CONFIDENCE_THRESHOLD and DOUBLET_THRESHOLD
  C. Candidate selection differences (full-mode weight threshold 0.01)
  D. Singlet scoring on sample pixels – R vs Python score comparison
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

VAL_DIR = Path("/home/pgueguen/git/rctd-py/data/xenium_validation")
REF_DIR = Path("/home/pgueguen/git/rctd-py/data")

# ─── Reload everything efficiently ────────────────────────────────────────────
from rctd import Reference, RCTDConfig
from rctd._rctd import RCTD
from rctd._likelihood import load_cached_q_matrices, compute_spline_coefficients
from rctd._full import run_full_mode
from rctd._doublet import run_doublet_mode
from rctd._types import SPOT_CLASS_NAMES, RCTDConfig
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
print(f"sigma={sigma}, cell_types={reference.cell_type_names[:3]}...")
print(f"norm_profiles shape: {rctd.norm_profiles.shape}")

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

# ─── Load R sample results ────────────────────────────────────────────────────
r_results = pd.read_csv(VAL_DIR / "r_results_df_sample.csv", index_col=0)
r_weights_sample = pd.read_csv(VAL_DIR / "r_pixel_weights_sample.csv", index_col=0)
r_barcodes_sample = [l.strip() for l in open(VAL_DIR / "r_pixel_barcodes_sample.txt") if l.strip()]

all_barcodes = np.array(spatial.obs_names)
py_barcodes  = all_barcodes[rctd._pixel_mask]
py_bc_map    = {b: i for i, b in enumerate(py_barcodes)}

matched_py = []
matched_r  = []
for j, bc in enumerate(r_barcodes_sample):
    bc2 = bc.replace("Region_3_35844-2_", "")
    if bc2 in py_bc_map:
        matched_py.append(py_bc_map[bc2])
        matched_r.append(j)

matched_py = np.array(matched_py)
matched_r  = np.array(matched_r)
print(f"Matched {len(matched_py)} sample pixels")

# ─────────────────────────────────────────────────────────────────────────────
# A. Spot class distribution deep dive
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("A. Spot class distribution analysis")
print("="*70)

print(f"\nConfig CONFIDENCE_THRESHOLD: {config.CONFIDENCE_THRESHOLD}")
print(f"Config DOUBLET_THRESHOLD:    {config.DOUBLET_THRESHOLD}")

# R reference:
# CONFIDENCE_THRESHOLD is the threshold for is_singlet decision:
#   is_singlet = (s_score - m_score) < DOUBLET_THRESHOLD
# CONFIDENCE_THRESHOLD is used for doublet_certain vs doublet_uncertain:
#   doublet_conf = min(sing_first - min_p_score, sing_second - min_p_score)
#   if doublet_conf >= CONFIDENCE_THRESHOLD -> doublet_certain, else doublet_uncertain
# Also for singlets: confidence = (second_singlet_score - best_singlet_score)
#   if confidence < CONFIDENCE_THRESHOLD -> reject, else singlet

# The R spacexr source uses:
# CONFIDENCE_THRESHOLD = 5 (from R defaults)
# DOUBLET_THRESHOLD    = 25
# These match our config.
print("\nR defaults match Python? CONFIDENCE=5.0, DOUBLET=25.0: checking R code comments")
print(f"  Python CONFIDENCE_THRESHOLD={config.CONFIDENCE_THRESHOLD}, DOUBLET_THRESHOLD={config.DOUBLET_THRESHOLD}")

# Let's manually run the doublet classification and inspect score distributions
# for the 1000 sample pixels

reg_gene_names = rctd._gene_list_reg
# After fit_platform_effects(), rctd.counts is already restricted to reg genes
# rctd.common_genes is the FULL common gene list (before restriction); rctd.counts
# has been sliced to reg genes only. So we use rctd.counts directly.
py_counts_m = rctd.counts[matched_py]   # already reg-gene-restricted
py_nUMI_m   = rctd.nUMI[matched_py]

print(f"\nRunning manual doublet scoring on {len(matched_py)} pixels for inspection...")

# Step 1: Full fit
full_res = run_full_mode(
    spatial_counts=py_counts_m.astype(np.float64),
    spatial_numi=py_nUMI_m.astype(np.float64),
    norm_profiles=rctd.norm_profiles.astype(np.float64),
    cell_type_names=reference.cell_type_names,
    q_mat=q_mat, sq_mat=sq_mat, x_vals=x_vals,
    batch_size=2000,
)
W_full = full_res.weights  # (N, K)

# Step 2: Candidate selection
N = len(matched_py)
K = len(reference.cell_type_names)
WEIGHT_THRESHOLD = 0.01
candidates_list = []
n_cands = []
for n in range(N):
    cands = np.where(W_full[n] > WEIGHT_THRESHOLD)[0].tolist()
    if len(cands) == 0:
        best = int(np.argmax(W_full[n]))
        alt  = (best + 1) % K
        cands = [best, alt]
    elif len(cands) == 1:
        t   = cands[0]
        alt = (t + 1) % K if t != 0 else 1
        cands = [t, alt]
    candidates_list.append(cands)
    n_cands.append(len(cands))

n_cands = np.array(n_cands)
print(f"\nCandidate count distribution:")
for v, c in sorted(Counter(n_cands).items()):
    print(f"  {v} candidates: {c} pixels")

# Compare candidate cell types to R's first_type/second_type
r_ft = r_results["first_type"].values[matched_r]
r_st = r_results["second_type"].values[matched_r]
r_sc = r_results["spot_class"].values[matched_r]

# Check if R's first/second types are in Python's candidates
K_names = reference.cell_type_names
k_name_map = {n: i for i, n in enumerate(K_names)}

r_ft_in_py_cands = 0
r_st_in_py_cands = 0
for n in range(N):
    ft_idx = k_name_map.get(r_ft[n], -1)
    st_idx = k_name_map.get(r_st[n], -1)
    if ft_idx in candidates_list[n]:
        r_ft_in_py_cands += 1
    if st_idx in candidates_list[n]:
        r_st_in_py_cands += 1

print(f"\nR first_type in Python candidates: {r_ft_in_py_cands}/{N} ({r_ft_in_py_cands/N*100:.1f}%)")
print(f"R second_type in Python candidates: {r_st_in_py_cands}/{N} ({r_st_in_py_cands/N*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# B. Full doublet mode on sample pixels with score inspection
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("B. Score distribution inspection")
print("="*70)

# Run full doublet mode and collect scores
# We'll do a manual reconstruction of doublet scoring for inspection
from rctd._irwls import solve_irwls_batch
from rctd._likelihood import calc_log_likelihood
import jax

P_gpu  = jnp.array(rctd.norm_profiles.astype(np.float64))
Q_gpu  = jnp.array(q_mat)
SQ_gpu = jnp.array(sq_mat)
X_gpu  = jnp.array(x_vals)

# Pairs scoring
from itertools import combinations
triples = []
for n, cands in enumerate(candidates_list):
    for t1, t2 in combinations(cands, 2):
        triples.append((n, t1, t2))

pair_log_l = {}
pair_weights_d = {}

triples_arr = np.array(triples, dtype=np.int32)
M = len(triples_arr)
bs_size = 2000

print(f"Scoring {len(triples)} pairs...")
for start in range(0, M, bs_size):
    end = min(start + bs_size, M)
    tr = triples_arr[start:end]
    pix_idx = tr[:, 0]
    t1_idx  = tr[:, 1]
    t2_idx  = tr[:, 2]

    nUMI_tr = jnp.array(py_nUMI_m[pix_idx])
    B_tr    = jnp.array(py_counts_m[pix_idx].astype(np.float64))
    P1 = P_gpu[:, t1_idx].T
    P2 = P_gpu[:, t2_idx].T
    P_pair = jnp.stack([P1, P2], axis=-1)
    S_pair = nUMI_tr[:, None, None] * P_pair

    weights_batch, _ = solve_irwls_batch(
        S_batch=S_pair, Y_batch=B_tr, nUMI_batch=nUMI_tr,
        Q_mat=Q_gpu, SQ_mat=SQ_gpu, x_vals=X_gpu,
        max_iter=25, min_change=0.001, constrain=True, bulk_mode=False,
    )
    expected = jnp.sum(S_pair * weights_batch[:, None, :], axis=-1)
    expected = jnp.maximum(expected, 1e-4)

    batched_nll = jax.vmap(
        lambda y, lam: calc_log_likelihood(y, lam, Q_gpu, SQ_gpu, X_gpu, config.K_val),
        in_axes=(0, 0),
    )
    scores = batched_nll(B_tr, expected)

    w_np = np.array(weights_batch)
    sc_np = np.array(scores)
    for i in range(len(tr)):
        n_idx, t1, t2 = triples[start + i]
        pair_log_l[(n_idx, t1, t2)] = sc_np[i]
        pair_weights_d[(n_idx, t1, t2)] = w_np[i]

# Singlet scoring
singles = list({(n, t) for n, cands in enumerate(candidates_list) for t in cands})
singlet_log_l = {}

singles_arr = np.array(singles, dtype=np.int32)
S_total = len(singles_arr)
print(f"Scoring {S_total} singlets...")
for start in range(0, S_total, bs_size):
    end = min(start + bs_size, S_total)
    sg = singles_arr[start:end]
    pix_idx = sg[:, 0]
    t_idx   = sg[:, 1]
    nUMI_sg = jnp.array(py_nUMI_m[pix_idx])
    B_sg    = jnp.array(py_counts_m[pix_idx].astype(np.float64))
    P_sg    = P_gpu[:, t_idx].T[..., None]
    S_sg    = nUMI_sg[:, None, None] * P_sg

    weights_batch, _ = solve_irwls_batch(
        S_batch=S_sg, Y_batch=B_sg, nUMI_batch=nUMI_sg,
        Q_mat=Q_gpu, SQ_mat=SQ_gpu, x_vals=X_gpu,
        max_iter=50, min_change=0.001, constrain=False, bulk_mode=False,
    )
    expected_sg = jnp.sum(S_sg * weights_batch[:, None, :], axis=-1)
    expected_sg = jnp.maximum(expected_sg, 1e-4)

    batched_nll = jax.vmap(
        lambda y, lam: calc_log_likelihood(y, lam, Q_gpu, SQ_gpu, X_gpu, config.K_val),
        in_axes=(0, 0),
    )
    scores = batched_nll(B_sg, expected_sg)
    sc_np = np.array(scores)
    for i in range(len(sg)):
        n_idx, t = singles[start + i]
        singlet_log_l[(n_idx, t)] = sc_np[i]

# ── Now reconstruct classification and collect per-pixel scores ──
INF = 1e18
s_scores_all = []
m_scores_all = []
doublet_thresh_vals = []
confidence_vals = []
py_classes = []
py_ft_names = []

for n in range(N):
    cands = candidates_list[n]
    C = len(cands)

    sing_scores = np.array([singlet_log_l.get((n, t), INF) for t in cands])
    best_si = int(np.argmin(sing_scores))
    best_singlet_type = cands[best_si]
    s_score = sing_scores[best_si]

    min_p_score = INF
    best_t1, best_t2 = cands[0], cands[1] if C > 1 else (cands[0], (cands[0]+1)%K)
    for i in range(C):
        for j in range(i+1, C):
            t1, t2 = cands[i], cands[j]
            sc = pair_log_l.get((n, t1, t2), INF)
            if sc < min_p_score:
                min_p_score = sc
                best_t1, best_t2 = t1, t2

    m_score = min(min_p_score, s_score)
    is_singlet = (s_score - m_score) < config.DOUBLET_THRESHOLD
    doublet_thresh_vals.append(s_score - m_score)

    s_scores_all.append(s_score)
    m_scores_all.append(m_score)

    if is_singlet:
        sorted_idx = np.argsort(sing_scores)
        confidence = (sing_scores[sorted_idx[1]] - s_score) if C > 1 else INF
        confidence_vals.append(confidence)
        s_class = "singlet" if confidence >= config.CONFIDENCE_THRESHOLD else "reject"
        py_ft_names.append(K_names[best_singlet_type])
    else:
        sing_first  = singlet_log_l.get((n, best_t1), INF)
        sing_second = singlet_log_l.get((n, best_t2), INF)
        doublet_conf = min(sing_first - min_p_score, sing_second - min_p_score)
        confidence_vals.append(doublet_conf)
        s_class = "doublet_certain" if doublet_conf >= config.CONFIDENCE_THRESHOLD else "doublet_uncertain"
        py_ft_names.append(K_names[best_t1])
    py_classes.append(s_class)

s_scores_all = np.array(s_scores_all)
m_scores_all = np.array(m_scores_all)
doublet_thresh_vals = np.array(doublet_thresh_vals)
confidence_vals = np.array(confidence_vals)
py_classes = np.array(py_classes)
py_ft_names = np.array(py_ft_names)

print(f"\nPython spot_class distribution (manual): {dict(Counter(py_classes))}")
print(f"R spot_class distribution: {dict(Counter(r_sc))}")

# The issue: Python has 0 doublet_uncertain. Why?
is_doublet = doublet_thresh_vals >= config.DOUBLET_THRESHOLD  # True = doublet
print(f"\nPixels classified as doublet (doublet_thresh >= {config.DOUBLET_THRESHOLD}): "
      f"{is_doublet.sum()} ({is_doublet.mean()*100:.1f}%)")
print(f"R doublets (certain+uncertain): {(r_sc == 'doublet_certain').sum() + (r_sc == 'doublet_uncertain').sum()}")

print(f"\nFor doublet pixels (Python), confidence distribution:")
doublet_conf_vals = confidence_vals[is_doublet]
if len(doublet_conf_vals) > 0:
    print(f"  n={len(doublet_conf_vals)}")
    print(f"  min={doublet_conf_vals.min():.2f}, max={doublet_conf_vals.max():.2f}")
    print(f"  mean={doublet_conf_vals.mean():.2f}, median={np.median(doublet_conf_vals):.2f}")
    print(f"  >= CONFIDENCE_THRESHOLD ({config.CONFIDENCE_THRESHOLD}): "
          f"{(doublet_conf_vals >= config.CONFIDENCE_THRESHOLD).sum()} (doublet_certain)")
    print(f"  < CONFIDENCE_THRESHOLD: "
          f"{(doublet_conf_vals < config.CONFIDENCE_THRESHOLD).sum()} (doublet_uncertain)")

print(f"\nFor singlet pixels (Python), confidence distribution:")
sing_mask = ~is_doublet
sing_conf_vals = confidence_vals[sing_mask]
print(f"  n={len(sing_conf_vals)}")
print(f"  min={sing_conf_vals.min():.2f}, max={sing_conf_vals.max():.2f}")
print(f"  mean={sing_conf_vals.mean():.2f}, median={np.median(sing_conf_vals):.2f}")
print(f"  >= CONFIDENCE_THRESHOLD ({config.CONFIDENCE_THRESHOLD}): "
      f"{(sing_conf_vals >= config.CONFIDENCE_THRESHOLD).sum()} (singlet)")
print(f"  < CONFIDENCE_THRESHOLD: "
      f"{(sing_conf_vals < config.CONFIDENCE_THRESHOLD).sum()} (reject)")

# ─────────────────────────────────────────────────────────────────────────────
# C. Compare doublet_thresh_vals (s_score - m_score) with R's min_score / singlet_score
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("C. Score comparison: Python vs R scores on sample pixels")
print("="*70)

r_min_score     = r_results["min_score"].values[matched_r]
r_singlet_score = r_results["singlet_score"].values[matched_r]

py_doublet_thresh_vals = doublet_thresh_vals  # s_score - m_score
py_s_scores = s_scores_all
py_m_scores = m_scores_all

# R: singlet_score = s_score (best singlet NLL)
# R: min_score = m_score (min of pair and singlet NLL)
# R: doublet_thresh = singlet_score - min_score

r_thresh = r_singlet_score - r_min_score

print(f"\nPython singlet_score (s_score):")
print(f"  min={py_s_scores.min():.2f}, max={py_s_scores.max():.2f}, "
      f"median={np.median(py_s_scores):.2f}")
print(f"R singlet_score:")
print(f"  min={r_singlet_score.min():.2f}, max={r_singlet_score.max():.2f}, "
      f"median={np.median(r_singlet_score):.2f}")

print(f"\nPython m_score (min of pair/singlet):")
print(f"  min={py_m_scores.min():.2f}, max={py_m_scores.max():.2f}, "
      f"median={np.median(py_m_scores):.2f}")
print(f"R min_score:")
print(f"  min={r_min_score.min():.2f}, max={r_min_score.max():.2f}, "
      f"median={np.median(r_min_score):.2f}")

print(f"\nPython doublet_thresh = s_score - m_score:")
print(f"  min={py_doublet_thresh_vals.min():.4f}, max={py_doublet_thresh_vals.max():.4f}")
print(f"  mean={py_doublet_thresh_vals.mean():.4f}, median={np.median(py_doublet_thresh_vals):.4f}")
print(f"  >= DOUBLET_THRESHOLD(25): "
      f"{(py_doublet_thresh_vals >= config.DOUBLET_THRESHOLD).sum()}")
print(f"R doublet_thresh = singlet_score - min_score:")
print(f"  min={r_thresh.min():.4f}, max={r_thresh.max():.4f}")
print(f"  mean={r_thresh.mean():.4f}, median={np.median(r_thresh):.4f}")
print(f"  >= DOUBLET_THRESHOLD(25): {(r_thresh >= config.DOUBLET_THRESHOLD).sum()}")

# Direct comparison pixel-by-pixel
diff_s = py_s_scores - r_singlet_score
diff_m = py_m_scores - r_min_score
diff_thresh = py_doublet_thresh_vals - r_thresh

print(f"\nDifferences (Python - R):")
print(f"  singlet_score: mean={diff_s.mean():.4f}, std={diff_s.std():.4f}, "
      f"max|diff|={np.abs(diff_s).max():.4f}")
print(f"  min_score:     mean={diff_m.mean():.4f}, std={diff_m.std():.4f}, "
      f"max|diff|={np.abs(diff_m).max():.4f}")
print(f"  doublet_thresh: mean={diff_thresh.mean():.4f}, std={diff_thresh.std():.4f}, "
      f"max|diff|={np.abs(diff_thresh).max():.4f}")

# Correlation
from numpy.ma import masked_invalid
def safe_corr(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() > 1:
        return np.corrcoef(a[mask], b[mask])[0, 1]
    return float('nan')

print(f"\nCorrelation (Python vs R):")
print(f"  singlet_score: r={safe_corr(py_s_scores, r_singlet_score):.6f}")
print(f"  min_score:     r={safe_corr(py_m_scores, r_min_score):.6f}")
print(f"  doublet_thresh: r={safe_corr(py_doublet_thresh_vals, r_thresh):.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# D. "doublet_uncertain" is 0 in Python - investigate CONFIDENCE_THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("D. doublet_uncertain analysis: why Python has none")
print("="*70)

# Python has 0 doublet_uncertain. This means: for all pixels Python classifies
# as doublet, doublet_conf >= 5.0. Let's check if the conf values are all very high.
print(f"\nPython doublet confidence values (for {is_doublet.sum()} doublet pixels):")
if is_doublet.sum() > 0:
    dc = doublet_conf_vals
    bins = [0, 1, 2, 5, 10, 25, 50, 100, 500, float('inf')]
    for lo, hi in zip(bins[:-1], bins[1:]):
        n = ((dc >= lo) & (dc < hi)).sum()
        print(f"  [{lo:.0f}, {hi:.0f}): {n}")

print(f"\nR doublet_uncertain pixels: {(r_sc == 'doublet_uncertain').sum()}")
print(f"For R doublet_uncertain pixels, what is Python's classification?")
r_du_mask = r_sc == "doublet_uncertain"
if r_du_mask.sum() > 0:
    py_class_for_r_du = py_classes[r_du_mask]
    print(f"  Python classes for R doublet_uncertain: {dict(Counter(py_class_for_r_du))}")
    py_thresh_for_r_du = py_doublet_thresh_vals[r_du_mask]
    py_conf_for_r_du = confidence_vals[r_du_mask]
    r_thresh_for_r_du = r_thresh[r_du_mask]
    print(f"  Python doublet_thresh: min={py_thresh_for_r_du.min():.2f}, "
          f"max={py_thresh_for_r_du.max():.2f}, mean={py_thresh_for_r_du.mean():.2f}")
    print(f"  R doublet_thresh: min={r_thresh_for_r_du.min():.2f}, "
          f"max={r_thresh_for_r_du.max():.2f}, mean={r_thresh_for_r_du.mean():.2f}")
    print(f"  Python confidence: min={py_conf_for_r_du.min():.2f}, "
          f"max={py_conf_for_r_du.max():.2f}, mean={py_conf_for_r_du.mean():.2f}")

print(f"\nR doublet_certain pixels: what is Python's classification?")
r_dc_mask = r_sc == "doublet_certain"
if r_dc_mask.sum() > 0:
    py_class_for_r_dc = py_classes[r_dc_mask]
    print(f"  Python classes for R doublet_certain: {dict(Counter(py_class_for_r_dc))}")

# ─────────────────────────────────────────────────────────────────────────────
# E. Look at pixels where Python=singlet, R=doublet (the main classification diff)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("E. Pixels where Python=singlet/reject, R=doublet_certain/uncertain")
print("="*70)

r_is_doublet = (r_sc == "doublet_certain") | (r_sc == "doublet_uncertain")
py_is_singlet = (py_classes == "singlet") | (py_classes == "reject")

disagreed = r_is_doublet & py_is_singlet
print(f"R doublet but Python singlet/reject: {disagreed.sum()}")

if disagreed.sum() > 0:
    # Inspect scores for these
    py_thresh_dis = py_doublet_thresh_vals[disagreed]
    r_thresh_dis  = r_thresh[disagreed]
    print(f"\n  Python doublet_thresh (s - m_score):")
    print(f"    min={py_thresh_dis.min():.2f}, max={py_thresh_dis.max():.2f}, "
          f"mean={py_thresh_dis.mean():.2f}")
    print(f"  R doublet_thresh (singlet - min_score):")
    print(f"    min={r_thresh_dis.min():.2f}, max={r_thresh_dis.max():.2f}, "
          f"mean={r_thresh_dis.mean():.2f}")

    # For these pixels: are R's types in Python's candidates?
    r_ft_dis = r_ft[disagreed]
    r_st_dis = r_st[disagreed]
    for_idx = np.where(disagreed)[0]
    r_ft_in = sum(1 for n in for_idx if k_name_map.get(r_ft[n], -1) in candidates_list[n])
    r_st_in = sum(1 for n in for_idx if k_name_map.get(r_st[n], -1) in candidates_list[n])
    print(f"\n  R first_type in Python candidates: {r_ft_in}/{disagreed.sum()}")
    print(f"  R second_type in Python candidates: {r_st_in}/{disagreed.sum()}")

    # nUMI distribution for disagreed pixels
    nUMI_dis = py_nUMI_m[disagreed]
    print(f"\n  nUMI for disagree pixels: min={nUMI_dis.min():.0f}, "
          f"max={nUMI_dis.max():.0f}, median={np.median(nUMI_dis):.0f}")

    # Check singlet score diff for disagreed pixels
    diff_s_dis = diff_s[disagreed]
    diff_m_dis = diff_m[disagreed]
    diff_thresh_dis = diff_thresh[disagreed]
    print(f"\n  Score diffs (Python-R) for disagreed pixels:")
    print(f"    singlet_score diff: mean={diff_s_dis.mean():.2f}, max|diff|={np.abs(diff_s_dis).max():.2f}")
    print(f"    min_score diff:     mean={diff_m_dis.mean():.2f}, max|diff|={np.abs(diff_m_dis).max():.2f}")
    print(f"    thresh diff:        mean={diff_thresh_dis.mean():.2f}, max|diff|={np.abs(diff_thresh_dis).max():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# F. Detailed inspection: R doublet_uncertain pixels
#    These are pixels where (1) R says doublet wins, but (2) doublet conf < 5
#    Python has 0 of these - why?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("F. Why does Python have 0 doublet_uncertain? Score analysis")
print("="*70)

# For pixels where R=doublet_uncertain: R says doublet wins (thresh>=25),
# but per-type confidence < 5. Let's see what Python computes for same pixels.
if r_du_mask.sum() > 0:
    print(f"\nR doublet_uncertain pixels ({r_du_mask.sum()} total):")
    r_sc_du = r_sc[r_du_mask]
    r_ft_du = r_ft[r_du_mask]
    r_st_du = r_st[r_du_mask]
    r_ms_du = r_min_score[r_du_mask]
    r_ss_du = r_singlet_score[r_du_mask]
    r_thr_du = r_thresh[r_du_mask]

    py_sc_du  = py_classes[r_du_mask]
    py_ft_du  = py_ft_names[r_du_mask]
    py_thr_du = py_doublet_thresh_vals[r_du_mask]
    py_ms_du  = py_m_scores[r_du_mask]
    py_ss_du  = py_s_scores[r_du_mask]
    py_conf_du = confidence_vals[r_du_mask]

    print(f"  Python spot_class: {dict(Counter(py_sc_du))}")
    print(f"\n  R thresholds (singlet_score - min_score):")
    print(f"    min={r_thr_du.min():.2f}, max={r_thr_du.max():.2f}, median={np.median(r_thr_du):.2f}")
    print(f"  Python thresholds:")
    print(f"    min={py_thr_du.min():.2f}, max={py_thr_du.max():.2f}, median={np.median(py_thr_du):.2f}")

    print(f"\n  R min_score: min={r_ms_du.min():.2f}, max={r_ms_du.max():.2f}, "
          f"median={np.median(r_ms_du):.2f}")
    print(f"  Python m_score: min={py_ms_du.min():.2f}, max={py_ms_du.max():.2f}, "
          f"median={np.median(py_ms_du):.2f}")

    print(f"\n  R singlet_score: min={r_ss_du.min():.2f}, max={r_ss_du.max():.2f}, "
          f"median={np.median(r_ss_du):.2f}")
    print(f"  Python s_score: min={py_ss_du.min():.2f}, max={py_ss_du.max():.2f}, "
          f"median={np.median(py_ss_du):.2f}")

    # Sample 5 specific pixels
    print(f"\n  First 5 R=doublet_uncertain pixels:")
    du_indices = np.where(r_du_mask)[0][:5]
    for i in du_indices:
        print(f"    Pixel {i}: R=[class={r_sc[i]}, ft={r_ft[i]}, st={r_st[i]}, "
              f"thresh={r_thresh[i]:.2f}, min_score={r_min_score[i]:.2f}, singlet_score={r_singlet_score[i]:.2f}]")
        print(f"             Py=[class={py_classes[i]}, ft={py_ft_names[i]}, "
              f"thresh={py_doublet_thresh_vals[i]:.2f}, m_score={py_m_scores[i]:.2f}, "
              f"s_score={py_s_scores[i]:.2f}, conf={confidence_vals[i]:.2f}]")
        print(f"             candidates: {[K_names[c] for c in candidates_list[i]]}")

# ─────────────────────────────────────────────────────────────────────────────
# G. What is the Python doublet_thresh distribution vs R's?
#    Key question: are Python scores on a different scale?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("G. Overall score scale comparison (all 1000 pixels)")
print("="*70)

print(f"\nPython singlet_score distribution:")
bins = [0, 10, 25, 50, 100, 250, 500, 1000, 5000, float('inf')]
for lo, hi in zip(bins[:-1], bins[1:]):
    n = ((py_s_scores >= lo) & (py_s_scores < hi)).sum()
    print(f"  [{lo:.0f}, {hi:.0f}): {n}")

print(f"\nR singlet_score distribution:")
for lo, hi in zip(bins[:-1], bins[1:]):
    n = ((r_singlet_score >= lo) & (r_singlet_score < hi)).sum()
    print(f"  [{lo:.0f}, {hi:.0f}): {n}")

print(f"\nPython doublet_thresh distribution:")
for lo, hi in zip(bins[:-1], bins[1:]):
    n = ((py_doublet_thresh_vals >= lo) & (py_doublet_thresh_vals < hi)).sum()
    print(f"  [{lo:.0f}, {hi:.0f}): {n}")

print(f"\nR doublet_thresh (singlet_score - min_score) distribution:")
for lo, hi in zip(bins[:-1], bins[1:]):
    n = ((r_thresh >= lo) & (r_thresh < hi)).sum()
    print(f"  [{lo:.0f}, {hi:.0f}): {n}")

# ─────────────────────────────────────────────────────────────────────────────
# H. nUMI scaling of scores
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("H. Score scaling: per-gene normalization")
print("="*70)

# Scores scale with number of genes and nUMI
# Python n_reg_genes = 206, R n_reg_genes = 206 (same!)
print(f"Number of reg genes: Python={len(reg_gene_names)}, R=206 (from r_gene_list_reg.txt)")
print(f"nUMI for sample: Python min={py_nUMI_m.min():.0f}, max={py_nUMI_m.max():.0f}, "
      f"median={np.median(py_nUMI_m):.0f}")

# Score per gene per UMI
py_s_per_gene = py_s_scores / len(reg_gene_names)
r_s_per_gene  = r_singlet_score / 206
print(f"\nPython singlet_score / n_genes: mean={py_s_per_gene.mean():.4f}")
print(f"R singlet_score / 206:          mean={r_s_per_gene.mean():.4f}")

py_thresh_per_gene = py_doublet_thresh_vals / len(reg_gene_names)
r_thresh_per_gene  = r_thresh / 206
print(f"\nPython doublet_thresh / n_genes: mean={py_thresh_per_gene.mean():.4f}, "
      f"median={np.median(py_thresh_per_gene):.4f}")
print(f"R doublet_thresh / n_genes: mean={r_thresh_per_gene.mean():.4f}, "
      f"median={np.median(r_thresh_per_gene):.4f}")

# What sigma was chosen vs R?
print(f"\nPython sigma: {sigma} (sigma_c = {sigma/100.0})")
print(f"(R sigma not directly available, but check r_norm_profiles.csv match implies R's sigma was similar)")

# K_val comparison
print(f"Python K_val: {config.K_val}")
print(f"R K_val: default is 1000 in spacexr")

print("\n" + "="*70)
print("DOUBLET INVESTIGATION COMPLETE")
print("="*70)
