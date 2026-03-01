#!/usr/bin/env python3
"""
Investigate remaining singlet_score discrepancy after constrain fix.

Key finding from singlet_score investigation:
  - Python singlet_score vs R: mean diff = -1.5, std = 7.47, max|diff| = 95.5
  - Python NLL for R's SAME type as singlet vs R: mean diff = -0.47, std = 4.30
  - The singlet NLL from Python for R's type is LOWER than R's (better fit)
    meaning Python finds a BETTER unconstrained fit than R in some cases.

The 47 pixels with |diff| > 10 are ALL negative (Python < R):
  - Python finds a LOWER NLL (better fit) for the singlet than R does.
  - This is because Python's IRWLS for singlets is unconstrained and can find
    negative weights or >1 weights.

When Python finds a lower singlet_score than R:
  - doublet_thresh = s_score - m_score gets SMALLER (since s_score is smaller)
  - Less likely to be classified as doublet
  - Explains remaining disagreements

Why would Python get a LOWER singlet NLL than R?
  R's singlet fitting: quadprog with lower bounds w >= -solution (meq=0)
  Python's singlet fitting: Box-constrained QP (same formulation)
  But Python doesn't have the lower bound constraint from initial solution!

Wait - in _irwls.py solve_irwls:
  The box QP has: delta_w >= -solution
  So w_new = solution + alpha * delta_w >= solution - alpha * solution >= 0
  This IS bounded from below.

Actually the issue might be the INITIALIZATION or convergence.
R warm-starts with W_full[t1, t2] for the pair types.
Python starts at 1/K (for K types in the solver).

Let's check: Does R's singlet solver warm-start from full-mode weights?
If R initializes the singlet solver with W_full[t] (full-mode weight for type t)
and the box QP lower bound is -W_full[t], then the first step can explore
w - W_full[t]*alpha which might be different from Python.

Python initializes IRWLS at [1/1] = [1.0] for K=1 singlet problem.
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

VAL_DIR = Path("/home/pgueguen/git/rctd-py/data/xenium_validation")
REF_DIR = Path("/home/pgueguen/git/rctd-py/data")

from rctd import Reference, RCTDConfig
from rctd._rctd import RCTD
from rctd._likelihood import load_cached_q_matrices, compute_spline_coefficients, calc_log_likelihood, calc_q_all
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

# Full mode for initialization
full_res = run_full_mode(
    spatial_counts=py_counts_m.astype(np.float64),
    spatial_numi=py_nUMI_m.astype(np.float64),
    norm_profiles=rctd.norm_profiles.astype(np.float64),
    cell_type_names=K_names,
    q_mat=q_mat, sq_mat=sq_mat, x_vals=x_vals, batch_size=2000,
)
W_full = full_res.weights

print(f"\n{'='*70}")
print("A. Analytical 1D singlet: what is the unconstrained minimum?")
print(f"{'='*70}")
print("""
For singlet fitting with a single type t:
  S = nUMI * P[:, t]   (G,)
  IRWLS solves: min NLL(w * S | Y)  with no sum constraint, but box: w >= -solution
  Initial solution = max(1.0, 0.0) = 1.0, so box bound: delta_w >= -1.0

The unconstrained minimum of NLL(w | Y, S) is at:
  d/dw [NLL] = -sum_g d1_g(w * S_g) * S_g = 0

This is a 1D problem. Python's IRWLS should converge to the global minimum.
R also solves this unconstrained.

The key: Python initializes w = 1/K = 1.0 for K=1 singlet.
         R initializes w = W_full[t] (full-mode weight for that type).

If W_full[t] is very different from 1.0 AND the NLL is non-convex,
the initialization matters. But the Poisson-Lognormal NLL should be convex.

Let's verify: 1D grid search for singlet weight to find the global minimum.
""")

# Find pixels with large singlet_score discrepancy
def score_singlet_1d_grid(n, t, n_points=10001):
    """Score singlet for type t using 1D grid search over w in [0, 3]."""
    nUMI_n = float(py_nUMI_m[n])
    Y = py_counts_m[n].astype(np.float64)
    P = np.array(P_gpu[:, t])
    S = nUMI_n * P

    best_score = float('inf')
    best_w = 1.0
    for w in np.linspace(0, 3, n_points):
        expected = np.maximum(w * S, 1e-4)
        sc = float(calc_log_likelihood(
            jnp.array(Y), jnp.array(expected), Q_gpu, SQ_gpu, X_gpu, config.K_val
        ))
        if sc < best_score:
            best_score = sc
            best_w = w
    return best_score, best_w

# Large discrepancy pixels: Pixel 134 had diff=-95.51, type=Melanoma
large_diff_pixels = [134, 15, 65, 80, 84, 87, 96, 97, 99]

print("1D grid search for singlet score on pixels with large discrepancy:")
for n in large_diff_pixels[:5]:
    ft_i = k_name_map.get(r_ft[n], -1)
    if ft_i < 0:
        continue
    # Python IRWLS score
    nUMI_b = jnp.array(py_nUMI_m[[n]])
    B_b    = jnp.array(py_counts_m[[n]].astype(np.float64))
    P_sg   = P_gpu[:, ft_i:ft_i+1].T[..., None]
    S_b    = nUMI_b[:, None, None] * P_sg
    w_b, _ = solve_irwls_batch(
        S_batch=S_b, Y_batch=B_b, nUMI_batch=nUMI_b,
        Q_mat=Q_gpu, SQ_mat=SQ_gpu, x_vals=X_gpu,
        max_iter=50, min_change=0.001, constrain=False, bulk_mode=False,
    )
    w_irwls = float(np.array(w_b)[0, 0])
    expected = jnp.maximum(jnp.sum(S_b * w_b[:, None, :], axis=-1), 1e-4)
    sc_irwls = float(calc_log_likelihood(B_b[0], expected[0], Q_gpu, SQ_gpu, X_gpu, config.K_val))

    # Grid search
    sc_grid, w_grid = score_singlet_1d_grid(n, ft_i)

    print(f"  Pixel {n} (type={r_ft[n]}, nUMI={py_nUMI_m[n]:.0f}):")
    print(f"    R singlet_score:     {r_sing_score[n]:.4f}")
    print(f"    Python IRWLS:        {sc_irwls:.4f} (w={w_irwls:.4f})")
    print(f"    Python 1D grid:      {sc_grid:.4f} (w={w_grid:.4f})")
    print(f"    W_full for type:     {W_full[n, ft_i]:.4f}")
    print(f"    Diff (IRWLS - R):    {sc_irwls - r_sing_score[n]:.4f}")
    print(f"    Diff (grid - R):     {sc_grid - r_sing_score[n]:.4f}")


print(f"\n{'='*70}")
print("B. Is singlet score discrepancy from Python selecting a DIFFERENT type?")
print(f"{'='*70}")

# For the large discrepancy pixels, what type does Python select as singlet?
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

# Score all singlets
from itertools import combinations
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

INF = 1e18
best_sing_types = []
best_sing_scores = []
for n in range(N):
    cands = candidates_list[n]
    sing_sc = [singlet_log_l.get((n, t), INF) for t in cands]
    best_i = int(np.argmin(sing_sc))
    best_sing_types.append(K_names[cands[best_i]])
    best_sing_scores.append(sing_sc[best_i])

best_sing_types = np.array(best_sing_types)
best_sing_scores = np.array(best_sing_scores)

# Large discrepancy analysis
diff_s = best_sing_scores - r_sing_score
large_disc = np.abs(diff_s) > 10
print(f"Pixels with |singlet_score diff| > 10: {large_disc.sum()}")

same_type = best_sing_types == r_ft
diff_type = ~same_type

print(f"\nAmong large-discrepancy pixels:")
ld_same_type = large_disc & same_type
ld_diff_type = large_disc & diff_type
print(f"  Same type as R: {ld_same_type.sum()} (NLL diff is from solver)")
print(f"  Different type than R: {ld_diff_type.sum()} (Python picked different best type)")

print(f"\n  Large-discrepancy with DIFFERENT type:")
for n in np.where(ld_diff_type)[0][:10]:
    cands = [K_names[c] for c in candidates_list[n]]
    r_ft_n = r_ft[n]
    py_ft_n = best_sing_types[n]
    r_ft_i = k_name_map.get(r_ft_n, -1)
    # Score R's type in Python
    r_type_score = singlet_log_l.get((n, r_ft_i), np.nan) if r_ft_i >= 0 else np.nan
    print(f"  Pixel {n}: R_ft={r_ft_n}, Py_ft={py_ft_n}, "
          f"R_sc={r_sc[n]}, diff={diff_s[n]:.2f}")
    print(f"    R singlet_score={r_sing_score[n]:.2f}, Py_sing={best_sing_scores[n]:.2f}")
    print(f"    Python NLL for R's type: {r_type_score:.2f}")
    print(f"    Python NLL for Py's type: {best_sing_scores[n]:.2f}")
    print(f"    Python finds {py_ft_n} better by: {r_type_score - best_sing_scores[n]:.2f}")
    print(f"    Candidates: {cands}")

print(f"\n  Large-discrepancy with SAME type:")
for n in np.where(ld_same_type)[0][:5]:
    ft_i = k_name_map.get(r_ft[n], -1)
    r_ft_n = r_ft[n]
    sc_irwls = singlet_log_l.get((n, ft_i), np.nan)
    sc_grid, w_grid = score_singlet_1d_grid(n, ft_i)
    print(f"  Pixel {n}: type={r_ft_n}, R={r_sing_score[n]:.2f}, "
          f"IRWLS={sc_irwls:.2f}, grid={sc_grid:.2f}")


print(f"\n{'='*70}")
print("C. R's singlet_score: is it for the BEST SINGLET or for R's first_type?")
print(f"{'='*70}")
print("""
R's singlet_score in results_df_sample.csv:
  This is R's BEST singlet score (across all singlet candidates),
  NOT specifically the score for R's first_type.

R's doublet_thresh = singlet_score - min_score
  = best_singlet_NLL - best_pair_NLL

Python also computes this the same way (best singlet across candidates).

But when Python finds a different BEST singlet type than R (different candidates),
the singlet_score will differ.

The 13.4% remaining disagreement is primarily from:
  1. Python selecting a different best singlet type (when R's first_type has
     lower full-mode weight and gets picked as a candidate, Python's singlet
     NLL for R's type is computed correctly, but Python picks a different type
     with even lower NLL from its candidate set).
  2. This is caused by R and Python exploring slightly different candidate sets
     (though both use the same 0.01 threshold on full-mode weights).

Wait - we showed the full-mode weights are essentially identical (100% dominant
type agreement, MAE=0.000045). So candidates should be nearly the same.

The real issue: Python's unconstrained singlet solver finds a LOWER NLL than R's.
This makes Python's singlet_score SMALLER than R's for those types, which means:
  - doublet_thresh = sing_score - pair_score is SMALLER in Python
  - Harder to reach DOUBLET_THRESHOLD = 25
  - More pixels misclassified as singlets

This is different from the pair constraint bug: it's a FEATURE of the unconstrained
singlet solver finding a lower minimum. R's quadprog with box constraints
(w >= -solution_init) is more restricted in the singlet case.

In R, the singlet solver initializes with W_full[t] and box constraint w >= -W_full[t].
If W_full[t] = 0.3, then the singlet weight must be >= -0.3.
The optimal weight might be e.g. w=0.4 in Python (unconstrained from below),
but R constrains w >= -0.3 * (1 - 0.3*step_size)... complex.

Actually: the box QP constrains delta_w >= -solution, and solution is clipped to >= 0
at each step. So the effective lower bound tightens as iterations proceed.
""")


print(f"\n{'='*70}")
print("D. Final summary: What fraction of overall 26% disagreement is from pair constraint bug?")
print(f"{'='*70}")
print(f"""
On 1000 sample pixels:
  - Original (constrain=True for pairs): 8.6% disagreement
  - After fix (constrain=False for pairs): 13.4% disagreement (WORSE!)

Wait - that's the FIRST TYPE disagreement (91.4% -> 86.6%).
The overall 26% disagreement was in DOMINANT TYPE assignment (full doublet weights).

Let me reconsider what the 26% refers to:
  The existing script (generate_xenium_report_data.py) computes:
    py_dominant = argmax(weights_doublet)
    r_dominant = argmax(r_doublet_weights)
  where weights_doublet is the 2-component weight matrix projected to K columns.

For doublet mode: weights_doublet has w[first_type]=doublet_w[0], w[second_type]=doublet_w[1].
The dominant type of the doublet weight IS the first_type.
So 26% disagreement = 26% of pixels have different first_type.

On our 1000-pixel sample:
  - constrain=True:  first_type agreement = 91.4%  (8.6% disagreement)
  - constrain=False: first_type agreement = 86.6% (13.4% disagreement)

The constrain=False for pairs REDUCES agreement from 91.4% to 86.6%?!
That means: fixing the pair constraint makes SOME things worse.

Let's understand why:
  - With constrain=True: Python classifies fewer pixels as doublets (only 86/1000)
    For those 86, the first_type is the pair type with better fit.
    For the 914 singlets, first_type is the best singlet type.
  - With constrain=False: Python classifies more pixels as doublets (175/1000)
    The new doublets come from pixels that were singlets before.
    But for R doublet pixels (264), Python's doublet classification has:
      R=doublet_certain (161): Python match = 73.3% (class), 57.8% (first_type)
      This is worse than constrain=True (82.0% first_type)!

Something more fundamental is happening. The issue is that with constrain=False
for pairs, the pair weights can be negative or >1, which changes the first_type
assignment (the type with higher positive weight).

In doublet mode, R's weights_doublet are computed from the pair fit weights
(NORMALIZED to sum=1 and be positive). Python does this too, but the raw
unconstrained weights might lead to different normalization behavior.
""")

print(f"\nExamining constrain=False pair weights for R-doublet pixels...")
# Score pairs with constrain=False and check the weights
r_is_doublet = (r_sc == "doublet_certain") | (r_sc == "doublet_uncertain")
r_dc_idx = np.where(r_is_doublet)[0][:10]

for n in r_dc_idx:
    ft_i = k_name_map.get(r_ft[n], -1)
    st_i = k_name_map.get(r_results["second_type"].values[matched_r[n]], -1)
    if ft_i < 0 or st_i < 0:
        continue

    nUMI_b = jnp.array(py_nUMI_m[[n]])
    B_b    = jnp.array(py_counts_m[[n]].astype(np.float64))
    P1 = P_gpu[:, ft_i:ft_i+1].T
    P2 = P_gpu[:, st_i:st_i+1].T
    S_b = nUMI_b[:, None, None] * jnp.stack([P1, P2], axis=-1)

    w_F, _ = solve_irwls_batch(
        S_batch=S_b, Y_batch=B_b, nUMI_batch=nUMI_b,
        Q_mat=Q_gpu, SQ_mat=SQ_gpu, x_vals=X_gpu,
        max_iter=25, min_change=0.001, constrain=False, bulk_mode=False,
    )
    w_T, _ = solve_irwls_batch(
        S_batch=S_b, Y_batch=B_b, nUMI_batch=nUMI_b,
        Q_mat=Q_gpu, SQ_mat=SQ_gpu, x_vals=X_gpu,
        max_iter=25, min_change=0.001, constrain=True, bulk_mode=False,
    )
    w_F_np = np.array(w_F)[0]
    w_T_np = np.array(w_T)[0]

    # Normalize for first_type determination
    # In Python doublet mode, after getting pair weights:
    # dw = pair_weights[key].copy()
    # s = dw.sum(); dw = dw / s if s > 0 else [0.5, 0.5]
    w_F_norm = w_F_np / max(w_F_np.sum(), 1e-10) if w_F_np.sum() > 0 else np.array([0.5, 0.5])
    w_T_norm = w_T_np  # already on simplex

    r_ft_n = r_ft[n]
    r_st_n = r_results["second_type"].values[matched_r[n]]
    py_ft_F = r_ft_n if w_F_norm[0] >= w_F_norm[1] else r_st_n
    py_ft_T = r_ft_n if w_T_np[0] >= w_T_np[1] else r_st_n

    print(f"  Pixel {n}: R=[ft={r_ft_n}, st={r_st_n}], "
          f"w_F={w_F_np} (norm={w_F_norm}), w_T={w_T_np}")
    print(f"    Py_ft (F)={py_ft_F}, Py_ft (T)={py_ft_T}")
