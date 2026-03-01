#!/usr/bin/env python3
"""Comprehensive investigation of R vs Python RCTD disagreements.
Tasks:
1. Gene list comparison
2. Normalized profile comparison
3. Full-mode weight comparison on sampled pixels
4. Doublet classification comparison
5. nUMI comparison
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

VAL_DIR = Path("/home/pgueguen/git/rctd-py/data/xenium_validation")
REF_DIR = Path("/home/pgueguen/git/rctd-py/data")

# ─────────────────────────────────────────────────────────────────────────────
# Helper to load reference and spatial
# ─────────────────────────────────────────────────────────────────────────────

def load_reference():
    counts = scipy.io.mmread(REF_DIR / "ref_counts.mtx").T.tocsc()
    meta = pd.read_csv(REF_DIR / "ref_meta.csv")
    genes = pd.read_csv(REF_DIR / "ref_genes.txt", header=None)[0].tolist()
    ref_adata = anndata.AnnData(
        X=counts,
        obs=meta.set_index(meta.columns[0]),
        var=pd.DataFrame(index=genes),
    )
    return ref_adata


def load_spatial():
    import scanpy as sc
    h5_path = (
        "/srv/gstore/projects/p36005/Xenium_Data/"
        "data_20240724__105930__o35844/"
        "output-XETG00404__0022381__Region_3__20240724__110035/"
        "cell_feature_matrix.h5"
    )
    spatial = sc.read_10x_h5(h5_path)
    spatial.var_names_make_unique()
    return spatial


# =============================================================================
# 1. GENE LIST COMPARISON
# =============================================================================
print("=" * 70)
print("TASK 1: Gene list comparison")
print("=" * 70)

# R gene lists
r_bulk_genes = [l.strip() for l in open(VAL_DIR / "r_gene_list_bulk.txt") if l.strip()]
r_reg_genes  = [l.strip() for l in open(VAL_DIR / "r_gene_list_reg.txt")  if l.strip()]
print(f"R gene_list_bulk: {len(r_bulk_genes)} genes")
print(f"R gene_list_reg:  {len(r_reg_genes)} genes")

# Python gene lists
from rctd import Reference, RCTDConfig
from rctd._rctd import RCTD

print("\nLoading reference and spatial for gene list computation...")
ref_adata = load_reference()
reference = Reference(ref_adata, cell_type_col="cell_type")
spatial = load_spatial()
config = RCTDConfig(UMI_min=20)
rctd = RCTD(spatial, reference, config)

# Reproduce the gene list computation from fit_platform_effects
common_gene_to_idx = {g: i for i, g in enumerate(rctd.common_genes)}
spatial_bulk_common = rctd.counts.sum(axis=0)
spatial_bulk_full = np.zeros(reference.n_genes, dtype=np.float32)
for g in rctd.common_genes:
    ref_idx = reference.gene_names.index(g)
    spatial_bulk_full[ref_idx] = spatial_bulk_common[common_gene_to_idx[g]]

py_bulk_raw = reference.get_de_genes(
    spatial_bulk=spatial_bulk_full,
    fc_thresh=config.fc_cutoff,
    expr_thresh=config.gene_cutoff,
    min_obs=config.MIN_OBS,
)
py_reg_raw = reference.get_de_genes(
    spatial_bulk=spatial_bulk_full,
    fc_thresh=config.fc_cutoff_reg,
    expr_thresh=config.gene_cutoff_reg,
    min_obs=config.MIN_OBS,
)
# Keep only common genes (same as fit_platform_effects does)
py_bulk_genes = sorted(set(py_bulk_raw) & set(rctd.common_genes))
py_reg_genes  = sorted(set(py_reg_raw)  & set(rctd.common_genes))

print(f"\nPython gene_list_bulk (after common filter): {len(py_bulk_genes)} genes")
print(f"Python gene_list_reg  (after common filter): {len(py_reg_genes)} genes")

r_bulk_set = set(r_bulk_genes)
r_reg_set  = set(r_reg_genes)
py_bulk_set = set(py_bulk_genes)
py_reg_set  = set(py_reg_genes)

bulk_only_r  = r_bulk_set - py_bulk_set
bulk_only_py = py_bulk_set - r_bulk_set
bulk_shared  = r_bulk_set & py_bulk_set

reg_only_r  = r_reg_set - py_reg_set
reg_only_py = py_reg_set - r_reg_set
reg_shared  = r_reg_set & py_reg_set

print(f"\n--- bulk gene list ---")
print(f"Shared: {len(bulk_shared)}")
print(f"Only in R: {len(bulk_only_r)}  -> {sorted(bulk_only_r)}")
print(f"Only in Python: {len(bulk_only_py)}  -> {sorted(bulk_only_py)}")

print(f"\n--- reg gene list ---")
print(f"Shared: {len(reg_shared)}")
print(f"Only in R: {len(reg_only_r)}  -> {sorted(reg_only_r)}")
print(f"Only in Python: {len(reg_only_py)}  -> {sorted(reg_only_py)}")


# =============================================================================
# 2. NORMALIZED PROFILE COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("TASK 2: Normalized profile comparison (reg genes)")
print("=" * 70)

# Load R normalized profiles on reg genes
r_norm_reg = pd.read_csv(VAL_DIR / "r_norm_profiles_reg.csv", index_col=0)
print(f"R norm profiles (reg): shape = {r_norm_reg.shape}  "
      f"(genes x cell_types)")
r_norm_reg_genes = list(r_norm_reg.index)
r_norm_reg_types = list(r_norm_reg.columns)
print(f"R reg profile genes[:5]: {r_norm_reg_genes[:5]}")
print(f"R cell types[:5]: {r_norm_reg_types[:5]}")

# Compute Python normalized profiles via fit_platform_effects
import jax.numpy as jnp
from rctd._normalize import fit_bulk

# Use Python's own gene lists (computed above)
bulk_profiles = reference.get_profiles_for_genes(py_bulk_genes)
bulk_gene_idx = [rctd.common_genes.index(g) for g in py_bulk_genes]
bulk_counts   = rctd.counts[:, bulk_gene_idx]

print("\nRunning fit_bulk to compute Python normalized profiles...")
bulk_weights_jax, norm_prof_bulk_jax = fit_bulk(
    cell_type_profiles=jnp.array(bulk_profiles),
    spatial_counts=jnp.array(bulk_counts),
    spatial_nUMI=jnp.array(rctd.nUMI),
    min_change=config.MIN_CHANGE_BULK,
    max_iter=config.max_iter,
)

norm_prof_bulk_np = np.array(norm_prof_bulk_jax)  # (G_bulk, K)
bulk_gene_map = {g: i for i, g in enumerate(py_bulk_genes)}

# Subset to py_reg_genes (only those that are also in bulk)
py_reg_in_bulk = [g for g in py_reg_genes if g in bulk_gene_map]
reg_in_bulk_idx = [bulk_gene_map[g] for g in py_reg_in_bulk]
py_norm_reg_np = norm_prof_bulk_np[reg_in_bulk_idx]  # (G_reg, K)

py_cell_types = reference.cell_type_names  # sorted list

print(f"Python norm profiles (reg): shape = {py_norm_reg_np.shape}")
print(f"Python reg genes[:5]: {py_reg_in_bulk[:5]}")
print(f"Python cell types[:5]: {py_cell_types[:5]}")

# Align genes and cell types between R and Python
common_profile_genes = sorted(set(r_norm_reg_genes) & set(py_reg_in_bulk))
common_profile_types = sorted(set(r_norm_reg_types) & set(py_cell_types))
print(f"\nCommon profile genes (R reg ∩ Py reg): {len(common_profile_genes)}")
print(f"Only in R reg profile: {sorted(set(r_norm_reg_genes) - set(py_reg_in_bulk))}")
print(f"Only in Python reg: {sorted(set(py_reg_in_bulk) - set(r_norm_reg_genes))}")
print(f"Common cell types: {len(common_profile_types)}")
print(f"Only in R types: {sorted(set(r_norm_reg_types) - set(py_cell_types))}")
print(f"Only in Python types: {sorted(set(py_cell_types) - set(r_norm_reg_types))}")

if len(common_profile_genes) > 0 and len(common_profile_types) > 0:
    # R matrix aligned
    r_aligned = r_norm_reg.loc[common_profile_genes, common_profile_types].values

    # Python matrix aligned
    py_reg_gene_map  = {g: i for i, g in enumerate(py_reg_in_bulk)}
    py_type_map      = {t: i for i, t in enumerate(py_cell_types)}
    py_g_idx = [py_reg_gene_map[g] for g in common_profile_genes]
    py_t_idx = [py_type_map[t]     for t in common_profile_types]
    py_aligned = py_norm_reg_np[np.ix_(py_g_idx, py_t_idx)]

    abs_diff = np.abs(r_aligned - py_aligned)
    print(f"\nElement-wise comparison (aligned {len(common_profile_genes)} genes x "
          f"{len(common_profile_types)} types):")
    print(f"  Max absolute difference: {abs_diff.max():.6f}")
    print(f"  Mean absolute difference: {abs_diff.mean():.6f}")
    print(f"  Median absolute difference: {np.median(abs_diff):.6f}")
    # Per-gene correlation
    per_gene_corr = []
    for i in range(len(common_profile_genes)):
        r_row = r_aligned[i]
        py_row = py_aligned[i]
        if r_row.std() > 1e-12 and py_row.std() > 1e-12:
            per_gene_corr.append(np.corrcoef(r_row, py_row)[0, 1])
    per_gene_corr = np.array(per_gene_corr)
    print(f"  Per-gene correlation (across cell types): mean={per_gene_corr.mean():.5f}, "
          f"min={per_gene_corr.min():.5f}, median={np.median(per_gene_corr):.5f}")
    # Per-type correlation
    per_type_corr = []
    for j in range(len(common_profile_types)):
        r_col = r_aligned[:, j]
        py_col = py_aligned[:, j]
        if r_col.std() > 1e-12 and py_col.std() > 1e-12:
            per_type_corr.append(np.corrcoef(r_col, py_col)[0, 1])
    per_type_corr = np.array(per_type_corr)
    print(f"  Per-type correlation (across genes): mean={per_type_corr.mean():.5f}, "
          f"min={per_type_corr.min():.5f}, median={np.median(per_type_corr):.5f}")

    # Ratio analysis
    ratio = py_aligned / np.where(r_aligned != 0, r_aligned, np.nan)
    finite_ratio = ratio[np.isfinite(ratio)]
    print(f"\n  Python/R ratio: mean={np.nanmean(finite_ratio):.5f}, "
          f"std={np.nanstd(finite_ratio):.5f}, "
          f"min={np.nanmin(finite_ratio):.5f}, max={np.nanmax(finite_ratio):.5f}")

    # Show largest disagreements
    flat_idx = np.argsort(abs_diff.ravel())[::-1][:10]
    print("\n  Top 10 largest absolute differences:")
    print(f"  {'Gene':<20} {'CellType':<30} {'R_val':>12} {'Py_val':>12} {'AbsDiff':>10}")
    for idx in flat_idx:
        gi, ti = np.unravel_index(idx, abs_diff.shape)
        print(f"  {common_profile_genes[gi]:<20} {common_profile_types[ti]:<30} "
              f"{r_aligned[gi,ti]:>12.6f} {py_aligned[gi,ti]:>12.6f} {abs_diff[gi,ti]:>10.6f}")

else:
    print("Not enough common genes/types to compare profiles.")


# =============================================================================
# 3. FULL-MODE WEIGHT COMPARISON ON SAMPLED PIXELS
# =============================================================================
print("\n" + "=" * 70)
print("TASK 3: Full-mode weight comparison on sampled pixels")
print("=" * 70)

# Load sample data
r_sample_weights = pd.read_csv(VAL_DIR / "r_pixel_weights_sample.csv", index_col=0)
r_sample_barcodes_raw = [l.strip() for l in open(VAL_DIR / "r_pixel_barcodes_sample.txt") if l.strip()]
print(f"R sample: {len(r_sample_barcodes_raw)} barcodes, {r_sample_weights.shape} weight matrix")
print(f"R weight columns[:5]: {list(r_sample_weights.columns)[:5]}")
print(f"Sample barcodes[0]: {r_sample_barcodes_raw[0]}")
print(f"Weight index[0]: {r_sample_weights.index[0]}")

# Barcodes: strip the prefix if needed
# R barcodes may have a prefix like "Region_3_35844-2_"
sample_bc_stripped = np.array([b.replace("Region_3_35844-2_", "") for b in r_sample_barcodes_raw])
print(f"Sample barcodes (stripped)[0]: {sample_bc_stripped[0]}")
print(f"Weight index (stripped)[0]: {str(r_sample_weights.index[0]).replace('Region_3_35844-2_', '')}")

# Try to match weight index with barcodes
r_weight_idx_stripped = [str(i).replace("Region_3_35844-2_", "") for i in r_sample_weights.index]

# Get all spatial barcodes
all_spatial_barcodes = np.array(spatial.obs_names)
py_pixel_mask = rctd._pixel_mask
py_barcodes = all_spatial_barcodes[py_pixel_mask]
print(f"\nAll spatial barcodes[0]: {all_spatial_barcodes[0]}")
print(f"Python pixel barcodes[0]: {py_barcodes[0]}")

# Match sample barcodes into Python pixel index
# strip prefix from sample barcodes
py_bc_map = {b: i for i, b in enumerate(py_barcodes)}

# Try different stripping strategies
matched_py_idx = []
matched_r_weight_idx = []
for j, bc_raw in enumerate(r_sample_barcodes_raw):
    bc = bc_raw.replace("Region_3_35844-2_", "")
    if bc in py_bc_map:
        matched_py_idx.append(py_bc_map[bc])
        matched_r_weight_idx.append(j)

print(f"\nMatched {len(matched_py_idx)}/{len(r_sample_barcodes_raw)} sample barcodes to Python pixels")

if len(matched_py_idx) == 0:
    print("  Trying alternative barcode matching...")
    # Try matching by stripping suffix
    sample_bc_nosuffix = [b.split("-")[0] if "-" in b else b for b in sample_bc_stripped]
    py_nosuffix = [b.split("-")[0] if "-" in b else b for b in py_barcodes]
    py_ns_map = {b: i for i, b in enumerate(py_nosuffix)}
    for j, bc in enumerate(sample_bc_nosuffix):
        if bc in py_ns_map:
            matched_py_idx.append(py_ns_map[bc])
            matched_r_weight_idx.append(j)
    print(f"  After suffix strip: matched {len(matched_py_idx)}")

if len(matched_py_idx) > 0:
    matched_py_idx = np.array(matched_py_idx)
    matched_r_weight_idx = np.array(matched_r_weight_idx)

    # Get Python counts and nUMI for matched pixels
    py_counts_matched = rctd.counts[matched_py_idx]  # shape before reg restriction
    # NOTE: rctd.counts may have been modified during fit_platform_effects
    # We need to rerun with the reg gene indices
    # Actually rctd has not had fit_platform_effects called, so rctd.counts is full common genes
    py_nUMI_matched = rctd.nUMI[matched_py_idx]

    print(f"\nPython nUMI for sample: min={py_nUMI_matched.min():.0f}, "
          f"median={np.median(py_nUMI_matched):.0f}, max={py_nUMI_matched.max():.0f}")

    # We need to re-run full mode on matched pixels with both:
    # (a) Python's own profiles (after fit_platform_effects)
    # (b) R's profiles (r_norm_profiles_reg.csv)

    # First, get the reg-gene-restricted counts for matched pixels
    # We need to apply the same gene restriction as fit_platform_effects does
    py_reg_gene_idx = [rctd.common_genes.index(g) for g in py_reg_in_bulk]
    py_counts_reg_matched = rctd.counts[matched_py_idx][:, py_reg_gene_idx]
    py_nUMI_matched = rctd.nUMI[matched_py_idx]

    print(f"Python reg gene counts for sample: shape={py_counts_reg_matched.shape}")

    # ── 3a. Python profiles ──
    from rctd._full import run_full_mode
    from rctd._likelihood import load_cached_q_matrices, compute_spline_coefficients

    # Load Q matrices for sigma computation
    # We need to load and use the same sigma as R; let's use what fit_platform_effects gives
    cache = load_cached_q_matrices()
    x_vals = cache.pop("X_vals")
    q_matrices = {k.replace("Q_", ""): v for k, v in cache.items()}

    # First run fit_platform_effects on the full rctd to get sigma
    import copy
    rctd2 = RCTD(spatial, reference, config)
    print("\nRunning fit_platform_effects to get sigma and normalized profiles...")
    rctd2.fit_platform_effects()
    sigma = rctd2.sigma
    print(f"Python sigma: {sigma} (sigma_c = {sigma/100.0})")

    # Python's norm profiles after fit_platform_effects
    py_norm_profiles = rctd2.norm_profiles  # (G_reg, K)
    py_reg_genes_from_rctd = rctd2._gene_list_reg
    print(f"Python norm_profiles shape: {py_norm_profiles.shape}")
    print(f"Python reg genes (from rctd2): {len(py_reg_genes_from_rctd)}")

    # Get reg-gene-restricted counts for matched pixels using rctd2's gene list
    py_reg_gene_idx2 = [rctd.common_genes.index(g) for g in py_reg_genes_from_rctd]
    py_counts_reg_matched2 = rctd.counts[matched_py_idx][:, py_reg_gene_idx2]

    print(f"\nRunning Python full mode on {len(matched_py_idx)} sample pixels "
          f"with Python profiles...")
    best_q_key = str(sigma)
    if best_q_key not in q_matrices:
        available = sorted([int(k) for k in q_matrices.keys() if k.isdigit()])
        nearest = min(available, key=lambda x: abs(x - sigma))
        best_q_key = str(nearest)
    q_mat = q_matrices[best_q_key]
    sq_mat = compute_spline_coefficients(q_mat, x_vals)

    full_res_py = run_full_mode(
        spatial_counts=py_counts_reg_matched2.astype(np.float64),
        spatial_numi=py_nUMI_matched.astype(np.float64),
        norm_profiles=py_norm_profiles.astype(np.float64),
        cell_type_names=reference.cell_type_names,
        q_mat=q_mat,
        sq_mat=sq_mat,
        x_vals=x_vals,
        batch_size=1000,
    )
    py_full_weights_a = full_res_py.weights  # (N_sample, K)
    print(f"Python full mode (own profiles) weights shape: {py_full_weights_a.shape}")

    # ── 3b. R's profiles ──
    # Get r_norm_profiles_reg.csv, align to Python's gene/type order
    r_norm_reg_df = pd.read_csv(VAL_DIR / "r_norm_profiles_reg.csv", index_col=0)
    r_genes_reg = list(r_norm_reg_df.index)
    r_types_reg = list(r_norm_reg_df.columns)
    print(f"\nR norm profiles (reg) for 3b: {r_norm_reg_df.shape} (genes x types)")

    # Align R profile to Python cell type order and Python reg genes
    common_3b_genes = [g for g in py_reg_genes_from_rctd if g in r_norm_reg_df.index]
    common_3b_types = [t for t in reference.cell_type_names if t in r_norm_reg_df.columns]
    print(f"Genes in Python-reg ∩ R-reg for 3b: {len(common_3b_genes)}")
    print(f"Types in Python ∩ R for 3b: {len(common_3b_types)}")

    # We need to use the SAME genes for both Python and R profiles
    # So extract counts restricted to common_3b_genes
    py_3b_gene_idx_in_common = [rctd.common_genes.index(g) for g in common_3b_genes]
    py_counts_3b = rctd.counts[matched_py_idx][:, py_3b_gene_idx_in_common]

    # Build R profile matrix aligned to (common_3b_genes x reference.cell_type_names)
    # Fill missing types with zeros
    r_prof_3b = np.zeros((len(common_3b_genes), len(reference.cell_type_names)), dtype=np.float64)
    for ti, ct in enumerate(reference.cell_type_names):
        if ct in r_norm_reg_df.columns:
            r_prof_3b[:, ti] = r_norm_reg_df.loc[common_3b_genes, ct].values

    print(f"\nRunning Python full mode on {len(matched_py_idx)} sample pixels "
          f"with R profiles ({len(common_3b_genes)} genes)...")
    full_res_r = run_full_mode(
        spatial_counts=py_counts_3b.astype(np.float64),
        spatial_numi=py_nUMI_matched.astype(np.float64),
        norm_profiles=r_prof_3b.astype(np.float64),
        cell_type_names=reference.cell_type_names,
        q_mat=q_mat,
        sq_mat=sq_mat,
        x_vals=x_vals,
        batch_size=1000,
    )
    py_full_weights_b = full_res_r.weights  # (N_sample, K)
    print(f"Python full mode (R profiles) weights shape: {py_full_weights_b.shape}")

    # ── Compare to R's sample weights ──
    # r_sample_weights is indexed by barcode, columns are cell types
    # matched_r_weight_idx gives which rows of r_sample_weights to use
    r_weight_types = list(r_sample_weights.columns)
    r_w_sample = r_sample_weights.iloc[matched_r_weight_idx].values  # (N_sample, K_r)

    # Align types between Python and R
    common_w_types = [t for t in reference.cell_type_names if t in r_weight_types]
    py_tw_idx = [reference.cell_type_names.index(t) for t in common_w_types]
    r_tw_idx  = [r_weight_types.index(t) for t in common_w_types]

    r_w_aligned = r_w_sample[:, r_tw_idx]          # (N_sample, K_common)
    py_wa_aligned = py_full_weights_a[:, py_tw_idx]  # (N_sample, K_common)
    py_wb_aligned = py_full_weights_b[:, py_tw_idx]  # (N_sample, K_common)

    # R full-mode weights can be negative; normalize for dominant type comparison
    def normalize_weights(w):
        w = np.clip(w, 0, None)
        s = w.sum(axis=1, keepdims=True)
        return np.where(s > 0, w / s, 1.0 / w.shape[1])

    r_w_norm   = normalize_weights(r_w_aligned)
    py_wa_norm = normalize_weights(py_wa_aligned)
    py_wb_norm = normalize_weights(py_wb_aligned)

    r_dominant  = np.array([common_w_types[i] for i in np.argmax(r_w_norm, axis=1)])
    pya_dominant = np.array([common_w_types[i] for i in np.argmax(py_wa_norm, axis=1)])
    pyb_dominant = np.array([common_w_types[i] for i in np.argmax(py_wb_norm, axis=1)])

    agree_a = (r_dominant == pya_dominant).mean()
    agree_b = (r_dominant == pyb_dominant).mean()
    agree_ab = (pya_dominant == pyb_dominant).mean()

    print(f"\nDominant type agreement (R vs Python-own-profiles): {agree_a:.3f} ({agree_a*100:.1f}%)")
    print(f"Dominant type agreement (R vs Python-R-profiles):   {agree_b:.3f} ({agree_b*100:.1f}%)")
    print(f"Dominant type agreement (Py-own vs Py-R-profiles):  {agree_ab:.3f} ({agree_ab*100:.1f}%)")

    # Weight correlation
    def per_pixel_corr(a, b):
        corrs = []
        for i in range(len(a)):
            if a[i].std() > 1e-10 and b[i].std() > 1e-10:
                corrs.append(np.corrcoef(a[i], b[i])[0, 1])
        return np.array(corrs)

    corr_a = per_pixel_corr(py_wa_norm, r_w_norm)
    corr_b = per_pixel_corr(py_wb_norm, r_w_norm)
    print(f"\nPer-pixel weight correlation (Python-own vs R): "
          f"mean={corr_a.mean():.4f}, median={np.median(corr_a):.4f}, "
          f"min={corr_a.min():.4f}")
    print(f"Per-pixel weight correlation (Python-R-prof vs R): "
          f"mean={corr_b.mean():.4f}, median={np.median(corr_b):.4f}, "
          f"min={corr_b.min():.4f}")

    # MAE
    mae_a = np.abs(py_wa_norm - r_w_norm).mean()
    mae_b = np.abs(py_wb_norm - r_w_norm).mean()
    print(f"\nMAE weights (Python-own vs R): {mae_a:.5f}")
    print(f"MAE weights (Python-R-prof vs R): {mae_b:.5f}")

else:
    print("Could not match sample barcodes to Python pixels. Skipping full-mode comparison.")
    matched_py_idx = None


# =============================================================================
# 4. DOUBLET CLASSIFICATION COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("TASK 4: Doublet classification comparison on sampled pixels")
print("=" * 70)

# Load R sample results
r_results_sample = pd.read_csv(VAL_DIR / "r_results_df_sample.csv", index_col=0)
print(f"R results sample shape: {r_results_sample.shape}")
print(f"R results columns: {list(r_results_sample.columns)}")
print(f"R spot_class values: {r_results_sample['spot_class'].value_counts().to_dict()}")

# Match barcodes
r_res_idx_stripped = [str(i).replace("Region_3_35844-2_", "") for i in r_results_sample.index]
print(f"\nR sample result barcodes[0]: {r_results_sample.index[0]}")
print(f"Python barcodes[0]: {py_barcodes[0]}")

matched_r4 = []
matched_py4 = []
for j, bc_raw in enumerate(r_res_idx_stripped):
    if bc_raw in py_bc_map:
        matched_py4.append(py_bc_map[bc_raw])
        matched_r4.append(j)

print(f"Matched {len(matched_py4)}/{len(r_results_sample)} result barcodes to Python pixels")

if len(matched_py4) > 0:
    matched_py4 = np.array(matched_py4)
    matched_r4 = np.array(matched_r4)

    # Run Python doublet mode on matched pixels
    py_counts_4 = rctd.counts[matched_py4][:, py_reg_gene_idx2]
    py_nUMI_4   = rctd.nUMI[matched_py4]

    from rctd._doublet import run_doublet_mode

    print(f"\nRunning Python doublet mode on {len(matched_py4)} pixels...")
    doublet_res = run_doublet_mode(
        spatial_counts=py_counts_4.astype(np.float64),
        spatial_numi=py_nUMI_4.astype(np.float64),
        norm_profiles=py_norm_profiles.astype(np.float64),
        cell_type_names=reference.cell_type_names,
        q_mat=q_mat,
        sq_mat=sq_mat,
        x_vals=x_vals,
        config=config,
        batch_size=1000,
    )

    from rctd._types import SPOT_CLASS_NAMES
    py_spot_class  = np.array([SPOT_CLASS_NAMES[i] for i in doublet_res.spot_class])
    py_first_type  = np.array([reference.cell_type_names[i] for i in doublet_res.first_type])
    py_second_type = np.array([reference.cell_type_names[i] for i in doublet_res.second_type])

    # R results for matched pixels
    r_spot_class_m  = r_results_sample["spot_class"].values[matched_r4]
    r_first_type_m  = r_results_sample["first_type"].values[matched_r4]
    r_second_type_m = r_results_sample["second_type"].values[matched_r4]

    # Map R spot class values to same naming
    # R uses: "singlet", "doublet_certain", "doublet_uncertain", "reject"
    print(f"\nPython spot_class distribution: {dict(zip(*np.unique(py_spot_class, return_counts=True)))}")
    print(f"R spot_class distribution: {dict(zip(*np.unique(r_spot_class_m, return_counts=True)))}")

    # Agreement on spot_class
    spot_agree = (py_spot_class == r_spot_class_m).mean()
    print(f"\nSpot class agreement: {spot_agree:.3f} ({spot_agree*100:.1f}%)")

    # Agreement on first_type (among all, among singlets, among doublets)
    first_agree = (py_first_type == r_first_type_m).mean()
    print(f"First type agreement: {first_agree:.3f} ({first_agree*100:.1f}%)")

    # Restrict to pixels where spot_class agrees
    sc_mask = py_spot_class == r_spot_class_m
    first_agree_sc = (py_first_type[sc_mask] == r_first_type_m[sc_mask]).mean() if sc_mask.sum() > 0 else float('nan')
    print(f"First type agreement (where spot_class agrees, n={sc_mask.sum()}): "
          f"{first_agree_sc:.3f} ({first_agree_sc*100:.1f}%)")

    # Among singlets
    r_singlet_mask = r_spot_class_m == "singlet"
    if r_singlet_mask.sum() > 0:
        first_agree_sing = (py_first_type[r_singlet_mask] == r_first_type_m[r_singlet_mask]).mean()
        print(f"First type agreement (R singlets, n={r_singlet_mask.sum()}): "
              f"{first_agree_sing:.3f} ({first_agree_sing*100:.1f}%)")

    # Among doublet_certain
    r_dc_mask = r_spot_class_m == "doublet_certain"
    if r_dc_mask.sum() > 0:
        first_agree_dc = (py_first_type[r_dc_mask] == r_first_type_m[r_dc_mask]).mean()
        print(f"First type agreement (R doublet_certain, n={r_dc_mask.sum()}): "
              f"{first_agree_dc:.3f} ({first_agree_dc*100:.1f}%)")

    # Breakdown: where do disagreements come from?
    disagree_mask = r_first_type_m != py_first_type
    n_disagree = disagree_mask.sum()
    print(f"\nFirst type disagreements: {n_disagree}/{len(matched_py4)} ({n_disagree/len(matched_py4)*100:.1f}%)")

    if n_disagree > 0:
        # What fraction of disagreements are purely from classification vs weight?
        # "Class disagreement" = spot_class differs
        class_diff = (py_spot_class != r_spot_class_m) & disagree_mask
        # "Type disagreement with same class" = same spot_class but different type
        type_diff_same_class = (py_spot_class == r_spot_class_m) & disagree_mask
        print(f"  Disagree due to different spot_class: {class_diff.sum()} "
              f"({class_diff.sum()/n_disagree*100:.1f}% of disagreements)")
        print(f"  Disagree despite same spot_class: {type_diff_same_class.sum()} "
              f"({type_diff_same_class.sum()/n_disagree*100:.1f}% of disagreements)")

        # Within same-class disagreements, what classes are they?
        if type_diff_same_class.sum() > 0:
            same_class_vals = r_spot_class_m[type_diff_same_class]
            print(f"  Same-class type-disagree breakdown: "
                  f"{dict(zip(*np.unique(same_class_vals, return_counts=True)))}")

        # What are the most common mismatched pairs?
        mismatched_r = r_first_type_m[disagree_mask]
        mismatched_py = py_first_type[disagree_mask]
        from collections import Counter
        pair_counts = Counter(zip(mismatched_r, mismatched_py))
        print(f"\n  Top 15 mismatched first_type pairs (R -> Python):")
        for (rt, pt), cnt in sorted(pair_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"    {rt:<35} -> {pt:<35}: {cnt}")

else:
    print("Could not match sample barcodes for doublet comparison.")


# =============================================================================
# 5. nUMI COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("TASK 5: nUMI comparison")
print("=" * 70)

r_nUMI_df = pd.read_csv(VAL_DIR / "r_puck_nUMI_all.csv", index_col=0)
print(f"R nUMI_all shape: {r_nUMI_df.shape}")
print(f"R nUMI_all columns: {list(r_nUMI_df.columns)}")
print(f"R nUMI_all index sample: {list(r_nUMI_df.index[:3])}")

# Determine nUMI column name
numi_col = r_nUMI_df.columns[0]
r_nUMI_vals = r_nUMI_df[numi_col].values
r_nUMI_barcodes = np.array([str(i).replace("Region_3_35844-2_", "") for i in r_nUMI_df.index])

print(f"R nUMI (all pixels): n={len(r_nUMI_vals)}, min={r_nUMI_vals.min():.0f}, "
      f"median={np.median(r_nUMI_vals):.0f}, max={r_nUMI_vals.max():.0f}, "
      f"sum={r_nUMI_vals.sum():.0f}")

# Python nUMI: computed from full spatial X across ALL genes (before UMI filter)
# rctd._pixel_mask gives which pixels pass the filter
# The original nUMI before filter is on the full spatial object
full_X = spatial.X
if sp.issparse(full_X):
    py_nUMI_all = np.array(full_X.sum(axis=1)).flatten().astype(np.float64)
else:
    py_nUMI_all = full_X.sum(axis=1).flatten().astype(np.float64)

all_barcodes_orig = np.array(spatial.obs_names)
print(f"\nPython nUMI (all pixels): n={len(py_nUMI_all)}, "
      f"min={py_nUMI_all.min():.0f}, median={np.median(py_nUMI_all):.0f}, "
      f"max={py_nUMI_all.max():.0f}, sum={py_nUMI_all.sum():.0f}")

# Match barcodes
r_bc_to_numi = dict(zip(r_nUMI_barcodes, r_nUMI_vals))
py_bc_to_numi = dict(zip(all_barcodes_orig, py_nUMI_all))

common_bc_numi = sorted(set(r_nUMI_barcodes) & set(all_barcodes_orig))
print(f"\nCommon barcodes for nUMI comparison: {len(common_bc_numi)}")

if len(common_bc_numi) > 0:
    r_numi_matched  = np.array([r_bc_to_numi[b] for b in common_bc_numi])
    py_numi_matched = np.array([py_bc_to_numi[b] for b in common_bc_numi])

    diff = py_numi_matched - r_numi_matched
    ratio = py_numi_matched / np.where(r_numi_matched > 0, r_numi_matched, np.nan)

    print(f"nUMI difference (Python - R): "
          f"mean={diff.mean():.4f}, std={diff.std():.4f}, "
          f"min={diff.min():.1f}, max={diff.max():.1f}")
    print(f"nUMI ratio (Python / R): "
          f"mean={np.nanmean(ratio):.6f}, std={np.nanstd(ratio):.6f}, "
          f"min={np.nanmin(ratio):.6f}, max={np.nanmax(ratio):.6f}")
    print(f"Pixels where Python nUMI != R nUMI: "
          f"{(diff != 0).sum()} ({(diff != 0).mean()*100:.2f}%)")
    print(f"Pixels where |diff| > 1: {(np.abs(diff) > 1).sum()} "
          f"({(np.abs(diff) > 1).mean()*100:.2f}%)")
    print(f"Max absolute nUMI difference: {np.abs(diff).max():.2f}")

    # Check: R nUMI count
    print(f"\nR total nUMI sum (matched): {r_numi_matched.sum():.0f}")
    print(f"Python total nUMI sum (matched): {py_numi_matched.sum():.0f}")
    print(f"R all-pixel nUMI sum: {r_nUMI_vals.sum():.0f}")
    print(f"Python all-pixel nUMI sum: {py_nUMI_all.sum():.0f}")

    # What does Python use as total nUMI for normalization?
    py_filtered_nUMI = py_nUMI_all[rctd._pixel_mask]
    print(f"\nPython filtered nUMI (UMI_min={config.UMI_min}): n={len(py_filtered_nUMI)}, "
          f"sum={py_filtered_nUMI.sum():.0f}")
    print(f"R all-pixel nUMI count: {len(r_nUMI_vals)}")
    print(f"Python all-pixel count: {len(py_nUMI_all)}")
    print(f"Python filtered count (UMI>={config.UMI_min}): {rctd._pixel_mask.sum()}")

else:
    print("No common barcodes found for nUMI comparison.")
    # Maybe barcodes are entirely different
    print(f"Sample R barcodes: {r_nUMI_barcodes[:5]}")
    print(f"Sample Python barcodes: {all_barcodes_orig[:5]}")

# Additional: what does R use for spatial_nUMI_total?
# In R's get_norm_ref: sum(puck@nUMI) — the sum over all pixels
print(f"\nKey normalization constants:")
print(f"  R sum(puck@nUMI): {r_nUMI_vals.sum():.0f}")
print(f"  Python sum(nUMI) used in fit_bulk (filtered): {rctd.nUMI.sum():.0f}")
print(f"  Note: Python uses filtered nUMI in fit_bulk (bulk_nUMI = sum(spatial_nUMI))")
print(f"  R likely uses ALL pixels' nUMI in fitBulk")

print("\n" + "=" * 70)
print("INVESTIGATION COMPLETE")
print("=" * 70)
