#!/usr/bin/env python3
"""Compare R spacexr RCTD results vs Python rctd-py on Xenium Region 3.

Runs rctd-py with matching R parameters (UMI_min=20, doublet mode),
then compares against extracted R results.
"""
import sys
import time
import numpy as np
import scipy.io
import scipy.sparse as sp
import pandas as pd
import anndata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_reference(ref_dir):
    counts = scipy.io.mmread(ref_dir / "ref_counts.mtx").T.tocsc()
    meta = pd.read_csv(ref_dir / "ref_meta.csv")
    genes = pd.read_csv(ref_dir / "ref_genes.txt", header=None)[0].tolist()
    return anndata.AnnData(
        X=counts,
        obs=meta.set_index(meta.columns[0]),
        var=pd.DataFrame(index=genes),
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", type=str,
                        default="/srv/gstore/projects/p36005/Xenium_Data/data_20240724__105930__o35844/output-XETG00404__0022381__Region_3__20240724__110035/cell_feature_matrix.h5")
    parser.add_argument("--sigma-override", type=int, default=None,
                        help="Override sigma (integer, e.g. 72 for sigma_c=0.72)")
    parser.add_argument("--n-pixels", type=int, default=None,
                        help="Subsample spatial to N random pixels (for fast validation)")
    args = parser.parse_args()

    from rctd import Reference, run_rctd, RCTDConfig

    ref_dir = Path("/home/pgueguen/git/rctd-py/data")
    val_dir = Path("/home/pgueguen/git/rctd-py/data/xenium_validation")

    # ── Load R results ──
    print("Loading R results...")
    r_meta = pd.read_csv(val_dir / "r_rctd_meta.csv")
    r_weights = pd.read_csv(val_dir / "r_rctd_weights.csv")
    r_barcodes = r_weights["barcode"].values
    r_weight_cols = [c for c in r_weights.columns if c != "barcode"]
    r_weight_matrix = r_weights[r_weight_cols].values.astype(np.float64)
    print(f"  R results: {len(r_barcodes)} cells, {len(r_weight_cols)} types")
    print(f"  R RCTD_Main top 5: {r_meta['RCTD_Main'].value_counts().head()}")

    # ── Load reference ──
    print("\nLoading reference...")
    ref_adata = load_reference(ref_dir)
    reference = Reference(ref_adata, cell_type_col="cell_type")
    print(f"  Reference: {reference.n_types} types, {reference.n_genes} genes")

    # ── Load spatial ──
    import scanpy as sc
    h5_path = args.spatial
    print(f"\nLoading spatial from {h5_path}...")
    spatial = sc.read_10x_h5(h5_path)
    spatial.var_names_make_unique()
    print(f"  Spatial: {spatial.n_obs} cells, {spatial.n_vars} genes")

    # Subsample for fast validation
    if args.n_pixels is not None and args.n_pixels < spatial.n_obs:
        rng = np.random.default_rng(42)
        keep = rng.choice(spatial.n_obs, args.n_pixels, replace=False)
        keep.sort()
        spatial = spatial[keep].copy()
        print(f"  Subsampled to {spatial.n_obs} pixels")

    # ── Run Python RCTD with matching R params ──
    # R uses UMI_min=20 (from param$rctdUMImin=20)
    config = RCTDConfig(UMI_min=20)
    print(f"\nRunning Python RCTD (doublet mode, UMI_min={config.UMI_min})...")

    # Diagnostic: show nUMI computation source
    import scipy.sparse as _sp
    _full_X = spatial.X
    if _sp.issparse(_full_X):
        _total_numi = np.array(_full_X.sum(axis=1)).flatten()
    else:
        _total_numi = np.array(_full_X.sum(axis=1)).flatten()
    _filtered = _total_numi[_total_numi >= config.UMI_min]
    print(f"  Total nUMI (all {spatial.n_vars} genes): range [{_filtered.min():.0f}, {_filtered.max():.0f}], "
          f"median={np.median(_filtered):.0f}, n_pixels={len(_filtered)}")

    t0 = time.time()
    sigma_ov = args.sigma_override
    if sigma_ov is not None:
        print(f"  Using sigma override: {sigma_ov} (sigma_c={sigma_ov/100.0})")
    result = run_rctd(spatial, reference, mode="doublet", config=config,
                      batch_size=5000, sigma_override=sigma_ov)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Python weights shape: {result.weights.shape}")

    # ── Match barcodes ──
    # Python barcodes come from spatial.obs_names, filtered by UMI
    # R barcodes have prefix "Region_3_35844-2_"
    # Reconstruct Python pixel barcodes after UMI filter
    from rctd._rctd import RCTD
    rctd_obj = RCTD(spatial, reference, config)
    py_all_barcodes = np.array(spatial.obs_names)
    py_barcodes = py_all_barcodes[rctd_obj._pixel_mask]
    print(f"  Python barcodes (after UMI filter): {len(py_barcodes)}")

    # Strip R prefix
    r_barcodes_stripped = np.array([b.replace("Region_3_35844-2_", "") for b in r_barcodes])

    # Find common barcodes
    py_set = set(py_barcodes)
    r_set = set(r_barcodes_stripped)
    common = py_set & r_set
    print(f"  Common barcodes: {len(common)}")

    if len(common) == 0:
        print("  ERROR: No common barcodes found!")
        print(f"  Python sample: {py_barcodes[:3]}")
        print(f"  R sample: {r_barcodes_stripped[:3]}")
        return

    # Build index maps
    py_idx_map = {b: i for i, b in enumerate(py_barcodes)}
    r_idx_map = {b: i for i, b in enumerate(r_barcodes_stripped)}
    common_sorted = sorted(common)
    py_idx = [py_idx_map[b] for b in common_sorted]
    r_idx = [r_idx_map[b] for b in common_sorted]

    # ── Compare weights ──
    # R's extracted weights are full-mode weights (from decompose_full, constrain=F,
    # then normalized to sum=1 by the extraction script). These are stored in
    # RCTD@results$weights which holds `all_weights` from process_bead_doublet.
    # For a fair comparison, use Python's full-mode weights (result.weights).
    K = len(reference.cell_type_names)
    N_py = len(py_barcodes)

    # Full-mode weights (primary comparison)
    py_full_w = result.weights  # (N_py, K) unconstrained full-mode IRWLS output

    # Doublet pair weights (secondary)
    py_doublet_full = np.zeros((N_py, K), dtype=np.float64)
    for i in range(N_py):
        ft = result.first_type[i]
        st = result.second_type[i]
        py_doublet_full[i, ft] = result.weights_doublet[i, 0]
        py_doublet_full[i, st] += result.weights_doublet[i, 1]

    py_w = py_full_w[py_idx]  # Use full-mode weights for primary comparison
    r_w = r_weight_matrix[r_idx]

    # Align cell type columns
    py_type_names = reference.cell_type_names
    # Map R column names to Python indices
    common_types = [t for t in r_weight_cols if t in py_type_names]
    print(f"  Common cell types: {len(common_types)}/{len(r_weight_cols)}")

    py_type_idx = [py_type_names.index(t) for t in common_types]
    r_type_idx = [r_weight_cols.index(t) for t in common_types]

    py_w_aligned = py_w[:, py_type_idx]
    r_w_aligned = r_w[:, r_type_idx]

    # Normalize both to proportions (R weights already sum to 1, Python full-mode doesn't)
    py_w_norm = py_w_aligned / np.maximum(py_w_aligned.sum(axis=1, keepdims=True), 1e-10)
    r_w_norm = r_w_aligned / np.maximum(r_w_aligned.sum(axis=1, keepdims=True), 1e-10)

    # Replace NaN in R weights with 0
    r_w_norm = np.nan_to_num(r_w_norm, nan=0.0)

    # ── Metrics ──
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Common pixels: {len(common_sorted)}")
    print(f"Common types: {len(common_types)}")

    # 1. Dominant type agreement
    py_dominant = np.array([common_types[i] for i in np.argmax(py_w_norm, axis=1)])
    r_dominant = np.array([common_types[i] for i in np.argmax(r_w_norm, axis=1)])
    agreement = (py_dominant == r_dominant).mean()
    print(f"\n1. Dominant type agreement: {agreement:.1%}")

    # Show top disagreements
    disagree_mask = py_dominant != r_dominant
    if disagree_mask.sum() > 0:
        print(f"   Disagreements: {disagree_mask.sum()} pixels")
        from collections import Counter
        py_disagree_types = Counter(py_dominant[disagree_mask])
        r_disagree_types = Counter(r_dominant[disagree_mask])
        print(f"   Python assigns (top 5): {py_disagree_types.most_common(5)}")
        print(f"   R assigns (top 5):      {r_disagree_types.most_common(5)}")

    # 2. Per-pixel weight correlation
    per_pixel_corr = []
    for i in range(len(common_sorted)):
        c = np.corrcoef(py_w_norm[i], r_w_norm[i])[0, 1]
        if not np.isnan(c):
            per_pixel_corr.append(c)
    per_pixel_corr = np.array(per_pixel_corr)
    print(f"\n2. Per-pixel weight correlation:")
    print(f"   Mean: {per_pixel_corr.mean():.4f}")
    print(f"   Median: {np.median(per_pixel_corr):.4f}")
    print(f"   >0.5: {(per_pixel_corr > 0.5).mean():.1%}")
    print(f"   >0.8: {(per_pixel_corr > 0.8).mean():.1%}")

    # 3. Per-type correlation
    print(f"\n3. Per-type weight correlation:")
    for i, t in enumerate(common_types):
        py_col = py_w_norm[:, i]
        r_col = r_w_norm[:, i]
        if py_col.std() > 1e-10 and r_col.std() > 1e-10:
            c = np.corrcoef(py_col, r_col)[0, 1]
            py_mean = py_col.mean()
            r_mean = r_col.mean()
            if abs(c) > 0.1 or py_mean > 0.01 or r_mean > 0.01:
                print(f"   {t:40s}: r={c:.3f} (py_mean={py_mean:.4f}, r_mean={r_mean:.4f})")

    # 4. Weight sum statistics
    print(f"\n4. Weight sums:")
    py_sums = py_w_aligned.sum(axis=1)
    r_sums = r_w_aligned.sum(axis=1)
    print(f"   Python: min={py_sums.min():.4f}, median={np.median(py_sums):.4f}, max={py_sums.max():.4f}")
    print(f"   R:      min={r_sums.min():.4f}, median={np.median(r_sums):.4f}, max={r_sums.max():.4f}")

    # 5. Python dominant type distribution
    print(f"\n5. Python dominant type distribution:")
    py_dom_counts = Counter(py_dominant)
    for t, c in sorted(py_dom_counts.items(), key=lambda x: -x[1])[:15]:
        r_count = (r_dominant == t).sum()
        print(f"   {t:40s}: Python={c:5d}, R={r_count:5d}")

    # 6. R dominant type distribution
    print(f"\n6. R dominant type distribution:")
    r_dom_counts = Counter(r_dominant)
    for t, c in sorted(r_dom_counts.items(), key=lambda x: -x[1])[:15]:
        py_count = (py_dominant == t).sum()
        print(f"   {t:40s}: R={c:5d}, Python={py_count:5d}")

    # 7. Spot class distribution (Python doublet mode)
    if hasattr(result, 'spot_class'):
        from rctd._types import SPOT_CLASS_NAMES
        print(f"\n7. Python spot class distribution:")
        sc_vals, sc_counts = np.unique(result.spot_class, return_counts=True)
        for v, c in zip(sc_vals, sc_counts):
            name = SPOT_CLASS_NAMES[v] if v < len(SPOT_CLASS_NAMES) else f"unknown_{v}"
            print(f"   {name}: {c} ({c/len(result.spot_class):.1%})")

    # 8. Doublet pair weights vs R full-mode (secondary comparison)
    py_dbl_aligned = py_doublet_full[py_idx][:, py_type_idx]
    py_dbl_norm = py_dbl_aligned / np.maximum(py_dbl_aligned.sum(axis=1, keepdims=True), 1e-10)
    py_dbl_dom = np.array([common_types[i] for i in np.argmax(py_dbl_norm, axis=1)])
    dbl_agree = (py_dbl_dom == r_dominant).mean()
    print(f"\n8. Doublet pair weights vs R full-mode dominant type: {dbl_agree:.1%}")

    print(f"\n{'='*60}")
    print("DONE")


if __name__ == "__main__":
    main()
