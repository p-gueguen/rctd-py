#!/usr/bin/env python3
"""Generate Xenium report data: run rctd-py (JAX) on Region 3 and save structured results."""
import sys
import time
import json
import numpy as np
import scipy.sparse as sp
import scipy.io
import pandas as pd
import anndata
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", type=str,
                        default="/srv/gstore/projects/p36005/Xenium_Data/data_20240724__105930__o35844/"
                                "output-XETG00404__0022381__Region_3__20240724__110035/cell_feature_matrix.h5")
    args = parser.parse_args()

    from rctd import Reference, run_rctd, RCTDConfig

    ref_dir = Path("data")
    val_dir = Path("data/xenium_validation")
    out_dir = Path("data")

    # ── Load R results ──
    print("Loading R results...")
    r_meta = pd.read_csv(val_dir / "r_rctd_meta.csv")
    r_weights = pd.read_csv(val_dir / "r_rctd_weights.csv")
    r_barcodes = r_weights["barcode"].values
    r_weight_cols = [c for c in r_weights.columns if c != "barcode"]
    r_weight_matrix = r_weights[r_weight_cols].values.astype(np.float64)
    print(f"  R results: {len(r_barcodes)} cells, {len(r_weight_cols)} types")

    # ── Load reference ──
    print("Loading reference...")
    counts = scipy.io.mmread(ref_dir / "ref_counts.mtx").T.tocsc()
    ref_meta = pd.read_csv(ref_dir / "ref_meta.csv")
    genes = pd.read_csv(ref_dir / "ref_genes.txt", header=None)[0].tolist()
    ref_adata = anndata.AnnData(X=counts, obs=ref_meta.set_index(ref_meta.columns[0]),
                                var=pd.DataFrame(index=genes))
    reference = Reference(ref_adata, cell_type_col="cell_type")
    print(f"  Reference: {reference.n_types} types, {reference.n_genes} genes")

    # ── Load spatial ──
    import scanpy as sc
    print(f"Loading spatial from {args.spatial}...")
    spatial = sc.read_10x_h5(args.spatial)
    spatial.var_names_make_unique()
    print(f"  Spatial: {spatial.n_obs} cells, {spatial.n_vars} genes")

    # ── Run rctd-py ──
    config = RCTDConfig(UMI_min=20)
    print(f"Running rctd-py (doublet mode, UMI_min={config.UMI_min})...")

    # Time sigma estimation and deconvolution separately
    from rctd._rctd import RCTD
    from rctd._likelihood import load_cached_q_matrices, compute_spline_coefficients
    from rctd._sigma import choose_sigma
    from rctd._doublet import run_doublet_mode as _run_doublet

    rctd_obj = RCTD(spatial, reference, config)

    t0 = time.time()
    t_sigma_start = time.time()
    rctd_obj.fit_platform_effects()
    t_sigma_end = time.time()
    sigma_elapsed = t_sigma_end - t_sigma_start

    t_deconv_start = time.time()
    result = _run_doublet(
        spatial_counts=rctd_obj.counts,
        spatial_numi=rctd_obj.nUMI,
        norm_profiles=rctd_obj.norm_profiles,
        cell_type_names=reference.cell_type_names,
        q_mat=rctd_obj.q_mat,
        sq_mat=rctd_obj.sq_mat,
        x_vals=rctd_obj.x_vals,
        config=config,
        batch_size=5000,
    )
    t_deconv_end = time.time()
    deconv_elapsed = t_deconv_end - t_deconv_start

    elapsed = time.time() - t0
    print(f"  Sigma estimation: {sigma_elapsed:.1f}s")
    print(f"  Deconvolution: {deconv_elapsed:.1f}s")
    print(f"  Completed in {elapsed:.1f}s")

    # ── Build Python classification DataFrame ──
    from rctd._types import SPOT_CLASS_NAMES
    py_all_barcodes = np.array(spatial.obs_names)
    pixel_mask = rctd_obj._pixel_mask
    py_barcodes = py_all_barcodes[pixel_mask]

    spot_classes = [SPOT_CLASS_NAMES[i] for i in np.array(result.spot_class)]
    first_types = [reference.cell_type_names[i] for i in np.array(result.first_type)]
    second_types = [reference.cell_type_names[i] for i in np.array(result.second_type)]

    df_py = pd.DataFrame(index=spatial.obs_names)
    df_py["spot_class"] = "unfiltered"
    df_py.loc[py_barcodes, "spot_class"] = spot_classes
    df_py.loc[py_barcodes, "first_type"] = first_types
    df_py.loc[py_barcodes, "second_type"] = second_types
    if hasattr(result, "min_score"):
        df_py.loc[py_barcodes, "min_score"] = result.min_score
    if hasattr(result, "singlet_score"):
        df_py.loc[py_barcodes, "singlet_score"] = result.singlet_score

    df_py.to_csv(out_dir / "res_xenium_jax_gpu.csv")
    print(f"  Saved res_xenium_jax_gpu.csv")

    # ── Build weight matrices ──
    K = len(reference.cell_type_names)
    N_py = len(py_barcodes)

    # Full-mode weights (primary comparison target, matches R's @results$weights)
    py_full_w = result.weights  # (N_py, K)

    # Doublet pair weights (secondary)
    py_doublet_w = np.zeros((N_py, K), dtype=np.float64)
    for i in range(N_py):
        ft = result.first_type[i]
        st = result.second_type[i]
        py_doublet_w[i, ft] = result.weights_doublet[i, 0]
        py_doublet_w[i, st] += result.weights_doublet[i, 1]

    # Save full-mode weight matrix (all spatial cells, NaN for unfiltered)
    full_weights = np.full((spatial.n_obs, K), np.nan)
    full_weights[pixel_mask] = py_full_w
    np.save(out_dir / "weights_xenium_jax_gpu.npy", full_weights)
    print(f"  Saved weights_xenium_jax_gpu.npy")

    # ── Save Python cell type names (column order) ──
    with open(out_dir / "xenium_cell_type_names.txt", "w") as f:
        for name in reference.cell_type_names:
            f.write(name + "\n")

    # ── Match barcodes and compute metrics ──
    r_barcodes_stripped = np.array([b.replace("Region_3_35844-2_", "") for b in r_barcodes])
    py_set, r_set = set(py_barcodes), set(r_barcodes_stripped)
    common = sorted(py_set & r_set)
    print(f"  Common barcodes: {len(common)}")

    py_idx_map = {b: i for i, b in enumerate(py_barcodes)}
    r_idx_map = {b: i for i, b in enumerate(r_barcodes_stripped)}
    py_idx = [py_idx_map[b] for b in common]
    r_idx = [r_idx_map[b] for b in common]

    # Align cell types
    common_types = [t for t in r_weight_cols if t in reference.cell_type_names]
    py_type_idx = [reference.cell_type_names.index(t) for t in common_types]
    r_type_idx = [r_weight_cols.index(t) for t in common_types]

    # Use full-mode weights for comparison (matches R's weight representation)
    py_w = py_full_w[py_idx][:, py_type_idx]
    r_w = r_weight_matrix[r_idx][:, r_type_idx]

    # Normalize both to proportions (R's already sum to 1, Python's don't)
    py_w_norm = py_w / np.maximum(py_w.sum(axis=1, keepdims=True), 1e-10)
    r_w_norm = r_w / np.maximum(r_w.sum(axis=1, keepdims=True), 1e-10)
    r_w_norm = np.nan_to_num(r_w_norm, nan=0.0)

    # Dominant type agreement
    py_dominant = np.array([common_types[i] for i in np.argmax(py_w_norm, axis=1)])
    r_dominant = np.array([common_types[i] for i in np.argmax(r_w_norm, axis=1)])
    dom_agreement = (py_dominant == r_dominant).mean()

    # Per-pixel correlation
    per_pixel_corr = []
    for i in range(len(common)):
        c = np.corrcoef(py_w_norm[i], r_w_norm[i])[0, 1]
        if not np.isnan(c):
            per_pixel_corr.append(c)
    per_pixel_corr = np.array(per_pixel_corr)

    # Per-type correlation
    type_corrs = {}
    for i, t in enumerate(common_types):
        py_col, r_col = py_w_norm[:, i], r_w_norm[:, i]
        if py_col.std() > 1e-10 and r_col.std() > 1e-10:
            type_corrs[t] = np.corrcoef(py_col, r_col)[0, 1]

    # Type distribution
    py_dom_counts = Counter(py_dominant)
    r_dom_counts = Counter(r_dominant)
    all_types_in_dist = sorted(set(list(py_dom_counts.keys()) + list(r_dom_counts.keys())))
    type_dist = pd.DataFrame({
        "cell_type": all_types_in_dist,
        "python_count": [py_dom_counts.get(t, 0) for t in all_types_in_dist],
        "r_count": [r_dom_counts.get(t, 0) for t in all_types_in_dist],
    })
    type_dist.to_csv(out_dir / "xenium_type_distribution.csv", index=False)

    # Spot class distribution
    sc_vals, sc_counts = np.unique(result.spot_class, return_counts=True)
    spot_dist = {SPOT_CLASS_NAMES[v]: int(c) for v, c in zip(sc_vals, sc_counts)}

    # ── Save metrics ──
    r_elapsed = None
    r_timing_path = val_dir / "r_xenium_timing.csv"
    if r_timing_path.exists():
        r_timing = pd.read_csv(r_timing_path)
        r_elapsed = float(r_timing["elapsed_s"].iloc[0])

    metrics = {
        "elapsed_s": round(elapsed, 1),
        "sigma_elapsed_s": round(sigma_elapsed, 1),
        "deconv_elapsed_s": round(deconv_elapsed, 1),
        "r_elapsed_s": round(r_elapsed, 1) if r_elapsed is not None else None,
        "n_spatial": int(spatial.n_obs),
        "n_filtered": int(pixel_mask.sum()),
        "n_common": len(common),
        "n_types": int(reference.n_types),
        "n_common_types": len(common_types),
        "dominant_type_agreement": round(dom_agreement, 4),
        "pixel_corr_mean": round(float(per_pixel_corr.mean()), 4),
        "pixel_corr_median": round(float(np.median(per_pixel_corr)), 4),
        "pixel_corr_gt_0.5": round(float((per_pixel_corr > 0.5).mean()), 4),
        "pixel_corr_gt_0.8": round(float((per_pixel_corr > 0.8).mean()), 4),
        "spot_class_distribution": spot_dist,
        "r_n_cells": len(r_barcodes),
    }
    with open(out_dir / "xenium_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved xenium_metrics.json")

    # Save per-type correlations
    pd.Series(type_corrs).to_csv(out_dir / "xenium_type_correlations.csv")

    # ── Doublet mode comparison (if R doublet data available) ──
    r_doublet_path = val_dir / "r_doublet_full.csv"
    if r_doublet_path.exists():
        print("\n=== Doublet mode comparison ===")
        r_dbl = pd.read_csv(r_doublet_path)
        r_dbl_barcodes = r_dbl["barcode"].values
        r_dbl_barcodes_stripped = np.array([
            b.replace("Region_3_35844-2_", "") for b in r_dbl_barcodes
        ])

        # Match barcodes
        r_dbl_idx = {b: i for i, b in enumerate(r_dbl_barcodes_stripped)}
        py_bc_set = set(py_barcodes)
        common_dbl = [b for b in r_dbl_barcodes_stripped if b in py_bc_set]
        print(f"  Common barcodes for doublet comparison: {len(common_dbl)}")

        py_dbl_idx = [py_idx_map[b] for b in common_dbl]
        r_dbl_idx_list = [r_dbl_idx[b] for b in common_dbl]

        # Extract Python doublet classifications
        py_spot = np.array([SPOT_CLASS_NAMES[result.spot_class[i]] for i in py_dbl_idx])
        py_ft = np.array([reference.cell_type_names[result.first_type[i]] for i in py_dbl_idx])
        py_st = np.array([reference.cell_type_names[result.second_type[i]] for i in py_dbl_idx])

        # Extract R doublet classifications
        r_spot = r_dbl["spot_class"].values[r_dbl_idx_list]
        r_ft = r_dbl["first_type"].values[r_dbl_idx_list]
        r_st = r_dbl["second_type"].values[r_dbl_idx_list]

        # Spot class agreement
        spot_agree = (py_spot == r_spot).mean()
        print(f"  Spot class agreement: {spot_agree:.1%}")

        # First type agreement (on non-reject pixels)
        non_reject = (py_spot != "reject") & (r_spot != "reject")
        ft_agree = (py_ft[non_reject] == r_ft[non_reject]).mean()
        print(f"  First type agreement (non-reject): {ft_agree:.1%}")

        # Second type agreement (on doublet pixels)
        both_doublet = (
            np.isin(py_spot, ["doublet_certain", "doublet_uncertain"]) &
            np.isin(r_spot, ["doublet_certain", "doublet_uncertain"])
        )
        if both_doublet.sum() > 0:
            st_agree = (py_st[both_doublet] == r_st[both_doublet]).mean()
            print(f"  Second type agreement (both doublet): {st_agree:.1%}")
        else:
            st_agree = float("nan")

        # Spot class distribution comparison
        from collections import Counter as Ctr
        py_spot_dist = Ctr(py_spot)
        r_spot_dist = Ctr(r_spot)
        print(f"\n  Spot class distribution:")
        for cls in ["reject", "singlet", "doublet_certain", "doublet_uncertain"]:
            print(f"    {cls:25s}: Python={py_spot_dist.get(cls, 0):6d}, R={r_spot_dist.get(cls, 0):6d}")

        # Save doublet metrics
        doublet_metrics = {
            "n_common_doublet": len(common_dbl),
            "spot_class_agreement": round(spot_agree, 4),
            "first_type_agreement": round(ft_agree, 4),
            "second_type_agreement": round(st_agree, 4) if not np.isnan(st_agree) else None,
            "py_spot_distribution": dict(py_spot_dist),
            "r_spot_distribution": dict(r_spot_dist),
        }
        with open(out_dir / "xenium_doublet_metrics.json", "w") as f:
            json.dump(doublet_metrics, f, indent=2)
        print(f"  Saved xenium_doublet_metrics.json")
    else:
        print(f"\n  R doublet data not found at {r_doublet_path}")
        print(f"  Run scripts/extract_r_doublet_full.R to generate it")

    print(f"\nDominant type agreement: {dom_agreement:.1%}")
    print(f"Median pixel correlation: {np.median(per_pixel_corr):.4f}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
