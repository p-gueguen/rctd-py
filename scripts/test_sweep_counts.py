#!/usr/bin/env python3
"""Test effect of QP sweep count on Xenium agreement."""
import sys, os, time
import numpy as np
import scipy.io, scipy.sparse
import pandas as pd
import anndata
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def load_and_run(n_sweeps, spatial, reference, config, sigma_override=None):
    """Monkey-patch sweep count and run."""
    import rctd._irwls as irwls_mod
    # Replace the default
    orig_fn = irwls_mod._solve_box_qp
    from functools import partial
    import jax

    # We need to modify the default and re-jit
    # Easiest: patch the source constant
    old_default = 30

    # Actually, _solve_box_qp is called inside solve_irwls which is jitted.
    # The n_sweeps is hardcoded as default=30 in _solve_box_qp.
    # Since it's a static value baked into the jit trace, we need to
    # clear the cache and redefine.

    # Simpler approach: directly modify the function default
    irwls_mod._solve_box_qp.__defaults__ = (n_sweeps,)

    # Clear jit cache for solve_irwls
    irwls_mod.solve_irwls = jax.jit(
        irwls_mod.solve_irwls.__wrapped__,
        static_argnames=("max_iter", "constrain", "bulk_mode")
    )

    from rctd import run_rctd
    t0 = time.time()
    result = run_rctd(spatial, reference, mode="doublet", config=config,
                      batch_size=5000, sigma_override=sigma_override)
    elapsed = time.time() - t0

    return result, elapsed


def main():
    from rctd import Reference, RCTDConfig
    import scanpy as sc

    ref_dir = Path("data")
    val_dir = Path("data/xenium_validation")

    # Load R results
    r_meta = pd.read_csv(val_dir / "r_rctd_meta.csv")
    r_weights = pd.read_csv(val_dir / "r_rctd_weights.csv")
    r_barcodes = r_weights["barcode"].values
    r_weight_cols = [c for c in r_weights.columns if c != "barcode"]
    r_weight_matrix = r_weights[r_weight_cols].values.astype(np.float64)

    # Load reference
    counts = scipy.io.mmread(ref_dir / "ref_counts.mtx").T.tocsc()
    ref_meta_df = pd.read_csv(ref_dir / "ref_meta.csv")
    genes = pd.read_csv(ref_dir / "ref_genes.txt", header=None)[0].tolist()
    ref_adata = anndata.AnnData(X=counts, obs=ref_meta_df.set_index(ref_meta_df.columns[0]),
                                var=pd.DataFrame(index=genes))
    reference = Reference(ref_adata, cell_type_col="cell_type")

    # Load spatial + subsample
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", type=str, default="/scratch/cell_feature_matrix.h5")
    args = parser.parse_args()

    spatial = sc.read_10x_h5(args.spatial)
    spatial.var_names_make_unique()

    rng = np.random.default_rng(42)
    keep = rng.choice(spatial.n_obs, 1000, replace=False)
    keep.sort()
    spatial = spatial[keep].copy()
    print(f"Spatial: {spatial.n_obs} pixels (subsampled)")

    config = RCTDConfig(UMI_min=20)

    # Prepare R comparison data
    r_barcodes_stripped = np.array([b.replace("Region_3_35844-2_", "") for b in r_barcodes])
    r_idx_map = {b: i for i, b in enumerate(r_barcodes_stripped)}

    def evaluate(result, reference, spatial, config):
        from rctd._rctd import RCTD
        rctd_obj = RCTD(spatial, reference, config)
        py_barcodes = np.array(spatial.obs_names)[rctd_obj._pixel_mask]

        K = len(reference.cell_type_names)
        N = len(py_barcodes)
        py_w = np.zeros((N, K), dtype=np.float64)
        for i in range(N):
            ft = result.first_type[i]
            st = result.second_type[i]
            py_w[i, ft] = result.weights_doublet[i, 0]
            py_w[i, st] += result.weights_doublet[i, 1]

        common = sorted(set(py_barcodes) & set(r_barcodes_stripped))
        py_idx_map = {b: i for i, b in enumerate(py_barcodes)}
        py_idx = [py_idx_map[b] for b in common]
        r_idx = [r_idx_map[b] for b in common]

        common_types = [t for t in r_weight_cols if t in reference.cell_type_names]
        py_type_idx = [reference.cell_type_names.index(t) for t in common_types]
        r_type_idx = [r_weight_cols.index(t) for t in common_types]

        pw = py_w[py_idx][:, py_type_idx]
        rw = r_weight_matrix[r_idx][:, r_type_idx]
        pw_n = pw / np.maximum(pw.sum(axis=1, keepdims=True), 1e-10)
        rw_n = rw / np.maximum(rw.sum(axis=1, keepdims=True), 1e-10)
        rw_n = np.nan_to_num(rw_n, nan=0.0)

        py_dom = np.array([common_types[i] for i in np.argmax(pw_n, axis=1)])
        r_dom = np.array([common_types[i] for i in np.argmax(rw_n, axis=1)])
        agreement = (py_dom == r_dom).mean()

        corrs = []
        for i in range(len(common)):
            c = np.corrcoef(pw_n[i], rw_n[i])[0, 1]
            if not np.isnan(c):
                corrs.append(c)
        corrs = np.array(corrs)

        return agreement, np.median(corrs), len(common)

    # Test different sweep counts
    # Use sigma_override=72 so sigma estimation doesn't add variance
    for n_sweeps in [30, 60, 100, 200]:
        print(f"\n{'='*50}")
        print(f"n_sweeps = {n_sweeps}")
        print(f"{'='*50}")
        result, elapsed = load_and_run(n_sweeps, spatial, reference, config, sigma_override=72)
        agreement, med_corr, n_common = evaluate(result, reference, spatial, config)
        print(f"  Agreement: {agreement:.1%}")
        print(f"  Median correlation: {med_corr:.4f}")
        print(f"  Common pixels: {n_common}")
        print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
