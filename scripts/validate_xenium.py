#!/usr/bin/env python3
"""Validate rctd-py on real Xenium data (duodenum Region 3).

Usage:
    python scripts/validate_xenium.py [--mode full|doublet|multi] [--batch-size 5000]
"""
import argparse
import time
import numpy as np
import scipy.io
import scipy.sparse as sp
import pandas as pd
import anndata
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_reference(ref_dir):
    """Load reference from converted files (ref_counts.mtx, ref_meta.csv, ref_genes.txt, ref_cells.txt)."""
    counts = scipy.io.mmread(ref_dir / "ref_counts.mtx").T.tocsc()  # cells x genes
    meta = pd.read_csv(ref_dir / "ref_meta.csv")
    genes = pd.read_csv(ref_dir / "ref_genes.txt", header=None)[0].tolist()
    cells = pd.read_csv(ref_dir / "ref_cells.txt", header=None)[0].tolist()

    adata = anndata.AnnData(
        X=counts,
        obs=meta.set_index(meta.columns[0]),
        var=pd.DataFrame(index=genes),
    )
    return adata


def load_xenium_spatial(h5_path):
    """Load Xenium cell-feature matrix."""
    import scanpy as sc
    adata = sc.read_10x_h5(str(h5_path))
    adata.var_names_make_unique()
    return adata


def main():
    parser = argparse.ArgumentParser(description="Validate rctd-py on Xenium data")
    parser.add_argument("--ref-dir", type=str,
                        default="/home/pgueguen/git/rctd-py/data",
                        help="Directory containing converted reference files")
    parser.add_argument("--spatial", type=str,
                        default="/srv/gstore/projects/p36005/Xenium_Data/data_20240724__105930__o35844/output-XETG00404__0022381__Region_3__20240724__110035/cell_feature_matrix.h5",
                        help="Path to Xenium cell_feature_matrix.h5")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "doublet", "multi"])
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-pixels", type=int, default=None,
                        help="Limit number of pixels for quick testing")
    parser.add_argument("--output-dir", type=str, default="data/xenium_validation")
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference
    print("Loading reference...")
    ref_adata = load_reference(ref_dir)
    print(f"  Reference: {ref_adata.n_obs} cells, {ref_adata.n_vars} genes")
    print(f"  Cell types: {ref_adata.obs['cell_type'].nunique()}")

    from rctd import Reference, run_rctd, RCTDConfig
    reference = Reference(ref_adata, cell_type_col="cell_type")
    print(f"  After filtering: {reference.n_types} types, {reference.n_genes} genes")

    # Load spatial
    print(f"\nLoading spatial data from {args.spatial}...")
    spatial = load_xenium_spatial(args.spatial)
    print(f"  Spatial: {spatial.n_obs} cells, {spatial.n_vars} genes")

    if args.max_pixels and spatial.n_obs > args.max_pixels:
        np.random.seed(42)
        idx = np.random.choice(spatial.n_obs, args.max_pixels, replace=False)
        spatial = spatial[sorted(idx)].copy()
        print(f"  Subsampled to {spatial.n_obs} pixels")

    # Run RCTD
    config = RCTDConfig()
    print(f"\nRunning RCTD in {args.mode} mode (batch_size={args.batch_size})...")
    t0 = time.time()
    result = run_rctd(spatial, reference, mode=args.mode, config=config, batch_size=args.batch_size)
    elapsed = time.time() - t0

    n_pixels = result.weights.shape[0]
    print(f"\nCompleted in {elapsed:.1f}s ({n_pixels / elapsed:.0f} pixels/sec)")
    print(f"  Result weights shape: {result.weights.shape}")

    # Summary statistics
    w = result.weights
    w_norm = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-10)
    dominant_types = np.argmax(w_norm, axis=1)

    print(f"\n  Weight sums: min={w.sum(1).min():.4f}, max={w.sum(1).max():.4f}, mean={w.sum(1).mean():.4f}")
    print(f"\n  Cell type distribution (dominant type):")
    from collections import Counter
    type_counts = Counter(dominant_types)
    for ct_idx, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        ct_name = reference.cell_type_names[ct_idx]
        print(f"    {ct_name}: {count} ({count/n_pixels:.1%})")

    if args.mode == "doublet":
        from rctd._types import SPOT_CLASS_NAMES
        print(f"\n  Spot class distribution:")
        sc_counts = Counter(result.spot_class)
        for sc_val, count in sorted(sc_counts.items()):
            name = SPOT_CLASS_NAMES[sc_val] if sc_val < len(SPOT_CLASS_NAMES) else f"unknown_{sc_val}"
            print(f"    {name}: {count} ({count/n_pixels:.1%})")

    if args.mode == "multi":
        print(f"\n  Types per pixel distribution:")
        nt_counts = Counter(result.n_types)
        for nt, count in sorted(nt_counts.items()):
            print(f"    {nt} types: {count} ({count/n_pixels:.1%})")

    # Save results
    np.save(output_dir / f"xenium_{args.mode}_weights.npy", w)
    print(f"\n  Saved weights to {output_dir / f'xenium_{args.mode}_weights.npy'}")

    print("\nValidation complete!")


if __name__ == "__main__":
    main()
