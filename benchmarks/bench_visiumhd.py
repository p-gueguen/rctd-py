#!/usr/bin/env python3
"""Benchmark rctd-py on VisiumHD Mouse Brain (8µm bins), full mode only.

Measures wall-clock time, peak VRAM, and peak RSS for GPU-accelerated RCTD
on a large VisiumHD dataset (~300k+ spots) using the Allen cortex reference (K=22).

Usage:
    python benchmarks/bench_visiumhd.py \
        --spatial /scratch/pgueguen/visiumhd_mouse_brain/square_008um/filtered_feature_bc_matrix.h5 \
        --ref-dir data/mouse_brain \
        --out-dir data/visiumhd_mouse_brain
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_reference(ref_dir: Path):
    """Load Allen cortex reference from mtx/csv/txt files."""
    import anndata

    counts = scipy.io.mmread(ref_dir / "ref_counts.mtx").T.tocsc()
    ref_meta = pd.read_csv(ref_dir / "ref_meta.csv")
    genes = pd.read_csv(ref_dir / "ref_genes.txt", header=None)[0].tolist()
    ref_adata = anndata.AnnData(
        X=counts,
        obs=ref_meta.set_index(ref_meta.columns[0]),
        var=pd.DataFrame(index=genes),
    )
    from rctd import Reference

    return Reference(ref_adata, cell_type_col="cell_type")


def main():
    parser = argparse.ArgumentParser(description="Benchmark rctd-py on VisiumHD (full mode)")
    parser.add_argument("--spatial", required=True, help="Path to filtered_feature_bc_matrix.h5")
    parser.add_argument("--ref-dir", required=True, help="Directory with ref_counts.mtx etc.")
    parser.add_argument("--out-dir", required=True, help="Output directory for benchmark JSON")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size (0 = auto)")
    parser.add_argument("--umi-min", type=int, default=20)
    args = parser.parse_args()

    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(args.ref_dir)

    print("=== rctd-py Benchmark: VisiumHD Mouse Brain (full mode) ===")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_vram:.0f} GB)")
    else:
        gpu_name = "CPU"
        print("Device: CPU")
    print()

    # Load data
    import scanpy as sc
    from rctd import RCTDConfig
    from rctd._rctd import RCTD

    print("Loading reference...")
    reference = load_reference(ref_dir)
    print(f"  {reference.n_types} types, {reference.n_genes} genes")

    print(f"Loading spatial from {args.spatial}...")
    spatial = sc.read_10x_h5(args.spatial)
    spatial.var_names_make_unique()
    print(f"  {spatial.n_obs} spots, {spatial.n_vars} genes")

    # Setup
    config = RCTDConfig(UMI_min=args.umi_min)
    rctd_obj = RCTD(spatial, reference, config)

    # Sigma estimation
    print("Fitting platform effects (sigma)...")
    t_sigma_start = time.time()
    rctd_obj.fit_platform_effects()
    sigma_elapsed = time.time() - t_sigma_start
    print(f"  Sigma: {sigma_elapsed:.1f}s")

    n_filtered = int(rctd_obj._pixel_mask.sum())
    print(f"  Filtered: {n_filtered}/{spatial.n_obs} spots")

    # Resolve batch size
    if args.batch_size > 0:
        batch_size = args.batch_size
    else:
        try:
            from rctd._types import auto_batch_size

            G = rctd_obj.counts.shape[1]
            K = reference.n_types
            # resident full mode: the whole N x G count matrix sits on the GPU
            batch_size = auto_batch_size(
                G,
                K,
                dtype_bytes=rctd_obj.counts.dtype.itemsize,
                mode="full",
                n_resident=rctd_obj.counts.shape[0],
            )
        except ImportError:
            batch_size = 10000
    print(f"  Batch size: {batch_size}")

    # Run full mode
    print("\n--- FULL mode ---")
    from rctd._full import run_full_mode

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    t0 = time.time()
    result = run_full_mode(
        spatial_counts=rctd_obj.counts,
        spatial_numi=rctd_obj.nUMI,
        norm_profiles=rctd_obj.norm_profiles,
        cell_type_names=reference.cell_type_names,
        q_mat=rctd_obj.q_mat,
        sq_mat=rctd_obj.sq_mat,
        x_vals=rctd_obj.x_vals,
        batch_size=batch_size,
        device=config.device,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    deconv_elapsed = time.time() - t0

    peak_vram_mb = 0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mb = rss_after / 1024

    total_s = sigma_elapsed + deconv_elapsed
    throughput = n_filtered / total_s

    print(f"  Deconv: {deconv_elapsed:.1f}s")
    print(f"  Total (sigma + deconv): {total_s:.1f}s")
    print(f"  VRAM: {peak_vram_mb:.0f} MB")
    print(f"  RSS: {peak_rss_mb:.0f} MB")
    print(f"  Throughput: {throughput:.0f} spots/s")
    print(f"  Converged: {result.converged.sum()}/{len(result.converged)}")

    # Save results
    benchmark = {
        "dataset": "visiumhd_mouse_brain_8um",
        "n_spatial": spatial.n_obs,
        "n_filtered": n_filtered,
        "n_types": reference.n_types,
        "n_genes_used": int(rctd_obj.counts.shape[1]),
        "gpu_name": gpu_name,
        "sigma_s": round(sigma_elapsed, 1),
        "deconv_s": round(deconv_elapsed, 1),
        "total_s": round(total_s, 1),
        "peak_vram_mb": round(peak_vram_mb),
        "peak_rss_mb": round(peak_rss_mb),
        "throughput_spots_per_s": round(throughput, 1),
        "converged_frac": round(float(result.converged.sum() / len(result.converged)), 4),
        "batch_size": str(batch_size),
        "umi_min": args.umi_min,
    }

    out_file = out_dir / "visiumhd_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\nSaved {out_file}")
    print("Done!")


if __name__ == "__main__":
    main()
