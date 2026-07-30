"""Minimal A/B benchmark: time deconvolution only, skip concordance."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_reference(ref_dir):
    """Load reference data from mtx/csv/txt files."""
    import anndata

    ref_dir = Path(ref_dir)
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


def run_benchmark(spatial_path, ref_dir, dataset_name, batch_size=5000):
    """Run full/doublet/multi and return timings."""
    from rctd._doublet import run_doublet_mode
    from rctd._full import run_full_mode
    from rctd._multi import run_multi_mode
    from rctd._rctd import RCTD
    from rctd._types import RCTDConfig

    import scanpy as sc

    config = RCTDConfig()
    reference = load_reference(ref_dir)

    spatial = sc.read_10x_h5(spatial_path)
    spatial.var_names_make_unique()

    rctd = RCTD(spatial, reference, config)
    rctd.fit_platform_effects()

    kwargs = {
        "spatial_counts": rctd.counts,
        "spatial_numi": rctd.nUMI,
        "norm_profiles": rctd.norm_profiles,
        "cell_type_names": rctd.reference.cell_type_names,
        "q_mat": rctd.q_mat,
        "sq_mat": rctd.sq_mat,
        "x_vals": rctd.x_vals,
        "batch_size": batch_size,
        "device": rctd.config.device,
    }

    results = {}
    N = rctd.counts.shape[0]
    K = rctd.norm_profiles.shape[1]

    for mode in ["full", "doublet", "multi"]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if mode == "full":
            run_full_mode(**kwargs)
        elif mode == "doublet":
            run_doublet_mode(**kwargs, config=rctd.config)
        elif mode == "multi":
            run_multi_mode(**kwargs, config=rctd.config)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        vram_mb = torch.cuda.max_memory_allocated() / 1e6

        results[mode] = {
            "deconv_s": round(elapsed, 2),
            "vram_mb": round(vram_mb),
            "cells_per_s": round(N / elapsed, 1),
        }
        print(f"  {mode:>8s}: {elapsed:6.2f}s  VRAM={vram_mb:.0f}MB  throughput={N/elapsed:.0f} cells/s")

    return {"dataset": dataset_name, "N": N, "K": K, "modes": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", required=True)
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    print(f"\n=== Benchmark: {args.dataset_name} ===")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    results = run_benchmark(args.spatial, args.ref_dir, args.dataset_name, args.batch_size)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {args.out_json}")


if __name__ == "__main__":
    main()
