"""Lighter benchmark for CPU-only environments.

Same methodology as benchmark.py but with smaller dataset sizes
that complete in reasonable time on CPU (~1-3 minutes).
"""

import hashlib
import multiprocessing as mp
import os
import platform
import resource
import sys
import time
import traceback

import anndata
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rctd._rctd import run_rctd
from rctd._reference import Reference
from rctd._types import RCTDConfig

BATCH_SIZE = 5000
CONFIG = {
    "n_genes": 500,
    "n_cells": 5000,
    "n_types": 8,
    "n_pixels": 5000,
    "seed": 2025,
}


def make_data(n_genes, n_cells, n_types, n_pixels, seed):
    rng = np.random.default_rng(seed)
    profiles = rng.exponential(0.0015, size=(n_genes, n_types)).astype(np.float64)
    block = max(1, n_genes // n_types)
    for k in range(n_types):
        start = k * block
        stop = min(n_genes, start + block)
        profiles[start:stop, k] *= 8.0
    profiles /= profiles.sum(axis=0, keepdims=True)

    base, rem = divmod(n_cells, n_types)
    ref_counts = np.zeros((n_cells, n_genes), dtype=np.float32)
    cell_types = []
    row = 0
    for k in range(n_types):
        n_k = base + (1 if k < rem else 0)
        for _ in range(n_k):
            n_umi = int(rng.integers(1200, 5000))
            ref_counts[row, :] = rng.poisson(profiles[:, k] * n_umi).astype(np.float32)
            cell_types.append(f"Type_{k}")
            row += 1

    ref = anndata.AnnData(X=ref_counts, obs={"cell_type": cell_types})
    ref.var_names = [f"Gene_{i}" for i in range(n_genes)]

    weights = rng.dirichlet(alpha=np.ones(n_types) * 0.7, size=n_pixels)
    spatial_counts = np.zeros((n_pixels, n_genes), dtype=np.float32)
    for i in range(n_pixels):
        n_umi = int(rng.integers(800, 3500))
        spatial_counts[i, :] = rng.poisson((profiles @ weights[i]) * n_umi).astype(np.float32)

    spatial = anndata.AnnData(
        X=spatial_counts,
        obs={"x": rng.uniform(0, 100, n_pixels), "y": rng.uniform(0, 100, n_pixels)},
    )
    spatial.var_names = ref.var_names.copy()
    return ref, spatial


def weights_hash(w):
    return hashlib.sha256(np.round(w, decimals=4).tobytes()).hexdigest()[:16]


def _run(queue):
    try:
        torch.set_num_threads(4)
        ref_ad, sp_ad = make_data(**CONFIG)
        reference = Reference(ref_ad, cell_min=10, min_UMI=10)
        config = RCTDConfig(dtype="float64", device="cpu", N_fit=40, N_epoch=4, UMI_min=20, UMI_min_sigma=50)

        t0 = time.perf_counter()
        result = run_rctd(spatial=sp_ad, reference=reference, mode="full", config=config, batch_size=BATCH_SIZE, sigma_override=100)
        elapsed = time.perf_counter() - t0
        maxrss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        queue.put({
            "elapsed_s": elapsed,
            "maxrss_mb": maxrss_kb / 1024,
            "n_pixels": result.weights.shape[0],
            "weights_hash": weights_hash(result.weights),
        })
    except Exception:
        queue.put({"error": traceback.format_exc()})


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run, args=(queue,), daemon=True)
    proc.start()
    proc.join(timeout=300)

    if proc.is_alive():
        proc.kill()
        proc.join()
        print("TIMEOUT")
        sys.exit(1)

    if queue.empty():
        print(f"ERROR: no result, exitcode={proc.exitcode}")
        sys.exit(1)

    result = queue.get()
    if "error" in result:
        print(f"CRASH:\n{result['error']}")
        sys.exit(1)

    print("---")
    print(f"elapsed_s:     {result['elapsed_s']:.2f}")
    print(f"maxrss_mb:     {result['maxrss_mb']:.1f}")
    print(f"n_pixels:      {result['n_pixels']}")
    print(f"weights_hash:  {result['weights_hash']}")
