"""Fixed benchmark for autoresearch experiments on rctd-py.

DO NOT MODIFY THIS FILE. This is the evaluation harness.

Generates deterministic synthetic data and benchmarks rctd-py across three
modes (full, doublet, multi). Reports elapsed time, peak memory, and a
correctness hash to detect regressions.

Usage:
    uv run autoresearch/benchmark.py              # run all benchmarks
    uv run autoresearch/benchmark.py --mode full  # run only full mode
"""

import argparse
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
from scipy import sparse

# ── Add src/ to path so rctd is importable ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rctd._rctd import run_rctd
from rctd._reference import Reference
from rctd._types import RCTDConfig

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

BATCH_SIZE = 10000
BENCHMARK_CONFIGS = {
    "full": {
        "n_genes": 1000,
        "n_cells": 50000,
        "n_types": 12,
        "n_pixels": 50000,
        "seed": 2025,
        "mode": "full",
    },
    "doublet": {
        "n_genes": 1000,
        "n_cells": 50000,
        "n_types": 12,
        "n_pixels": 50000,
        "seed": 2026,
        "mode": "doublet",
    },
    "multi": {
        "n_genes": 1000,
        "n_cells": 50000,
        "n_types": 12,
        "n_pixels": 50000,
        "seed": 2027,
        "mode": "multi",
    },
}

# ---------------------------------------------------------------------------
# Data generation (fixed, do not modify)
# ---------------------------------------------------------------------------


def make_reference_and_spatial(
    n_genes: int,
    n_cells: int,
    n_types: int,
    n_pixels: int,
    seed: int,
):
    """Create deterministic synthetic reference + spatial data."""
    rng = np.random.default_rng(seed)

    profiles = rng.exponential(0.0015, size=(n_genes, n_types)).astype(np.float64)
    block = max(1, n_genes // n_types)
    for k in range(n_types):
        start = k * block
        stop = min(n_genes, start + block)
        profiles[start:stop, k] *= 8.0
    profiles /= profiles.sum(axis=0, keepdims=True)

    base_cells_per_type, remainder = divmod(n_cells, n_types)
    ref_counts = np.zeros((n_cells, n_genes), dtype=np.float32)
    cell_types = []
    row = 0
    for k in range(n_types):
        n_cells_k = base_cells_per_type + (1 if k < remainder else 0)
        for _ in range(n_cells_k):
            n_umi = int(rng.integers(1200, 5000))
            lam = profiles[:, k] * n_umi
            ref_counts[row, :] = rng.poisson(lam).astype(np.float32)
            cell_types.append(f"Type_{k}")
            row += 1

    ref = anndata.AnnData(X=ref_counts, obs={"cell_type": cell_types})
    ref.var_names = [f"Gene_{i}" for i in range(n_genes)]

    weights = rng.dirichlet(alpha=np.ones(n_types) * 0.7, size=n_pixels)
    spatial_counts = np.zeros((n_pixels, n_genes), dtype=np.float32)
    for i in range(n_pixels):
        n_umi = int(rng.integers(800, 3500))
        lam = (profiles @ weights[i]) * n_umi
        spatial_counts[i, :] = rng.poisson(lam).astype(np.float32)

    spatial = anndata.AnnData(
        X=spatial_counts,
        obs={
            "x": rng.uniform(0.0, 100.0, size=n_pixels),
            "y": rng.uniform(0.0, 100.0, size=n_pixels),
        },
    )
    spatial.var_names = ref.var_names.copy()

    return ref, spatial


# ---------------------------------------------------------------------------
# Correctness hash (fixed, do not modify)
# ---------------------------------------------------------------------------


def weights_hash(weights: np.ndarray) -> str:
    """Compute a short hash of the output weights for correctness checking.

    Uses rounded weights (4 decimal places) to tolerate minor floating-point
    variations across platforms while catching real regressions.
    """
    rounded = np.round(weights, decimals=4)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Benchmark runner (fixed, do not modify)
# ---------------------------------------------------------------------------


def _run_benchmark(mode: str, queue):
    """Run a single benchmark in a subprocess."""
    try:
        cfg = BENCHMARK_CONFIGS[mode]
        torch.set_num_threads(1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = "float32" if device == "cuda" else "float64"

        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Generate data
        ref_adata, spatial_adata = make_reference_and_spatial(
            n_genes=cfg["n_genes"],
            n_cells=cfg["n_cells"],
            n_types=cfg["n_types"],
            n_pixels=cfg["n_pixels"],
            seed=cfg["seed"],
        )
        reference = Reference(ref_adata, cell_min=10, min_UMI=10)

        config = RCTDConfig(
            dtype=dtype,
            device=device,
            N_fit=60,
            N_epoch=4,
            UMI_min=20,
            UMI_min_sigma=50,
        )

        # Benchmark
        t0 = time.perf_counter()
        result = run_rctd(
            spatial=spatial_adata,
            reference=reference,
            mode=mode,
            config=config,
            batch_size=BATCH_SIZE,
            sigma_override=100,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        cuda_peak = 0
        if device == "cuda":
            cuda_peak = torch.cuda.max_memory_allocated()

        maxrss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        w_hash = weights_hash(result.weights.numpy() if isinstance(result.weights, torch.Tensor) else result.weights)

        queue.put({
            "mode": mode,
            "elapsed_s": elapsed,
            "peak_vram_mb": cuda_peak / (1024 * 1024) if cuda_peak else 0.0,
            "maxrss_mb": maxrss_kb / 1024,
            "n_pixels": result.weights.shape[0],
            "device": device,
            "dtype": dtype,
            "weights_hash": w_hash,
        })

    except Exception:
        queue.put({"mode": mode, "error": traceback.format_exc()})


def run_benchmark(mode: str, timeout_s: int = 600) -> dict:
    """Run benchmark in isolated subprocess with timeout."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run_benchmark, args=(mode, queue), daemon=True)
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return {"mode": mode, "error": f"TIMEOUT after {timeout_s}s"}

    if queue.empty():
        return {"mode": mode, "error": f"No result; exitcode={proc.exitcode}"}

    return queue.get()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rctd-py autoresearch benchmark")
    parser.add_argument(
        "--mode",
        choices=["full", "doublet", "multi", "all"],
        default="all",
        help="Which benchmark mode to run (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per benchmark in seconds (default: 600)",
    )
    args = parser.parse_args()

    modes = list(BENCHMARK_CONFIGS.keys()) if args.mode == "all" else [args.mode]

    print(f"rctd-py autoresearch benchmark")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    for mode in modes:
        cfg = BENCHMARK_CONFIGS[mode]
        print(f"--- {mode} mode ---")
        print(f"Dataset: {cfg['n_pixels']} pixels, {cfg['n_genes']} genes, {cfg['n_types']} types")

        result = run_benchmark(mode, timeout_s=args.timeout)

        if "error" in result:
            print(f"FAILED: {result['error']}")
            print()
            continue

        print(f"---")
        print(f"elapsed_s:     {result['elapsed_s']:.2f}")
        print(f"peak_vram_mb:  {result['peak_vram_mb']:.1f}")
        print(f"maxrss_mb:     {result['maxrss_mb']:.1f}")
        print(f"n_pixels:      {result['n_pixels']}")
        print(f"device:        {result['device']}")
        print(f"dtype:         {result['dtype']}")
        print(f"weights_hash:  {result['weights_hash']}")
        print()
