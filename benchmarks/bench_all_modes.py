"""All-modes A/B benchmark: baseline vs CUDA-optimized across full/doublet/multi.

Tests accuracy (weights hash) and speed for each RCTD mode.

Usage:
    uv run python benchmarks/bench_all_modes.py [--n-pixels 100] [--runs 3]
"""

import argparse
import hashlib
import time

import numpy as np
import torch

from rctd._full import run_full_mode
from rctd._doublet import run_doublet_mode
from rctd._multi import run_multi_mode
from rctd._likelihood import build_x_vals, compute_q_matrix, compute_spline_coefficients
from rctd._types import RCTDConfig


def generate_data(n_pixels, n_genes, n_types, seed=2025):
    rng = np.random.default_rng(seed)
    profiles = rng.exponential(0.01, size=(n_genes, n_types)).astype(np.float64)
    profiles = profiles / profiles.sum(axis=0, keepdims=True)
    nUMIs = rng.integers(200, 5000, size=n_pixels).astype(np.float64)
    counts = np.zeros((n_pixels, n_genes), dtype=np.float64)
    for i in range(n_pixels):
        true_w = rng.dirichlet(np.ones(n_types))
        lam = (profiles @ true_w) * nUMIs[i]
        counts[i] = rng.poisson(np.clip(lam, 0, 1e6))
    return profiles, counts, nUMIs


def hash_result(result):
    """Hash the primary weights from any result type."""
    w = result.weights
    return hashlib.md5(np.round(w, 8).tobytes()).hexdigest()[:16]


def hash_doublet_extra(result):
    """Hash the doublet-specific outputs."""
    parts = [
        np.round(result.weights_doublet, 8).tobytes(),
        result.spot_class.tobytes(),
        result.first_type.tobytes(),
        result.second_type.tobytes(),
    ]
    return hashlib.md5(b"".join(parts)).hexdigest()[:16]


def hash_multi_extra(result):
    """Hash the multi-specific outputs."""
    parts = [
        np.round(result.sub_weights, 8).tobytes(),
        result.cell_type_indices.tobytes(),
        result.n_types.tobytes(),
    ]
    return hashlib.md5(b"".join(parts)).hexdigest()[:16]


def run_full(profiles, counts, nUMIs, q_mat, sq_mat, x_vals, device="cpu"):
    t0 = time.perf_counter()
    result = run_full_mode(
        spatial_counts=counts,
        spatial_numi=nUMIs,
        norm_profiles=profiles,
        cell_type_names=[f"type_{i}" for i in range(profiles.shape[1])],
        q_mat=q_mat,
        sq_mat=sq_mat,
        x_vals=x_vals,
        batch_size=10000,
        device=device,
    )
    elapsed = time.perf_counter() - t0
    return result, elapsed


def run_doublet(profiles, counts, nUMIs, q_mat, sq_mat, x_vals, config, device="cpu"):
    t0 = time.perf_counter()
    result = run_doublet_mode(
        spatial_counts=counts,
        spatial_numi=nUMIs,
        norm_profiles=profiles,
        cell_type_names=[f"type_{i}" for i in range(profiles.shape[1])],
        q_mat=q_mat,
        sq_mat=sq_mat,
        x_vals=x_vals,
        config=config,
        batch_size=10000,
        device=device,
    )
    elapsed = time.perf_counter() - t0
    return result, elapsed


def run_multi(profiles, counts, nUMIs, q_mat, sq_mat, x_vals, config, device="cpu"):
    t0 = time.perf_counter()
    result = run_multi_mode(
        spatial_counts=counts,
        spatial_numi=nUMIs,
        norm_profiles=profiles,
        cell_type_names=[f"type_{i}" for i in range(profiles.shape[1])],
        q_mat=q_mat,
        sq_mat=sq_mat,
        x_vals=x_vals,
        config=config,
        batch_size=10000,
        device=device,
    )
    elapsed = time.perf_counter() - t0
    return result, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pixels", type=int, default=100)
    parser.add_argument("--n-genes", type=int, default=100)
    parser.add_argument("--n-types", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    config = RCTDConfig()

    print("=== All-Modes CUDA Optimization Benchmark ===")
    print(f"Config: {args.n_pixels} pixels, {args.n_genes} genes, {args.n_types} types")
    print()

    profiles, counts, nUMIs = generate_data(args.n_pixels, args.n_genes, args.n_types)
    x_vals = build_x_vals()
    q_mat = compute_q_matrix(1.0, x_vals)
    sq_mat = compute_spline_coefficients(q_mat, x_vals)

    modes = {
        "full": lambda: run_full(profiles, counts, nUMIs, q_mat, sq_mat, x_vals),
        "doublet": lambda: run_doublet(profiles, counts, nUMIs, q_mat, sq_mat, x_vals, config),
        "multi": lambda: run_multi(profiles, counts, nUMIs, q_mat, sq_mat, x_vals, config),
    }

    for mode_name, run_fn in modes.items():
        print(f"\n{'='*60}")
        print(f"  Mode: {mode_name.upper()}")
        print(f"{'='*60}")

        # Warmup
        for i in range(args.warmup):
            _, t = run_fn()
            print(f"  warmup {i+1}: {t:.3f}s")

        # Timed runs
        times = []
        hashes = []
        extra_hashes = []
        for i in range(args.runs):
            result, t = run_fn()
            times.append(t)
            h = hash_result(result)
            hashes.append(h)

            if mode_name == "doublet":
                eh = hash_doublet_extra(result)
                extra_hashes.append(eh)
                print(f"  run {i+1}: {t:.3f}s  weights_hash={h}  doublet_hash={eh}")
            elif mode_name == "multi":
                eh = hash_multi_extra(result)
                extra_hashes.append(eh)
                print(f"  run {i+1}: {t:.3f}s  weights_hash={h}  multi_hash={eh}")
            else:
                print(f"  run {i+1}: {t:.3f}s  weights_hash={h}")

        consistent = len(set(hashes)) == 1
        extra_consistent = len(set(extra_hashes)) <= 1 if extra_hashes else True
        print(f"\n  Median: {np.median(times):.3f}s")
        print(f"  Hash consistent across runs: {'YES' if consistent else 'NO'}")
        if extra_hashes:
            print(f"  Extra hash consistent: {'YES' if extra_consistent else 'NO'}")

    print(f"\n{'='*60}")
    print("  DONE — all modes tested")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
