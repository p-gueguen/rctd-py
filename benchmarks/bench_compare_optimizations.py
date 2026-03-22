"""A/B benchmark: baseline vs CUDA-optimized IRWLS solver.

Compares speed and verifies bit-identical accuracy between:
  - Baseline: separate prediction → calc_q_all → grad → hess
  - Optimized: fused torch.compile graph + compiled simplex/K=2 solvers

Usage:
    uv run python benchmarks/bench_compare_optimizations.py [--n-pixels 500] [--runs 3]
"""

import argparse
import hashlib
import time

import numpy as np
import torch

from rctd._irwls import solve_irwls_batch_shared
from rctd._likelihood import (
    _calc_q_all_impl,
    build_x_vals,
    calc_q_all,
    compute_q_matrix,
    compute_spline_coefficients,
)


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


def weights_hash(weights):
    w_rounded = np.round(weights, 8)
    return hashlib.md5(w_rounded.tobytes()).hexdigest()[:16]


def run_solver(P, Y, nUMI, Q, SQ, xv, calc_q_fn=None, constrain=True):
    """Run solver, return (weights_np, elapsed_s)."""
    torch.manual_seed(0)
    t0 = time.perf_counter()
    w, conv = solve_irwls_batch_shared(
        P, Y, nUMI, Q, SQ, xv,
        constrain=constrain,
        _calc_q_fn=calc_q_fn,
    )
    elapsed = time.perf_counter() - t0
    return w.numpy(), conv.numpy(), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pixels", type=int, default=500)
    parser.add_argument("--n-genes", type=int, default=200)
    parser.add_argument("--n-types", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    print("=== CUDA Optimization A/B Benchmark ===")
    print(f"Config: {args.n_pixels} pixels, {args.n_genes} genes, {args.n_types} types")
    print(f"Device: cpu (torch.compile generates C++ kernels)")
    print()

    # Generate data
    profiles, counts, nUMIs = generate_data(
        args.n_pixels, args.n_genes, args.n_types
    )
    x_vals_np = build_x_vals()
    q_mat_np = compute_q_matrix(1.0, x_vals_np)
    sq_mat_np = compute_spline_coefficients(q_mat_np, x_vals_np)

    P = torch.tensor(profiles)
    Y = torch.tensor(counts)
    nUMI = torch.tensor(nUMIs)
    Q = torch.tensor(q_mat_np)
    SQ = torch.tensor(sq_mat_np)
    xv = torch.tensor(x_vals_np)

    # --- Baseline: force non-fused path via _calc_q_fn ---
    # Passing _calc_q_fn disables the fused path
    print("--- Baseline (separate kernels, eager calc_q_all) ---")
    for i in range(args.warmup):
        _, _, t = run_solver(P, Y, nUMI, Q, SQ, xv, calc_q_fn=_calc_q_all_impl)
        print(f"  warmup {i+1}: {t:.3f}s")

    baseline_times = []
    baseline_hash = None
    for i in range(args.runs):
        w, conv, t = run_solver(P, Y, nUMI, Q, SQ, xv, calc_q_fn=_calc_q_all_impl)
        h = weights_hash(w)
        baseline_times.append(t)
        if baseline_hash is None:
            baseline_hash = h
        converged_pct = conv.mean() * 100
        print(f"  run {i+1}: {t:.3f}s  hash={h}  converged={converged_pct:.1f}%")

    # --- Optimized: fused torch.compile path (default) ---
    print("\n--- Optimized (fused compiled graph) ---")
    for i in range(args.warmup):
        _, _, t = run_solver(P, Y, nUMI, Q, SQ, xv)
        print(f"  warmup {i+1}: {t:.3f}s")

    opt_times = []
    opt_hash = None
    for i in range(args.runs):
        w, conv, t = run_solver(P, Y, nUMI, Q, SQ, xv)
        h = weights_hash(w)
        opt_times.append(t)
        if opt_hash is None:
            opt_hash = h
        converged_pct = conv.mean() * 100
        print(f"  run {i+1}: {t:.3f}s  hash={h}  converged={converged_pct:.1f}%")

    # --- Comparison ---
    print("\n=== Results ===")
    baseline_med = np.median(baseline_times)
    opt_med = np.median(opt_times)
    speedup = baseline_med / opt_med if opt_med > 0 else float("inf")

    print(f"Baseline median: {baseline_med:.3f}s")
    print(f"Optimized median: {opt_med:.3f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Baseline hash: {baseline_hash}")
    print(f"Optimized hash: {opt_hash}")
    print(f"Hash match: {'YES' if baseline_hash == opt_hash else 'NO — ACCURACY REGRESSION'}")


if __name__ == "__main__":
    main()
