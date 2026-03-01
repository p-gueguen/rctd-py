#!/usr/bin/env python3
"""GPU benchmark for rctd-py IRWLS solver.

Run on GPU node: python scripts/benchmark_gpu.py
"""

import time
import sys
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

REF_DIR = Path(__file__).parent / "r_reference"


def load(name):
    return np.load(REF_DIR / f"{name}.npy", allow_pickle=True)


def main():
    print("=" * 60)
    print("GPU BENCHMARK: rctd-py IRWLS solver")
    print("=" * 60)
    print(f"  JAX backend: {jax.default_backend()}")
    print(f"  Devices: {jax.devices()}")

    from rctd._irwls import solve_irwls_batch
    from rctd._likelihood import compute_spline_coefficients, load_cached_q_matrices

    # Load data
    renorm_profiles = load("renorm_profiles")
    gene_list_reg = load("gene_list_reg")
    gene_list_bulk = load("gene_list_bulk")
    sp_counts = load("sp_counts")
    sp_gene_names = load("sp_gene_names")
    sp_nUMI = load("sp_nUMI")
    sp_pixel_names = load("sp_pixel_names")
    r_full_pixels = load("full_pixel_names")

    cached = load_cached_q_matrices()
    x_vals_cached = cached["X_vals"]

    # Gene matching
    bulk_gene_to_idx = {str(g): i for i, g in enumerate(gene_list_bulk)}
    reg_in_bulk_idx = np.array([bulk_gene_to_idx[str(g)] for g in gene_list_reg if str(g) in bulk_gene_to_idx])
    gene_list_reg_matched = np.array([str(g) for g in gene_list_reg if str(g) in bulk_gene_to_idx])
    profiles_reg = renorm_profiles[reg_in_bulk_idx, :]

    sp_gene_to_idx = {str(g): i for i, g in enumerate(sp_gene_names)}
    gene_idx_matched = np.array([sp_gene_to_idx[str(g)] for g in gene_list_reg_matched])
    sp_counts_matched = sp_counts[:, gene_idx_matched]

    r_pixel_set = set(str(p) for p in r_full_pixels)
    py_pixel_indices = [i for i, p in enumerate(sp_pixel_names) if str(p) in r_pixel_set]

    counts_base = sp_counts_matched[py_pixel_indices]
    numi_base = sp_nUMI[py_pixel_indices]
    P = jnp.array(profiles_reg)

    sigma_int = int(round(float(load("sigma")) * 100))
    Q_sigma = jnp.array(cached[f"Q_{sigma_int}"])
    SQ_sigma = jnp.array(compute_spline_coefficients(
        np.array(cached[f"Q_{sigma_int}"]), x_vals_cached
    ))
    xv = jnp.array(x_vals_cached)

    N_base = counts_base.shape[0]
    G = counts_base.shape[1]
    K = P.shape[1]
    print(f"  Base data: {N_base} pixels, {G} genes, {K} cell types")

    # Benchmark at increasing scale
    scale_factors = [1, 10, 100, 500, 1000]
    if jax.default_backend() == "gpu":
        scale_factors.extend([5000, 10000, 20000])

    n_repeats = 5

    print(f"\n{'Pixels':>8} | {'Median (s)':>10} | {'Min (s)':>10} | {'Pixels/s':>10} | {'Conv':>6}")
    print("-" * 60)

    for scale in scale_factors:
        cb = jnp.array(np.tile(counts_base, (scale, 1)))
        nb = jnp.array(np.tile(numi_base, (scale,)))
        S_batch = nb[:, None, None] * P[None, :, :]
        n = cb.shape[0]

        # Warmup
        w, c = solve_irwls_batch(
            S_batch[:min(2, n)], cb[:min(2, n)], nb[:min(2, n)],
            Q_sigma, SQ_sigma, xv,
            max_iter=50, min_change=0.001, constrain=False,
        )
        w.block_until_ready()

        times = []
        for r in range(n_repeats):
            t0 = time.perf_counter()
            weights, conv = solve_irwls_batch(
                S_batch, cb, nb,
                Q_sigma, SQ_sigma, xv,
                max_iter=50, min_change=0.001, constrain=False,
            )
            weights.block_until_ready()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        median_t = np.median(times)
        min_t = np.min(times)
        conv_count = int(conv.sum())
        print(f"{n:>8} | {median_t:>10.4f} | {min_t:>10.4f} | {n/median_t:>10.0f} | {conv_count:>6}/{n}")

        # Don't run too long
        if median_t > 60:
            print("  (skipping larger sizes, taking >60s)")
            break


if __name__ == "__main__":
    main()
