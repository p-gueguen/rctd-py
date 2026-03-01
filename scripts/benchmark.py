#!/usr/bin/env python3
"""Benchmark rctd-py IRWLS solver vs R spacexr.

Run: python scripts/benchmark.py
Prerequisites: Run scripts/generate_r_reference.R first.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

REF_DIR = Path("scripts/r_reference")


def load(name):
    return np.load(REF_DIR / f"{name}.npy", allow_pickle=True)


def benchmark_python(n_repeats=5, scale_factors=None):
    """Benchmark Python IRWLS on vignette data."""
    from rctd._irwls import solve_irwls, solve_irwls_batch
    from rctd._likelihood import compute_spline_coefficients, load_cached_q_matrices

    # Load data (same setup as compare_r_python.py TEST 8)
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

    counts_batch = jnp.array(sp_counts_matched[py_pixel_indices])
    numi_batch = jnp.array(sp_nUMI[py_pixel_indices])
    P = jnp.array(profiles_reg)

    sigma_int = int(round(float(load("sigma")) * 100))
    Q_sigma = jnp.array(cached[f"Q_{sigma_int}"])
    SQ_sigma = jnp.array(compute_spline_coefficients(
        np.array(cached[f"Q_{sigma_int}"]), x_vals_cached
    ))
    xv = jnp.array(x_vals_cached)

    N_pixels = counts_batch.shape[0]
    G_genes = counts_batch.shape[1]
    K_types = P.shape[1]
    print(f"  Data: {N_pixels} pixels, {G_genes} genes, {K_types} cell types")
    print(f"  Backend: {jax.default_backend()}")

    if scale_factors is None:
        scale_factors = [1]

    for scale in scale_factors:
        if scale == 1:
            cb = counts_batch
            nb = numi_batch
        else:
            # Tile to simulate larger datasets
            cb = jnp.tile(counts_batch, (scale, 1))
            nb = jnp.tile(numi_batch, (scale,))

        S_batch = nb[:, None, None] * P[None, :, :]
        n = cb.shape[0]

        # Warmup (JIT compilation)
        warmup_w, _ = solve_irwls_batch(
            S_batch[:2], cb[:2], nb[:2],
            Q_sigma, SQ_sigma, xv,
            max_iter=50, min_change=0.001, constrain=False,
        )
        # Block until done
        warmup_w.block_until_ready()

        # Timed runs
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
        print(f"  {n:>6} pixels: median={median_t:.4f}s  min={min_t:.4f}s  "
              f"({n/median_t:.0f} pixels/s)  converged={int(conv.sum())}/{n}")

    return median_t  # Return last timing


def benchmark_r(n_repeats=5):
    """Time R spacexr on the same vignette data."""
    import subprocess
    r_script = """
library(spacexr)

# Load vignette data (matching spacexr vignette exactly)
refdir <- system.file("extdata", "Reference", "Vignette", package = "spacexr")
counts <- as.matrix(read.csv(file.path(refdir, "dge.csv"), row.names=1))
meta <- read.csv(file.path(refdir, "meta_data.csv"))
cell_types <- setNames(as.factor(meta$cluster), meta$barcode)
nUMI_ref <- setNames(meta$nUMI, meta$barcode)
reference <- Reference(counts, cell_types, nUMI_ref)

spdir <- system.file("extdata", "SpatialRNA", "Vignette", package = "spacexr")
sp_counts <- as.matrix(read.csv(file.path(spdir, "MappedDGEForR.csv"), row.names=1))
coords <- read.csv(file.path(spdir, "BeadLocationsForR.csv"), row.names=1)
sp_nUMI <- colSums(sp_counts)
puck <- SpatialRNA(coords, sp_counts, sp_nUMI)

# Time full RCTD pipeline (create + fitBulk + choose_sigma + fitPixels)
times <- numeric(NREPS)
for (i in seq_len(NREPS)) {
    t0 <- proc.time()
    myRCTD <- create.RCTD(puck, reference, max_cores = 1)
    myRCTD <- run.RCTD(myRCTD, doublet_mode = "full")
    t1 <- proc.time()
    times[i] <- (t1 - t0)["elapsed"]
}
cat("R_TIMES:", paste(times, collapse=","), "\\n")
"""
    r_script = r_script.replace("NREPS", str(n_repeats))

    result = subprocess.run(
        ["bash", "-c", "source /usr/share/lmod/lmod/init/bash && module load Dev/R/4.5.0 && Rscript -e " + repr(r_script)],
        capture_output=True, text=True, timeout=600,
    )
    # Parse times from output
    for line in result.stdout.split("\n"):
        if line.startswith("R_TIMES:"):
            times = [float(x) for x in line.split(":")[1].strip().split(",")]
            return times
    # If parsing failed, print full output
    print("  R stdout:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    print("  R stderr:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
    return None


def main():
    print("=" * 60)
    print("BENCHMARK: rctd-py vs R spacexr")
    print("=" * 60)

    # Python benchmark
    print("\n--- Python (JAX) ---")
    py_time = benchmark_python(n_repeats=5, scale_factors=[1, 10, 100])

    # R benchmark
    print("\n--- R (spacexr, single core) ---")
    try:
        r_times = benchmark_r(n_repeats=3)
        if r_times:
            r_median = np.median(r_times)
            print(f"  71 pixels: median={r_median:.4f}s  times={[f'{t:.2f}' for t in r_times]}")
        else:
            r_median = None
            print("  Failed to parse R times")
    except Exception as e:
        r_median = None
        print(f"  R benchmark failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if r_median:
        # Compare at 71 pixels
        py_71 = benchmark_python(n_repeats=5, scale_factors=[1])
        speedup = r_median / py_71
        print(f"  R (71 pixels, 1 core):  {r_median:.4f}s")
        print(f"  Python (71 pixels, CPU): {py_71:.4f}s")
        print(f"  Speedup: {speedup:.1f}x")


if __name__ == "__main__":
    main()
