"""Quick profiler to find bottlenecks in the IRWLS solver."""
import cProfile
import os
import pstats
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rctd._rctd import run_rctd
from rctd._reference import Reference
from rctd._types import RCTDConfig

# Small dataset for profiling
import anndata

def make_data():
    rng = np.random.default_rng(2025)
    n_genes, n_types, n_cells, n_pixels = 500, 8, 5000, 2000
    profiles = rng.exponential(0.0015, size=(n_genes, n_types)).astype(np.float64)
    block = n_genes // n_types
    for k in range(n_types):
        profiles[k*block:(k+1)*block, k] *= 8.0
    profiles /= profiles.sum(axis=0, keepdims=True)

    ref_counts = np.zeros((n_cells, n_genes), dtype=np.float32)
    cell_types = []
    base, rem = divmod(n_cells, n_types)
    row = 0
    for k in range(n_types):
        for _ in range(base + (1 if k < rem else 0)):
            ref_counts[row] = rng.poisson(profiles[:, k] * rng.integers(1200, 5000)).astype(np.float32)
            cell_types.append(f"Type_{k}")
            row += 1

    ref = anndata.AnnData(X=ref_counts, obs={"cell_type": cell_types})
    ref.var_names = [f"Gene_{i}" for i in range(n_genes)]

    weights = rng.dirichlet(np.ones(n_types) * 0.7, size=n_pixels)
    sp = np.zeros((n_pixels, n_genes), dtype=np.float32)
    for i in range(n_pixels):
        sp[i] = rng.poisson((profiles @ weights[i]) * rng.integers(800, 3500)).astype(np.float32)

    spatial = anndata.AnnData(X=sp, obs={"x": rng.uniform(0, 100, n_pixels), "y": rng.uniform(0, 100, n_pixels)})
    spatial.var_names = ref.var_names.copy()
    return ref, spatial

ref, sp = make_data()
reference = Reference(ref, cell_min=10, min_UMI=10)
config = RCTDConfig(dtype="float64", device="cpu", N_fit=40, N_epoch=4, UMI_min=20, UMI_min_sigma=50)

# Warm up torch.compile
run_rctd(sp, reference, mode="full", config=config, batch_size=5000, sigma_override=100)

# Profile second run
profiler = cProfile.Profile()
profiler.enable()
run_rctd(sp, reference, mode="full", config=config, batch_size=5000, sigma_override=100)
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
stats.print_stats(30)
print("\n--- BY TOTTIME ---")
stats.sort_stats("tottime")
stats.print_stats(30)
