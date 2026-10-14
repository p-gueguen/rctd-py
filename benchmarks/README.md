# Benchmarks

Development benchmarks used while optimising rctd-py. The published runtime and memory benchmark
against R spacexr (15 datasets, three modes) is in [docs/benchmarks.md](../docs/benchmarks.md),
with its setup; its scripts are not in this repository.

## Files

| File | Purpose |
|------|---------|
| `bench_gpu.py` | Fixed synthetic benchmark (50k pixels, 1000 genes, 12 types) for the autoresearch loop |
| `bench_gpu_baseline.py` | Same benchmark with the original `solve_irwls_batch`, for comparison |
| `bench_ab_compare.py` | Minimal A/B timing of the deconvolution step |
| `profile_irwls.py` | Times the components of the IRWLS inner loop |
| `bench_visiumhd.py`, `SBATCH_benchmark_visiumhd.sh` | Visium HD mouse brain, full mode |
| `visiumhd_proportions.py`, `SBATCH_visiumhd_proportions.sh` | Cell type proportions from that run |
| `program.md` | Guide for the autonomous optimisation loop (autoresearch agent) |

## Development Benchmarks

The `bench_gpu.py` script runs a fixed synthetic benchmark (50k pixels, 1000 genes, 12 types) for the autoresearch optimization loop. It is READ-ONLY — optimizations are validated by checking that the weights hash is unchanged.

Key optimizations found by the autoresearch agent's loop (branch `autoresearch/mar14`):
- Eager sigma estimation (avoid torch.compile overhead for small batches)
- GPU eigh for K<=16 (pairwise fits in doublet mode) — **85% faster**, biggest single win
- Shared-profile P matrix (avoid materializing per-pixel S_batch)
- Active pixel compaction (skip converged pixels in IRWLS)
- torch.compile on `calc_q_all` (5% faster, 60% less VRAM)
- Fix graph break in `calc_q_all` (`.item()` → tensor)
- Fuse PSD reconstruction `V*sqrt(λ)` (1.4% faster)
- Precompute `P.T` contiguous (marginal, cleaner)

## Optimizations tried and discarded

| Experiment | Result | Reason discarded |
|------------|--------|------------------|
| Adaptive QP with early exit | 9.7% slower | Slower + hash changed |
| QP sweeps 50→30 | No change | No measurable improvement at 1.2s scale |
| Remove active pixel compaction | 15% slower | Compaction helps significantly |
| Diagonal loading PSD (skip eigh) | 54% faster | Hash changed — eigh required for correctness |
| float32 eigh | 41% faster | Hash changed — precision matters |

## Post-autoresearch optimizations (feat/cli branch)

- **Analytical K=2 PSD projection** (`_psd_2x2`): Closed-form eigendecomposition for 2×2 symmetric matrices avoids cuSOLVER entirely. Used in doublet mode pairwise fits.
- **Analytical K=2 box-QP** (`_solve_box_qp_2`): Cramer's rule + clamping replaces iterative Gauss-Seidel coordinate descent for 2-variable problems.
- **Auto batch size** (`auto_batch_size`): Estimates optimal batch size from available VRAM budget.

## Results log

Summary of the kept optimizations (the full experiment log was not kept in the repository):

| Commit | elapsed_s | VRAM (MB) | Description |
|--------|-----------|-----------|-------------|
| baseline | 8.294 | 2386 | Shared-profile P_outer Hessian + active pixel compaction |
| `27d2671` | 7.884 | 953 | torch.compile on calc_q_all |
| `423b974` | 7.839 | 1027 | Fix graph break in calc_q_all |
| `460cc7c` | **1.200** | 6335 | **GPU-side eigh for K<=16 (85% faster)** |
| `4a21101` | 1.200 | 6335 | Avoid .item() sync in compaction |
| `86ad695` | 1.200 | 6641 | Pre-transfer all spatial data to GPU |
| `35bf64b` | 1.183 | 6641 | Fuse PSD reconstruction V*sqrt(λ) |
| `c4c4cc5` | 1.182 | 6641 | Precompute P.T contiguous |

Cross-GPU: L40S achieved 1.52s on the same benchmark (5.4x speedup over the CPU-eigh baseline of 8.25s).
