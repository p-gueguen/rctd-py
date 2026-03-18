# Benchmarks

Benchmark infrastructure for measuring rctd-py performance against R spacexr.

## Directory Structure

```
benchmarks/
├── README.md                          # This file
├── program.md                         # Autonomous optimization loop guide (autoresearch agent)
│
├── ## Production Benchmarks (3 datasets, 3 modes)
├── SBATCH_benchmark_mouse_brain.sh    # GPU: mouse brain (36k cells, K=22), Blackwell
├── SBATCH_benchmark_region1.sh        # GPU: Xenium Region 1 (14k cells, K=45), Blackwell
├── SBATCH_benchmark_region3.sh        # GPU: Xenium Region 3 (58k cells, K=45), Blackwell
├── SBATCH_r_mouse_brain.sh            # CPU: R spacexr baseline for mouse brain (8 cores)
├── SBATCH_r_regions_all_modes.sh      # CPU: R spacexr full+multi for Regions 1/3 (8 cores)
│
├── ## Development Benchmarks (optimization experiments)
├── bench_gpu.py                       # Fixed synthetic benchmark (READ-ONLY, for autoresearch)
├── bench_gpu_baseline.py              # Baseline benchmark for comparison
├── profile_irwls.py                   # Profiling script for IRWLS solver
├── sbatch_cli_demo.sh                 # CLI demo on GPU (all 3 modes + JSON output)
├── SBATCH_region1_blackwell.sh        # Early Region 1 benchmark (v1)
├── SBATCH_region1_blackwell_v2.sh     # Early Region 1 benchmark (v2, with eager sigma)
├── SBATCH_region3_blackwell.sh        # Early Region 3 benchmark (v1)
├── SBATCH_region3_blackwell_v2.sh     # Early Region 3 benchmark (v2, with eager sigma)
│
├── ## Experiment Logs (autoresearch optimization loop)
├── experiment_*.log / .err            # Individual experiment runs on GPU nodes
├── batch_*.log / .err                 # Batch experiment runs
├── bench_*.log / .err                 # Benchmark comparison runs
├── profile_*.log / .err               # Profiling runs
└── test_*.log / .err                  # Test runs
```

## Production Benchmark Scheme

### Datasets

| Dataset | Cells (filtered) | Cell Types (K) | Species | Reference |
|---------|-----------------|----------------|---------|-----------|
| Mouse Brain CTX_HP | 36,362 | 22 | Mouse | Allen cortex (spacexr vignette) |
| Xenium Region 1 | 13,940 | 45 | Human | Duodenum (p36005) |
| Xenium Region 3 | 58,191 | 45 | Human | Duodenum (p36005) |

### Compute Configurations

**GPU (rctd-py):**
- Node: fgcz-c-056 (NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM)
- SBATCH: `--partition=GPU --gres=gpu:1 --cpus-per-task=8 --mem-per-cpu=16G`
- Scripts: `SBATCH_benchmark_{mouse_brain,region1,region3}.sh`
- Output: `data/{dataset}/{dataset}_benchmark.json`

**CPU (R spacexr):**
- Nodes: fgcz-c-041, fgcz-c-054 (employee partition, 8 CPU cores)
- SBATCH: `--partition=employee --cpus-per-task=8 --mem-per-cpu=16G`
- Scripts: `SBATCH_r_mouse_brain.sh`, `SBATCH_r_regions_all_modes.sh`
- Output: `data/{dataset}/r_xenium_timing.csv`, `data/{dataset}/r_rctd_weights.csv`

### Pipeline

1. **R baselines** (CPU, employee partition): Run R spacexr in all 3 modes (full, doublet, multi) per dataset. Produces timing CSVs and weight matrices.
2. **rctd-py benchmarks** (GPU, Blackwell): Run `scripts/benchmark_all_modes.py` per dataset. Measures wall-clock time, peak VRAM, peak RSS, concordance vs R. Reads R timing CSVs for speedup calculation.
3. **Figure generation** (local): Run `scripts/generate_benchmark_figure.py` to produce `docs/benchmark.png` (README curves) and `docs/benchmark_comprehensive.png` (4-panel detailed figure).

### Key Scripts (in `scripts/`)

| Script | Purpose |
|--------|---------|
| `benchmark_all_modes.py` | Main benchmark: runs full/doublet/multi, measures time/memory/concordance |
| `generate_benchmark_figure.py` | Generates scalability + memory curve figures |
| `export_allen_cortex_ref.R` | Exports Allen cortex RDS to mtx/csv format for rctd-py |
| `run_r_rctd_mouse_brain.R` | Runs R spacexr baseline (all 3 modes) on mouse brain |
| `run_r_rctd_region1.R` | Runs R spacexr doublet mode on Xenium Region 1 |

### Metrics Collected

Per dataset, per mode (full/doublet/multi):
- **Wall-clock time**: total (sigma + deconvolution), sigma-only, deconvolution-only
- **Peak VRAM**: `torch.cuda.max_memory_allocated()` (GPU only)
- **Peak RSS**: `resource.getrusage(RUSAGE_SELF).ru_maxrss` (host RAM)
- **Throughput**: cells/second
- **Concordance vs R**: dominant type agreement (%), per-pixel Pearson correlation, RMSE of weight vectors
- **Speedup**: R time / rctd-py time (when R baseline available)

### Development Benchmarks

The `bench_gpu.py` script runs a fixed synthetic benchmark (50k pixels, 1000 genes, 12 types) for the autoresearch optimization loop. It is READ-ONLY — optimizations are validated by checking that the weights hash is unchanged.

The `experiment_*.log` files are from the autoresearch agent's optimization loop on the `autoresearch/mar14` branch. Key optimizations found:
- Eager sigma estimation (avoid torch.compile overhead for small batches)
- GPU eigh for K<=16 (pairwise fits in doublet mode) — **85% faster**, biggest single win
- Shared-profile P matrix (avoid materializing per-pixel S_batch)
- Active pixel compaction (skip converged pixels in IRWLS)
- torch.compile on `calc_q_all` (5% faster, 60% less VRAM)
- Fix graph break in `calc_q_all` (`.item()` → tensor)
- Fuse PSD reconstruction `V*sqrt(λ)` (1.4% faster)
- Precompute `P.T` contiguous (marginal, cleaner)

### Optimizations tried and discarded

| Experiment | Result | Reason discarded |
|------------|--------|------------------|
| Adaptive QP with early exit | 9.7% slower | Slower + hash changed |
| QP sweeps 50→30 | No change | No measurable improvement at 1.2s scale |
| Remove active pixel compaction | 15% slower | Compaction helps significantly |
| Diagonal loading PSD (skip eigh) | 54% faster | Hash changed — eigh required for correctness |
| float32 eigh | 41% faster | Hash changed — precision matters |

### Post-autoresearch optimizations (feat/cli branch)

- **Analytical K=2 PSD projection** (`_psd_2x2`): Closed-form eigendecomposition for 2×2 symmetric matrices avoids cuSOLVER entirely. Used in doublet mode pairwise fits.
- **Analytical K=2 box-QP** (`_solve_box_qp_2`): Cramer's rule + clamping replaces iterative Gauss-Seidel coordinate descent for 2-variable problems.
- **Auto batch size** (`auto_batch_size`): Estimates optimal batch size from available VRAM budget.

### Results log

Full experiment history is in `/results.tsv` (tab-separated). Summary of kept optimizations:

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
