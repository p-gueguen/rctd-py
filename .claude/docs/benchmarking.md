# Benchmarking and autoresearch

Moved out of `CLAUDE.md`, which links here.

## Benchmarking

GPU benchmarks are in `benchmarks/`. Submit via SLURM:

```bash
sbatch benchmarks/sbatch_bench_compare.sh   # baseline vs optimized comparison
sbatch benchmarks/sbatch_bench_gpu.sh        # optimized-only timing
```

## Autoresearch

Autonomous optimization framework inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The agent modifies solver code, benchmarks, keeps/discards based on speed + correctness.

### How to run

1. Read `benchmarks/program.md` for full instructions
2. Create branch: `git checkout -b autoresearch/<tag>`
3. Establish baseline: run benchmark as-is
4. Loop: modify → commit → benchmark → keep/discard

### Key rules

- **Metric**: `elapsed_s` (lower is better)
- **Correctness**: `weights_hash` must match baseline — if it changes, discard
- **Files to modify**: `src/rctd/_irwls.py`, `src/rctd/_likelihood.py`, `src/rctd/_simplex.py`
- **Read-only**: `benchmarks/bench_gpu.py`, `tests/`
- **Log results** to `results.tsv` (tab-separated)
- **GPU partition**: target an L40S or Blackwell (sm_100+) node via your scheduler (e.g. `--partition=GPU --gres=gpu:1` for SLURM clusters that expose GPUs as a generic resource)

### Profiling hot spots

Pre-optimization CPU profiling (2k pixels): `calc_q_all` 36%, QP solver 27%, eigh (PSD) 14%, bmm 9%.

Post-optimization: `calc_q_all` is now a fused Triton kernel (single launch), K=2 PSD/QP are analytical (zero cuSOLVER calls in doublet mode). Run `torch.profiler` to identify remaining GPU-specific bottlenecks.

### Benchmarking tips

- **GPU benchmarks must pin to a specific node** via `--nodelist=<your-node>` for reproducible comparisons (different GPU SKUs and even different nodes of the same SKU produce different timings)
- **Warmup is mandatory**: torch.compile and Triton JIT have ~60s first-call overhead. Run a warmup dataset before the timed benchmarks
- **A/B comparisons**: Use `benchmarks/SBATCH_compare_optimizations.sh` — checks out baseline vs optimized source files, runs both on the same GPU node back-to-back
- **Spatial data loading**: Use `sc.read_10x_h5()` (scanpy), NOT `anndata.read_h5()` which doesn't exist
