# autoresearch — rctd-py performance optimization

This is an experiment to have the LLM autonomously optimize rctd-py for speed and memory efficiency, inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar14`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The source is compact. Read these files for full context:
   - `autoresearch/program.md` — this file, your instructions.
   - `autoresearch/benchmark.py` — fixed benchmark harness. **Do not modify.**
   - `src/rctd/_irwls.py` — IRWLS solver (core hot path, ~60% of runtime).
   - `src/rctd/_likelihood.py` — Poisson-Lognormal likelihood + spline interpolation.
   - `src/rctd/_doublet.py` — Doublet mode deconvolution.
   - `src/rctd/_multi.py` — Multi mode deconvolution.
   - `src/rctd/_full.py` — Full mode deconvolution.
   - `src/rctd/_sigma.py` — Sigma estimation.
   - `src/rctd/_normalize.py` — Platform effect normalization.
   - `src/rctd/_simplex.py` — Simplex projection.
   - `src/rctd/_rctd.py` — Pipeline orchestration.
   - `src/rctd/_types.py` — Data structures and config.
   - `src/rctd/_reference.py` — Reference class.
4. **Run baseline**: `uv run autoresearch/benchmark.py --mode full > run.log 2>&1` to establish the baseline.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. Record the baseline.
6. **Confirm and go**: Confirm setup looks good, then begin the experiment loop.

## What you optimize

rctd-py is a GPU-accelerated Python reimplementation of the RCTD cell type deconvolution algorithm for spatial transcriptomics. The pipeline:

1. **Reference processing** — compute mean cell type profiles from scRNA-seq
2. **Platform effect normalization** — `fit_bulk()` estimates bulk composition
3. **Sigma estimation** — `choose_sigma()` auto-calibrates noise (grid search over 91 candidates)
4. **Per-pixel deconvolution** — IRWLS solver decomposes each pixel into cell type weights

### Performance profile

The hot path is the **IRWLS solver** (`_irwls.py`), specifically:
- `solve_irwls_batch()` — vectorized solver for N pixels simultaneously
- `_get_derivatives_batch()` — batched gradient/Hessian computation via `calc_q_all()`
- `_psd_batch()` — eigendecomposition (currently offloaded to CPU for small matrices)
- `_solve_box_qp_batch_compiled()` — JIT-compiled Gauss-Seidel coordinate descent

Secondary hotspots:
- `calc_q_all()` in `_likelihood.py` — cubic spline interpolation (called N×G times)
- `run_doublet_mode()` — evaluates all K×(K-1)/2 cell type pairs per pixel
- `run_multi_mode()` — greedy forward selection, up to 4 types per pixel

### Architecture constraints

- **PyTorch only** — no custom CUDA kernels, no new dependencies
- **Numerical correctness** — output weights must match baseline (verified via `weights_hash`)
- **Same public API** — `run_rctd()` signature and return types must not change

## Experimentation

Each experiment runs the benchmark: `uv run autoresearch/benchmark.py --mode full > run.log 2>&1`

A typical run takes 1–5 minutes depending on hardware.

**What you CAN do:**
- Modify any file under `src/rctd/` — all source code is fair game.
- Change algorithms, data layouts, batching strategies, torch.compile usage, memory management.
- Add new internal modules under `src/rctd/` if needed.

**What you CANNOT do:**
- Modify `autoresearch/benchmark.py`. It is read-only.
- Install new packages or add dependencies beyond what's in `pyproject.toml`.
- Change the public API (`run_rctd` signature, `Reference` class, result types).
- Break numerical correctness — the `weights_hash` must match baseline, or the change is invalid.

**The goal is simple: get the lowest elapsed_s while maintaining correctness.**

Memory (peak_vram_mb) is a soft constraint. Some increase is acceptable for meaningful speed gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing code and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh complexity cost against improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline.

## Output format

The benchmark prints results like:

```
--- full mode ---
Dataset: 50000 pixels, 1000 genes, 12 types
---
elapsed_s:     42.15
peak_vram_mb:  1234.5
maxrss_mb:     3456.7
n_pixels:      50000
device:        cuda
dtype:         float32
weights_hash:  a1b2c3d4e5f6g7h8
```

Extract the key metrics:
```bash
grep "^elapsed_s:\|^peak_vram_mb:\|^weights_hash:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

The TSV has a header row and 6 columns:

```
commit	elapsed_s	memory_mb	hash	status	description
```

1. git commit hash (short, 7 chars)
2. elapsed_s achieved (e.g. 42.15) — use 0.00 for crashes
3. peak memory in MB, round to .1f — use 0.0 for crashes
4. weights_hash (first 16 hex chars) — use "none" for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	elapsed_s	memory_mb	hash	status	description
a1b2c3d	42.15	1234.5	a1b2c3d4e5f6g7h8	keep	baseline
b2c3d4e	38.90	1240.2	a1b2c3d4e5f6g7h8	keep	reduce QP sweeps from 50 to 30
c3d4e5f	41.80	1234.5	a1b2c3d4e5f6g7h8	discard	vectorize psd_batch on GPU (slower)
d4e5f6g	0.00	0.0	none	crash	custom triton kernel (import error)
e5f6g7h	35.20	1234.5	ff00112233445566	discard	skip convergence check (wrong hash)
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: current branch/commit.
2. Think about what optimization to try. Consider:
   - Reducing IRWLS iterations (fewer QP sweeps, early stopping, better warm starts)
   - Better batching / memory layout (contiguous access patterns, fewer allocations)
   - torch.compile coverage (more functions, different compile modes)
   - Algorithmic improvements (skip unnecessary computation, better convergence)
   - Reducing CPU↔GPU transfers (keep more on GPU, avoid _psd_batch CPU roundtrip)
   - Likelihood computation optimization (calc_q_all vectorization)
   - Doublet/multi mode pruning (skip unlikely cell type pairs early)
3. Modify source files under `src/rctd/`.
4. git commit.
5. Run the benchmark: `uv run autoresearch/benchmark.py --mode full > run.log 2>&1`
6. Read results: `grep "^elapsed_s:\|^peak_vram_mb:\|^weights_hash:" run.log`
7. If grep output is empty, the run crashed. Run `tail -n 50 run.log` for the traceback.
8. **Check correctness**: weights_hash MUST match the baseline. If it doesn't, the optimization broke correctness — discard it.
9. Record results in `results.tsv` (do NOT commit results.tsv).
10. If elapsed_s improved AND hash matches: keep the commit, advance the branch.
11. If elapsed_s is worse OR hash doesn't match: `git reset --hard HEAD~1`.

**Timeout**: Each benchmark should take at most 10 minutes. If it exceeds that, kill it and treat as failure.

**Crashes**: If it's a typo or import error, fix and re-run. If the idea is fundamentally broken, skip it.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of ideas, re-read the source files, look for new angles, try combining previous near-misses. The loop runs until you are manually stopped.

## Optimization ideas to get you started

These are suggestions — you're free to try anything:

1. **Reduce QP sweeps**: `n_sweeps=50` in `_solve_box_qp_batch_compiled` may be overkill. Try 20-30.
2. **Better warm start**: Initialize IRWLS weights from previous batch instead of uniform 1/K.
3. **Vectorized PSD on GPU**: The CPU roundtrip in `_psd_batch` adds latency. Try Cholesky-based PSD fix on GPU.
4. **Fuse calc_q_all**: The spline interpolation does many small tensor ops. Could be compiled.
5. **Skip converged pixels earlier**: Current code still computes gradients for all pixels each iteration.
6. **Reduce max_iter**: Many pixels converge in 5-10 iterations. Adaptive iteration count.
7. **Mixed precision**: Use float16 for intermediate computation, float32 for accumulation.
8. **Memory pre-allocation**: Avoid per-iteration allocations in the IRWLS loop.
9. **Batch size tuning**: The default 10000 may not be optimal for all GPU architectures.
10. **Profile-guided optimization**: Identify the actual bottleneck with torch.profiler before optimizing.
