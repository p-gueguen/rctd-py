# Changelog

All notable changes to rctd-py are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] — 2026-05-02

### Added
- **Hierarchical cell type fallback (`class_df`)** for doublet mode (#14). When a granular reference makes type-level resolution ambiguous, RCTD now reports the best two subtypes alongside `first_class` / `second_class` boolean flags indicating that the assignment is only trustworthy at the parent-class level — mirroring R spacexr exactly.
  - Python API: `RCTDConfig(class_df={"T_CD4": "T_cell", ...})`
  - CLI: `--class-df path.tsv` (TSV with columns `cell_type` and `class`)
  - New result fields on `DoubletResult`: `first_class_name`, `second_class_name` (string arrays, populated only when `class_df` is provided)
  - When `class_df` is omitted (default), behavior is bit-identical to v0.3.1 — verified by an explicit identity-mapping regression test.
- **Arch-gated GPU eigh threshold** (`_psd_batch`). On Hopper (sm_90+) and Blackwell (sm_100+), the K-cutoff for staying on GPU eigh is bumped from 16 to 128 via `torch.cuda.get_device_capability`. Older architectures (Volta, Turing, Ampere, Ada / L40S) keep the K≤16 cutoff that earlier benchmarks showed wins via CPU OpenBLAS.
- **TorchScript-fused box-QP** as the `compile=False` path. The eager Python Gauss-Seidel loop is replaced by `_solve_box_qp_batch_adaptive_jit` (`@torch.jit.script`, separate from torch.compile / Inductor). Fuses 50 sweeps × K coords into a single TorchScript graph and adds batch-level early exit, eliminating the kernel-launch storm at K>16.
- 23 new perf-regression tests (`tests/test_blackwell_perf.py`) covering arch-detection across 8 GPU architectures, eager/JIT numerical equivalence at K=3,8,16,32,78,100, ill-conditioned matrices, active lower-bound constraints, CPU path preservation, CPU perf neutrality, and end-to-end full + doublet integration regression.

### Fixed
- **Blackwell perf cliff at K>16, doublet mode** (reported by @litj). Before this release, doublet mode at K≈78 with ~100k pixels would stall for 8+ hours on Blackwell + CUDA 13 with `--no-compile`, despite GPU memory being allocated and the process running. Root cause: the K>16 path in `_psd_batch` unconditionally CPU-offloaded eigendecomposition, which then oversubscribed all CPU cores via OpenBLAS while the GPU sat at 0% utilization. Reproduced on FGCZ Blackwell node and verified end-to-end:

  | Configuration | Step 1 (full-mode fit) on K=78, 102k pixels |
  | --- | --- |
  | v0.3.0 + `--no-compile`, no env caps | >8h, killed without completing |
  | **v0.3.2 patched** (K≤128 GPU eigh + JIT box-QP + `OMP_NUM_THREADS=1`) | **2800 s (~47 min)** |

  Convergence rate 1.000, GPU util sustained at 85%. The `OMP_NUM_THREADS=1` env var is still recommended on Blackwell to prevent OpenBLAS from spawning threads for incidental CPU LAPACK calls.

- **L40S / Ampere unaffected by the dispatch change.** Regression-tested on fgcz-r-023 (L40S, sm_89): the JIT box-QP path is **1.6–1.85× faster than the previous eager Python loop** at K=45 and K=78, with max numerical diff ~1e-6. Arch gating preserves the CPU eigh offload that earlier L40S benchmarks validated.

### Notes
- `--no-compile` and `RCTDConfig(compile=False)` semantics are unchanged. Users who previously selected this path now get the JIT-script box-QP automatically; no API change.
- For users on Blackwell hitting the K>16 perf cliff on v0.3.0 / v0.3.1: upgrading to v0.3.2 is sufficient; the recommended env vars (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`) still apply as belt-and-suspenders.

## [0.3.0] — 2026-03-31

### Added
- `--no-compile` CLI flag and `RCTDConfig(compile: bool)` field for environments without CUDA development headers (#10). Lazy-compile-with-fallback pattern in both `_solve_box_qp_batch` and `calc_q_all`.
- `pixel_mask` field on result types for barcode mapping (#8, #9).

### Fixed
- Marimo tutorial: figures render in static HTML export.

## [0.2.2] — earlier release
See git history for details.
