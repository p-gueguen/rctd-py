# Changelog

All notable changes to rctd-py are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.5] — 2026-06-05

### Fixed
- **CPU-eigh thread oversubscription** (issue #22, reported by @meisproject). `_psd_batch` now caps PyTorch's intra-op thread count to 1 around the CPU `torch.linalg.eigh` call and restores the caller's previous count on exit. On hosts with many CPU cores, default OpenBLAS thread count oversubscribed under batched `syevd` — V100 + 64 cores at K=38 stalled at Step 1 = 3086 s in v0.3.4. The auto-cap now produces the bounded-threads behavior out of the box, no env vars required.

  **Empirically confirmed on Tesla V100** (smei, [#22 comment](https://github.com/p-gueguen/rctd-py/issues/22#issuecomment-4621228332)): with the auto-cap on `main` and **no `OMP_NUM_THREADS` env var and no `--eigh-threshold` flag**, Step 1 = **27.4 s** (down from 3086 s in v0.3.4 with the same bare command — **~113× speedup**). Total doublet-mode wall time on smei's 6113-pixel × K=38 workload: 57.7 s, vs the original ~52 min.

  Numerical output is bit-identical to v0.3.4 (existing K=78 atol=1e-9 equivalence test passes). Users on Hopper/Blackwell are unaffected — they stay on GPU eigh and never enter the CPU branch. Users who already set `OMP/MKL/OPENBLAS_NUM_THREADS=1` see no behavior change.

  Note: smei's earlier A/B also disconfirmed bumping the per-arch K threshold default for `sm_<9` — forcing GPU eigh at K=38 on Volta was 4× *slower* (126.2 s) than CPU eigh with bounded threads (33.8 s). The `--eigh-threshold` flag from v0.3.4 remains as a diagnostic / power-user knob; the default behavior is now correct on every architecture we have empirical data for (Volta + V100, Ada + L40S, Hopper, Blackwell).

### Internal
- New `.pre-commit-config.yaml` mirroring the CI lint + format checks (`ruff-format` + `ruff-check --fix`, pinned to v0.15.6 matching the dev extra). Install once after cloning: `uv pip install pre-commit && pre-commit install`. CONTRIBUTING.md updated with the workflow.

## [0.3.4] — 2026-06-02

Bundles two `_psd_batch` improvements. v0.3.3 was prepared and merged to `main` (CPU eigh crash fix for #20) but never tagged to PyPI; both changes ship together here.

### Fixed
- **`_LinAlgError` crash in doublet mode at K≈49 on the CPU eigh path** (reported by @EduardGhemes-ICR, #20). `_psd_batch` previously called `torch.linalg.eigh` raw on the CPU branch; a single non-finite or near-degenerate batch element would crash LAPACK `syevd` with "error code: 99" and kill multi-hour Xenium runs. The CPU branch now mirrors the GPU branch's NaN guard (extended to ±Inf) and adds a small-diagonal-jitter retry ladder (1e-6 → 1e-4 → ε·I last resort). Happy-path output is bit-identical to v0.3.2 — only previously-crashing inputs are affected. Triggered most often on older arches (Volta / Turing / Ampere / Ada / L40S) where K > 16 falls through to CPU eigh, but the guard is unconditional and applies to CPU-only deployments as well.

### Added
- **`--eigh-threshold` CLI flag and `RCTDConfig.eigh_threshold`** (reported by @meisproject, #22). Manually override the K cutoff for staying on GPU eigh inside `_psd_batch`. The arch-based default (`K≤16` on sm_<9, `K≤128` on sm_≥9) was derived from L40S benchmarks at K=45 where CPU OpenBLAS won — but only with `OMP_NUM_THREADS` capped. Users on Volta (V100, sm_70), Turing, Ampere, or Ada (L20/L40S, sm_89) who hit Step 1 perf cliffs at K∈[17, 64] (e.g. K=38 reported at 3086 s for 6113 pixels) can now force GPU eigh via `--eigh-threshold 64` without waiting on a per-arch benchmark / release. Setting `--eigh-threshold 0` forces CPU eigh on every arch (diagnostic counter-case). Default `None` preserves v0.3.2 arch-gated behavior bit-for-bit.

  Caveat: this ships the *override mechanism*, not a confirmed perf win on V100/L20. The maintainer has no V100 or L20 hardware to bench against; whether GPU eigh actually beats CPU offload at K=38 on those arches is unverified. Recommended diagnostic sequence: try `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` alone first (often the real fix is the BLAS thread cap, not the dispatch), then layer `--eigh-threshold 64` if Step 1 is still slow.

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
