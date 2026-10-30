# Changelog

All notable changes to rctd-py are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Lower GPU memory use in full mode.** `run_rctd(..., mode="full", stream_gpu=True)` can now keep the complete spatial count matrix in host memory and transfer only the current batch to the GPU. The existing resident path remains the default. Doublet and multi mode still pre-stage the full spatial count matrix.
- **Lower GPU memory use during sigma estimation.** Q/SQ likelihood tables are staged one sigma pair at a time instead of retaining all 126 pairs on the GPU simultaneously.
- **Lower GPU residency during bulk platform-effect fitting.** Spatial counts and nUMI remain CPU-resident for the bulk reductions; only the reduced gene vector and scalar are moved to the compute device.
- **Reduced temporary allocations in shared IRWLS.** Derivative workspaces are reused in place and released after their last use. Numerical output remains bit-identical to 0.4.0 for the tested full, doublet and multi workflows.
- **Documented CUDA allocator fragmentation mitigation.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` can reduce reserved CUDA memory on large workloads where allocator fragmentation is significant.

## [0.4.0] — 2026-10-19

rctd-py 0.4.0 matches R spacexr: wherever both estimate the same sigma, every spot class and cell-type call is the same as R's. Getting there changed a few details of the fit (see Fixed), so **results can differ from 0.3.9**. On Visium about 15% of spots get a different dominant cell type, and on Visium HD and StrataMap 7-19% of spot classes. On imaging platforms (Xenium, CosMx, Atera) fewer than 1% of spot classes change, plus a few percent of first-type labels. Every difference we checked moves the call to R's. To reproduce 0.3.9 results exactly, pin `rctd-py==0.3.9`.

### Changed
- **Results match R spacexr.** Together with the two fit fixes below, an audit against R spacexr 2.2.1 on 18 datasets found four more places where rctd-py did something R does not (listed under Fixed). With R's own per-pixel inputs fed in (normalised profiles, sigma, counts), rctd-py now returns the same spot class, `first_type` and `second_type` as R on every pixel tested (5 datasets, 50,661 pixels, weights within 6e-4). End to end, the one remaining source of difference is sigma: R and rctd-py both estimate it from a random subset of pixels, and R does not fix its seed. Two R runs that land on different sigmas agree on only 99.3% of spot classes (Xenium liver, R seeds 1-5 give 0.61 or 0.62), the same level as rctd-py vs R; where both pick the same sigma, agreement is 99.99-100%. To reproduce a given R run exactly, pass its sigma: `run_rctd(..., sigma_override=round(100 * rctd@internal_vars$sigma))`.
- **Upgrading from 0.3.9.** The numbers above were measured v0.3.9 vs 0.4.0 in doublet mode (first-type changes on imaging data are mostly singlet labels). `RCTDConfig(N_fit=100)` restores the old sigma sample size. `cell_type_names`, and so the weight columns, can come in a different order (R's sorting, which now applies to `.h5ad` and CLI inputs too): index weights by name, not by position.

### Fixed
- **PyTorch 2.3 or later is now required** (was `>=2.0`). A new environment gets NumPy 2, which torch 2.0-2.2 cannot read: every run failed with `RuntimeError: Could not infer dtype of numpy.float32` (checked on Python 3.10 with torch 2.0.1 and 2.2.2; 2.3.1 passes). 0.3.9 had the same problem.
- **Input handling.** Missing cell-type labels (NaN / None) raise a clear error, and `rctd validate` now fails on them instead of passing; cell types dropped for having fewer than `cell_min` cells are named in a warning; integer labels work again (they crashed the new R ordering) and no longer crash the CLI's output write when pixels are filtered (0.3.9 too); when `UMI_min` removes every pixel the error says so.
- **`rctd info` printed a FutureWarning with anndata 0.13**, which deprecates `anndata.__version__`; it now reads the version from the package metadata.
- **Mitochondrial genes were filtered differently from R.** R's `get_de_genes` drops `grep("mt-", gene_list)`, case-sensitive: mouse `mt-` genes go and human `MT-` genes stay DE candidates. rctd-py lowercased the names first, so it also dropped human `MT-` genes; on human data that changed the bulk gene list and, through `counts_MIN`, which pixels were kept (issue #11). No change on the 19 benchmark datasets, whose references hold no human `MT-` genes. `tests/test_reference.py::test_de_genes_filters_mt_genes` checks R's rule.
- **`first_type` of singlets was not R's.** For a pixel called `singlet`, rctd-py replaced the first type with the candidate that had the best single-type score; R's `process_bead_doublet` only relabels the spot class and keeps the first and second type (and class flags) of the best-scoring pair. The label disagreed with R on up to 6% of pixels even where the spot class agreed (Xenium Prime 5K breast: 93.9% `first_type` agreement at 99.8% spot-class agreement). `tests/test_r_parity.py` feeds R's per-pixel inputs for 300 Slide-seq beads and requires identical calls; it fails on the old code (150 of 300 beads).
- **Cell types were ordered differently from R.** spacexr indexes types by `levels(cell_types)` and falls back to "the first type" by index when a pixel has a single candidate, so the order changes calls, not just column order. rctd-py sorted names by code point (`B cell` before `adipocyte`); R sorts them with ICU collation (`adipocyte` before `B cell`) or keeps a factor's levels. A categorical `cell_type` column with a deliberate category order (an R factor's levels) keeps it; anything else, including a categorical sorted by code point (pandas) or naturally (`Type_1` before `Type_17`, as anndata writes every string column) (so every `.h5ad` and every CLI run), is sorted as R's `factor()` would (`r_sort_key`, checked against R on 2,435 names; numeric labels sort by value). The order of `cell_type_names`, and so of weight columns, can change, `.h5ad` and CLI inputs included.
- **Sigma was estimated on 100 pixels; R uses 1000** (`N_fit`), and the README listed R's default as 100. The smaller sample made sigma noisier than R's own: Visium HD mouse brain, 10 seeds, 0.40-0.48 at 100 pixels vs 0.43-0.47 at 1000 (R: 0.44-0.46). It adds 5-20 s of GPU time per run.
- **The box-constrained QP solve was approximate.** R uses `quadprog::solve.QP`, which is exact; rctd-py ran 50 Gauss-Seidel sweeps, which stop short on ill-conditioned problems at high K and occasionally flipped a candidate type (1 pixel in ~13,000 on Xenium liver, K=45). Problems that fail the KKT check after the sweeps now get an active-set solve warm-started from the sweep result. Measured on an L40S, doublet mode: VisiumHD K=78 1,718 s vs 1,726 s, Atera breast 2,126 s vs 2,124 s, no change on smaller sets. `tests/test_irwls.py::test_box_qp_is_exact_like_quadprog` checks against scipy NNLS.
- **`eigh_threshold` and `compile=False` leaked into later runs in the same process.** Both are applied by `RCTD.__init__` to module-level flags, which were set but never reset, so one run with `RCTDConfig(eigh_threshold=64)` kept every later run in that Python session on GPU eigh, and one `compile=False` kept it eager. Found while benchmarking: on an L40S (StrataMap breast, 21,425 cells, K=40, doublet mode) a leaked `eigh_threshold=64` made the full-fit step of a later default-config float32 run take 263 s; the same float32 full fit takes 44 s in a fresh process. Every `RCTD` now sets both flags from its own config; a compile fallback triggered by an actual compile failure still stays sticky. `tests/test_config_state.py` covers both leaks and the sticky fallback.
- **Bulk platform-effect fit used genes R leaves out, lowering concordance with R on sparse data.** R's `fitBulk` (`prepareBulkData`) decomposes the bulk only on genes with at least 10 total counts across the tissue (`MIN_OBS = 10`, hard-coded) and then renormalises every bulk gene; `fit_bulk` decomposed on all of them. Imaging panels have almost no such genes, so Xenium/CosMx/Atera concordance was unaffected (99.7-100%), but on sequencing-based data the extra genes moved the bulk proportions by up to 7 points (StrataMap breast: keratinocyte 33.3% vs R's 26.1%) and every cell was then fitted against shifted profiles. Against R spacexr 2.2.1 doublet calls (K=40 breast reference, same inputs), spot-class agreement goes from 80.7% to 99.5% on StrataMap breast (21,425 cells), 85.9% to 100% on Visium HD breast segmented cells (38,990) and 91.5% to 100% on Visium v1 breast (3,798 spots); bulk proportions now match R to 1e-7. `tests/test_normalize.py::test_fit_bulk_drops_genes_below_min_obs` covers it.

### Documentation
- **Documentation site** at [rctd-py.readthedocs.io](https://rctd-py.readthedocs.io): installation, quick start, tutorial, CLI and API reference, and a page for R spacexr users (#33, #34).
- **Benchmarks refreshed**: 19 datasets from eight platforms (2,518 to 714,535 cells) in all three modes, rctd-py on one NVIDIA L40S and on 8 CPU threads against R spacexr 2.2.1 on 8 cores, median of 3 runs. GPU vs R: 10-57x in doublet mode, 5-119x in full mode, 5-21x in multi mode.
- **Agreement with R** on the docs site now covers 19 datasets with the 0.4.0 code: identical calls where both sides pick the same sigma (11 datasets), 98.99-99.88% of spot classes where sigma differs by 0.01-0.02.
- **Corrections from a docs audit**: R spacexr does not normalise full-mode weights either; the CLI's `--batch-size` defaults to 10000 (`"auto"` is Python-only); mode recommendations now follow spacexr (doublet for segmented cells and high-resolution data); float32 speed-up re-measured (up to 2.3x, median 1.6x); `max_iter` removed from the tutorial, since it is not read yet; README links and images work on PyPI.

## [0.3.9] — 2026-10-02

### Added
- **`--table NAME` for `rctd validate` / `rctd run`** (issue #28 follow-up). A SpatialData store with several tables used to stop with an error asking for an explicit `<store>/tables/<name>` path; `--table` now picks the one to read from the spatial store, so the store root stays the single thing you pass. An unknown name lists the tables that exist, and `--table` on an `.h5ad` or on a path already inside a table is a usage error rather than a silent no-op. It applies to the spatial argument only; a multi-table reference store still takes the explicit table path.

## [0.3.8] — 2026-09-10

### Added
- **SpatialData / Zarr inputs in the CLI** (issue #28). `rctd validate` and `rctd run` now accept a `.h5ad` file, an AnnData Zarr store, or a SpatialData Zarr store for either positional argument. A directory is read as Zarr; if it holds a `tables/` group its single table is used, and a store with several tables errors with their names and asks for an explicit `<store>/tables/<name>` path. Reading goes through one shared `_read_adata()` used by both subcommands, so the two paths cannot drift.
- **`RCTD_Q_MATRICES` for offline / HPC installs** (issue #29). `q_matrices.npz` is 404 MB and does not compress (measured: `savez_compressed` returns the same 404 MB, because it is float64 spline tables), so it cannot be vendored in the wheel — PyPI caps a single file at 100 MB. Instead the lookup now consults `$RCTD_Q_MATRICES` (a `.npz`, or a directory holding one) before the package `data/` dir and the `~/.cache/rctd` download, so one staged read-only copy serves every user on a cluster rather than one 404 MB download per home directory. When there is no network and nothing is staged, the failure is now a `RuntimeError` naming the URL, the exact cache path to drop the file at, and the env var — previously a bare `urllib` traceback, which is what sent the reporter into the source to find the cache path. A partial download is deleted rather than left to fail as a corrupt cache later, and a file the caller pointed at deliberately (via the env var or `data_dir`) is reported as unreadable instead of being silently re-downloaded over.

### Fixed
- **A Zarr-read AnnData could not be written back out to `.h5ad`.** SpatialData encodes `obs`/`var` names as `nullable-string-array`, which anndata reads as a pandas `StringArray` and then refuses to write to h5ad unless `allow_write_nullable_strings` is set. The failure landed at the very last step, after the full deconvolution had run. `_read_adata()` now normalises nullable string dtypes to the plain object dtype the `.h5ad` path already yields, so downstream code cannot tell the two containers apart. Verified against stores written by spatialdata 0.8.0 (including its own `datasets.blobs()`); `tests/test_cli.py` builds real SpatialData stores and asserts that the names read back as a nullable string array, so the tests cannot silently stop covering this.
- **The default output path no longer lands inside the input store.** With `<store>/tables/table` as input, `<spatial_stem>_rctd.h5ad` would have been written into the Zarr store's `tables/` group; it now resolves to the store's parent directory.

## [0.3.7] — 2026-07-15

### Fixed
- **Inductor codegen WARNING spam on Apple Silicon / torch ≥ 2.10** (issue #27). On some platforms inductor fails to codegen the box-QP / likelihood reductions, logs a large recoverable `Error in codegen` dump, then falls back to eager *without raising* — so the existing `except RuntimeError` auto-fallback never tripped and `torch.compile` was re-attempted (re-dumping) for every new batch shape. rctd-py now installs a targeted filter on the `torch._inductor.scheduler` logger that drops those recoverable records and flags the degradation; the first compiled call detects the flag and switches `calc_q_all` / `_solve_box_qp_batch` to their quiet non-compiled paths for the rest of the run. Numerical output is unchanged (inductor was already falling back to eager). Set `RCTDConfig(compile=False)` to skip the compile probe entirely.

## [0.3.6] — 2026-06-16

### Changed
- **Pre-stage `spatial_counts` on the device once** in `_doublet.py` (steps 3/4/6) and `_multi.py` (forward-selection iterations). Previously each batch did `torch.tensor(spatial_counts[pix_idx], device=device)` — a pixel that appears in many triples/tasks was gathered and copied repeatedly. Now both `spatial_counts` and `spatial_numi` move to the device once at the start of each mode and per-batch access is an on-device gather. Numerical output is unchanged: a cross-worktree replay against unmodified `main` reproduced the v0.3.5 output bit-for-bit on a synthetic doublet workload. GPU bench on fgcz-r-023 (L40S, sm_89, N=20000, K=30, G=500): 76.2 s → 75.0 s (1.5% — modest because the H2D was already memcpy-bound).

### Documented (no code change)
- **fp32 concordance on GPU is now empirically verified.** The fp32 path has been exposed via `RCTDConfig(dtype="float32")` / CLI `--dtype float32` since the initial release, and the perf test suite already times it for all three modes — but no test ever asserted numerical agreement with fp64, and `CLAUDE.md` flagged the spline-index `floor(sqrt(lam/delta))` in `_calc_q_all_impl` as a precision-sensitive site. New `tests/test_fp32_concordance.py` asserts `spot_class` agreement ≥ 99% (observed: 100%), `first_type` agreement ≥ 99% (observed: 100%), and `weights_doublet` max diff < 1e-2 (observed: <1e-6 on the synthetic fixture). On L40S sm_89 at N=20000, K=30 the fp32 doublet run produced an *identical* `spot_class` hash to fp64 (`63dfa94a10f7aa93`) while cutting wall time from 76 s → 39 s (~2×). README now points consumer-GPU users at `float32` with the empirical numbers.

  Caveat: only synthetic data was stress-tested. The `floor(sqrt(...))` spline-index sensitivity could shift on real-world `lam` distributions near integer boundaries; the assertion is set at ≥99% to admit a small drift before failing.

### Internal
- New `tests/test_doublet_prestage.py` runs doublet mode through the pre-staged path and asserts shape, simplex sums, valid class/type ranges, and that ≥90% of pixels reach a non-degenerate split. Tolerance-based rather than byte-hash — early CI on this branch caught that hash equality across PyTorch/numpy minor versions is too brittle.

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

- **L40S (Ada) and Ampere unaffected by the dispatch change.** Regression-tested on fgcz-r-023 (L40S, sm_89): the JIT box-QP path is **1.6–1.85× faster than the previous eager Python loop** at K=45 and K=78, with max numerical diff ~1e-6. Arch gating preserves the CPU eigh offload that earlier L40S benchmarks validated.

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
