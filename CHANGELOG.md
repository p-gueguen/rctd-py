# Changelog

All notable changes to rctd-py are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — branch `feat/protein-celltune` (stacks on `feat/multimodal-protein`, PR #25)

Measures the protein modality instead of assuming it, and stops trusting protein
that looks like bleed-through from a neighbour. The ideas are lifted from CellTune
(Bussi et al., Nat Methods 2026, doi:10.1038/s41592-026-03162-2), which faced the
same problem in spatial proteomics: no cell has a label the model did not produce.
Everything here is additive and off by default; `protein_weight=0.0` remains
byte-identical to the RNA-only solver.

### Added

- **Landmark cells as internal truth** (`_protein.gate_landmarks`). Strict
  two-sided gates on the protein channels, relaxed on a ladder from the 95th
  percentile until each type reaches `protein_landmark_min_cells`. A cell claimed
  by two types is dropped rather than assigned - truth has to be unambiguous - and
  types that never reach the target are reported, never silently missing from a
  macro average.
- **`protein_weight="landmark"`**: pick lambda by measured accuracy instead of the
  `"auto"` gradient heuristic (which lands ~0.1 on a 62-plex CosMx panel where 4
  was best-in-class). Markers are split into folds; truth is gated with fold *f*
  and the fit for that fold runs with fold *f* removed from both the observation
  matrix and the profile. Scored on `first_type` with rejects counted as misses,
  on the landmark cells the RNA-only fit was willing to call. The full curve lands
  in `RCTD.protein_lambda_curve`.
- **A permutation null on that choice.** The same sweep runs on
  `protein_lambda_nulls` (default 6) column-shuffled copies of the panel, and the
  observed gain must beat the largest null gain. This is not ceremony: on the test
  fixture a noise panel scored gain +0.039 while a single shuffled sweep drew
  +0.038, so both "beats lambda=0" and "beats one null" selected lambda=4 on pure
  noise. The null gain is centred at zero (mean -0.001) but spreads as wide as a
  real effect when landmarks are few (sd 0.08 at 36 landmarks/fold). Informative
  panel: gain +0.122, z=2.5, selected. Noise panel: +0.039, z=0.5, rejected.
- **`protein_signature_magnitude="calibrated"`**: per-marker +/- levels measured
  from landmark cells (`_protein.calibrate_signed_levels`) instead of one global
  +/-1.5 z for every marker of every type. On a planted separation of z=6 the
  calibrated level finds 6.6-7.1; the fixed 1.5 understates it, and a lambda large
  enough to compensate over-weights the whole panel rather than that one marker.
  `tests/test_protein_landmarks.py` checks that a correctly scaled profile at
  lambda=1 does what an understated one needs lambda=4 for.
- **`protein_reliability="neighbour_ratio"`**: per-cell protein reliability from
  CellTune's anti-spillover test - a marker value is credible as cell-intrinsic
  only if it stands above the same marker in the cell's neighbours. There was no
  spatial term anywhere in the package before this; the only background handling
  was a global per-marker constant that no config field could even reach. Uses
  `scipy.spatial.cKDTree` (already a dependency), NOT the dense N x N kNN in
  `_multimodal._morans_i`, which would OOM on a 60k-cell section.
- **Per-cell RNA-vs-protein disagreement**, free of any extra fit. `_doublet.py`
  already formed `score = score_RNA + lambda * score_protein`; it now keeps both
  terms, so `DoubletResult` carries `rna_first_type`, `prot_first_type`,
  `modality_conflict` and a `(K, K)` `modality_confusion` matrix. This is
  CellTune's query-by-committee, and it generalises the dataset-level MECR proxy
  used in the validation report to a per-cell flag. Caveat stated in the code: both
  components are read off the same joint single-type fits, so it is a decomposition
  of the joint score, not two independent solves.

- **Class-level landmarks** (`protein_landmark_classes={cell_type: class}`,
  `protein_landmark_signatures={class: gates}`). A 27-plex panel can adjudicate
  T vs B vs myeloid vs epithelium; it cannot adjudicate CD4 vs Treg or PT vs
  tumour, and gating 25 fine types on it left 72 landmark cells with 10 types
  empty (10x renal, 2026-08-20), which is why that measurement could only say
  "lambda = 0". Landmarks are now gated per class and `first_type` is mapped
  through the class map before scoring, so confusion inside a class is not an
  error. `protein_signatures` stays per-type for the curated PROFILE; a type left
  out of the map never matches a landmark (a call there scores as wrong). Not
  combinable with `protein_signature_magnitude="calibrated"` (per-type levels
  need per-type landmarks; it raises).

### Changed

- **`--protein-weight landmark` now works from the CLI.** The option parsed
  `"auto"` and `float()`-ed everything else, so the measured selector was
  Python-API only. `_parse_protein_weight` accepts a number, `auto` or
  `landmark` and rejects anything else with a `BadParameter`.
- **`prot_mask` is now a per-cell WEIGHT, not just a mask.** It was already
  multiplied into both the protein gradient and the protein Hessian, so accepting a
  float in [0, 1] cost one `torch.where` -> `*` swap in each solver and preserves
  the shared `(K, K)` Hessian precompute (it becomes a broadcast scale). An all-True
  bool is arithmetically identical to before, and a test asserts an all-ones float
  matches it exactly. `calc_protein_log_likelihood_batch` applies the same weight,
  so the fit and the classification cannot disagree about how much protein counts.
- **`run_rctd_multimodal(fusion=...)` now defaults to `"protein_wls"`, and
  `"count"` raises on continuous input.** `tests/test_fusion_switch.py` has always
  asserted that count fusion COLLAPSES on intensity data (one constant weight
  vector for every pixel); shipping that as the default meant anyone omitting the
  argument got a plausible-looking constant answer. PR #25 was never released, so
  nothing downstream depends on the old default.
- **The CLI keeps the confidence it computes.** `rctd_min_score`,
  `rctd_singlet_score` and their difference `rctd_singlet_gap` (the margin the
  reject/singlet decision is actually made on) now reach `obs`, along with the
  modality-disagreement columns. Before this a CLI user saw the verdict and never
  the margin.

### Fixed

- **A tuple-valued config field killed a CLI run at the final `write_h5ad`**, after
  all the compute: `uns` has no writer for a tuple. `config._asdict()` is now
  normalized on the way out, so future tuple fields are safe too.

### Measured on real data (2026-09-08)

10x ccRCC Xenium Protein section (`Xenium_V1_Human_Clear_Cell_Renal_Cell_Carcinoma_FFPE_Protein`,
XOA 4.0, 477 genes, 27-plex), 150k-cell subset, reference = DISCO kidney + a `Tumour_ccRCC`
type from Zhang et al. 2021 PNAS. Landmarks gated per LINEAGE (8 classes, 839 cells) with the
gate markers (CD3E, CD20, CD68, CD31, PanCK, alphaSMA, CD45, CD138, CD16) held out of every
protein fit. Bootstrap protein, `protein_weight="landmark"`: lambda 1 selected, held-out F1
0.480 -> 0.549, gain +0.069 vs null max -0.002 (z 4.1); agreement with 10x's WNN lineages
0.449 -> 0.697; rejects 31.7% -> 17.8%. Curated scGate overrides (lambda 0, z 0.3), CLR
normalisation (lambda 0) and a post-hoc lineage override (no gain) did not help. The two
August negatives on the same tissue were measured against a reference with no tumour type
and 72 fine-type landmarks - the reference, not the fusion, was the problem. Full arm table:
`/srv/GT/analysis/pgueguen/rctd-py/ccrcc_protein/FINDINGS.md`.

Two things that measurement caught in the docs: (1) `marker_folds` splits by column PARITY,
so a panel whose lineage markers sit at even indices gates everything in fold 0 and leaves
fold 1 with a junk truth - order the protein columns (or the folds) so each fold can gate;
(2) scaling `CONFIDENCE_THRESHOLD`/`DOUBLET_THRESHOLD` changed `first_type` in 19.8% of
cells, not only `spot_class` - the singlet branch and the doublet branch pick different types.

### Known limitation (measured, not suspected)

Marker folds stop a marker from scoring itself. They do NOT make landmark truth
independent of the protein modality as a whole. If a panel is internally consistent
but attached to the wrong cells - mis-registration, a segmentation offset, protein
assigned to the neighbour - then the gated truth and the fit read the same wrong
assignment and agree with each other, and the sweep can still land on lambda > 0.
Verified both ways on synthetic data: shuffling each marker independently (no
information at all) correctly selects lambda=0, while shuffling whole rows (a
mis-assigned panel) does not. What separates them is the ABSOLUTE macro F1 (0.40 vs
0.85 on that fixture), because a mis-assigned panel's gated truth contradicts the
RNA the fit also sees. Read the absolute F1, not only the argmax. The only real fix
is truth that does not come from protein, which is what CellTune buys with human
labelling.

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
