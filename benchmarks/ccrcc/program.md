# autoresearch - RNA+protein cell typing on the 10x ccRCC Xenium Protein section

Goal: raise `sqrt(heldout_lineage_F1 * tenx_fine_agreement)` on the 80k-cell TUNE split (west half).
Budget: an experiment must finish its GPU eval in < 8 min (the Verify call is capped at 10); a
slower one is a crash, not a keep.
Branch `autoresearch/protein-ccrcc` (off `feat/protein-celltune`). Baseline = arm A2 of
`/srv/GT/analysis/pgueguen/rctd-py/ccrcc_protein/02_run_arms.py` (hybrid reference, doublet mode,
bootstrap protein, arcsinh-robust, lambda 1).

## Files

| file | role | editable by the loop |
|---|---|---|
| `benchmarks/ccrcc/pipeline.py` | THE experiment: config, reference composition, post-processing | **yes** |
| `src/rctd/{_protein,_protein_eval,_irwls,_doublet,_rctd,_types,_likelihood}.py` | the protein path of the solver | **yes** |
| `benchmarks/ccrcc/eval.py` | the metric, the truth maps, the frozen eval set | no |
| `benchmarks/ccrcc/prepare.py`, `verify.py`, `guard.py`, `eval.sbatch`, `prepare.sbatch` | harness | no |
| `/srv/GT/analysis/pgueguen/rctd-py/ccrcc_protein/autoresearch_data/*` | tune / test h5ads, references | no |
| `tests/`, `pyproject.toml` | | no |

`guard.py` pins the harness hashes at loop start and fails if any harness file changes, if
`pipeline.py` references the truth (`tenx`, `landmark`, `eval_cell`, `GATE_MARKERS`, the data
dir, `.h5ad`), or if `tests/test_protein_regression.py` (RNA path byte-identity) fails.

## What the pipeline sees

`run(ad, refs, device)`: counts (477 genes), `obsm["protein"]` = the 18 non-gate markers
(PD-1, VISTA, PD-L1, LAG-3, GranzymeB, CD163, CD4, CD8A, HLA-DR, CD11c, CD45RA, PCNA, CD45RO, Ki-67,
Beta-catenin, PTEN, Vimentin, E-Cadherin), `obsm["spatial"]`, `obs[transcript_counts, cell_area,
segmentation_method]`, and the two reference paths (`refs["hybrid"]`, `refs["disco"]`). The 9 gate
markers (CD3E, CD20, CD68, CD31, PanCK, alphaSMA, CD45, CD138, CD16) and the 10x labels are stripped
by the harness before the call. Return one label per input cell; dropped cells = "filtered".

## Metric (fixed)

1. Held-out lineage macro-F1: landmark cells gated on the 9 gate markers (8 lineages), ALL landmarks,
   rejects/filtered = miss (precision punishes wrong new calls); `first_type` mapped to
   lineage by `eval.py:LINEAGE` (a type not in the map counts as wrong - add new reference types there
   is NOT allowed; use names already in the map).
2. 10x fine-type agreement: macro over 10x WNN groups (>= 50 cells) of the fraction of cells whose
   `first_type` is in the group's accepted set (`eval.py:TENX_ACCEPT`). 10x's WNN used protein, so this
   axis favours protein arms; that is why it is paired with (1).

Verify: `uv run python benchmarks/ccrcc/verify.py` -> GPU job on fgcz-r-023 (~5-6 min), prints the
metric. Per-class F1, per-group agreement, reject fraction, 10x lineage agreement, CA9-positive fraction
of tumour calls and runtime land in `benchmarks/ccrcc/out/eval_<tag>.json` - READ THEM before choosing
the next change; a metric that rose because one class collapsed is a discard in spirit.

## Loop rules (uditgoenka/autoresearch Classic mode)

One atomic change -> `git commit -am "experiment: <what>"` -> Verify -> Guard -> keep, or
`git revert HEAD --no-edit`. TSV row per iteration in `autoresearch/loop-YYMMDD-HHMM/results.tsv`.
Never edit a read-only file. Never touch the test split until the loop is over.

## Idea list (start here, one per iteration; pick by expected value / cost)

1. lambda 0.5 / 1.5 / 2 (1 was chosen on lineage F1 alone; the fine axis may want more).
2. Scale CONFIDENCE_THRESHOLD/DOUBLET_THRESHOLD by 477/5000 (x0.1) and by x0.3: A1t took 10x lineage
   agreement 0.449 -> 0.644 RNA-only, rejects 31.7% -> 1.4%, at some rare-class precision.
3. `gene_cutoff_reg=0, fc_cutoff_reg=0`: only 232/477 curated panel genes currently reach the pixel fit.
4. `protein_var_model="unit"`; `protein_arcsinh_cofactor` 1 / 20; `protein_singlet_purity` 0.6 / 0.9.
5. Drop the checkpoint markers (PD-1, VISTA, PD-L1, LAG-3) from the fit set inside `run()`.
6. `class_df` over T subtypes and myeloid subtypes (recovers singlets; does not change the winner).
7. Reference: drop rare DISCO types that absorb calls (Type_A/B_intercalated, distal tubule), add Zhang
   2021 `Endothelial` / `VSMC` (names already in `LINEAGE`), 300 vs 500 cells per type.
8. UMI_min 20; `protein_reliability="neighbour_ratio"`.
9. Post-processing in `run()`: same-lineage neighbour smoothing of labels; two-stage typing (lineage by
   RNA+protein, fine type by RNA weights within the lineage).
10. Solver: per-marker informativeness weights (`inv_tau2` scaling), doublet-score fusion changes.
