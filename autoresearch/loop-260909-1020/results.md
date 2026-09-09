# autoresearch loop 260909-1020 - ledger (copy of results.tsv, which the repo's artifact rule ignores)

| iteration | timestamp | commit | metric | delta | guard | guard-metric | status | description |
|---|---|---|---|---|---|---|---|---|
| 0 | 2026-09-09T10:18:52+02:00 | 4cf1a09 | 0.457453 | 0.0 | pass | - | baseline | A2: hybrid ref, doublet, bootstrap protein arcsinh-robust, lambda 1 (F1 0.528, fine 0.396, reject 0.283) |
| 1 | 2026-09-09T10:25:12+02:00 | 9367fe8 | 0.492111 | +0.034658 | pass | - | keep | scale CONFIDENCE/DOUBLET thresholds by 477/5000 (F1 0.626, fine 0.387, reject 0.101; MDM/TAM agreement collapsed 0.33/0.41 -> 0.02/0.05) |
| 2 | 2026-09-09T10:31:06+02:00 | 8fc18a8 | 0.566285 | +0.074174 | pass | - | keep | drop cDC1/cDC2 from reference (F1 0.639, fine 0.502; MDM 0.94 TAM 0.79, pDC 0.00) |
| 3 | 2026-09-09T10:33:24+02:00 | fb0edd0 | nan | - | pass | - | crash | gene_cutoff_reg=fc_cutoff_reg=0 alone -> KeyError ABCC11: reg genes must be a subset of the bulk gene list (bulk cutoffs still default) |
| 4 | 2026-09-09T10:40:23+02:00 | 2dcee6b | 0.498219 | -0.068066 | pass | - | discard | all gene filters off (bulk+reg cutoffs 0): F1 0.599, fine 0.414, CA9 purity of tumour calls 0.71 -> 0.46, CD8 groups collapsed - the DE filter matters on this panel (reverted) |
| 5 | 2026-09-09T10:51:56+02:00 | 6c84853 | 0.554799 | -0.011486 | pass | - | discard | protein lambda 2 (F1 0.611 vs 0.639: NK 0.16->0.0, B 0.93->0.86; fine 0.503 flat) - reverted |
| 6 | 2026-09-09T11:00:33+02:00 | 77f63c0 | 0.557964 | -0.008321 | pass | - | discard | tumour as 3 k-means sub-states merged back (F1 0.632, fine 0.492, CA9 purity 0.71->0.62, tumour groups unchanged) - reverted |
| 7 | 2026-09-09T11:09:31+02:00 | 09f2865 | 0.545185 | -0.021100 | pass | - | discard | protein_var_model unit (F1 0.630, fine 0.472; 10x lineage agreement 0.729->0.663) - reverted. Diagnostic: Cycling_myeloid_cell (135 ref cells) is a top-4 call in 14/16 10x groups |
| 8 | 2026-09-09T11:17:46+02:00 | e1f87f2 | 0.569658 | +0.003373 | pass | - | keep | drop Cycling_myeloid_cell too (F1 0.642, fine 0.506, Myeloid F1 0.84; Helper T -> Treg 86%, exhausted CD8 -> CD4 31%) |
| 9 | 2026-09-09T11:26:14+02:00 | 9d63515 | 0.511798 | -0.057860 | pass | - | discard | signed CD4/CD8A gates for CD4_T/CD8_T with bootstrap elsewhere: Treg (bootstrap) absorbs the gated types (exhausted CD8 -> Treg 57%, Helper -> Treg 93%) - reverted. Treg_cell is the T-cell absorber in every state |
| 10 | 2026-09-09T11:35:32+02:00 | 8b0ea13 | 0.507073 | -0.062585 | pass | - | discard | T/NK types from Krishna 2021: its CD8 label is NAIVE CD8 (exhausted CD8 groups 0.46 -> 0.01), fine 0.420 - reverted |
| 11 | 2026-09-09T11:43:40+02:00 | 7b5b84a | 0.527200 | -0.042458 | pass | - | discard | signed CD4/CD8A gates on all three T types: gated T types LOSE to bootstrap Macrophage/Plasma/B (Tregs 0.84 -> 0.46, B F1 0.93 -> 0.72) - reverted. Curated gates in hybrid mode hurt twice; stop |
| 12 | 2026-09-09T11:51:58+02:00 | 6fdfc9e | 0.571682 | +0.002024 | pass | - | keep | thresholds x0.05 (F1 0.650, fine 0.503, reject 0.096) - threshold lever is exhausted |
| 13 | 2026-09-09T12:00:25+02:00 | 72a3f9b | 0.585989 | +0.014307 | pass | - | keep | bootstrap singlet purity 0.9 (F1 0.643, fine 0.534; exhausted CD8 0.46 -> 0.70, vascular CD8 0.41 -> 0.62) but Helper T 0.06 -> 0.00: CD4_T_cell calls vanish into Treg |
| 14 | 2026-09-09T12:08:56+02:00 | c8f4443 | 0.580175 | -0.005814 | pass | - | discard | CD4_T_cell profile from Krishna 2021 (activated CD4 TILs): Helper T still -> Treg 93%, NK F1 0.15 -> 0.08 - reverted. The DISCO Treg profile is the attractor, not the CD4 profile |

> --- Eval Checkpoint (iterations 1-6) --- Metric: 0.4575 -> 0.5663 (+0.1088) | Kept: 2/6 | Trend: up | reference composition and threshold scaling carried the gain; protein-side knobs (lambda, var model) did not


> --- Eval Checkpoint (iterations 7-12) --- Metric: 0.5663 -> 0.5717 (+0.0054) | Kept: 2/6 | Trend: flat | curated protein gates and reference swaps for T cells all hurt; remaining lever is bootstrap quality

| 15 | 2026-09-09T12:17:03+02:00 | 318e566 | 0.590391 | +0.004402 | pass | - | keep | arcsinh cofactor 20 (F1 0.637, fine 0.547; NK F1 0.15 -> 0.08 on 22 landmarks) |
| 16 | 2026-09-09T12:25:22+02:00 | 7a6ba0c | 0.590391 | 0.000000 | pass | - | discard | UMI_min 5: metric identical to 6 decimals, filtered still 7463 - the filtered cells come from counts_MIN, not UMI_min (also a determinism check: identical rerun) - reverted |
| 17 | 2026-09-09T12:33:47+02:00 | 566c268 | 0.592375 | +0.001984 | pass | - | keep | counts_MIN 5 (filtered 7463 -> 5940, reject 0.077, fine 0.551, 10x lineage agreement 0.770) |
| 18 | 2026-09-09T12:41:57+02:00 | 016b542 | 0.592637 | +0.000262 | pass | - | keep | counts_MIN 3 (filtered 5940 -> 5923; marginal) |

> --- Eval Checkpoint (iterations 13-18) --- Metric: 0.5717 -> 0.5926 (+0.0209) | Kept: 4/6 | Trend: up, flattening | bootstrap purity and arcsinh cofactor were the protein-side wins; counts_MIN is saturating

| 19 | 2026-09-09T12:50:04+02:00 | 0548b89 | 0.593705 | +0.001068 | pass | - | keep | drop Cycling_T_NK_cell (F1 0.644, T_cell F1 0.978, Stromal 0.65; fine 0.547) |
| 20 | 2026-09-09T12:58:19+02:00 | 10e24c2 | 0.597240 | +0.003535 | pass | - | keep | protein lambda 1.5 at the final state (F1 0.643, fine 0.555; NK F1 0.08 -> 0.00 on 22 landmarks) |
