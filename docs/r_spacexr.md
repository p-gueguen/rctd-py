# Coming from R spacexr

rctd-py implements the same algorithm as R spacexr's RCTD, with R's parameter names and R's
defaults.

## Agreement with R

Measured against R spacexr 2.2.1 on 19 datasets (doublet mode, `UMI_min=20`), each side estimating
its own sigma, the noise parameter of the Poisson-lognormal model:

- On the 11 datasets where rctd-py picks the same sigma as R, the calls match R on every pixel:
  100% of spot classes and first types.
- On the other 8, the two sigmas differ by 0.01-0.02, and 98.99-99.88% of spot classes agree.

| Dataset | Pixels | Sigma (rctd-py / R) | Spot class | First type | Dominant type |
|---|---:|---|---:|---:|---:|
| Visium v1 FFPE Breast (55 µm spots) | 2,518 | 0.37 / 0.37 | 100% | 100% | 100% |
| Visium v1 Breast (55 µm spots) | 3,798 | 0.30 / 0.30 | 100% | 100% | 100% |
| Visium v2 CytAssist Breast (55 µm spots) | 4,992 | 0.41 / 0.42 | 99.64% | 99.88% | 99.94% |
| Slide-seq Mouse Cerebellum | 11,626 | 0.88 / 0.88 | 100% | 100% | 100% |
| Xenium Liver (small) | 13,936 | 0.62 / 0.62 | 100% | 100% | 100% |
| Slide-seq Mouse Hippocampus | 17,747 | 0.52 / 0.52 | 100% | 100% | 100% |
| StrataMap Breast (1.5 mm window) | 21,425 | 0.69 / 0.69 | 100% | 100% | 100% |
| Xenium Mouse Brain | 36,362 | 0.69 / 0.69 | 100% | 100% | 100% |
| VisiumHD Breast (segmented cells) | 38,990 | 0.67 / 0.68 | 99.47% | 99.77% | 99.71% |
| Xenium Liver (large) | 58,187 | 0.72 / 0.70 | 99.05% | 99.42% | 99.73% |
| VisiumHD Human (8 µm, K=78) | 102,971 | 0.47 / 0.47 | 100% | 100% | >99.99% |
| CosMx WTX Breast | 151,281 | 1.00 / 1.00 | 100% | 100% | 100% |
| Xenium Breast | 162,023 | 0.62 / 0.63 | 99.46% | 99.72% | 99.86% |
| Atera WTA Breast | 169,965 | 0.74 / 0.74 | 100% | 100% | 100% |
| VisiumHD 3' Mouse Brain (8 µm) | 371,461 | 0.40 / 0.41 | 99.85% | 99.88% | 99.89% |
| VisiumHD Mouse Brain (8 µm) | 391,894 | 0.43 / 0.44 | 99.84% | 99.88% | 99.91% |
| Xenium Prime 5K Breast | 503,411 | 0.65 / 0.64 | 99.88% | 99.86% | 99.91% |
| VisiumHD Breast (8 µm bins) | 557,300 | 0.80 / 0.78 | 98.99% | 99.42% | 99.52% |
| Atera WTA Cervical | 714,535 | 1.12 / 1.12 | 100% | 100% | >99.99% |

The remaining difference is sigma alone. Both R and rctd-py estimate it from a random sample of
pixels, and R does not fix its random seed, so two R runs on the same data can also pick sigmas
0.01 apart and then agree on only about 99.3% of spot classes (Xenium liver small). Given R's
sigma, rctd-py matches R on 100% of spot classes even on the dataset with the largest gap above
(Visium HD breast, 557,300 bins).

Before 0.4.0, rctd-py agreed less on sequencing-based data (84.9% dominant-type agreement on
Visium v1 breast) because its bulk fit used genes R leaves out; see the changelog.

### Reproducing a given R run

To match a specific R result, take its sigma and pass it to rctd-py, which then skips sigma
estimation:

```r
# R
sigma <- round(100 * myRCTD@internal_vars$sigma)   # e.g. 63
```

```python
result = run_rctd(spatial, reference, mode="doublet", sigma_override=63)
```

`sigma_override` is not needed for normal use; the sigma rctd-py estimates is as valid as R's.
It is for validation against R, for migrating a project that already has R results, and for
keeping sigma fixed when you rerun on a subset of the data.

Neither package normalises full-mode weights: R's `myRCTD@results$weights` and rctd-py's
`result.weights` are both the raw fit. To compare them as proportions, normalise both per pixel
(in R, `normalize_weights()`). Doublet-mode `weights_doublet` and multi-mode `sub_weights` are
already normalised in both.

## Parameter correspondence

### Reference

| R spacexr | rctd-py {class}`~rctd.Reference` | Default |
|---|---|---|
| `cell_types` argument of `Reference()` | `cell_type_col` (a column of `.obs`) | `"cell_type"` |
| `create.RCTD(CELL_MIN_INSTANCE = )` | `cell_min` | `25` |
| `n_max_cells` | `n_max_cells` | `10000` |
| `min_UMI` | `min_UMI` | `100` |

R stops with an error when a cell type has fewer than `CELL_MIN_INSTANCE` cells; rctd-py drops
the type and continues.

### `create.RCTD()`

These `create.RCTD()` arguments are fields of the same name and default in
{class}`~rctd.RCTDConfig`: `gene_cutoff`, `fc_cutoff`, `gene_cutoff_reg`, `fc_cutoff_reg`,
`UMI_min`, `UMI_max`, `counts_MIN`, `UMI_min_sigma`, `MAX_MULTI_TYPES`,
`CONFIDENCE_THRESHOLD` (5), `DOUBLET_THRESHOLD` (20) and `class_df`. `CELL_MIN_INSTANCE` is
`cell_min` on {class}`~rctd.Reference`. `class_df` is a dict `{cell_type: class}` in rctd-py
instead of R's data frame with the cell types as row names and a `class` column.

R's internal config entries are fields too: `N_epoch`, `MIN_CHANGE_BULK`, `MIN_CHANGE_REG`,
`MIN_OBS` and `N_fit`, with R's defaults (`MIN_CHANGE_REG` is accepted but not read yet).
`N_fit`, the number of pixels sampled for sigma estimation, is 1000 as in R since 0.4.0 (100
before). The {class}`~rctd.RCTDConfig` reference lists what each field does.

`max_cores` has no equivalent: rctd-py vectorises across pixels instead of forking workers. On
CPU, set the thread count with `OMP_NUM_THREADS`. A GPU is used automatically when PyTorch sees
one (`device="auto"`); force a device with `RCTDConfig(device="cuda")` or `--device cuda`.

### `run.RCTD(doublet_mode = ...)`

| R | rctd-py |
|---|---|
| `run.RCTD(rctd, doublet_mode = "full")` | `run_rctd(spatial, reference, mode="full")` |
| `run.RCTD(rctd, doublet_mode = "doublet")` | `run_rctd(spatial, reference, mode="doublet")` |
| `run.RCTD(rctd, doublet_mode = "multi")` | `run_rctd(spatial, reference, mode="multi")` |
| `rctd@results$results_df` | {class}`~rctd.DoubletResult` fields (R's `conv_all` and `conv_doublet` have no equivalent) |
| `rctd@results$weights` | `result.weights` |

Where R stops with an error, rctd-py carries on:

- R's multi mode stops the whole run when a pixel has no cell type above the weight threshold
  (in the {doc}`benchmarks <benchmarks>`, after 3.2 hours on a 714,535-cell dataset); rctd-py
  keeps the type with the highest weight for that pixel.
- R stops when fewer than 10 differentially expressed genes are selected; rctd-py prints a warning
  and uses all common genes (bulk fit) or the bulk gene list (pixel fits) instead.
