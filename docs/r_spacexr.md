# Coming from R spacexr

rctd-py implements the same algorithm as R spacexr's RCTD with the same parameters and the same
defaults, so an R configuration carries over one to one.

## Agreement with R

Measured against R spacexr on the same inputs (doublet mode, `UMI_min=20`), dominant cell type
agreement:

| Dataset | Pixels | Dominant type agreement |
|---|---|---|
| Xenium liver (small) | 13,940 | 99.73% |
| Xenium liver (large) | 58,191 | 99.71% |
| Atera WTA breast (FFPE, subsample) | 4,998 | 99.22% |
| Atera WTA cervical (FFPE, subsample) | 5,977 | 99.26% |

Most of the remaining difference comes from **sigma**, the noise parameter of the
Poisson-lognormal model. Both R and rctd-py estimate it from a random sample of pixels, and R
does not fix its random seed, so two R runs on the same data can pick sigmas 0.01 apart and
then disagree with each other on a fraction of a percent of pixels.

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
It is for validation, for migrating a project that already has R results, and for pinning sigma
across runs.

When comparing weights, normalise rctd-py's full-mode weights per pixel first: spacexr stores
normalised weights (rows sum to 1), rctd-py returns the raw fit.

## Parameter correspondence

### Reference

| R spacexr `Reference()` | rctd-py {class}`~rctd.Reference` | Default |
|---|---|---|
| cell type column of meta.data | `cell_type_col` | `"cell_type"` |
| `CELL_MIN_INSTANCE` | `cell_min` | `25` |
| `n_max_cells` | `n_max_cells` | `10000` |
| `min_UMI` | `min_UMI` | `100` |

R stops with an error when a cell type has fewer than `CELL_MIN_INSTANCE` cells; rctd-py drops
the type and continues.

### `create.RCTD()`

Every `create.RCTD()` argument has a field of the same name in {class}`~rctd.RCTDConfig`, with
the same default: `gene_cutoff`, `fc_cutoff`, `gene_cutoff_reg`, `fc_cutoff_reg`, `UMI_min`,
`UMI_max`, `counts_MIN`, `UMI_min_sigma`, `N_epoch`, `N_fit`, `MIN_CHANGE_BULK`,
`MIN_CHANGE_REG`, `MIN_OBS`, `MAX_MULTI_TYPES`, `CONFIDENCE_THRESHOLD`, `DOUBLET_THRESHOLD`,
`class_df`. The {class}`~rctd.RCTDConfig` reference lists what each one does.

`max_cores` has no equivalent: rctd-py vectorises across pixels instead of forking workers. On
CPU, set the thread count with `OMP_NUM_THREADS`; on GPU, pass `device="cuda"`.

### `run.RCTD(doublet_mode = ...)`

| R | rctd-py |
|---|---|
| `run.RCTD(rctd, doublet_mode = "full")` | `run_rctd(spatial, reference, mode="full")` |
| `run.RCTD(rctd, doublet_mode = "doublet")` | `run_rctd(spatial, reference, mode="doublet")` |
| `run.RCTD(rctd, doublet_mode = "multi")` | `run_rctd(spatial, reference, mode="multi")` |
| `rctd@results$results_df` | {class}`~rctd.DoubletResult` fields |
| `rctd@results$weights` | `result.weights` |

R's multi mode stops the whole run with an error when a pixel has no cell type above the weight
threshold; rctd-py keeps the type with the highest weight for that pixel.
