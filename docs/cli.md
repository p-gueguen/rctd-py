# Command line

Installing rctd-py adds an `rctd` command with three subcommands:

```bash
rctd info                                   # versions, GPU availability (--json for machines)
rctd validate spatial.h5ad reference.h5ad   # fast input checks, no GPU needed
rctd run spatial.h5ad reference.h5ad        # deconvolve (default: doublet mode)
```

```bash
rctd run spatial.h5ad reference.h5ad \
    --mode doublet --output results.h5ad --device cuda \
    --batch-size 5000 --umi-min 20 --cell-type-col cell_type

rctd run spatial.h5ad reference.h5ad --json --quiet   # JSON summary for pipelines
```

Both positional arguments accept a `.h5ad` file, an AnnData Zarr store or a SpatialData Zarr
store. In a SpatialData store with several tables, name one with `--table` (or point at
`store.zarr/tables/<name>`). Results are always written as `.h5ad`; the input is never modified.

## Output

Results are written into a copy of the spatial input, by default `<spatial_stem>_rctd.h5ad`.

| Slot | Content | Modes |
|---|---|---|
| `.obsm["rctd_weights"]` | cell type weights (N x K) | all |
| `.obs["rctd_dominant_type"]` | top cell type per pixel | all |
| `.obs["rctd_spot_class"]` | singlet / doublet_certain / doublet_uncertain / reject | doublet |
| `.obs["rctd_first_type"]`, `.obs["rctd_second_type"]` | primary and secondary type | doublet |
| `.obsm["rctd_weights_doublet"]` | doublet weights (N x 2) | doublet |
| `.obs["rctd_converged"]` | convergence flag | full |
| `.obs["rctd_n_types"]` | number of types per pixel | multi |
| `.obsm["rctd_sub_weights"]`, `.obsm["rctd_cell_type_indices"]` | selected types and their weights | multi |
| `.uns["rctd_mode"]`, `.uns["rctd_config"]`, `.uns["rctd_cell_type_names"]` | run metadata | all |

Pixels that were not fitted get `NaN` weights and `"filtered"` labels: total UMI outside
`--umi-min` / `--umi-max`, or fewer than `counts_MIN` (10) counts on the bulk gene list.

## Reference

```{click} rctd.cli:main
:prog: rctd
:nested: full
```
