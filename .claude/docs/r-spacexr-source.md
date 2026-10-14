# R spacexr source reference

Moved out of `CLAUDE.md`, which links here.

## R spacexr Source Reference

For citations and cross-checks (avoid re-grepping each time):

| Topic | R location |
|-------|------------|
| IRWLS damping (alpha=0.3 → our `step_size`) | [`R/IRWLS.R:95`](https://github.com/dmcable/spacexr/blob/master/R/IRWLS.R#L95) |
| Multi candidate threshold (0.01 → our `WEIGHT_THRESHOLD`) | [`R/RCTD_helper.R:76, 153`](https://github.com/dmcable/spacexr/blob/master/R/RCTD_helper.R) |
| `normalize_weights` definition | [`R/postProcessing.R:135-137`](https://github.com/dmcable/spacexr/blob/master/R/postProcessing.R#L135-L137) |
| Tutorial Assay construction (the `# normalize the cell type proportions to sum to 1.` line) | [`vignettes/spatial-transcriptomics.Rmd:135-143`](https://github.com/dmcable/spacexr/blob/master/vignettes/spatial-transcriptomics.Rmd#L135-L143) |
| `solveOLS` / `solveIRWLS.weights` | `R/IRWLS.R:1-50` |
| `process_bead_doublet` / `check_pairs_type` | `R/RCTD_helper.R:44-168` |
