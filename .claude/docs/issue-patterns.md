# Common user confusion (issue tracker patterns)

Moved out of `CLAUDE.md`, which links here.

## Common User Confusion (issue tracker patterns)

### `result.weights` on `MultiResult` / `DoubletResult` is the **full-mode** matrix

External users (e.g. issue #18) consistently expect `MultiResult.weights`
to be the multi-mode answer. It's not — it's the full-mode candidate
matrix used internally for type selection (`_multi.py:173, 348`),
unnormalized.

The actual multi-mode answer is `sub_weights[n, :n_types[n]]` paired
with `cell_type_indices[n, :n_types[n]]` and `n_types[n]` — these are
normalized to sum=1. Same shape for doublet: `weights_doublet` is the
(N, 2) normalized two-type result; `weights` is the unnormalized
full-mode candidate matrix.

This is preserved for R spacexr compatibility but it's a footgun. When
extending the README "Output schema" table or doublet/multi tutorial
sections, call this out explicitly.

### "R weights are on the simplex" misconception

R `myRCTD@results$weights` is **also** unnormalized — same
`constrain=FALSE` NNLS output as ours. The R vignette explicitly calls
`normalize_weights(results$weights)` before stuffing into the Seurat
Assay. Users reading R Section 6 (about the Assay) easily miss the
normalize step earlier and assume the raw `results$weights` is on the
simplex.

When responding to such questions, link to the R source line that does
the normalize — it's the most convincing evidence. See R spacexr
reference table below.
