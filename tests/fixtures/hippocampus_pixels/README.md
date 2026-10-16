# Pixel-level R parity fixture

300 Slide-seq V2 mouse hippocampus beads (public data from the spacexr paper, Cable et al. 2022)
with everything R spacexr 2.2.1 fed into its per-pixel doublet step: the bulk-normalised reference
profiles on `gene_list_reg`, the chosen sigma, counts and nUMI, plus R's per-bead results.

Generated from one `create.RCTD()` + `run.RCTD(doublet_mode = "doublet")` run (seed 1, sigma 0.52,
K = 17). 150 beads are ones where rctd-py <= 0.3.9 reported a different `first_type` than R; 150 are
random others. Feeding R's own inputs removes sigma estimation and the bulk fit, so any difference
left is in the per-pixel code.
