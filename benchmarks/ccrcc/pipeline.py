"""THE EXPERIMENT. This is the file the autoresearch loop edits (with the protein path in
src/rctd). Contract:

    run(ad, refs, device) -> {"first_type": (N,) str, "spot_class": (N,) str,
                              "weights": (N, K) float | None, "config": <anything printable>}

`ad`: X = raw counts (cells x 477 genes), obsm["protein"] = the 18 fit markers as a DataFrame
(the 9 held-out markers the metric is built on never reach this function), obsm["spatial"] (N, 2),
obs[transcript_counts, cell_area, segmentation_method]. `refs` = {"hybrid": path, "disco": path}
of the two reference h5ads (raw counts, obs.cell_type); reference edits belong HERE.
Labels for cells the fit drops must be "filtered"; spot_class values: singlet / doublet_certain /
doublet_uncertain / reject / filtered.

Baseline = arm A2 of ccrcc_protein/02_run_arms.py: hybrid reference, doublet mode, bootstrap
protein profiles, arcsinh-robust, lambda 1.
"""

import anndata
import numpy as np

from rctd import RCTDConfig, Reference, run_rctd

CONFIG = dict(
    compile=False,
    eigh_threshold=0,
    UMI_min=10,
    protein_weight=1.0,
    protein_norm="arcsinh_robust",
    # singlet/reject thresholds are absolute log-likelihood gaps tuned for ~5k genes;
    # scale them to the 477-gene panel (A1t: rejects 31.7% -> 1.4% RNA-only)
    CONFIDENCE_THRESHOLD=5.0 * 477 / 5000,
    DOUBLET_THRESHOLD=20.0 * 477 / 5000,
)
BATCH = 4000
SPOT = np.array(["reject", "singlet", "doublet_certain", "doublet_uncertain"])


def run(ad, refs, device="auto"):
    ref = Reference(anndata.read_h5ad(refs["hybrid"]), cell_type_col="cell_type")
    cfg = RCTDConfig(device=device, **CONFIG)
    res = run_rctd(ad, ref, mode="doublet", config=cfg, batch_size=BATCH)
    n = ad.n_obs
    names = np.array(res.cell_type_names)
    first = np.array(["filtered"] * n, dtype=object)
    first[res.pixel_mask] = names[res.first_type]
    spot = np.array(["filtered"] * n, dtype=object)
    spot[res.pixel_mask] = SPOT[res.spot_class]
    W = np.full((n, len(names)), np.nan)
    W[res.pixel_mask] = res.weights
    return {"first_type": first, "spot_class": spot, "weights": W, "config": CONFIG}
