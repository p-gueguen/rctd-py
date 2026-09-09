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
    # 32% of 10x's exhausted CD8 cells are called CD4_T_cell: cleaner bootstrap singlets should give
    # sharper CD4 / CD8A protein profiles for the two types
    protein_singlet_purity=0.9,
    # singlet/reject thresholds are absolute log-likelihood gaps tuned for ~5k genes;
    # scale them to the 477-gene panel (A1t: rejects 31.7% -> 1.4% RNA-only)
    CONFIDENCE_THRESHOLD=5.0 * 477 / 5000 / 2,  # x0.05: rejects still 10% at x0.1 (Memory T 34%)
    DOUBLET_THRESHOLD=20.0 * 477 / 5000 / 2,
)
BATCH = 4000
SPOT = np.array(["reject", "singlet", "doublet_certain", "doublet_uncertain"])


# cDC1/cDC2 absorb the macrophage groups on this 477-gene panel; Cycling_myeloid_cell (135 DISCO
# cells, a proliferation profile) is a top-4 call in 14 of 16 10x groups - it steals T and tumour cells
DROP_TYPES = {"cDC1", "cDC2", "Cycling_myeloid_cell"}


# DISCO's CD4_T_cell is a normal-kidney (naive-like) profile; 92% of 10x's Helper T cells go to
# Treg_cell. Replace only that one type with the activated/effector CD4 T cells of the tissue-matched
# Krishna 2021 ccRCC atlas (iteration 10 swapped all four T/NK types and lost CD8 to a naive CD8 label).
SWAP_FROM_KRISHNA = {"CD4_T_cell"}


def build_reference(refs):
    ref_ad = anndata.read_h5ad(refs["hybrid"])
    ct = ref_ad.obs["cell_type"].astype(str)
    ref_ad = ref_ad[~ct.isin(DROP_TYPES | SWAP_FROM_KRISHNA)].copy()
    kr = anndata.read_h5ad(refs["krishna_immune"])
    kr = kr[kr.obs["cell_type"].astype(str).isin(SWAP_FROM_KRISHNA)].copy()
    genes = sorted(set(ref_ad.var_names) & set(kr.var_names))
    out = anndata.concat(
        [ref_ad[:, genes].copy(), kr[:, genes].copy()], join="inner", index_unique="-"
    )
    out.obs["cell_type"] = out.obs["cell_type"].astype(str)
    return out


def run(ad, refs, device="auto"):
    ref = Reference(build_reference(refs), cell_type_col="cell_type")
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
    return {
        "first_type": first,
        "spot_class": spot,
        "weights": W,
        "config": {
            **CONFIG,
            "drop_types": sorted(DROP_TYPES),
            "swap_from_krishna": sorted(SWAP_FROM_KRISHNA),
        },
    }
