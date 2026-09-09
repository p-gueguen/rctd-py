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


DROP_TYPES = {"cDC1", "cDC2"}  # on this 477-gene panel they absorb the macrophage groups
TUMOUR_SUBSTATES = 3  # RCTD fits ONE mean profile per type; ccRCC has hypoxic / proliferative /
# differentiated states that one profile misses (10x "Growth Signaling" tumour recall 0.32).
# Split the Zhang tumour cells into k expression sub-states for the fit, merge the labels back.


def split_tumour(ref_ad, k, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    ct = ref_ad.obs["cell_type"].astype(str).to_numpy()
    m = ct == "Tumour_ccRCC"
    X = ref_ad.X[m]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e4)
    Z = PCA(n_components=20, random_state=seed).fit_transform(X)
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    ct[m] = [f"Tumour_ccRCC_s{i}" for i in lab]
    ref_ad.obs["cell_type"] = ct
    return ref_ad


def run(ad, refs, device="auto"):
    ref_ad = anndata.read_h5ad(refs["hybrid"])
    ref_ad = ref_ad[~ref_ad.obs["cell_type"].astype(str).isin(DROP_TYPES)].copy()
    ref_ad = split_tumour(ref_ad, TUMOUR_SUBSTATES)
    ref = Reference(ref_ad, cell_type_col="cell_type")
    cfg = RCTDConfig(device=device, **CONFIG)
    res = run_rctd(ad, ref, mode="doublet", config=cfg, batch_size=BATCH)
    n = ad.n_obs
    names = np.array(res.cell_type_names)
    merged = np.array([t.split("_s")[0] if t.startswith("Tumour_ccRCC_s") else t for t in names])
    first = np.array(["filtered"] * n, dtype=object)
    first[res.pixel_mask] = merged[res.first_type]
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
            "tumour_substates": TUMOUR_SUBSTATES,
        },
    }
