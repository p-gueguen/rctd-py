#!/usr/bin/env python3
"""External transfer section: 10x Xenium_V1_Human_Kidney_FFPE_Protein_updated (ccRCC, a different
patient block, 405-gene panel sharing 387 genes with the tuning section, the same 27-plex; CA9 is NOT
in this panel). 150k-cell subsample (seed 2). Truth columns: protein-gated landmark classes exactly
as prepare.py, plus obs["region"] from the pathologist annotation geojson (H&E pixel space ->
alignment matrix -> x 0.2125 um/px): Tumor / Immune infiltration / Necrosis / Adipose tissue.
Needs the GPU (RNA-only baseline fit for eval_cell). Never used by the loop."""

import json
import sys
from pathlib import Path

import anndata
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import DATA, GATE_MARKERS, LINEAGE, LINEAGE_SIGS, MIN_LM  # noqa: E402

from rctd import RCTDConfig, Reference, run_rctd  # noqa: E402
from rctd._protein import gate_landmarks, normalize_protein  # noqa: E402
from rctd._types import SPOT_CLASS_REJECT  # noqa: E402


def contains_points(poly, pts):
    """Even-odd ray casting, vectorised over points (no matplotlib in the venv)."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    xj, yj = poly[-1]
    for xi, yi in poly:
        cross = (yi > y) != (yj > y)
        xint = (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
        inside ^= cross & (x < xint)
        xj, yj = xi, yi
    return inside


SRC = Path("/srv/GT/analysis/pgueguen/rctd-py/ccrcc_protein/external_kidney/data")
B = "Xenium_V1_Human_Kidney_FFPE_Protein_updated"
N = 150_000
PX_UM = 0.2125

with h5py.File(SRC / f"{B}_cell_feature_matrix.h5") as f:
    m = f["matrix"]
    mat = csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:]))
    names = np.array([x.decode() for x in m["features"]["name"][:]])
    ftype = np.array([x.decode() for x in m["features"]["feature_type"][:]])
    barcodes = np.array([x.decode() for x in m["barcodes"][:]])
X = mat.T.tocsr()
is_rna, is_prot = ftype == "Gene Expression", ftype == "Protein Expression"
cells = pd.read_parquet(SRC / f"{B}_cells.parquet").set_index("cell_id").loc[barcodes]
rna_tot = np.asarray(X[:, is_rna].sum(1)).ravel()
keep = np.where(rna_tot >= 15)[0]
idx = np.sort(np.random.default_rng(2).choice(keep, size=min(N, len(keep)), replace=False))

ad = anndata.AnnData(
    X=X[idx][:, is_rna].tocsr().astype(np.int32),
    obs=cells.iloc[idx][["transcript_counts", "cell_area", "segmentation_method"]].copy(),
)
ad.var_names = list(names[is_rna])
ad.obs_names = barcodes[idx]
ad.obsm["spatial"] = cells.iloc[idx][["x_centroid", "y_centroid"]].to_numpy()
prot = pd.DataFrame(
    np.asarray(X[idx][:, is_prot].todense()), index=ad.obs_names, columns=list(names[is_prot])
)
fit_markers = [c for c in prot.columns if c not in GATE_MARKERS]
assert len(fit_markers) == 18
ad.obsm["protein_fit"], ad.obsm["protein_truth"] = prot[fit_markers], prot[GATE_MARKERS]
ad.obs["tenx_group"] = np.nan  # no 10x cell groups for this section
ad.uns["fit_markers"], ad.uns["truth_markers"], ad.uns["split"] = (
    fit_markers,
    GATE_MARKERS,
    "external",
)

# region annotation: geojson (H&E pixels) -> alignment matrix -> Xenium morphology pixels -> um
M = np.loadtxt(SRC / f"{B}_he_imagealignment.csv", delimiter=",")
g = json.load(open(SRC / f"{B}_annotation.geojson"))
pts = ad.obsm["spatial"]
region = np.array([""] * ad.n_obs, dtype=object)
for ft in g["features"]:
    poly = np.array(ft["geometry"]["coordinates"][0], float)
    h = np.c_[poly, np.ones(len(poly))]
    poly = ((M @ h.T).T[:, :2]) * PX_UM
    inside = contains_points(poly, pts)
    region[inside & (region == "")] = ft["properties"]["name"]  # first polygon wins (0.6% overlap)
ad.obs["region"] = region
print("regions:", pd.Series(region).value_counts().to_dict())

classes = sorted(set(LINEAGE.values()))
P_std, _, _ = normalize_protein(ad.obsm["protein_truth"].to_numpy(), method="arcsinh_robust")
labels, info = gate_landmarks(P_std, GATE_MARKERS, LINEAGE_SIGS, classes, min_cells=MIN_LM)
ad.obs["landmark_class"] = np.where(labels >= 0, np.array(classes)[np.maximum(labels, 0)], "")
ref = Reference(anndata.read_h5ad(DATA / "reference_hybrid_ccrcc.h5ad"), cell_type_col="cell_type")
base = anndata.AnnData(X=ad.X.copy(), obs=pd.DataFrame(index=ad.obs_names))
base.var_names = ad.var_names
res = run_rctd(
    base,
    ref,
    mode="doublet",
    batch_size=4000,
    config=RCTDConfig(compile=False, device="auto", eigh_threshold=0, UMI_min=10),
)
called = np.zeros(ad.n_obs, dtype=bool)
called[res.pixel_mask] = res.spot_class != SPOT_CLASS_REJECT
ad.obs["eval_cell"] = (labels >= 0) & called
bf = np.array(["filtered"] * ad.n_obs, dtype=object)
bf[res.pixel_mask] = np.array(res.cell_type_names)[res.first_type]
ad.obs["baseline_rna_first_type"] = pd.Categorical(bf)
ad.write_h5ad(DATA / "external.h5ad")
per = {
    c: int(((labels >= 0) & (np.array(classes)[np.maximum(labels, 0)] == c)).sum()) for c in classes
}
print(
    f"[external] {ad.n_obs} cells, genes {ad.n_vars}, landmarks {int((labels >= 0).sum())} {per}, "
    f"eval cells {int(ad.obs['eval_cell'].sum())}, baseline reject {float((~called).mean()):.3f}"
)
print("DONE")
