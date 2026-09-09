#!/usr/bin/env python3
"""Region-annotation checks on the external section, from the labels each eval run saved.
Two expectations that use neither protein nor 10x: inside pathologist 'Immune infiltration'
polygons the calls should be immune lineages (a Tumour_ccRCC call there is wrong); inside 'Tumor'
polygons a normal-nephron call (PT / LoH / DCT / intercalated / podocyte) is wrong. Writes
out/external_metrics.json."""

import glob
import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DATA = Path("/srv/GT/analysis/pgueguen/rctd-py/ccrcc_protein/autoresearch_data")
IMMUNE = {
    "CD4_T_cell",
    "CD8_T_cell",
    "Treg_cell",
    "NK_cell",
    "B_cell",
    "Plasma_cell",
    "Macrophage",
    "CD14_monocyte",
    "CD16_monocyte",
    "cDC1",
    "cDC2",
    "Cycling_myeloid_cell",
    "Cycling_T_NK_cell",
    "Mast_cell",
}
NEPHRON = {
    "Renal_epithelial_cell_proximal_tubule",
    "Renal_epithelial_cell_Loop_of_Henle",
    "Renal_epithelial_cell_distal_tubule",
    "Type_A_intercalated_cell",
    "Type_B_intercalated_cell",
    "Podocyte",
}

ad = anndata.read_h5ad(DATA / "external.h5ad", backed="r")
region = ad.obs["region"].astype(str).to_numpy()
res = {}
for f in sorted(glob.glob(str(OUT / "labels_external_*.npz"))):
    tag = Path(f).stem[len("labels_") :]
    z = np.load(f, allow_pickle=True)
    assert (z["obs_names"] == np.asarray(ad.obs_names, dtype=object)).all()
    ft, sc = z["first_type"].astype(str), z["spot_class"].astype(str)
    called = ~np.isin(sc, ["reject", "filtered"])
    tum, imm = region == "Tumor", region == "Immune infiltration"
    ej = json.load(open(OUT / f"eval_{tag}.json"))
    res[tag] = {
        "n_tumor_region": int(tum.sum()),
        "n_immune_region": int(imm.sum()),
        "tumor_region_called_Tumour_ccRCC": float((ft[tum] == "Tumour_ccRCC").mean()),
        "tumor_region_called_nephron": float(np.isin(ft[tum], list(NEPHRON)).mean()),
        "tumor_region_called_immune": float(np.isin(ft[tum], list(IMMUNE)).mean()),
        "tumor_region_reject_or_filtered": float((~called[tum]).mean()),
        "immune_region_called_immune": float(np.isin(ft[imm], list(IMMUNE)).mean()),
        "immune_region_called_Tumour_ccRCC": float((ft[imm] == "Tumour_ccRCC").mean()),
        "immune_region_called_nephron": float(np.isin(ft[imm], list(NEPHRON)).mean()),
        "immune_region_reject_or_filtered": float((~called[imm]).mean()),
        "heldout_lineage_f1": ej.get("heldout_lineage_f1"),
        "reject_frac": ej.get("reject_frac"),
        "per_class_f1": {k: v["f1"] for k, v in ej.get("per_class_f1", {}).items()},
        "calls_top": pd.Series(ft).value_counts().head(8).to_dict(),
    }
json.dump(res, open(OUT / "external_metrics.json", "w"), indent=2)
for tag, r in res.items():
    print(
        f"{tag:26s} F1 {r['heldout_lineage_f1']:.3f} | tumour region: tumour {r['tumor_region_called_Tumour_ccRCC']:.3f} nephron {r['tumor_region_called_nephron']:.3f} "
        f"immune {r['tumor_region_called_immune']:.3f} rej {r['tumor_region_reject_or_filtered']:.3f} | immune region: immune {r['immune_region_called_immune']:.3f} "
        f"tumour {r['immune_region_called_Tumour_ccRCC']:.3f} nephron {r['immune_region_called_nephron']:.3f}"
    )
