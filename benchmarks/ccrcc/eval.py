#!/usr/bin/env python3
"""READ-ONLY metric for the ccRCC RNA+protein autoresearch loop. The loop edits pipeline.py
(and the protein path in src/rctd); it never edits this file, prepare.py, verify.py,
guard.py, eval.sbatch or the data.

    uv run python benchmarks/ccrcc/eval.py --split tune --tag <sha>

Prints ONE number on the last stdout line: sqrt(heldout_lineage_F1 * tenx_fine_agreement).
Everything else goes to benchmarks/ccrcc/out/eval_<tag>.json.

Metric 1, held-out lineage F1: landmark cells were gated in prepare.py on the 9 gate markers
(obs["landmark_class"]); the pipeline receives only the other 18 markers. Scored on ALL
landmark cells, first_type mapped through LINEAGE, rejects and "filtered" count as misses (so calling more cells only pays if the
calls are right - precision falls otherwise), macro over the classes present. The called-only F1
(landmarks the RNA-only baseline called, obs["eval_cell"]) is reported alongside.

Metric 2, 10x fine-type agreement: for each 10x WNN group, the fraction of its cells whose
first_type is in the group's accepted set (TENX_ACCEPT); macro over groups with >= 50 cells.
Rejects/filtered are misses. 10x's labels used protein, so this axis favours protein arms -
which is why it is paired with metric 1, not used alone.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = Path("/srv/GT/analysis/pgueguen/rctd-py/ccrcc_protein/autoresearch_data")
REF_HYBRID_SRC = "/srv/GT/analysis/pgueguen/rctd-py/ccrcc_protein/reference_hybrid_ccrcc.h5ad"
REF_DISCO_SRC = "/srv/GT/analysis/pgueguen/rctd-py/renal_protein_test/kidney_ref.h5ad"
MIN_LM = 20
GATE_MARKERS = ["CD3E", "CD20", "CD68", "CD31", "PanCK", "alphaSMA", "CD45", "CD138", "CD16"]

LINEAGE = {
    "Tumour_ccRCC": "Epithelial",
    "Renal_epithelial_cell_proximal_tubule": "Epithelial",
    "Renal_epithelial_cell_Loop_of_Henle": "Epithelial",
    "Renal_epithelial_cell_distal_tubule": "Epithelial",
    "Type_A_intercalated_cell": "Epithelial",
    "Type_B_intercalated_cell": "Epithelial",
    "Podocyte": "Epithelial",
    "CD4_T_cell": "T_cell",
    "CD8_T_cell": "T_cell",
    "Treg_cell": "T_cell",
    "Cycling_T_NK_cell": "T_cell",
    "NK_cell": "NK",
    "B_cell": "B_cell",
    "Plasma_cell": "Plasma_cell",
    "Macrophage": "Myeloid",
    "CD14_monocyte": "Myeloid",
    "CD16_monocyte": "Myeloid",
    "cDC1": "Myeloid",
    "cDC2": "Myeloid",
    "Cycling_myeloid_cell": "Myeloid",
    "Arterial_EC": "Endothelial",
    "Capillary_EC": "Endothelial",
    "Venous_EC": "Endothelial",
    "Lymphatic_EC": "Endothelial",
    "Fibroblast": "Stromal",
    "Pericyte": "Stromal",
    # Zhang 2021 types a pipeline may add to the reference
    "Endothelial": "Endothelial",
    "VSMC": "Stromal",
    "Myofibroblast": "Stromal",
    "Mast_cell": "Myeloid",
}
LINEAGE_SIGS = {
    "Epithelial": {"positive": ["PanCK"], "negative": ["CD45", "CD31", "CD3E", "CD68", "CD20"]},
    "T_cell": {"positive": ["CD3E"], "negative": ["CD20", "CD68", "PanCK", "CD31"]},
    "B_cell": {"positive": ["CD20"], "negative": ["CD3E", "CD68", "CD138"]},
    "Plasma_cell": {"positive": ["CD138"], "negative": ["CD20", "CD3E", "PanCK", "CD31"]},
    "Myeloid": {"positive": ["CD68"], "negative": ["CD3E", "CD20", "PanCK", "CD31"]},
    "Endothelial": {"positive": ["CD31"], "negative": ["CD45", "PanCK", "alphaSMA", "CD3E"]},
    "Stromal": {"positive": ["alphaSMA"], "negative": ["CD31", "CD45", "PanCK", "CD68"]},
    "NK": {"positive": ["CD16"], "negative": ["CD3E", "CD68", "CD20"]},
}
EC = {"Arterial_EC", "Capillary_EC", "Venous_EC", "Lymphatic_EC", "Endothelial"}
MYELOID_MONO = {"Macrophage", "CD14_monocyte", "CD16_monocyte", "Cycling_myeloid_cell"}
T_ANY = {"CD4_T_cell", "CD8_T_cell", "Treg_cell"}
TENX_ACCEPT = {
    "Regulatory T Cells (Tregs)": {"Treg_cell"},
    "Exhausted Cytotoxic T Cells": {"CD8_T_cell"},
    "Vascular-Localized Cytotoxic T Cells": {"CD8_T_cell"},
    "Cycling Exhausted T Cells": {"CD8_T_cell", "Cycling_T_NK_cell"},
    "Proliferating T Cells": {"CD8_T_cell", "CD4_T_cell", "Cycling_T_NK_cell"},
    "Helper T Cells": {"CD4_T_cell"},
    "Memory T Cells": {"CD4_T_cell", "CD8_T_cell"},
    "Lymphatic-Associated T Cells": T_ANY,
    "Follicular B Cells and Interacting Tfh Cells": {"B_cell", "CD4_T_cell"},
    "Endothelial-Associated Plasma Cells": {"Plasma_cell"} | EC,
    "Monocyte-Derived Macrophages": MYELOID_MONO,
    "Tumor-Associated Macrophages (M2)": MYELOID_MONO,
    "Plasmacytoid Dendritic Cells (pDCs)": {"cDC1", "cDC2"},
    "Hypoxic Tumor": {"Tumour_ccRCC"},
    "Differentiated Proximal-Like Tumor": {"Tumour_ccRCC"},
    "Growth Signaling-Enriched Tumor": {"Tumour_ccRCC"},
    "Stressed/Dedifferentiating Proximal Tubule": {
        "Renal_epithelial_cell_proximal_tubule",
        "Tumour_ccRCC",
        "Renal_epithelial_cell_Loop_of_Henle",
    },
    # "Tumor-Associated Neutrophils": excluded - no neutrophil type in either reference
}
TENX_MIN_CELLS = 50
FORBIDDEN_OBS = {"tenx_group", "landmark_class", "eval_cell", "baseline_rna_first_type"}


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True
        ).strip()
    except Exception:
        return "nogit"


def strip_for_pipeline(ad):
    """What the pipeline is allowed to see: counts, the 18 fit markers, coordinates, 4 obs columns."""
    keep = ["transcript_counts", "cell_area", "segmentation_method"]
    out = anndata.AnnData(X=ad.X.copy(), obs=ad.obs[keep].copy())
    out.var_names = ad.var_names
    out.obsm["spatial"] = np.asarray(ad.obsm["spatial"])
    out.obsm["protein"] = ad.obsm["protein_fit"].copy()
    out.uns["protein_feature_names"] = list(ad.obsm["protein_fit"].columns)
    assert not (set(out.obs.columns) & FORBIDDEN_OBS)
    assert not (set(out.obsm["protein"].columns) & set(GATE_MARKERS)), (
        "gate marker leaked into the fit set"
    )
    assert "protein_truth" not in out.obsm
    return out


def macro_f1(true_cls, pred_cls, classes):
    per = {}
    for c in classes:
        tp = int(((true_cls == c) & (pred_cls == c)).sum())
        fp = int(((true_cls != c) & (pred_cls == c)).sum())
        fn = int(((true_cls == c) & (pred_cls != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per[c] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "support": int((true_cls == c).sum()),
        }
    return float(np.mean([v["f1"] for v in per.values()])), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="tune", choices=["tune", "test"])
    ap.add_argument("--pipeline", default=str(HERE / "pipeline.py"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--shuffle-protein-seed",
        type=int,
        default=None,
        help="permutation null: shuffle each fit marker independently before the pipeline sees it",
    )
    a = ap.parse_args()
    t0 = time.time()
    tag = a.tag or f"{git_sha()}_{int(t0)}"
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ad = anndata.read_h5ad(DATA / f"{a.split}.h5ad")
    ad_in = strip_for_pipeline(ad)
    if a.shuffle_protein_seed is not None:
        rng = np.random.default_rng(a.shuffle_protein_seed)
        P = ad_in.obsm["protein"].to_numpy().copy()
        for j in range(P.shape[1]):
            P[:, j] = P[rng.permutation(P.shape[0]), j]
        ad_in.obsm["protein"] = pd.DataFrame(
            P, index=ad_in.obs_names, columns=ad_in.obsm["protein"].columns
        )
    refs = {
        "hybrid": str(DATA / "reference_hybrid_ccrcc.h5ad"),
        "disco": str(DATA / "kidney_ref.h5ad"),
    }

    spec = importlib.util.spec_from_file_location("pipeline", a.pipeline)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    res = pipeline.run(ad_in, refs, a.device)
    first = np.asarray(res["first_type"], dtype=object)
    spot = np.asarray(res["spot_class"], dtype=object)
    assert first.shape == (ad.n_obs,) and spot.shape == (ad.n_obs,), (
        "pipeline must return one label per input cell"
    )
    called = ~np.isin(spot, ["reject", "filtered"])
    pred_lin = pd.Series(first).map(LINEAGE).fillna("__none__").to_numpy()
    pred_lin = np.where(called, pred_lin, "__reject__")

    # ---- metric 1: held-out lineage F1 on ALL landmark cells (rejects/filtered = misses) ----
    true_cls = ad.obs["landmark_class"].astype(str).to_numpy()
    ev = true_cls != ""
    classes = sorted(c for c in set(true_cls[ev]) if c)
    f1, per_class = macro_f1(true_cls[ev], pred_lin[ev], classes)
    ev_called = (
        ad.obs["eval_cell"].to_numpy().astype(bool)
    )  # landmarks the RNA-only baseline called
    f1_called, _ = macro_f1(true_cls[ev_called], pred_lin[ev_called], classes)

    # ---- metric 2: 10x fine-type agreement (macro over groups) ----
    g = ad.obs["tenx_group"].astype(object)
    per_group = {}
    for grp, acc in TENX_ACCEPT.items():
        m = (g == grp).to_numpy()
        if m.sum() < TENX_MIN_CELLS:
            continue
        hit = np.isin(first[m], list(acc)) & called[m]
        per_group[grp] = {"n": int(m.sum()), "agreement": round(float(hit.mean()), 4)}
    fine = (
        float(np.mean([v["agreement"] for v in per_group.values()])) if per_group else float("nan")
    )

    metric = float(np.sqrt(max(f1, 0.0) * max(fine, 0.0)))

    # ---- extras (reported, never optimised) ----
    from_lin = {
        "Regulatory T Cells (Tregs)": "T_cell",
        "Exhausted Cytotoxic T Cells": "T_cell",
        "Memory T Cells": "T_cell",
        "Cycling Exhausted T Cells": "T_cell",
        "Helper T Cells": "T_cell",
        "Vascular-Localized Cytotoxic T Cells": "T_cell",
        "Proliferating T Cells": "T_cell",
        "Lymphatic-Associated T Cells": "T_cell",
        "Monocyte-Derived Macrophages": "Myeloid",
        "Tumor-Associated Macrophages (M2)": "Myeloid",
        "Plasmacytoid Dendritic Cells (pDCs)": "Myeloid",
        "Tumor-Associated Neutrophils": "Myeloid",
        "Follicular B Cells and Interacting Tfh Cells": "B_cell",
        "Endothelial-Associated Plasma Cells": "Plasma_cell",
        "Hypoxic Tumor": "Epithelial",
        "Differentiated Proximal-Like Tumor": "Epithelial",
        "Growth Signaling-Enriched Tumor": "Epithelial",
        "Stressed/Dedifferentiating Proximal Tubule": "Epithelial",
    }
    lab = g.notna().to_numpy()
    tenx_lin = g[lab].map(from_lin).to_numpy()
    lineage_agreement = float((pred_lin[lab] == tenx_lin).mean())
    ca9 = np.asarray(ad[:, "CA9"].X.todense()).ravel() if "CA9" in ad.var_names else None
    tum = (first == "Tumour_ccRCC") & called
    ca9_pos_frac_tumour = (
        float((ca9[tum] > 0).mean()) if ca9 is not None and tum.any() else float("nan")
    )
    tenx_tum = g.isin(
        ["Hypoxic Tumor", "Differentiated Proximal-Like Tumor", "Growth Signaling-Enriched Tumor"]
    ).to_numpy()
    rec = {
        "tag": tag,
        "split": a.split,
        "git_sha": git_sha(),
        "pipeline_sha256": hashlib.sha256(Path(a.pipeline).read_bytes()).hexdigest()[:12],
        "metric": metric,
        "heldout_lineage_f1": f1,
        "tenx_fine_agreement": fine,
        "per_class_f1": per_class,
        "per_group_agreement": per_group,
        "heldout_lineage_f1_called_only": f1_called,
        "n_landmarks": int(ev.sum()),
        "n_eval_cells_called": int(ev_called.sum()),
        "reject_frac": float((~called).mean()),
        "tenx_lineage_agreement": lineage_agreement,
        "tumour_groups_called_Tumour_ccRCC": float((first[tenx_tum] == "Tumour_ccRCC").mean()),
        "ca9_pos_frac_of_tumour_calls": ca9_pos_frac_tumour,
        "n_tumour_calls": int(tum.sum()),
        "calls": pd.Series(first).value_counts().to_dict(),
        "shuffle_protein_seed": a.shuffle_protein_seed,
        "config_repr": str(res.get("config", "")),
        "elapsed_s": round(time.time() - t0, 1),
    }
    # ---- spatial-anno-metrics (Paul's package; REPORTED, never part of the loop metric) ----
    sam = {}
    try:
        from spatial_anno_metrics.eval_metrics import (
            external_scores,
            hierarchical_accuracy,
            internal_validity,
        )

        ext = external_scores(pred_lin[lab], tenx_lin)
        sam["lineage_vs_tenx"] = {k: v for k, v in ext.items() if k != "per_class_f1"}
        primary = {  # one representative reference type per 10x group, for subtype accuracy
            "Regulatory T Cells (Tregs)": "Treg_cell",
            "Exhausted Cytotoxic T Cells": "CD8_T_cell",
            "Vascular-Localized Cytotoxic T Cells": "CD8_T_cell",
            "Cycling Exhausted T Cells": "CD8_T_cell",
            "Proliferating T Cells": "Cycling_T_NK_cell",
            "Helper T Cells": "CD4_T_cell",
            "Memory T Cells": "CD4_T_cell",
            "Lymphatic-Associated T Cells": "CD4_T_cell",
            "Follicular B Cells and Interacting Tfh Cells": "B_cell",
            "Endothelial-Associated Plasma Cells": "Plasma_cell",
            "Monocyte-Derived Macrophages": "Macrophage",
            "Tumor-Associated Macrophages (M2)": "Macrophage",
            "Plasmacytoid Dendritic Cells (pDCs)": "cDC2",
            "Hypoxic Tumor": "Tumour_ccRCC",
            "Differentiated Proximal-Like Tumor": "Tumour_ccRCC",
            "Growth Signaling-Enriched Tumor": "Tumour_ccRCC",
            "Stressed/Dedifferentiating Proximal Tubule": "Renal_epithelial_cell_proximal_tubule",
        }
        m = g.isin(list(primary)).to_numpy()
        pred_fine = np.where(called, first, "__reject__")
        sam["hierarchical_vs_tenx_primary"] = hierarchical_accuracy(
            pred_fine[m], g[m].map(primary).to_numpy(), LINEAGE
        )
        # reference-free coherence of the fine labels in expression space (log-normalised counts)
        sub = ad_in[called].copy()
        X = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
        X = np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 100.0)
        coh = anndata.AnnData(
            X=X.astype(np.float32), obs=pd.DataFrame({"ft": first[called]}, index=sub.obs_names)
        )
        sam["internal_validity_fine"] = internal_validity(
            coh, label_key="ft", embedding=None, subsample=3000
        )
    except Exception as e:  # reporting must never sink an iteration
        sam["error"] = repr(e)
    rec["spatial_anno_metrics"] = sam

    (out_dir / f"eval_{tag}.json").write_text(json.dumps(rec, indent=2, default=float))
    print(
        json.dumps(
            {
                k: rec[k]
                for k in [
                    "heldout_lineage_f1",
                    "tenx_fine_agreement",
                    "reject_frac",
                    "tenx_lineage_agreement",
                    "ca9_pos_frac_of_tumour_calls",
                    "elapsed_s",
                ]
            }
        ),
        file=sys.stderr,
    )
    print(f"{metric:.6f}")


if __name__ == "__main__":
    main()
