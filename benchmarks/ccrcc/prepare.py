#!/usr/bin/env python3
"""Build the frozen tune/test splits for the ccRCC RNA+protein autoresearch loop. Run ONCE.

READ-ONLY for the loop. Everything the metric depends on is decided here:
  - spatial split (west half = tune, east half = test) so bootstrap profiles, platform
    effects and sigma fitted on tune cells never see test cells;
  - the 9 gate markers go to obsm["protein_truth"], the other 18 to obsm["protein_fit"];
    the pipeline only ever receives protein_fit;
  - landmark classes gated on protein_truth (frozen LINEAGE_SIGS from eval.py);
  - obs["eval_cell"] = landmark cells the RNA-only baseline (hybrid reference, lambda 0) called;
    eval.py scores ALL landmarks (rejects = misses) and reports the called-only F1 alongside.
Needs the GPU (one RNA-only doublet fit per split).
"""

import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import (  # noqa: E402
    DATA,
    GATE_MARKERS,
    LINEAGE,
    LINEAGE_SIGS,
    MIN_LM,
    REF_DISCO_SRC,
    REF_HYBRID_SRC,
)

from rctd import RCTDConfig, Reference, run_rctd  # noqa: E402
from rctd._protein import gate_landmarks, normalize_protein  # noqa: E402
from rctd._types import SPOT_CLASS_REJECT  # noqa: E402

SRC = "/srv/GT/analysis/pgueguen/rctd-py/ccrcc_protein/spatial_ccrcc_full.h5ad"
N_TUNE, N_TEST = 80_000, 100_000
OBS_KEEP = ["tenx_group", "transcript_counts", "cell_area", "segmentation_method"]

DATA.mkdir(parents=True, exist_ok=True)
for src, dst in [
    (REF_HYBRID_SRC, DATA / "reference_hybrid_ccrcc.h5ad"),
    (REF_DISCO_SRC, DATA / "kidney_ref.h5ad"),
]:
    if not dst.exists():
        anndata.read_h5ad(src).write_h5ad(dst)

full = anndata.read_h5ad(SRC)
prot = full.obsm["protein"]
assert set(GATE_MARKERS) <= set(prot.columns), "gate marker missing from the 27-plex"
fit_markers = [m for m in prot.columns if m not in GATE_MARKERS]
assert len(fit_markers) == 18 and len(GATE_MARKERS) == 9
x = full.obs["x_centroid"].to_numpy()
west = x < np.median(x)
rng = np.random.default_rng(0)
idx_tune = np.sort(rng.choice(np.where(west)[0], N_TUNE, replace=False))
rng = np.random.default_rng(1)
idx_test = np.sort(rng.choice(np.where(~west)[0], N_TEST, replace=False))
ref = Reference(anndata.read_h5ad(DATA / "reference_hybrid_ccrcc.h5ad"), cell_type_col="cell_type")
classes = sorted(set(LINEAGE.values()))

for name, idx in [("tune", idx_tune), ("test", idx_test)]:
    sub = full[idx].copy()
    ad = anndata.AnnData(X=sub.X.copy(), obs=sub.obs[OBS_KEEP].copy())
    ad.var_names = sub.var_names
    ad.obsm["spatial"] = np.asarray(sub.obsm["spatial"])
    ad.obsm["protein_fit"] = (
        prot.iloc[idx][fit_markers].reset_index(drop=True).set_index(ad.obs_names)
    )
    ad.obsm["protein_truth"] = (
        prot.iloc[idx][GATE_MARKERS].reset_index(drop=True).set_index(ad.obs_names)
    )
    ad.uns["fit_markers"], ad.uns["truth_markers"], ad.uns["split"] = (
        fit_markers,
        GATE_MARKERS,
        name,
    )

    # landmarks from the truth markers alone (standardised within this split)
    P_std, _, _ = normalize_protein(ad.obsm["protein_truth"].to_numpy(), method="arcsinh_robust")
    labels, info = gate_landmarks(P_std, GATE_MARKERS, LINEAGE_SIGS, classes, min_cells=MIN_LM)
    ad.obs["landmark_class"] = np.where(labels >= 0, np.array(classes)[np.maximum(labels, 0)], "")

    # frozen evaluation set: landmark cells the RNA-only baseline was willing to call
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
    base_first = np.array(["filtered"] * ad.n_obs, dtype=object)
    base_first[res.pixel_mask] = np.array(res.cell_type_names)[res.first_type]
    ad.obs["baseline_rna_first_type"] = pd.Categorical(
        base_first
    )  # for the report only; eval.py never reads it
    ad.write_h5ad(DATA / f"{name}.h5ad")
    per = {
        c: int(((labels >= 0) & (np.array(classes)[np.maximum(labels, 0)] == c)).sum())
        for c in classes
    }
    print(
        f"[{name}] {ad.n_obs} cells, x range {sub.obs['x_centroid'].min():.0f}-{sub.obs['x_centroid'].max():.0f}, "
        f"10x-labelled {int(ad.obs['tenx_group'].notna().sum())}, landmarks {int((labels >= 0).sum())} {per}, "
        f"eval cells {int(ad.obs['eval_cell'].sum())}, baseline reject {float((~called).mean()):.3f}",
        flush=True,
    )
print("DONE")
