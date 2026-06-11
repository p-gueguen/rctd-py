"""Real-data smoke test on p39836 T-VEC Xenium region p1_Pre.

Validates the biological signature of the joint model: among cells called NK,
the joint RNA+protein deconvolution should LOWER mean CD3E protein (less T-cell
contamination) and RAISE mean CD16 protein (cleaner NK identity) versus RNA-only.

Skipped unless the exported prototype data and the converted reference exist.
Slow (41k cells, two doublet passes); marked accordingly and run opt-in.
"""

from pathlib import Path

import numpy as np
import pytest

PROTO = Path("/srv/GT/analysis/p39836/Analyses_Paul/protein_prior_prototype")
SPATIAL = PROTO / "p1pre_spatial.h5ad"
PROTEIN = PROTO / "p1pre_protein_clr.csv"
REFERENCE = PROTO / "skin_ref_noTCM.h5ad"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.protein,
    pytest.mark.skipif(
        not (SPATIAL.exists() and PROTEIN.exists() and REFERENCE.exists()),
        reason="p1_Pre prototype data / converted reference not present",
    ),
]


def test_nk_contamination_drops_with_protein():
    import anndata
    import pandas as pd

    from rctd import RCTDConfig, Reference, run_rctd

    spatial = anndata.read_h5ad(SPATIAL)
    prot = pd.read_csv(PROTEIN, index_col=0)
    spatial.obsm["protein"] = prot.reindex(spatial.obs_names)
    spatial.uns["protein_feature_names"] = list(prot.columns)
    ref_adata = anndata.read_h5ad(REFERENCE)

    # eigh_threshold=0 avoids the Blackwell cuSOLVER crash at K=28; batch_size pinned
    # because the GPU is often shared. lambda=1.0 is where the NK signal is clear
    # (the auto per-feature value ~0.4 is gentler; see smoke_nk_summary.csv).
    cfg_kw = dict(compile=False, UMI_min=20, device="auto", eigh_threshold=0)
    rna = run_rctd(
        spatial,
        Reference(ref_adata, cell_type_col="cell_type"),
        mode="doublet",
        config=RCTDConfig(**cfg_kw),
        batch_size=4000,
    )
    joint = run_rctd(
        spatial,
        Reference(ref_adata, cell_type_col="cell_type"),
        mode="doublet",
        config=RCTDConfig(**cfg_kw, protein_weight=1.0, protein_norm="arcsinh_robust"),
        batch_size=4000,
    )

    names = rna.cell_type_names
    nk = names.index("NK")
    bc = np.asarray(spatial.obs_names)

    def nk_means(res):
        full = np.zeros(spatial.n_obs, bool)
        full[res.pixel_mask] = res.first_type == nk
        ids = bc[full]
        return prot.loc[ids, "CD3E"].mean(), prot.loc[ids, "CD16"].mean(), int(full.sum())

    cd3e_rna, cd16_rna, n_rna = nk_means(rna)
    cd3e_joint, cd16_joint, n_joint = nk_means(joint)
    print(
        f"\nNK calls: rna={n_rna} joint={n_joint}\n"
        f"CD3E protein in NK: rna={cd3e_rna:.3f} joint={cd3e_joint:.3f}\n"
        f"CD16 protein in NK: rna={cd16_rna:.3f} joint={cd16_joint:.3f}"
    )
    assert cd3e_joint < cd3e_rna, "CD3E contamination did not drop in NK calls"
    assert cd16_joint >= cd16_rna - 1e-6, "CD16 did not hold/rise in NK calls"
