"""Curated / scGate-derived signed protein profiles (the planned fix for the bootstrap
circularity on RNA-ambiguous types like NK vs cytotoxic T)."""

import numpy as np
import pandas as pd
import pytest

from rctd import RCTDConfig, Reference
from rctd._protein import build_signed_profile, scgate_signatures
from rctd._rctd import RCTD


@pytest.mark.protein
def test_build_signed_profile_signs_and_mask():
    names = ["A", "B", "C"]
    feats = ["m0", "m1", "m2"]
    sig = {"A": {"positive": ["m0"], "negative": ["m1"]}, "C": {"negative": ["m2"]}}
    P, mask = build_signed_profile(names, feats, sig, magnitude=1.5)
    assert P.shape == (3, 3)
    assert mask.tolist() == [True, False, True]  # B has no signature
    assert P[0, 0] == 1.5 and P[1, 0] == -1.5 and P[2, 0] == 0.0  # A: m0+, m1-
    assert np.allclose(P[:, 1], 0.0)  # B neutral
    assert P[2, 2] == -1.5  # C: m2-


@pytest.mark.protein
def test_scgate_signatures_parses_negative_markers(tmp_path):
    # tiny scGate-format fixture: master_table (signature -> genes, '-' = negative) + an NK model
    mt = tmp_path / "master_table.tsv"
    mt.write_text(
        "name\tsignature\n"
        "Immune\tPTPRC\n"
        "NK\tFCGR3A;NKG7;CD3D-;CD3E-;CD8A-\n"
        "Tcell\tCD3D;CD3E;CD3G\n"
        "Epithelial\tKRT8;KRT18\n"
    )
    nk = tmp_path / "NK.tsv"
    nk.write_text(
        "levels\tuse_as\tname\tsignature\n"
        "level1\tpositive\tImmune\t\n"
        "level1\tpositive\tNK\t\n"
        "level1\tnegative\tTcell\t\n"
        "level1\tnegative\tEpithelial\t\n"
    )
    g2p = {
        "PTPRC": "CD45",
        "FCGR3A": "CD16",
        "NKG7": "GranzymeB",
        "CD3D": "CD3E",
        "CD3E": "CD3E",
        "CD3G": "CD3E",
        "CD8A": "CD8A",
        "KRT8": "PanCK",
        "KRT18": "PanCK",
    }
    sig = scgate_signatures(str(mt), {"NK": str(nk)}, g2p)
    nkp = sig["NK"]
    assert "CD16" in nkp["positive"] and "CD45" in nkp["positive"]
    # CD3E must be NEGATIVE: NK's own CD3D/E- AND the negative Tcell gate both push it down
    assert "CD3E" in nkp["negative"]
    assert "CD8A" in nkp["negative"]
    assert "PanCK" in nkp["negative"]  # negative Epithelial gate
    assert "CD3E" not in nkp["positive"]


def _synth_with_protein(synthetic_data, markers=("mA", "mB", "mC")):
    spatial = synthetic_data["spatial"].copy()
    rng = np.random.default_rng(0)
    spatial.obsm["protein"] = pd.DataFrame(
        rng.standard_normal((spatial.n_obs, len(markers))),
        index=spatial.obs_names,
        columns=list(markers),
    )
    return spatial


@pytest.mark.protein
def test_curated_source_requires_signatures(synthetic_data):
    spatial = _synth_with_protein(synthetic_data)
    ref = Reference(synthetic_data["reference"], cell_min=10, min_UMI=10)
    rctd = RCTD(
        spatial,
        ref,
        RCTDConfig(compile=False, protein_weight=1.0, protein_profile_source="curated"),
    )
    rctd.fit_platform_effects()
    with pytest.raises(ValueError, match="requires config.protein_signatures"):
        rctd.prepare_protein()


@pytest.mark.protein
def test_signature_override_applied_hybrid(synthetic_data):
    """bootstrap source + protein_signatures => listed type's column is the curated gate,
    other types keep their bootstrapped profile."""
    spatial = _synth_with_protein(synthetic_data)
    ref = Reference(synthetic_data["reference"], cell_min=10, min_UMI=10)
    sigs = {"Type_0": {"positive": ["mA"], "negative": ["mB"]}}
    rctd = RCTD(
        spatial,
        ref,
        RCTDConfig(
            compile=False,
            protein_weight=1.0,
            protein_signatures=sigs,
            protein_signature_magnitude=1.5,
        ),
    )
    rctd.fit_platform_effects()
    rctd.prepare_protein()
    P = rctd.reference.protein_profiles
    feats = rctd.reference.protein_feature_names
    k0 = rctd.reference.cell_type_names.index("Type_0")
    assert P[feats.index("mA"), k0] == 1.5
    assert P[feats.index("mB"), k0] == -1.5
    assert P[feats.index("mC"), k0] == 0.0
