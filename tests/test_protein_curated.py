"""Curated / scGate-derived signed protein profiles (the planned fix for the bootstrap
circularity on RNA-ambiguous types like NK vs cytotoxic T)."""

import numpy as np
import pandas as pd
import pytest

from rctd import RCTDConfig, Reference
from rctd._protein import build_signed_profile, cognate_profile, scgate_signatures
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


@pytest.mark.protein
def test_build_signed_profile_matches_marker_aliases():
    """Panel names are not gene symbols: 'CD3e' / 'cd3-e' must hit a signature
    written as 'CD3E'. Exact string matching silently ignored such markers."""
    P, _ = build_signed_profile(
        ["T"], ["CD3e", "pan-CK"], {"T": {"positive": ["CD3E"], "negative": ["panCK"]}}
    )
    assert P[0, 0] == 1.5 and P[1, 0] == -1.5


@pytest.mark.protein
def test_cognate_profile_from_reference(synthetic_data):
    """VirTues reads a channel by the protein it measures, not by its column name.
    The cognate prior does the same in miniature: marker mA is protein of Gene_0,
    so its profile is Gene_0's z-scored expression across the reference types -
    positive on the type that expresses it, negative elsewhere, with no bootstrap
    pass and no hand-written negatives."""
    ref = Reference(synthetic_data["reference"], cell_min=10, min_UMI=10)
    feats = ["mA", "mB", "mC"]
    P, has_gene = cognate_profile(ref, feats, marker_genes={"mA": ["Gene_0"]})
    k0 = ref.cell_type_names.index("Type_0")
    assert P.shape == (3, len(ref.cell_type_names))
    assert has_gene.tolist() == [True, False, False]
    others = [k for k in range(P.shape[1]) if k != k0]
    assert P[0, k0] > 1.0 and P[0, k0] == P[0].max()
    assert P[0, others].mean() < 0  # the fixture's non-marker expression is exponential noise
    assert np.allclose(P[1:], 0.0)

    spatial = _synth_with_protein(synthetic_data)
    rctd = RCTD(
        spatial,
        ref,
        RCTDConfig(
            compile=False,
            protein_weight=1.0,
            protein_profile_source="cognate",
            protein_marker_genes={"mA": ["Gene_0"]},
        ),
    )
    rctd.fit_platform_effects()
    rctd.prepare_protein()
    assert np.allclose(rctd.reference.protein_profiles, P)


def _toy_citeseq(seed=0):
    """T cells: CD3 high, CD45 high. B cells: CD3 at isotype level, CD45 high."""
    rng = np.random.default_rng(seed)
    n = 200
    labels = np.array(["T1"] * n + ["B1"] * n + ["Junk"] * n)
    iso = rng.poisson(5, 3 * n)
    cd3 = np.r_[rng.poisson(200, n), rng.poisson(5, n), rng.poisson(50, n)]
    cd45 = np.r_[rng.poisson(300, n), rng.poisson(280, n), rng.poisson(5, n)]
    adt = np.column_stack([cd3, cd45, iso]).astype(float)
    return adt, labels, ["CD3-1", "CD45-1", "IgG-iso"]


@pytest.mark.protein
def test_reference_protein_levels_use_isotype_floor():
    from rctd._protein import reference_protein_levels

    adt, labels, names = _toy_citeseq()
    lv = reference_protein_levels(
        adt,
        labels,
        names,
        isotype_names=["IgG-iso"],
        type_map={"T1": "T", "B1": "B"},
        marker_to_adt={"CD3E": "CD3-1", "CD45": "CD45-1", "PanCK": "not-measured"},
    )
    assert list(lv.columns) == ["T", "B"] or set(lv.columns) == {"T", "B"}
    assert lv.loc["CD3E", "T"] == pytest.approx(1.0)
    assert lv.loc["CD3E", "B"] < 0.15
    # pan-immune marker stays high for BOTH types (a min-max scale would push B to 0)
    assert lv.loc["CD45", "B"] > 0.8 and lv.loc["CD45", "T"] > 0.8
    assert "PanCK" not in lv.index  # unmeasured markers are absent, not zero


@pytest.mark.protein
def test_apply_reference_levels_overrides_only_covered_cells():
    import pandas as pd

    from rctd._protein import apply_reference_levels

    rng = np.random.default_rng(1)
    P_std = rng.normal(0, 1, (500, 3))
    lv = pd.DataFrame({"T": [1.0, 1.0], "B": [0.0, np.nan]}, index=["cd3e", "CD45"])
    base = np.full((3, 3), 7.0)
    P, mask = apply_reference_levels(base, lv, P_std, ["CD3E", "CD45", "m3"], ["T", "B", "Other"])
    hi, lo = np.percentile(P_std[:, 0], 90), np.percentile(P_std[:, 0], 10)
    assert P[0, 0] == pytest.approx(hi) and P[0, 1] == pytest.approx(lo)
    assert P[1, 1] == 7.0  # NaN level -> base kept
    assert (P[2] == 7.0).all() and (P[:, 2] == 7.0).all()
    assert mask.tolist() == [True, True, False]
    assert (base == 7.0).all()  # input not mutated


@pytest.mark.protein
def test_prepare_protein_applies_reference_levels(synthetic_data):
    import pandas as pd

    spatial = _synth_with_protein(synthetic_data)
    ref = Reference(synthetic_data["reference"], cell_min=10, min_UMI=10)
    lv = pd.DataFrame({"Type_0": [1.0, 0.0]}, index=["mA", "mB"])
    rctd = RCTD(
        spatial, ref, RCTDConfig(compile=False, protein_weight=1.0, protein_reference_levels=lv)
    )
    rctd.fit_platform_effects()
    kw = rctd.prepare_protein()
    feats = rctd.reference.protein_feature_names
    k0 = rctd.reference.cell_type_names.index("Type_0")
    Y = kw["protein_intensity"]
    iA = feats.index("mA")
    assert rctd.reference.protein_profiles[iA, k0] == pytest.approx(
        np.percentile(Y[:, iA], 90), rel=1e-5
    )


@pytest.mark.protein
def test_reference_protein_levels_needs_an_exact_antibody_name():
    """'CD31' and 'CD3-1' share a punctuation-free key. Handing CD31 the CD3 antibody
    gave the endothelial marker a T-cell profile (PBMC panel, 2026-09-17)."""
    from rctd._protein import reference_protein_levels

    adt, labels, names = _toy_citeseq()
    names = ["CD3-1", "CD31", "IgG-iso"]  # column 1 is now the CD45-like pan marker
    lv = reference_protein_levels(
        adt,
        labels,
        names,
        isotype_names=["IgG-iso"],
        type_map={"T1": "T", "B1": "B"},
        marker_to_adt={"CD3E": "CD3-1", "CD31": "CD31"},
    )
    assert lv.loc["CD3E", "B"] < 0.15  # the real CD3 antibody
    assert lv.loc["CD31", "B"] > 0.8  # the pan marker, NOT a copy of CD3
    with pytest.warns(UserWarning, match="matches several antibodies"):
        reference_protein_levels(
            adt,
            labels,
            ["CD3-1", "CD3.1", "IgG-iso"],
            isotype_names=["IgG-iso"],
            type_map={"T1": "T"},
            marker_to_adt={"CD3E": "cd31"},
        )


@pytest.mark.protein
def test_reference_protein_levels_drop_background_antibodies():
    """An antibody that never leaves isotype background carries noise, not levels."""
    from rctd._protein import reference_protein_levels

    adt, labels, names = _toy_citeseq()
    adt[:, 0] = np.random.default_rng(0).poisson(6, adt.shape[0])  # CD3 at isotype level
    lv = reference_protein_levels(
        adt,
        labels,
        names,
        isotype_names=["IgG-iso"],
        type_map={"T1": "T", "B1": "B"},
        marker_to_adt={"CD3E": "CD3-1", "CD45": "CD45-1"},
    )
    assert "CD3E" not in lv.index and "CD45" in lv.index
