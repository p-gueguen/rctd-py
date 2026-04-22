"""Tests for class_df hierarchical fallback in doublet mode (GitHub #14)."""

import numpy as np
import pytest

from rctd._doublet import run_doublet_mode
from rctd._likelihood import compute_spline_coefficients, load_cached_q_matrices
from rctd._types import (
    SPOT_CLASS_SINGLET,
    DoubletResult,
    RCTDConfig,
)

# --- API-surface tests (tiny, verify fields exist) ---


def test_rctdconfig_accepts_class_df():
    """RCTDConfig must accept a class_df keyword argument."""
    cfg = RCTDConfig(class_df={"A": "X", "B": "X"})
    assert cfg.class_df == {"A": "X", "B": "X"}


def test_rctdconfig_class_df_defaults_to_none():
    """Default class_df is None (no hierarchical fallback)."""
    cfg = RCTDConfig()
    assert cfg.class_df is None


def test_doubletresult_accepts_class_name_fields():
    """DoubletResult must accept first_class_name and second_class_name kwargs."""
    N = 3
    res = DoubletResult(
        weights=np.zeros((N, 2)),
        weights_doublet=np.zeros((N, 2)),
        spot_class=np.zeros(N, dtype=np.int32),
        first_type=np.zeros(N, dtype=np.int32),
        second_type=np.zeros(N, dtype=np.int32),
        first_class=np.zeros(N, dtype=bool),
        second_class=np.zeros(N, dtype=bool),
        min_score=np.zeros(N),
        singlet_score=np.zeros(N),
        cell_type_names=["A", "B"],
        first_class_name=np.array(["X", "X", "Y"]),
        second_class_name=np.array(["Y", "X", "Y"]),
    )
    assert res.first_class_name.shape == (N,)
    assert res.second_class_name.shape == (N,)


def test_doubletresult_class_name_defaults_to_none():
    """Backward-compat: existing unpacking works without class_name kwargs."""
    N = 2
    res = DoubletResult(
        weights=np.zeros((N, 2)),
        weights_doublet=np.zeros((N, 2)),
        spot_class=np.zeros(N, dtype=np.int32),
        first_type=np.zeros(N, dtype=np.int32),
        second_type=np.zeros(N, dtype=np.int32),
        first_class=np.zeros(N, dtype=bool),
        second_class=np.zeros(N, dtype=bool),
        min_score=np.zeros(N),
        singlet_score=np.zeros(N),
        cell_type_names=["A", "B"],
    )
    assert res.first_class_name is None
    assert res.second_class_name is None


# --- Behavioral tests ---


def _run(synthetic_data, class_df=None, seed_config=None):
    """Shared helper to run doublet mode on synthetic_data."""
    cache = load_cached_q_matrices()
    x_vals = cache.pop("X_vals")
    q_mat = cache["Q_100"]
    sq_mat = compute_spline_coefficients(q_mat, x_vals)

    profiles = synthetic_data["profiles"]
    spatial_counts = synthetic_data["spatial"].X
    spatial_numi = np.array(spatial_counts.sum(axis=1)).flatten()
    cell_type_names = synthetic_data["cell_type_names"]

    config = seed_config or RCTDConfig(class_df=class_df)
    return run_doublet_mode(
        spatial_counts=spatial_counts,
        spatial_numi=spatial_numi,
        norm_profiles=profiles,
        cell_type_names=cell_type_names,
        q_mat=q_mat,
        sq_mat=sq_mat,
        x_vals=x_vals,
        config=config,
        batch_size=10,
    )


def test_class_df_populates_first_class_name(synthetic_data):
    """When class_df is provided, first_class_name is populated with correct class strings."""
    cell_type_names = synthetic_data["cell_type_names"]
    # Split 5 types into two classes: Type_0, Type_1 -> ClassA; Type_2..4 -> ClassB
    class_df = {ct: ("ClassA" if i < 2 else "ClassB") for i, ct in enumerate(cell_type_names)}

    res = _run(synthetic_data, class_df=class_df)

    N = res.first_type.shape[0]
    assert res.first_class_name is not None
    assert res.second_class_name is not None
    assert res.first_class_name.shape == (N,)
    assert res.second_class_name.shape == (N,)

    # Every entry must equal the class_df lookup for the corresponding type
    for n in range(N):
        ft = cell_type_names[res.first_type[n]]
        st = cell_type_names[res.second_type[n]]
        assert res.first_class_name[n] == class_df[ft]
        assert res.second_class_name[n] == class_df[st]


def test_class_df_none_does_not_populate_class_names(synthetic_data):
    """When class_df is None, first_class_name/second_class_name remain None."""
    res = _run(synthetic_data, class_df=None)
    assert res.first_class_name is None
    assert res.second_class_name is None


def test_class_df_none_matches_identity_mapping(synthetic_data):
    """Regression guard: class_df=None and identity mapping produce same classification.

    With an identity class_df ({ct: ct for ct in names}), the class-level logic reduces
    to the type-level logic, so spot_class / first_type / second_type / first_class /
    second_class must match bit-identically to class_df=None.
    """
    cell_type_names = synthetic_data["cell_type_names"]
    identity = {ct: ct for ct in cell_type_names}

    res_none = _run(synthetic_data, class_df=None)
    res_id = _run(synthetic_data, class_df=identity)

    np.testing.assert_array_equal(res_none.spot_class, res_id.spot_class)
    np.testing.assert_array_equal(res_none.first_type, res_id.first_type)
    np.testing.assert_array_equal(res_none.second_type, res_id.second_type)
    np.testing.assert_array_equal(res_none.first_class, res_id.first_class)
    np.testing.assert_array_equal(res_none.second_class, res_id.second_class)
    np.testing.assert_allclose(res_none.weights_doublet, res_id.weights_doublet, atol=1e-6)


def test_class_df_validates_missing_types(synthetic_data):
    """class_df missing a cell type from the reference raises ValueError."""
    cell_type_names = synthetic_data["cell_type_names"]
    bad_class_df = {ct: "ClassA" for ct in cell_type_names[:-1]}  # last type missing

    with pytest.raises(ValueError, match="class_df"):
        _run(synthetic_data, class_df=bad_class_df)


def test_class_df_affects_first_class_flag(synthetic_data):
    """class_df must actually change first_class/second_class flag distribution.

    With identity class_df (each type = own class), first_class == False always (no class
    ambiguity possible). With all types collapsed into one class, the flag can flip True
    for pixels where competitive pairs involve different types within the class.
    """
    cell_type_names = synthetic_data["cell_type_names"]
    identity = {ct: ct for ct in cell_type_names}
    one_class = {ct: "OneClass" for ct in cell_type_names}

    res_identity = _run(synthetic_data, class_df=identity)
    res_one_class = _run(synthetic_data, class_df=one_class)

    # Identity: flags must be all False (no within-class ambiguity possible)
    assert not res_identity.first_class.any(), "Identity class_df must never flag first_class=True"
    assert not res_identity.second_class.any(), (
        "Identity class_df must never flag second_class=True"
    )

    # Single-class: flags must flip True for at least some pixels (otherwise class_df logic
    # isn't firing). This is a behavioral smoke test that class_df changes decision flow.
    any_flagged = res_one_class.first_class.any() or res_one_class.second_class.any()
    assert any_flagged, (
        "Single-class class_df must flip at least one first/second_class flag to True"
    )


def test_class_df_cross_class_pixels_keep_certain_classification(synthetic_data):
    """Pixels that are clear cross-class doublets should remain doublet_certain without class flag.

    A singlet pixel (one type dominant) under a two-class scheme should NOT have class flags
    set -- the classification is confident at type level.
    """
    cell_type_names = synthetic_data["cell_type_names"]
    two_class = {ct: ("ClassA" if i < 2 else "ClassB") for i, ct in enumerate(cell_type_names)}

    res = _run(synthetic_data, class_df=two_class)

    # For clear singlets, class flags should be False (no ambiguity)
    singlet_mask = res.spot_class == SPOT_CLASS_SINGLET
    if singlet_mask.any():
        assert not res.first_class[singlet_mask].any(), "Singlets should not have first_class=True"
