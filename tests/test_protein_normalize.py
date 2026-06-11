"""Unit tests for protein normalization (_protein.normalize_protein)."""

import numpy as np
import pytest

from rctd._protein import normalize_protein


@pytest.mark.protein
def test_arcsinh_robust_is_per_marker_centered():
    rng = np.random.default_rng(0)
    # Three markers on very different intensity scales.
    P = np.column_stack(
        [
            rng.lognormal(3, 1, size=500),
            rng.lognormal(7, 1, size=500),
            rng.lognormal(0, 1, size=500),
        ]
    )
    P_std, tau, valid = normalize_protein(P, method="arcsinh_robust")
    assert P_std.shape == P.shape
    assert valid.all()
    # Each marker is independently centered (robust): per-marker median ~ 0.
    assert np.allclose(np.median(P_std, axis=0), 0.0, atol=1e-6)
    # Per-marker spread is comparable across markers despite raw-scale differences.
    spreads = P_std.std(axis=0)
    assert spreads.max() / spreads.min() < 3.0
    assert tau.shape == (3,)


@pytest.mark.protein
def test_constant_marker_does_not_blow_up():
    P = np.ones((100, 4))
    P[:, 1] = np.linspace(1, 50, 100)  # one informative marker
    P_std, tau, valid = normalize_protein(P, method="arcsinh_robust")
    assert np.isfinite(P_std).all()
    # Constant markers collapse to ~0 (no spurious signal), not NaN/Inf.
    assert np.allclose(P_std[:, 0], 0.0)


@pytest.mark.protein
def test_nan_rows_are_zeroed_and_flagged_invalid():
    rng = np.random.default_rng(1)
    P = rng.lognormal(2, 1, size=(50, 5))
    P[10] = np.nan  # a cell with no protein measurement
    P_std, tau, valid = normalize_protein(P)
    assert not valid[10]
    assert valid.sum() == 49
    assert np.allclose(P_std[10], 0.0)
    assert np.isfinite(P_std).all()


@pytest.mark.protein
def test_background_subtraction_hook():
    P = np.full((20, 3), 5.0)
    bg = np.array([5.0, 0.0, 0.0])  # marker 0 is entirely background
    P_std, _, _ = normalize_protein(P, background=bg)
    # After subtracting its background, marker 0 is all-zero -> standardized 0.
    assert np.allclose(P_std[:, 0], 0.0)


@pytest.mark.protein
def test_clr_centers_each_cell():
    rng = np.random.default_rng(2)
    P = rng.lognormal(2, 1, size=(30, 6))
    P_std, _, _ = normalize_protein(P, method="clr")
    # CLR: per-cell (row) values are centered about their own geometric mean.
    assert np.allclose(P_std.mean(axis=1), 0.0, atol=1e-9)


@pytest.mark.protein
def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown protein normalization"):
        normalize_protein(np.ones((4, 3)), method="zscore")
