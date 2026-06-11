"""Unit tests for protein-profile bootstrap (_protein.bootstrap_protein_profiles)."""

import numpy as np
import pytest

from rctd._protein import bootstrap_protein_profiles


@pytest.mark.protein
def test_profiles_recover_per_type_means():
    rng = np.random.default_rng(0)
    n_types, M = 3, 4
    # Distinct per-type protein means.
    true_mu = np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]])  # (K, M)
    rows, idx = [], []
    for k in range(n_types):
        block = true_mu[k][None, :] + rng.standard_normal((200, M)) * 0.3
        rows.append(block)
        idx += [k] * 200
    protein_std = np.vstack(rows)
    singlet_idx = np.array(idx)

    P_prot, tau, n_used = bootstrap_protein_profiles(protein_std, singlet_idx, n_types)
    assert P_prot.shape == (M, n_types)
    assert (n_used == 200).all()
    # Columns of P_prot recover the per-type means.
    np.testing.assert_allclose(P_prot.T, true_mu, atol=0.1)
    # Pooled residual tau is positive and ~0.3 (the injected noise).
    assert np.all(tau > 0) and np.isfinite(tau).all()
    np.testing.assert_allclose(tau, 0.3, atol=0.1)


@pytest.mark.protein
def test_undersupported_type_gets_zero_profile():
    rng = np.random.default_rng(1)
    n_types, M = 3, 4
    protein_std = rng.standard_normal((250, M))
    # Type 2 has only 5 confident singlets (< min_cells=25) -> zero profile.
    idx = np.array([0] * 120 + [1] * 125 + [2] * 5)
    P_prot, tau, n_used = bootstrap_protein_profiles(protein_std, idx, n_types, min_cells=25)
    assert n_used.tolist() == [120, 125, 5]
    assert np.allclose(P_prot[:, 2], 0.0)
    assert not np.allclose(P_prot[:, 0], 0.0)


@pytest.mark.protein
def test_non_confident_cells_excluded():
    rng = np.random.default_rng(2)
    n_types, M = 2, 3
    protein_std = rng.standard_normal((100, M))
    # Half the cells are -1 (not confident singlets) and must be ignored.
    idx = np.array([0] * 30 + [1] * 30 + [-1] * 40)
    P_prot, tau, n_used = bootstrap_protein_profiles(protein_std, idx, n_types)
    assert n_used.tolist() == [30, 30]
    np.testing.assert_allclose(P_prot[:, 0], protein_std[:30].mean(axis=0))
