"""Tests for IRWLS solver."""

import jax.numpy as jnp
import numpy as np
import pytest

from rctd._irwls import solve_irwls, solve_irwls_batch
from rctd._likelihood import build_x_vals, compute_q_matrix, compute_spline_coefficients


@pytest.fixture(scope="module")
def likelihood_tables():
    x_vals = build_x_vals()
    Q_mat = compute_q_matrix(sigma=1.0, x_vals=x_vals, K_val=100)
    SQ_mat = compute_spline_coefficients(Q_mat, x_vals)
    return jnp.array(Q_mat), jnp.array(SQ_mat), jnp.array(x_vals)


class TestSolveIRWLS:
    def test_single_type_recovers_identity(self, likelihood_tables):
        """With 1 cell type, weight should be ~1.0."""
        Q_mat, SQ_mat, x_vals = likelihood_tables
        rng = np.random.default_rng(42)
        G = 50
        profile = rng.exponential(0.01, size=G)
        profile = profile / profile.sum()
        nUMI = 1000.0
        S = jnp.array((profile * nUMI).reshape(-1, 1))
        Y = jnp.array(rng.poisson(profile * nUMI).astype(np.float32))

        weights, converged = solve_irwls(S, Y, nUMI, Q_mat, SQ_mat, x_vals)
        np.testing.assert_allclose(float(weights[0]), 1.0, atol=0.1)
        assert converged

    def test_two_types_recover_mixture(self, likelihood_tables):
        """With 2 cell types, should recover approximate mixture weights."""
        Q_mat, SQ_mat, x_vals = likelihood_tables
        rng = np.random.default_rng(42)
        G = 100
        K = 2
        profiles = rng.exponential(0.01, size=(G, K))
        profiles = profiles / profiles.sum(axis=0, keepdims=True)
        true_w = np.array([0.7, 0.3])
        nUMI = 2000.0
        lam = (profiles @ true_w) * nUMI
        Y = jnp.array(rng.poisson(lam).astype(np.float32))
        S = jnp.array(profiles * nUMI)

        weights, converged = solve_irwls(S, Y, nUMI, Q_mat, SQ_mat, x_vals)
        np.testing.assert_allclose(np.array(weights), true_w, atol=0.15)

    def test_weights_on_simplex(self, likelihood_tables):
        """Constrained weights should be non-negative and sum to 1."""
        Q_mat, SQ_mat, x_vals = likelihood_tables
        rng = np.random.default_rng(42)
        G, K = 80, 5
        profiles = rng.exponential(0.01, size=(G, K))
        profiles = profiles / profiles.sum(axis=0, keepdims=True)
        nUMI = 1500.0
        true_w = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
        lam = (profiles @ true_w) * nUMI
        Y = jnp.array(rng.poisson(lam).astype(np.float32))
        S = jnp.array(profiles * nUMI)

        weights, converged = solve_irwls(S, Y, nUMI, Q_mat, SQ_mat, x_vals)
        assert jnp.all(weights >= -1e-6)
        np.testing.assert_allclose(float(jnp.sum(weights)), 1.0, atol=1e-4)

    def test_unconstrained_nonnegative(self, likelihood_tables):
        """With constrain=False, weights should be non-negative but not necessarily sum to 1."""
        Q_mat, SQ_mat, x_vals = likelihood_tables
        rng = np.random.default_rng(42)
        G, K = 60, 3
        profiles = rng.exponential(0.01, size=(G, K))
        profiles = profiles / profiles.sum(axis=0, keepdims=True)
        nUMI = 1500.0
        true_w = np.array([0.5, 0.3, 0.2])
        lam = (profiles @ true_w) * nUMI
        Y = jnp.array(rng.poisson(lam).astype(np.float32))
        S = jnp.array(profiles * nUMI)

        weights, converged = solve_irwls(S, Y, nUMI, Q_mat, SQ_mat, x_vals, constrain=False)
        assert jnp.all(weights >= -1e-6)


class TestSolveIRWLSBatch:
    def test_batch_matches_single(self, likelihood_tables):
        """Batch results should match single-pixel results."""
        Q_mat, SQ_mat, x_vals = likelihood_tables
        rng = np.random.default_rng(42)
        G, K, N = 60, 3, 10
        profiles = rng.exponential(0.01, size=(G, K))
        profiles = profiles / profiles.sum(axis=0, keepdims=True)

        nUMIs = rng.integers(500, 3000, size=N).astype(np.float32)
        Y_batch = np.zeros((N, G), dtype=np.float32)
        for i in range(N):
            true_w = rng.dirichlet(np.ones(K))
            lam = (profiles @ true_w) * nUMIs[i]
            Y_batch[i] = rng.poisson(lam)

        # Build S_batch: (N, G, K) = profiles[None] * nUMIs[:, None, None]
        S_batch = jnp.array(profiles)[None, :, :] * jnp.array(nUMIs)[:, None, None]
        batch_weights, batch_conv = solve_irwls_batch(
            S_batch,
            jnp.array(Y_batch),
            jnp.array(nUMIs),
            Q_mat,
            SQ_mat,
            x_vals,
        )
        assert batch_weights.shape == (N, K)

        # Compare with single-pixel results
        for i in range(N):
            single_w, single_c = solve_irwls(
                S_batch[i],
                jnp.array(Y_batch[i]),
                nUMIs[i],
                Q_mat,
                SQ_mat,
                x_vals,
            )
            np.testing.assert_allclose(
                np.array(batch_weights[i]),
                np.array(single_w),
                atol=1e-5,
            )

    def test_batch_simplex(self, likelihood_tables):
        """All batch results should sum to 1 when constrained."""
        Q_mat, SQ_mat, x_vals = likelihood_tables
        rng = np.random.default_rng(99)
        G, K, N = 50, 4, 8
        profiles = rng.exponential(0.01, size=(G, K))
        profiles = profiles / profiles.sum(axis=0, keepdims=True)

        nUMIs = rng.integers(500, 3000, size=N).astype(np.float32)
        Y_batch = np.zeros((N, G), dtype=np.float32)
        for i in range(N):
            true_w = rng.dirichlet(np.ones(K))
            lam = (profiles @ true_w) * nUMIs[i]
            Y_batch[i] = rng.poisson(lam)

        S_batch = jnp.array(profiles)[None, :, :] * jnp.array(nUMIs)[:, None, None]
        batch_weights, batch_conv = solve_irwls_batch(
            S_batch,
            jnp.array(Y_batch),
            jnp.array(nUMIs),
            Q_mat,
            SQ_mat,
            x_vals,
        )
        np.testing.assert_allclose(jnp.sum(batch_weights, axis=1), jnp.ones(N), atol=1e-4)
