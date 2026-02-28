"""Tests for Poisson-Lognormal likelihood computation."""

import jax.numpy as jnp
import numpy as np
import pytest

from rctd._likelihood import (
    build_x_vals,
    calc_log_likelihood,
    calc_q_all,
    compute_q_matrix,
    compute_spline_coefficients,
)


class TestBuildXVals:
    def test_length(self):
        x_vals = build_x_vals()
        assert len(x_vals) == 439

    def test_range(self):
        x_vals = build_x_vals()
        np.testing.assert_allclose(x_vals[0], 1e-4, rtol=1e-3)
        np.testing.assert_allclose(x_vals[-1], 1600.0, rtol=1e-2)

    def test_monotonic(self):
        x_vals = build_x_vals()
        assert np.all(np.diff(x_vals) > 0)

    def test_matches_r_xvals(self):
        """Validate against the R spacexr X_vals.rds file."""
        import subprocess

        result = subprocess.run(
            [
                "Rscript",
                "-e",
                'cat(formatC(readRDS("~/git/spacexr/inst/extdata/Qmat/X_vals.rds"), '
                'format="e", digits=15), sep="\\n")',
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip("R/spacexr not available for cross-validation")
        r_x_vals = np.array([float(x) for x in result.stdout.strip().split("\n")])
        x_vals = build_x_vals()
        np.testing.assert_allclose(x_vals, r_x_vals, atol=1e-12)


class TestComputeQMatrix:
    def test_shape(self):
        x_vals = build_x_vals()
        Q = compute_q_matrix(sigma=1.0, x_vals=x_vals, K_val=100)
        assert Q.shape == (103, 439)  # K_val+3 rows, N_X cols

    def test_finite(self):
        x_vals = build_x_vals()
        Q = compute_q_matrix(sigma=1.0, x_vals=x_vals, K_val=100)
        assert np.all(np.isfinite(Q))

    def test_negative_log_probs(self):
        """Q stores log-probabilities, so all values should be <= 0."""
        x_vals = build_x_vals()
        Q = compute_q_matrix(sigma=1.0, x_vals=x_vals, K_val=100)
        assert np.all(Q <= 0)


class TestSplineCoefficients:
    def test_shape(self):
        x_vals = build_x_vals()
        Q = compute_q_matrix(sigma=1.0, x_vals=x_vals, K_val=100)
        SQ = compute_spline_coefficients(Q, x_vals)
        assert SQ.shape == Q.shape

    def test_boundary_zeros(self):
        """Natural spline: second derivatives are zero at boundaries."""
        x_vals = build_x_vals()
        Q = compute_q_matrix(sigma=1.0, x_vals=x_vals, K_val=100)
        SQ = compute_spline_coefficients(Q, x_vals)
        np.testing.assert_array_equal(SQ[:, 0], 0.0)
        np.testing.assert_array_equal(SQ[:, -1], 0.0)


class TestCalcQAll:
    @pytest.fixture(scope="class")
    def likelihood_tables(self):
        x_vals = build_x_vals()
        Q_mat = compute_q_matrix(sigma=1.0, x_vals=x_vals, K_val=100)
        SQ_mat = compute_spline_coefficients(Q_mat, x_vals)
        return (
            jnp.array(Q_mat),
            jnp.array(SQ_mat),
            jnp.array(x_vals),
        )

    def test_derivatives_shape(self, likelihood_tables):
        Q_mat, SQ_mat, x_vals = likelihood_tables
        Y = jnp.array([0, 1, 5, 10, 50])
        lam = jnp.array([0.5, 1.0, 5.0, 10.0, 50.0])
        d0, d1, d2 = calc_q_all(Y, lam, Q_mat, SQ_mat, x_vals)
        assert d0.shape == (5,)
        assert d1.shape == (5,)
        assert d2.shape == (5,)

    def test_d0_is_log_likelihood(self, likelihood_tables):
        """d0 is log P(Y|lambda), so should be higher (less negative)
        when lambda matches Y."""
        Q_mat, SQ_mat, x_vals = likelihood_tables
        # Y=10 is more likely when lambda=10 than lambda=100
        Y = jnp.array([10, 10])
        lam = jnp.array([10.0, 100.0])
        d0, _, _ = calc_q_all(Y, lam, Q_mat, SQ_mat, x_vals)
        # d0[0] should be greater (less negative) than d0[1]
        assert float(d0[0]) > float(d0[1])

    def test_finite_output(self, likelihood_tables):
        Q_mat, SQ_mat, x_vals = likelihood_tables
        Y = jnp.array([0, 1, 50, 100])
        lam = jnp.array([0.001, 1.0, 50.0, 100.0])
        d0, d1, d2 = calc_q_all(Y, lam, Q_mat, SQ_mat, x_vals)
        assert jnp.all(jnp.isfinite(d0))
        assert jnp.all(jnp.isfinite(d1))
        assert jnp.all(jnp.isfinite(d2))

    def test_jit_compatible(self, likelihood_tables):
        import jax

        Q_mat, SQ_mat, x_vals = likelihood_tables
        jitted = jax.jit(calc_q_all, static_argnums=())
        Y = jnp.array([5])
        lam = jnp.array([5.0])
        d0, d1, d2 = jitted(Y, lam, Q_mat, SQ_mat, x_vals)
        assert jnp.isfinite(d0[0])


class TestCalcLogLikelihood:
    @pytest.fixture(scope="class")
    def likelihood_tables(self):
        x_vals = build_x_vals()
        Q_mat = compute_q_matrix(sigma=1.0, x_vals=x_vals, K_val=100)
        SQ_mat = compute_spline_coefficients(Q_mat, x_vals)
        return (
            jnp.array(Q_mat),
            jnp.array(SQ_mat),
            jnp.array(x_vals),
        )

    def test_scalar_output(self, likelihood_tables):
        Q_mat, SQ_mat, x_vals = likelihood_tables
        Y = jnp.array([5, 10, 3])
        lam = jnp.array([5.0, 10.0, 3.0])
        ll = calc_log_likelihood(Y, lam, Q_mat, SQ_mat, x_vals)
        assert ll.ndim == 0  # scalar
        assert jnp.isfinite(ll)

    def test_positive_nll(self, likelihood_tables):
        """Negative log-likelihood should be positive (since log P < 0)."""
        Q_mat, SQ_mat, x_vals = likelihood_tables
        Y = jnp.array([5, 10, 3])
        lam = jnp.array([5.0, 10.0, 3.0])
        nll = calc_log_likelihood(Y, lam, Q_mat, SQ_mat, x_vals)
        assert float(nll) > 0

    def test_better_fit_lower_nll(self, likelihood_tables):
        """Better predictions should give lower NLL."""
        Q_mat, SQ_mat, x_vals = likelihood_tables
        Y = jnp.array([5, 10, 20])
        lam_good = jnp.array([5.0, 10.0, 20.0])
        lam_bad = jnp.array([50.0, 100.0, 200.0])
        nll_good = calc_log_likelihood(Y, lam_good, Q_mat, SQ_mat, x_vals)
        nll_bad = calc_log_likelihood(Y, lam_bad, Q_mat, SQ_mat, x_vals)
        assert float(nll_good) < float(nll_bad)
