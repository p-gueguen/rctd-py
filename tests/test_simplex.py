"""Tests for simplex projection."""

import jax.numpy as jnp
import numpy as np

from rctd._simplex import project_simplex


class TestProjectSimplex:
    def test_already_on_simplex(self):
        v = jnp.array([0.3, 0.5, 0.2])
        result = project_simplex(v)
        np.testing.assert_allclose(result, v, atol=1e-6)

    def test_negative_values(self):
        v = jnp.array([-0.5, 1.0, 0.5])
        result = project_simplex(v)
        assert jnp.all(result >= -1e-7)
        np.testing.assert_allclose(float(jnp.sum(result)), 1.0, atol=1e-6)

    def test_all_equal(self):
        v = jnp.array([1.0, 1.0, 1.0, 1.0])
        result = project_simplex(v)
        np.testing.assert_allclose(result, jnp.ones(4) / 4, atol=1e-6)

    def test_single_dominant(self):
        v = jnp.array([10.0, 0.0, 0.0])
        result = project_simplex(v)
        np.testing.assert_allclose(float(jnp.sum(result)), 1.0, atol=1e-6)
        assert float(result[0]) > 0.9

    def test_all_negative(self):
        v = jnp.array([-1.0, -2.0, -3.0])
        result = project_simplex(v)
        assert jnp.all(result >= -1e-7)
        np.testing.assert_allclose(float(jnp.sum(result)), 1.0, atol=1e-6)

    def test_closest_point(self):
        """Projection should be the closest point on the simplex."""
        v = jnp.array([0.5, 0.5, 0.5])
        result = project_simplex(v)
        # Any other point on simplex should be farther
        other = jnp.array([1.0, 0.0, 0.0])
        assert float(jnp.sum((result - v) ** 2)) <= float(jnp.sum((other - v) ** 2)) + 1e-6

    def test_jit_compatible(self):
        import jax

        f = jax.jit(project_simplex)
        v = jnp.array([0.3, 0.5, 0.2])
        result = f(v)
        np.testing.assert_allclose(result, v, atol=1e-6)

    def test_vmap_compatible(self):
        import jax

        batch = jnp.array([[0.3, 0.5, 0.2], [-1.0, 2.0, 0.0], [1.0, 1.0, 1.0]])
        f = jax.vmap(project_simplex)
        result = f(batch)
        assert result.shape == (3, 3)
        np.testing.assert_allclose(jnp.sum(result, axis=1), jnp.ones(3), atol=1e-6)
