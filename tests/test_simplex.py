"""Tests for simplex projection."""

import numpy as np
import torch

from rctd._simplex import project_simplex


class TestProjectSimplex:
    def test_already_on_simplex(self):
        v = torch.tensor([0.3, 0.5, 0.2])
        result = project_simplex(v)
        np.testing.assert_allclose(result.numpy(), v.numpy(), atol=1e-6)

    def test_negative_values(self):
        v = torch.tensor([-0.5, 1.0, 0.5])
        result = project_simplex(v)
        assert (result >= -1e-7).all()
        np.testing.assert_allclose(float(result.sum()), 1.0, atol=1e-6)

    def test_all_equal(self):
        v = torch.tensor([1.0, 1.0, 1.0, 1.0])
        result = project_simplex(v)
        np.testing.assert_allclose(result.numpy(), (torch.ones(4) / 4).numpy(), atol=1e-6)

    def test_single_dominant(self):
        v = torch.tensor([10.0, 0.0, 0.0])
        result = project_simplex(v)
        np.testing.assert_allclose(float(result.sum()), 1.0, atol=1e-6)
        assert float(result[0]) > 0.9

    def test_all_negative(self):
        v = torch.tensor([-1.0, -2.0, -3.0])
        result = project_simplex(v)
        assert (result >= -1e-7).all()
        np.testing.assert_allclose(float(result.sum()), 1.0, atol=1e-6)

    def test_closest_point(self):
        """Projection should be the closest point on the simplex."""
        v = torch.tensor([0.5, 0.5, 0.5])
        result = project_simplex(v)
        # Any other point on simplex should be farther
        other = torch.tensor([1.0, 0.0, 0.0])
        assert float(((result - v) ** 2).sum()) <= float(((other - v) ** 2).sum()) + 1e-6

    def test_batch_via_loop(self):
        """Verify project_simplex works on multiple inputs via a loop."""
        batch = torch.tensor([[0.3, 0.5, 0.2], [-1.0, 2.0, 0.0], [1.0, 1.0, 1.0]])
        results = torch.stack([project_simplex(batch[i]) for i in range(3)])
        assert results.shape == (3, 3)
        np.testing.assert_allclose(results.sum(dim=1).numpy(), np.ones(3), atol=1e-6)
