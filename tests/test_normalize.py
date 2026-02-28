import jax.numpy as jnp
import numpy as np

from rctd._normalize import fit_bulk


def test_fit_bulk_runs(synthetic_data):
    """Test that fit_bulk runs without errors and produces valid outputs."""
    profiles = synthetic_data["profiles"]  # (G, K)
    spatial_adata = synthetic_data["spatial"]  # (N, G)

    spatial_counts = spatial_adata.X
    spatial_numi = np.array(spatial_counts.sum(axis=1)).flatten()

    bulk_weights, norm_profiles = fit_bulk(
        cell_type_profiles=jnp.array(profiles),
        spatial_counts=jnp.array(spatial_counts),
        spatial_nUMI=jnp.array(spatial_numi),
    )

    # Weights should be finite, non-negative (R does NOT normalize to sum=1)
    assert np.all(np.isfinite(bulk_weights))
    assert np.all(bulk_weights >= 0)
    assert np.sum(bulk_weights) > 0

    # Normalized profiles should have the same shape
    assert norm_profiles.shape == profiles.shape
    assert np.all(np.isfinite(norm_profiles))
    assert np.all(norm_profiles >= 0)
