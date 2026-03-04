import numpy as np
import torch

from rctd._normalize import fit_bulk


def test_fit_bulk_runs(synthetic_data):
    """Test that fit_bulk runs without errors and produces valid outputs."""
    profiles = synthetic_data["profiles"]  # (G, K)
    spatial_adata = synthetic_data["spatial"]  # (N, G)

    spatial_counts = spatial_adata.X
    spatial_numi = np.array(spatial_counts.sum(axis=1)).flatten()

    bulk_weights, norm_profiles = fit_bulk(
        cell_type_profiles=torch.tensor(profiles),
        spatial_counts=torch.tensor(spatial_counts),
        spatial_nUMI=torch.tensor(spatial_numi),
    )

    bulk_weights_np = bulk_weights.cpu().numpy()
    norm_profiles_np = norm_profiles.cpu().numpy()

    # Weights should be finite, non-negative (R does NOT normalize to sum=1)
    assert np.all(np.isfinite(bulk_weights_np))
    assert np.all(bulk_weights_np >= 0)
    assert np.sum(bulk_weights_np) > 0

    # Normalized profiles should have the same shape
    assert norm_profiles_np.shape == profiles.shape
    assert np.all(np.isfinite(norm_profiles_np))
    assert np.all(norm_profiles_np >= 0)
