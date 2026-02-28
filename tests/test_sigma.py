import numpy as np

from rctd._likelihood import load_cached_q_matrices
from rctd._sigma import choose_sigma


def test_choose_sigma_runs(synthetic_data):
    """Test that choose_sigma completes successfully."""
    # We need q_matrices and x_vals
    cache = load_cached_q_matrices()
    x_vals = cache.pop("X_vals")
    # remaining cache is exactly q_matrices
    q_matrices = {k.replace("Q_", ""): v for k, v in cache.items()}

    profiles = synthetic_data["profiles"]  # (G, K)
    spatial_adata = synthetic_data["spatial"]
    spatial_counts = spatial_adata.X
    spatial_numi = np.array(spatial_counts.sum(axis=1)).flatten()

    # Run with small n_epoch and n_fit just to check if it runs without crashing
    best_sigma = choose_sigma(
        spatial_counts=spatial_counts,
        spatial_numi=spatial_numi,
        norm_profiles=profiles,  # using original profiles as mock norm_profiles
        q_matrices=q_matrices,
        x_vals=x_vals,
        sigma_init=100,
        min_umi=50,
        n_fit=5,
        n_epoch=2,
    )

    assert isinstance(best_sigma, (int, np.integer))
    assert best_sigma > 0
