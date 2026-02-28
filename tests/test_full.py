import numpy as np

from rctd._full import run_full_mode
from rctd._likelihood import compute_spline_coefficients, load_cached_q_matrices


def test_full_mode_runs(synthetic_data):
    """Test that full mode runs without errors."""
    cache = load_cached_q_matrices()
    x_vals = cache.pop("X_vals")
    q_mat = cache["Q_100"]
    sq_mat = compute_spline_coefficients(q_mat, x_vals)

    profiles = synthetic_data["profiles"]
    spatial_adata = synthetic_data["spatial"]
    spatial_counts = spatial_adata.X
    spatial_numi = np.array(spatial_counts.sum(axis=1)).flatten()
    cell_type_names = synthetic_data["cell_type_names"]

    res = run_full_mode(
        spatial_counts=spatial_counts,
        spatial_numi=spatial_numi,
        norm_profiles=profiles,
        cell_type_names=cell_type_names,
        q_mat=q_mat,
        sq_mat=sq_mat,
        x_vals=x_vals,
        batch_size=10,
    )

    N, K = spatial_counts.shape[0], profiles.shape[1]

    assert res.weights.shape == (N, K)
    assert res.weights.dtype == np.float32 or res.weights.dtype == np.float64
    assert res.converged.shape == (N,)
    assert res.converged.dtype == bool
    assert len(res.cell_type_names) == K
