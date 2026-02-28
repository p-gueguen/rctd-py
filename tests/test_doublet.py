import numpy as np

from rctd._doublet import run_doublet_mode
from rctd._likelihood import compute_spline_coefficients, load_cached_q_matrices
from rctd._types import RCTDConfig


def test_doublet_mode_runs(synthetic_data):
    """Test that doublet mode runs without errors and produces valid outputs."""
    cache = load_cached_q_matrices()
    x_vals = cache.pop("X_vals")
    q_mat = cache["Q_100"]
    sq_mat = compute_spline_coefficients(q_mat, x_vals)

    profiles = synthetic_data["profiles"]
    spatial_adata = synthetic_data["spatial"]
    spatial_counts = spatial_adata.X
    spatial_numi = np.array(spatial_counts.sum(axis=1)).flatten()
    cell_type_names = synthetic_data["cell_type_names"]

    config = RCTDConfig()

    res = run_doublet_mode(
        spatial_counts=spatial_counts,
        spatial_numi=spatial_numi,
        norm_profiles=profiles,
        cell_type_names=cell_type_names,
        q_mat=q_mat,
        sq_mat=sq_mat,
        x_vals=x_vals,
        config=config,
        batch_size=10,
    )

    N, K = spatial_counts.shape[0], profiles.shape[1]

    assert res.weights.shape == (N, K)
    assert res.weights_doublet.shape == (N, 2)
    assert res.spot_class.shape == (N,)
    assert res.first_type.shape == (N,)
    assert res.second_type.shape == (N,)

    # Values check
    assert np.all((res.spot_class >= 0) & (res.spot_class <= 3))
    assert np.all((res.first_type >= 0) & (res.first_type < K))
    assert np.all((res.second_type >= 0) & (res.second_type < K))
    assert np.all(res.weights_doublet >= 0.0)
    np.testing.assert_allclose(res.weights_doublet.sum(axis=1), 1.0, atol=1e-5)
    assert len(res.cell_type_names) == K
