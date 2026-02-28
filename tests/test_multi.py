import numpy as np

from rctd._likelihood import compute_spline_coefficients, load_cached_q_matrices
from rctd._multi import run_multi_mode
from rctd._types import RCTDConfig


def test_multi_mode_runs(synthetic_data):
    """Test that multi mode runs without errors and produces valid outputs."""
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

    res = run_multi_mode(
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
    assert res.sub_weights.shape == (N, config.MAX_MULTI_TYPES)
    assert res.cell_type_indices.shape == (N, config.MAX_MULTI_TYPES)
    assert res.n_types.shape == (N,)
    assert res.conf_list.shape == (N, config.MAX_MULTI_TYPES)

    # Check values
    assert np.all((res.n_types >= 1) & (res.n_types <= config.MAX_MULTI_TYPES))

    for n in range(N):
        active_k = res.n_types[n]
        assert np.all(res.cell_type_indices[n, :active_k] >= 0)
        assert np.all(res.cell_type_indices[n, active_k:] == -1)
        sub_w = res.sub_weights[n, :active_k]
        assert np.all(sub_w >= 0.0)
        np.testing.assert_allclose(sub_w.sum(), 1.0, atol=1e-5)
