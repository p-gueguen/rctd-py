"""Regression test: doublet/multi pre-stage of spatial_counts to GPU.

Locks in the weights_hash for a synthetic doublet run so a future
refactor of the H2D path can't silently change numerical output.
"""

import hashlib

import numpy as np

from rctd._doublet import run_doublet_mode
from rctd._likelihood import compute_spline_coefficients, load_cached_q_matrices
from rctd._types import RCTDConfig


def _hash(arr):
    return hashlib.sha256(np.ascontiguousarray(arr.astype(np.float64)).tobytes()).hexdigest()[:16]


def test_doublet_weights_hash_locked(synthetic_data):
    """Output of doublet mode must stay byte-identical across the H2D refactor."""
    cache = load_cached_q_matrices()
    x_vals = cache.pop("X_vals")
    q_mat = cache["Q_100"]
    sq_mat = compute_spline_coefficients(q_mat, x_vals)

    profiles = synthetic_data["profiles"]
    spatial_adata = synthetic_data["spatial"]
    spatial_counts = spatial_adata.X
    spatial_numi = np.array(spatial_counts.sum(axis=1)).flatten()
    cell_type_names = synthetic_data["cell_type_names"]

    res = run_doublet_mode(
        spatial_counts=spatial_counts,
        spatial_numi=spatial_numi,
        norm_profiles=profiles,
        cell_type_names=cell_type_names,
        q_mat=q_mat,
        sq_mat=sq_mat,
        x_vals=x_vals,
        config=RCTDConfig(),
        batch_size=10,
        device="cpu",
    )

    # Sanity, not value: just confirm the run is internally consistent.
    np.testing.assert_allclose(res.weights_doublet.sum(axis=1), 1.0, atol=1e-5)
    assert np.all((res.spot_class >= 0) & (res.spot_class <= 3))

    # Locked under the synthetic_data fixture (seed=42, n_pixels=100, n_types=5).
    # If the fixture seed/parameters change these need regeneration; otherwise any
    # change to the H2D staging path or IRWLS numerics must preserve them.
    assert _hash(res.weights_doublet) == "f9361efb22ff0af4"
    assert _hash(res.spot_class.astype(np.float64)) == "15f73b91f424f559"
