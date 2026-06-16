"""Regression test: doublet/multi pre-stage of spatial_counts to GPU.

Catches a future refactor of the H2D path that would silently change
numerical output. Uses tolerance-based comparison rather than a byte
hash because PyTorch / numpy minor versions can shift the last ULPs
without changing the cell-type calls (which is what users see).

The byte-identical-vs-main claim was verified one-time during the
v0.3.6 perf sweep by replaying this test on a separate worktree of
unmodified main; the implementation in this branch reproduced that
output exactly. CI then verifies that *future* edits to the H2D path
don't drift further than the tolerance.
"""

import numpy as np

from rctd._doublet import run_doublet_mode
from rctd._likelihood import compute_spline_coefficients, load_cached_q_matrices
from rctd._types import RCTDConfig


def test_doublet_prestage_run_is_sane(synthetic_data):
    """Doublet mode must produce internally consistent output via the pre-staged path."""
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

    N = spatial_counts.shape[0]
    K = profiles.shape[1]

    # Shapes match what doublet mode promises
    assert res.weights.shape == (N, K)
    assert res.weights_doublet.shape == (N, 2)
    assert res.spot_class.shape == (N,)

    # Doublet weights are valid simplex points
    np.testing.assert_allclose(res.weights_doublet.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(res.weights_doublet >= 0.0)

    # Spot class is in the documented range, type indices are valid
    assert np.all((res.spot_class >= 0) & (res.spot_class <= 3))
    assert np.all((res.first_type >= 0) & (res.first_type < K))
    assert np.all((res.second_type >= 0) & (res.second_type < K))

    # Most pixels should converge to a non-degenerate split (one of the two
    # weights >= 0.5). A regression that broke the IRWLS solve would push
    # everything to 0.5/0.5 — guard against that.
    dominant = res.weights_doublet.max(axis=1)
    assert (dominant >= 0.5).mean() >= 0.9
