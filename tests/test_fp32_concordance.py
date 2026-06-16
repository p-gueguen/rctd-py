"""fp32 vs fp64 concordance assertion.

The fp32 code path has been exposed via ``RCTDConfig(dtype="float32")``
and CLI ``--dtype float32`` since the initial release, and the perf test
suite times it for all three modes -- but no test asserts that fp32 and
fp64 produce equivalent cell-type classifications. This file fills that
gap.

We run doublet mode twice on the same synthetic fixture (once in each
dtype) and assert:

* spot_class agreement >= 99% (ideally 100%)
* first_type agreement >= 99%
* weights_doublet max absolute diff < 1e-2

The known precision-sensitive site is the spline index in
``_likelihood._calc_q_all_impl`` (``floor(sqrt(lam / delta))``) -- CLAUDE.md
flags it. Empirically (this test + a GPU run at N=20000, K=30 on L40S sm_89)
the index does not shift on either synthetic Slide-seq-like data or large
random Dirichlet mixtures. If a real dataset ever breaks this assertion,
that's the signal to revisit (and likely add an integer-stable index
formulation).
"""

import numpy as np

from rctd._doublet import run_doublet_mode
from rctd._likelihood import compute_spline_coefficients, load_cached_q_matrices
from rctd._types import RCTDConfig


def test_doublet_mode_fp32_matches_fp64(synthetic_data):
    """fp32 must agree with fp64 on cell-type calls."""
    cache = load_cached_q_matrices()
    x_vals = cache.pop("X_vals")
    q_mat = cache["Q_100"]
    sq_mat = compute_spline_coefficients(q_mat, x_vals)

    profiles = synthetic_data["profiles"]
    spatial_counts = synthetic_data["spatial"].X
    spatial_numi = np.array(spatial_counts.sum(axis=1)).flatten()
    cell_type_names = synthetic_data["cell_type_names"]

    def _run(dt):
        target = np.float32 if dt == "float32" else np.float64
        return run_doublet_mode(
            spatial_counts=spatial_counts.astype(target),
            spatial_numi=spatial_numi.astype(target),
            norm_profiles=profiles.astype(target),
            cell_type_names=cell_type_names,
            q_mat=q_mat.astype(target),
            sq_mat=sq_mat.astype(target),
            x_vals=x_vals.astype(target),
            config=RCTDConfig(dtype=dt, compile=False),
            batch_size=10,
            device="cpu",
        )

    res64 = _run("float64")
    res32 = _run("float32")

    spot_agree = float((res64.spot_class == res32.spot_class).mean())
    first_agree = float((res64.first_type == res32.first_type).mean())
    w_diff = float(np.abs(res64.weights_doublet - res32.weights_doublet).max())

    assert spot_agree >= 0.99, f"spot_class agreement {spot_agree:.3f} < 0.99"
    assert first_agree >= 0.99, f"first_type agreement {first_agree:.3f} < 0.99"
    assert w_diff < 1e-2, f"weights_doublet max diff {w_diff:.4g} >= 1e-2"
