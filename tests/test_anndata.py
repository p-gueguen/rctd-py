"""rctd.deconvolve writes run_rctd's results into the AnnData, scanpy-style."""

import numpy as np
import pytest

from rctd import RCTDConfig, Reference, deconvolve, run_rctd


def _inputs(synthetic_data):
    spatial = synthetic_data["spatial"].copy()
    X = spatial.X.toarray() if hasattr(spatial.X, "toarray") else np.array(spatial.X)
    X[:3] = 0  # below UMI_min: these pixels are filtered
    spatial.X = X
    return spatial, Reference(synthetic_data["reference"], cell_min=10, min_UMI=10)


@pytest.mark.parametrize("mode", ["full", "doublet", "multi"])
def test_deconvolve_writes_run_rctd_results(synthetic_data, mode):
    spatial, ref = _inputs(synthetic_data)
    kw = dict(mode=mode, config=RCTDConfig(), batch_size=10, sigma_override=50)
    res = run_rctd(spatial, ref, **kw)
    out = spatial.copy()
    assert deconvolve(out, ref, **kw) is None

    m = res.pixel_mask
    assert (~m).sum() == 3
    names = np.array([str(c) for c in res.cell_type_names])
    assert list(out.uns["rctd_cell_type_names"]) == list(names)
    assert out.uns["rctd_mode"] == mode

    w = out.obsm["rctd_weights"]
    assert w.shape == (out.n_obs, len(names))
    np.testing.assert_array_equal(w[m], res.weights.astype(np.float32))
    assert np.isnan(w[~m]).all()
    dominant = out.obs["rctd_dominant_type"].astype(str).values
    assert (dominant[m] == names[res.weights.argmax(axis=1)]).all()
    assert (dominant[~m] == "filtered").all()

    if mode == "full":
        np.testing.assert_array_equal(out.obs["rctd_converged"].values[m], res.converged)
    elif mode == "doublet":
        first = out.obs["rctd_first_type"].astype(str).values
        assert (first[m] == names[res.first_type]).all()
        assert (first[~m] == "filtered").all()
        second = out.obs["rctd_second_type"].astype(str).values
        assert (second[m] == names[res.second_type]).all()
        np.testing.assert_array_equal(
            out.obsm["rctd_weights_doublet"][m], res.weights_doublet.astype(np.float32)
        )
        assert (out.obs["rctd_spot_class"].astype(str).values[~m] == "filtered").all()
    else:
        np.testing.assert_array_equal(out.obs["rctd_n_types"].values[m], res.n_types)
        np.testing.assert_array_equal(
            out.obsm["rctd_sub_weights"][m], res.sub_weights.astype(np.float32)
        )
        np.testing.assert_array_equal(out.obsm["rctd_cell_type_indices"][m], res.cell_type_indices)


def test_deconvolve_rerun_drops_previous_mode_keys(synthetic_data):
    spatial, ref = _inputs(synthetic_data)
    deconvolve(spatial, ref, mode="doublet", batch_size=10, sigma_override=50)
    assert "rctd_spot_class" in spatial.obs
    deconvolve(spatial, ref, mode="full", batch_size=10, sigma_override=50)
    assert spatial.uns["rctd_mode"] == "full"
    assert "rctd_spot_class" not in spatial.obs and "rctd_weights_doublet" not in spatial.obsm
    assert "rctd_converged" in spatial.obs


def test_deconvolve_copy_leaves_input_unchanged(synthetic_data):
    spatial, ref = _inputs(synthetic_data)
    keys = (list(spatial.obs.columns), list(spatial.obsm.keys()), list(spatial.uns.keys()))
    out = deconvolve(spatial, ref, mode="full", batch_size=10, sigma_override=50, copy=True)
    assert out is not spatial
    assert "rctd_weights" in out.obsm and out.uns["rctd_mode"] == "full"
    assert (list(spatial.obs.columns), list(spatial.obsm.keys()), list(spatial.uns.keys())) == keys
