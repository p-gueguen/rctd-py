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


def test_deconvolve_rerun_drops_previous_mode_keys_only(synthetic_data):
    spatial, ref = _inputs(synthetic_data)
    spatial.obs["leiden"] = "0"
    spatial.obs["rctd_my_label"] = "x"  # a user's own rctd_* key must survive
    spatial.obsm["spatial"] = np.zeros((spatial.n_obs, 2))
    spatial.uns["rctd_notes"] = "kept"
    deconvolve(spatial, ref, mode="doublet", batch_size=10, sigma_override=50)
    assert "rctd_spot_class" in spatial.obs
    deconvolve(spatial, ref, mode="full", batch_size=10, sigma_override=50)
    assert spatial.uns["rctd_mode"] == "full"
    assert "rctd_spot_class" not in spatial.obs and "rctd_weights_doublet" not in spatial.obsm
    assert "rctd_converged" in spatial.obs
    assert {"leiden", "rctd_my_label"} <= set(spatial.obs.columns)
    assert "spatial" in spatial.obsm and spatial.uns["rctd_notes"] == "kept"


def test_deconvolve_class_df_names(synthetic_data):
    spatial, ref = _inputs(synthetic_data)
    class_df = {n: ("A" if i % 2 else "B") for i, n in enumerate(ref.cell_type_names)}
    cfg = RCTDConfig(class_df=class_df)
    res = run_rctd(spatial, ref, mode="doublet", config=cfg, batch_size=10, sigma_override=50)
    deconvolve(spatial, ref, mode="doublet", config=cfg, batch_size=10, sigma_override=50)
    m = res.pixel_mask
    got = spatial.obs["rctd_first_class_name"].astype(str).values
    assert (got[m] == np.asarray(res.first_class_name).astype(str)).all()
    assert (got[~m] == "filtered").all()


def test_deconvolve_rejects_backed(synthetic_data, tmp_path):
    import anndata

    spatial, ref = _inputs(synthetic_data)
    spatial.write_h5ad(tmp_path / "s.h5ad")
    backed = anndata.read_h5ad(tmp_path / "s.h5ad", backed="r")
    with pytest.raises(ValueError, match="to_memory"):
        deconvolve(backed, ref, mode="full", batch_size=10, sigma_override=50)


def test_deconvolve_copy_leaves_input_unchanged(synthetic_data):
    spatial, ref = _inputs(synthetic_data)
    spatial.uns["meta"] = {"a": 1}
    keys = (list(spatial.obs.columns), list(spatial.obsm.keys()), list(spatial.uns.keys()))
    out = deconvolve(spatial, ref, mode="full", batch_size=10, sigma_override=50, copy=True)
    assert out is not spatial
    assert "rctd_weights" in out.obsm and out.uns["rctd_mode"] == "full"
    assert (list(spatial.obs.columns), list(spatial.obsm.keys()), list(spatial.uns.keys())) == keys
    out.uns["meta"]["a"] = 2
    assert spatial.uns["meta"]["a"] == 1


# --- scverse-style namespaces: rctd.pp.reference, rctd.tl.deconvolve, layer, key_added ---

_KW = dict(batch_size=10, sigma_override=50)
# The keys 0.4.2 wrote, per mode; the default key_added must keep them exactly
_KEYS_042 = {
    "full": ({"rctd_dominant_type", "rctd_converged"}, {"rctd_weights"}),
    "doublet": (
        {
            "rctd_dominant_type",
            "rctd_spot_class",
            "rctd_first_type",
            "rctd_second_type",
            "rctd_first_class",
            "rctd_second_class",
        },
        {"rctd_weights", "rctd_weights_doublet"},
    ),
    "multi": (
        {"rctd_dominant_type", "rctd_n_types"},
        {"rctd_weights", "rctd_sub_weights", "rctd_cell_type_indices"},
    ),
}
_UNS_042 = {"rctd_mode", "rctd_config", "rctd_version", "rctd_cell_type_names"}


def _assert_same_results(a, b):
    assert list(a.obs.columns) == list(b.obs.columns)
    for k in a.obs.columns:
        np.testing.assert_array_equal(np.asarray(a.obs[k]), np.asarray(b.obs[k]))
    assert set(a.obsm.keys()) == set(b.obsm.keys())
    for k in a.obsm.keys():
        np.testing.assert_array_equal(a.obsm[k], b.obsm[k])
    for k in ("rctd_mode", "rctd_cell_type_names"):
        assert list(np.atleast_1d(a.uns[k])) == list(np.atleast_1d(b.uns[k]))


def test_namespaces():
    import rctd

    assert rctd.tl.deconvolve is rctd.deconvolve
    assert callable(rctd.pp.reference)


@pytest.mark.parametrize("mode", ["full", "doublet", "multi"])
def test_tl_deconvolve_with_reference_anndata(synthetic_data, mode):
    import rctd

    spatial, _ = _inputs(synthetic_data)
    ref_adata = synthetic_data["reference"].copy()
    ref_adata.obs["ct"] = ref_adata.obs.pop("cell_type")
    a = rctd.tl.deconvolve(spatial, ref_adata, mode=mode, cell_type_key="ct", copy=True, **_KW)
    b = deconvolve(spatial, Reference(ref_adata, cell_type_col="ct"), mode=mode, copy=True, **_KW)
    _assert_same_results(a, b)
    obs_keys, obsm_keys = _KEYS_042[mode]
    assert set(a.obs.columns) - set(spatial.obs.columns) == obs_keys
    assert set(a.obsm.keys()) - set(spatial.obsm.keys()) == obsm_keys
    assert set(a.uns.keys()) - set(spatial.uns.keys()) == _UNS_042


def test_pp_reference_matches_reference_class(synthetic_data):
    import rctd

    ref_adata = synthetic_data["reference"]
    a = rctd.pp.reference(ref_adata, cell_min=10, min_UMI=10)
    b = Reference(ref_adata, cell_min=10, min_UMI=10)
    assert list(a.cell_type_names) == list(b.cell_type_names)
    np.testing.assert_array_equal(a.profiles, b.profiles)


def _move_counts_to_layer(adata):
    """Counts into layers["counts"], log-normalised values into X, as after sc.pp.log1p."""
    adata = adata.copy()
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    adata.layers["counts"] = X.copy()
    adata.X = np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e4)
    return adata


def test_layer_reads_counts_from_layer(synthetic_data):
    import rctd

    spatial, ref = _inputs(synthetic_data)
    want = deconvolve(spatial, ref, mode="doublet", copy=True, **_KW)
    norm = _move_counts_to_layer(spatial)
    got = rctd.tl.deconvolve(norm, ref, mode="doublet", layer="counts", copy=True, **_KW)
    _assert_same_results(got.copy(), want)
    assert "counts" in got.layers  # the input's own slots are kept
    # Positive control: on the log-normalised X the same call gives another answer
    wrong = rctd.tl.deconvolve(norm, ref, mode="doublet", copy=True, **_KW)
    assert not np.array_equal(
        np.nan_to_num(wrong.obsm["rctd_weights"]), np.nan_to_num(want.obsm["rctd_weights"])
    )

    ref_norm = _move_counts_to_layer(synthetic_data["reference"])
    a = rctd.pp.reference(ref_norm, layer="counts", cell_min=10, min_UMI=10)
    np.testing.assert_array_equal(a.profiles, ref.profiles)


def test_unknown_layer_and_bad_key_added_raise(synthetic_data):
    spatial, ref = _inputs(synthetic_data)
    with pytest.raises(KeyError, match="nope"):
        deconvolve(spatial, ref, layer="nope", **_KW)
    with pytest.raises(ValueError, match="key_added"):
        deconvolve(spatial, ref, key_added="", **_KW)
    assert "rctd_weights" not in spatial.obsm


def test_key_added_runs_sit_side_by_side(synthetic_data):
    spatial, ref = _inputs(synthetic_data)
    want_dbl = deconvolve(spatial, ref, mode="doublet", copy=True, **_KW)
    deconvolve(spatial, ref, mode="doublet", key_added="dbl", **_KW)
    deconvolve(spatial, ref, mode="full", key_added="full", **_KW)
    spatial.obs["dbl_mine"] = "x"
    deconvolve(spatial, ref, mode="doublet", key_added="dbl", **_KW)  # re-run keeps the others
    for slot in (spatial.obs.columns, spatial.obsm.keys(), spatial.uns.keys()):
        assert not any(k.startswith("rctd_") for k in slot)
    assert spatial.uns["dbl_mode"] == "doublet" and spatial.uns["full_mode"] == "full"
    assert "full_converged" in spatial.obs and "dbl_mine" in spatial.obs
    np.testing.assert_array_equal(spatial.obsm["dbl_weights"], want_dbl.obsm["rctd_weights"])
    np.testing.assert_array_equal(
        np.asarray(spatial.obs["dbl_first_type"]), np.asarray(want_dbl.obs["rctd_first_type"])
    )

    # A re-run in another mode with the same prefix drops that prefix's doublet-only keys
    deconvolve(spatial, ref, mode="multi", key_added="dbl", **_KW)
    assert not {"dbl_first_type", "dbl_spot_class"} & set(spatial.obs.columns)
    assert "dbl_weights_doublet" not in spatial.obsm and "dbl_n_types" in spatial.obs
    assert "full_converged" in spatial.obs and "dbl_mine" in spatial.obs
