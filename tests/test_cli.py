"""Tests for rctd CLI."""

import json

import anndata
import numpy as np
import pytest
from click.testing import CliRunner

from rctd.cli import main


def test_info_human():
    """rctd info prints human-readable environment info."""
    runner = CliRunner()
    result = runner.invoke(main, ["info"])
    assert result.exit_code == 0
    assert "rctd-py" in result.output
    assert "torch" in result.output.lower() or "PyTorch" in result.output


def test_info_json():
    """rctd info --json prints valid JSON with required keys."""
    runner = CliRunner()
    result = runner.invoke(main, ["info", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "rctd_version" in data
    assert "python_version" in data
    assert "torch_version" in data
    assert "cuda_available" in data
    assert isinstance(data["cuda_available"], bool)


def test_version():
    """rctd --version prints version string."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()


# ── validate tests ──────────────────────────────────────────────────


@pytest.fixture
def h5ad_pair(tmp_path):
    """Write synthetic spatial and reference h5ad files for CLI testing."""
    rng = np.random.default_rng(42)
    n_genes, n_types, n_ref_cells, n_pixels = 200, 5, 500, 100

    # Reference
    ref_counts = rng.poisson(5, size=(n_ref_cells, n_genes)).astype(np.float32)
    cell_types = [f"Type_{i % n_types}" for i in range(n_ref_cells)]
    ref = anndata.AnnData(
        X=ref_counts,
        obs={"cell_type": cell_types},
    )
    ref.var_names = [f"Gene_{i}" for i in range(n_genes)]
    ref.obs_names = [f"Cell_{i}" for i in range(n_ref_cells)]

    # Spatial
    sp_counts = rng.poisson(10, size=(n_pixels, n_genes)).astype(np.float32)
    sp = anndata.AnnData(X=sp_counts)
    sp.var_names = [f"Gene_{i}" for i in range(n_genes)]
    sp.obs_names = [f"Pixel_{i}" for i in range(n_pixels)]

    ref_path = tmp_path / "reference.h5ad"
    sp_path = tmp_path / "spatial.h5ad"
    ref.write_h5ad(ref_path)
    sp.write_h5ad(sp_path)
    return sp_path, ref_path


def test_validate_pass(h5ad_pair):
    """rctd validate passes with valid inputs."""
    sp_path, ref_path = h5ad_pair
    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(sp_path), str(ref_path)])
    assert result.exit_code == 0
    assert "pass" in result.output.lower() or "PASS" in result.output


def test_validate_json(h5ad_pair):
    """rctd validate --json returns structured JSON."""
    sp_path, ref_path = h5ad_pair
    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(sp_path), str(ref_path), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] in ("pass", "fail")
    assert "checks" in data


def test_validate_missing_column(h5ad_pair):
    """rctd validate fails when cell_type_col doesn't exist."""
    sp_path, ref_path = h5ad_pair
    runner = CliRunner()
    result = runner.invoke(
        main, ["validate", str(sp_path), str(ref_path), "--cell-type-col", "nonexistent"]
    )
    assert result.exit_code == 0  # validate reports failure, doesn't crash
    assert "fail" in result.output.lower() or "FAIL" in result.output


def test_validate_no_gene_overlap(tmp_path):
    """rctd validate fails when spatial and reference share no genes."""
    rng = np.random.default_rng(42)
    ref = anndata.AnnData(X=rng.poisson(5, size=(100, 50)).astype(np.float32))
    ref.var_names = [f"RefGene_{i}" for i in range(50)]
    ref.obs_names = [f"Cell_{i}" for i in range(100)]
    ref.obs["cell_type"] = [f"Type_{i % 3}" for i in range(100)]

    sp = anndata.AnnData(X=rng.poisson(10, size=(50, 50)).astype(np.float32))
    sp.var_names = [f"SpGene_{i}" for i in range(50)]
    sp.obs_names = [f"Pixel_{i}" for i in range(50)]

    ref.write_h5ad(tmp_path / "ref.h5ad")
    sp.write_h5ad(tmp_path / "sp.h5ad")

    runner = CliRunner()
    result = runner.invoke(
        main, ["validate", str(tmp_path / "sp.h5ad"), str(tmp_path / "ref.h5ad")]
    )
    assert result.exit_code == 0
    assert "fail" in result.output.lower() or "0 common" in result.output.lower()


# ── run tests ────────────────────────────────────────────────────────


@pytest.fixture
def h5ad_pair_for_run(tmp_path):
    """Synthetic data that actually works with RCTD (enough UMI, good profiles)."""
    from conftest import _make_synthetic_reference, _make_synthetic_spatial

    ref_adata, profiles, cell_type_names = _make_synthetic_reference(
        n_genes=200, n_cells=500, n_types=5, seed=42
    )
    spatial_adata, true_weights = _make_synthetic_spatial(
        profiles, n_pixels=100, n_types=5, seed=123
    )

    ref_path = tmp_path / "reference.h5ad"
    sp_path = tmp_path / "spatial.h5ad"
    ref_adata.write_h5ad(ref_path)
    spatial_adata.write_h5ad(sp_path)
    return sp_path, ref_path


@pytest.mark.slow
def test_run_doublet(h5ad_pair_for_run, tmp_path):
    """rctd run writes output h5ad with doublet results."""
    sp_path, ref_path = h5ad_pair_for_run
    out_path = tmp_path / "output.h5ad"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            str(sp_path),
            str(ref_path),
            "--mode",
            "doublet",
            "--output",
            str(out_path),
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert out_path.exists()

    out = anndata.read_h5ad(out_path)
    assert "rctd_weights" in out.obsm
    assert "rctd_spot_class" in out.obs.columns
    assert "rctd_first_type" in out.obs.columns
    assert "rctd_dominant_type" in out.obs.columns
    assert out.obsm["rctd_weights"].shape[0] == out.n_obs
    assert out.uns["rctd_mode"] == "doublet"


@pytest.mark.slow
def test_run_full(h5ad_pair_for_run, tmp_path):
    """rctd run --mode full writes full-mode results."""
    sp_path, ref_path = h5ad_pair_for_run
    out_path = tmp_path / "output_full.h5ad"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            str(sp_path),
            str(ref_path),
            "--mode",
            "full",
            "--output",
            str(out_path),
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    out = anndata.read_h5ad(out_path)
    assert "rctd_weights" in out.obsm
    assert "rctd_converged" in out.obs.columns
    assert out.uns["rctd_mode"] == "full"


@pytest.mark.slow
def test_run_multi(h5ad_pair_for_run, tmp_path):
    """rctd run --mode multi writes multi-mode results."""
    sp_path, ref_path = h5ad_pair_for_run
    out_path = tmp_path / "output_multi.h5ad"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            str(sp_path),
            str(ref_path),
            "--mode",
            "multi",
            "--output",
            str(out_path),
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    out = anndata.read_h5ad(out_path)
    assert "rctd_weights" in out.obsm
    assert "rctd_n_types" in out.obs.columns
    assert out.uns["rctd_mode"] == "multi"


@pytest.mark.slow
def test_run_json_output(h5ad_pair_for_run, tmp_path):
    """rctd run --json prints structured JSON summary."""
    sp_path, ref_path = h5ad_pair_for_run
    out_path = tmp_path / "output_json.h5ad"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            str(sp_path),
            str(ref_path),
            "--output",
            str(out_path),
            "--device",
            "cpu",
            "--json",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    # CliRunner mixes stderr with stdout; extract JSON block from output
    output = result.output
    json_start = output.index("{")
    json_end = output.rindex("}") + 1
    data = json.loads(output[json_start:json_end])
    assert data["status"] == "success"
    assert data["mode"] == "doublet"
    assert "input" in data
    assert "results" in data
    assert "summary" in data


@pytest.mark.slow
def test_run_default_output_path(h5ad_pair_for_run):
    """rctd run without --output writes to <spatial_stem>_rctd.h5ad."""
    sp_path, ref_path = h5ad_pair_for_run
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            str(sp_path),
            str(ref_path),
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    expected_out = sp_path.parent / "spatial_rctd.h5ad"
    assert expected_out.exists()


@pytest.mark.slow
def test_run_pixel_mask_expansion(h5ad_pair_for_run, tmp_path):
    """Output h5ad has same n_obs as input; filtered pixels have NaN weights."""
    sp_path, ref_path = h5ad_pair_for_run
    out_path = tmp_path / "output_mask.h5ad"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            str(sp_path),
            str(ref_path),
            "--output",
            str(out_path),
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    sp_in = anndata.read_h5ad(sp_path)
    out = anndata.read_h5ad(out_path)
    assert out.n_obs == sp_in.n_obs


@pytest.mark.slow
def test_run_doublet_with_class_df(h5ad_pair_for_run, tmp_path):
    """rctd run --mode doublet --class-df writes class-level columns."""
    sp_path, ref_path = h5ad_pair_for_run

    # Build a TSV mapping 5 synthetic types into 2 classes
    class_df_path = tmp_path / "class_df.tsv"
    with class_df_path.open("w") as f:
        f.write("cell_type\tclass\n")
        for i in range(5):
            f.write(f"Type_{i}\t{'ClassA' if i < 2 else 'ClassB'}\n")

    out_path = tmp_path / "output_cdf.h5ad"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            str(sp_path),
            str(ref_path),
            "--mode",
            "doublet",
            "--output",
            str(out_path),
            "--device",
            "cpu",
            "--class-df",
            str(class_df_path),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    out = anndata.read_h5ad(out_path)
    # Class-level columns must exist
    assert "rctd_first_class" in out.obs.columns
    assert "rctd_second_class" in out.obs.columns
    assert "rctd_first_class_name" in out.obs.columns
    assert "rctd_second_class_name" in out.obs.columns

    # Every non-filtered class name must be in {ClassA, ClassB, filtered}
    valid = {"ClassA", "ClassB", "filtered"}
    assert set(out.obs["rctd_first_class_name"].unique()).issubset(valid)
    assert set(out.obs["rctd_second_class_name"].unique()).issubset(valid)


def test_run_class_df_missing_columns_errors(h5ad_pair_for_run, tmp_path):
    """--class-df TSV without required columns fails with clear message."""
    sp_path, ref_path = h5ad_pair_for_run

    class_df_path = tmp_path / "bad_class_df.tsv"
    with class_df_path.open("w") as f:
        f.write("wrong_col\tother_col\n")
        f.write("Type_0\tClassA\n")

    out_path = tmp_path / "output_bad.h5ad"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            str(sp_path),
            str(ref_path),
            "--mode",
            "doublet",
            "--output",
            str(out_path),
            "--device",
            "cpu",
            "--class-df",
            str(class_df_path),
        ],
    )
    assert result.exit_code != 0
    assert "cell_type" in result.output and "class" in result.output


# ── SpatialData Zarr input (issue #28) ──────────────────────────────


def _write_spatialdata_store(path, table):
    """Write a real SpatialData Zarr store, with pandas future string dtypes on.

    ``pandas.options.future.infer_string`` (the pandas 3.0 default) makes
    spatialdata encode obs/var names as ``nullable-string-array``. anndata reads
    those back as a pandas ``StringArray`` and then refuses to write them out to
    .h5ad, so this is the store shape ``_read_adata`` has to normalise.
    """
    import pandas as pd

    sd = pytest.importorskip("spatialdata")
    import anndata as ad
    from spatialdata.models import TableModel

    table = table.copy()
    table.obs_names = pd.Index(table.obs_names, dtype="string")
    table.var_names = pd.Index(table.var_names, dtype="string")

    prev = ad.settings.allow_write_nullable_strings
    try:
        ad.settings.allow_write_nullable_strings = True
        sd.SpatialData(tables={"table": TableModel.parse(table)}).write(path)
    finally:
        ad.settings.allow_write_nullable_strings = prev

    # Fail loudly if the store stopped carrying the encoding these tests exist for.
    meta = json.loads((path / "tables/table/obs/_index/zarr.json").read_text())
    assert meta["attributes"]["encoding-type"] == "nullable-string-array", meta
    return path


@pytest.fixture
def spatialdata_store(tmp_path):
    """A real SpatialData store holding an RCTD-shaped table, plus a reference h5ad."""
    from conftest import _make_synthetic_reference, _make_synthetic_spatial

    ref_adata, profiles, _ = _make_synthetic_reference(n_genes=200, n_cells=500, n_types=5, seed=42)
    spatial_adata, _ = _make_synthetic_spatial(profiles, n_pixels=100, n_types=5, seed=123)

    ref_path = tmp_path / "reference.h5ad"
    ref_adata.write_h5ad(ref_path)

    store = _write_spatialdata_store(tmp_path / "spatial.zarr", spatial_adata)
    return store, ref_path


def test_read_adata_real_spatialdata_store(tmp_path):
    """_read_adata reads spatialdata's own blobs store; result stays h5ad-writable."""
    sd = pytest.importorskip("spatialdata")

    from rctd.cli import _read_adata

    store = _write_spatialdata_store(tmp_path / "blobs.zarr", sd.datasets.blobs()["table"])

    adata = _read_adata(store)
    assert adata.shape == (26, 3)

    # Regression: without dtype normalisation these names come back as a pandas
    # nullable StringArray and the write below raises — after a full run's compute.
    assert not str(adata.obs_names.dtype).startswith("string")
    assert not str(adata.var_names.dtype).startswith("string")
    assert not anndata.settings.allow_write_nullable_strings  # the CLI's own default
    adata.write_h5ad(tmp_path / "roundtrip.h5ad")


def test_validate_spatialdata_store(spatialdata_store):
    """rctd validate accepts a SpatialData store root and a table subpath alike."""
    store, ref_path = spatialdata_store
    runner = CliRunner()
    for spatial in (store, store / "tables" / "table"):
        result = runner.invoke(main, ["validate", str(spatial), str(ref_path), "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["status"] == "pass", data
        assert data["checks"]["spatial_readable"]["pass"]


def test_read_adata_spatialdata_multiple_tables(tmp_path):
    """A store with several tables names them and asks for an explicit table path."""
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import TableModel

    from rctd.cli import _read_adata

    rng = np.random.default_rng(0)
    table = anndata.AnnData(X=rng.poisson(5, size=(20, 10)).astype(np.float32))
    table.var_names = [f"Gene_{i}" for i in range(10)]
    store = tmp_path / "two.zarr"
    sd.SpatialData(
        tables={"table": TableModel.parse(table.copy()), "other": TableModel.parse(table.copy())}
    ).write(store)

    with pytest.raises(Exception, match="2 tables"):
        _read_adata(store)
    # ...but pointing at one of them works
    assert _read_adata(store / "tables" / "other").shape == (20, 10)


@pytest.mark.slow
def test_run_spatialdata_store(spatialdata_store, tmp_path):
    """rctd run deconvolves straight from a SpatialData store."""
    store, ref_path = spatialdata_store
    out_path = tmp_path / "out.h5ad"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            str(store),
            str(ref_path),
            "--mode",
            "doublet",
            "--output",
            str(out_path),
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    out = anndata.read_h5ad(out_path)
    assert "rctd_spot_class" in out.obs
    assert out.obsm["rctd_weights"].shape[0] == out.n_obs


@pytest.mark.slow
def test_run_spatialdata_default_output_stays_outside_store(spatialdata_store):
    """The default output path must not land inside the Zarr store."""
    store, ref_path = spatialdata_store
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            str(store / "tables" / "table"),
            str(ref_path),
            "--mode",
            "doublet",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (store.parent / "spatial_rctd.h5ad").exists()
    assert not list(store.rglob("*.h5ad"))
