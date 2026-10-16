"""Tests for reference processing."""

import numpy as np
import pytest

from rctd._reference import Reference


class TestReference:
    def test_creates_from_anndata(self, synthetic_reference):
        ref_adata, _, _ = synthetic_reference
        ref = Reference(ref_adata, cell_type_col="cell_type")
        assert ref.n_types > 0
        assert ref.profiles.shape[0] > 0

    def test_profiles_sum_to_one(self, synthetic_reference):
        ref_adata, _, _ = synthetic_reference
        ref = Reference(ref_adata, cell_type_col="cell_type")
        col_sums = ref.profiles.sum(axis=0)
        np.testing.assert_allclose(col_sums, np.ones(ref.n_types), atol=0.01)

    def test_gene_selection(self, synthetic_reference):
        ref_adata, _, _ = synthetic_reference
        ref = Reference(ref_adata, cell_type_col="cell_type")
        de_genes = ref.get_de_genes(fc_thresh=0.5, expr_thresh=1e-4)
        assert len(de_genes) > 0
        assert all(g in ref_adata.var_names for g in de_genes)

    def test_filters_low_count_types(self):
        """Cell types with < cell_min cells should be rejected."""
        import anndata

        rng = np.random.default_rng(42)
        n_genes, n_cells = 100, 30
        counts = rng.poisson(5, size=(n_cells, n_genes)).astype(np.float32)
        # Type_A has 25 cells, Type_B has only 5
        cell_types = ["Type_A"] * 25 + ["Type_B"] * 5
        adata = anndata.AnnData(X=counts, obs={"cell_type": cell_types})
        adata.var_names = [f"Gene_{i}" for i in range(n_genes)]

        with pytest.raises(ValueError, match="fewer than"):
            Reference(adata, cell_type_col="cell_type", cell_min=25)

    def test_profiles_shape_matches(self, synthetic_reference):
        ref_adata, _, cell_type_names = synthetic_reference
        ref = Reference(ref_adata, cell_type_col="cell_type")
        assert ref.profiles.shape == (ref.n_genes, ref.n_types)
        assert ref.n_types == len(cell_type_names)
        assert ref.n_genes == ref_adata.n_vars

    def test_get_profiles_for_genes(self, synthetic_reference):
        ref_adata, _, _ = synthetic_reference
        ref = Reference(ref_adata, cell_type_col="cell_type")
        gene_subset = ref.gene_names[:10]
        sub = ref.get_profiles_for_genes(gene_subset)
        assert sub.shape == (10, ref.n_types)
        np.testing.assert_array_equal(sub, ref.profiles[:10, :])

    def test_normalize_to_spatial(self, synthetic_reference):
        ref_adata, _, _ = synthetic_reference
        ref = Reference(ref_adata, cell_type_col="cell_type")
        gene_names = ref.gene_names[:50]
        # Simulate spatial bulk counts
        rng = np.random.default_rng(99)
        spatial_bulk = rng.poisson(100, size=ref.n_genes).astype(np.float64)
        spatial_nUMI_total = float(spatial_bulk.sum())
        proportions = np.ones(ref.n_types) / ref.n_types

        renorm = ref.normalize_to_spatial(spatial_bulk, spatial_nUMI_total, gene_names, proportions)
        assert renorm.shape == (50, ref.n_types)
        # All values should be finite and non-negative
        assert np.all(np.isfinite(renorm))
        assert np.all(renorm >= 0)

    def test_de_genes_filters_mt_genes(self):
        """Genes whose name contains "mt-" are excluded, case-sensitively, as in R.

        spacexr's get_de_genes drops grep("mt-", gene_list): mouse mt-Co1 and any name holding
        "mt-" go, human MT-ND1 stays a candidate.
        """
        import anndata

        rng = np.random.default_rng(42)
        n_genes, n_cells = 50, 200

        # Create two distinct types
        counts = np.zeros((n_cells, n_genes), dtype=np.float32)
        cell_types = []
        for i in range(100):
            counts[i, :25] = rng.poisson(10, size=25)
            counts[i, 25:] = rng.poisson(1, size=25)
            cell_types.append("Type_A")
        for i in range(100, 200):
            counts[i, :25] = rng.poisson(1, size=25)
            counts[i, 25:] = rng.poisson(10, size=25)
            cell_types.append("Type_B")

        gene_names = [f"Gene_{i}" for i in range(n_genes)]
        gene_names[0] = "mt-Co1"
        gene_names[1] = "MT-ND1"
        gene_names[2] = "Xmt-1"

        adata = anndata.AnnData(X=counts, obs={"cell_type": cell_types})
        adata.var_names = gene_names
        adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

        ref = Reference(adata, cell_type_col="cell_type")
        de_genes = ref.get_de_genes(fc_thresh=0.5, expr_thresh=1e-4)

        assert "mt-Co1" not in de_genes
        assert "Xmt-1" not in de_genes
        assert "MT-ND1" in de_genes  # positive control: an equally DE gene that R keeps

    def test_missing_column_raises(self, synthetic_reference):
        ref_adata, _, _ = synthetic_reference
        with pytest.raises(ValueError, match="not found"):
            Reference(ref_adata, cell_type_col="nonexistent")

    def test_min_umi_filter(self):
        """Cells below min_UMI should be filtered out."""
        import anndata

        rng = np.random.default_rng(42)
        n_genes, n_cells = 50, 200
        # Low UMI cells (sum ~50 per cell)
        counts = rng.poisson(1, size=(n_cells, n_genes)).astype(np.float32)
        cell_types = ["Type_A"] * 100 + ["Type_B"] * 100
        adata = anndata.AnnData(X=counts, obs={"cell_type": cell_types})
        adata.var_names = [f"Gene_{i}" for i in range(n_genes)]

        # With high min_UMI, should fail
        with pytest.raises(ValueError):
            Reference(adata, cell_type_col="cell_type", min_UMI=10000)

    def test_downsample(self):
        """When n_max_cells < cells per type, should downsample."""
        import anndata

        rng = np.random.default_rng(42)
        n_genes, n_cells = 50, 200
        counts = rng.poisson(10, size=(n_cells, n_genes)).astype(np.float32)
        cell_types = ["Type_A"] * 100 + ["Type_B"] * 100
        adata = anndata.AnnData(X=counts, obs={"cell_type": cell_types})
        adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
        adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

        ref = Reference(adata, cell_type_col="cell_type", n_max_cells=50)
        # Profiles should still be valid and sum to ~1
        col_sums = ref.profiles.sum(axis=0)
        np.testing.assert_allclose(col_sums, np.ones(ref.n_types), atol=0.02)

    @pytest.mark.parametrize("n_max_cells", [10_000, 50])
    def test_sparse_matches_dense(self, n_max_cells):
        """Sparse and dense inputs must produce identical profiles.

        Tests both with and without downsampling (n_max_cells < cells/type).
        """
        import anndata
        from scipy import sparse

        rng = np.random.default_rng(42)
        n_genes, n_cells = 100, 300
        counts = rng.poisson(5, size=(n_cells, n_genes)).astype(np.float32)
        cell_types = ["Type_A"] * 100 + ["Type_B"] * 100 + ["Type_C"] * 100

        adata_dense = anndata.AnnData(
            X=counts.copy(),
            obs={"cell_type": cell_types},
        )
        adata_dense.var_names = [f"Gene_{i}" for i in range(n_genes)]
        adata_dense.obs_names = [f"Cell_{i}" for i in range(n_cells)]

        adata_sparse = anndata.AnnData(
            X=sparse.csr_matrix(counts),
            obs={"cell_type": cell_types},
        )
        adata_sparse.var_names = [f"Gene_{i}" for i in range(n_genes)]
        adata_sparse.obs_names = [f"Cell_{i}" for i in range(n_cells)]

        ref_dense = Reference(adata_dense, cell_type_col="cell_type", n_max_cells=n_max_cells)
        ref_sparse = Reference(adata_sparse, cell_type_col="cell_type", n_max_cells=n_max_cells)

        np.testing.assert_allclose(ref_sparse.profiles, ref_dense.profiles, rtol=1e-12)


def test_cell_type_order_follows_r():
    """Type indices must match spacexr's levels(cell_types): R's sort for plain strings, the
    category order for a categorical (an R factor's levels). Expected orders are R 4.5's sort()."""
    import anndata
    import pandas as pd

    from rctd._reference import r_sort_key

    assert "".join(sorted("$+-_,.09aAbB", key=r_sort_key)) == "_-,.+$09aAbB"
    r_order = [
        "adipocyte",
        "B cell",
        "blood vessel endothelial cell",
        "CD4-positive helper T cell",
        "CD4-positive, alpha-beta T cell",
        "gamma_delta_T_cells",
        "Germinal_center_B_cells",
        "Serpinf1",
        "SMC",
    ]
    assert sorted(r_order[::-1], key=r_sort_key) == r_order

    rng = np.random.default_rng(0)
    types = ["Serpinf1", "SMC", "adipocyte", "B cell"]
    X = rng.poisson(5, size=(400, 50)).astype(np.float32)
    labels = np.repeat(types, 100)
    ad = anndata.AnnData(
        X=X, obs=pd.DataFrame({"cell_type": labels}, index=[f"c{i}" for i in range(400)])
    )
    ad.var_names = [f"g{i}" for i in range(50)]
    assert Reference(ad, cell_min=25, min_UMI=1).cell_type_names == [
        "adipocyte",
        "B cell",
        "Serpinf1",
        "SMC",
    ]
    ad.obs["cell_type"] = pd.Categorical(
        labels, categories=["SMC", "B cell", "Serpinf1", "adipocyte"]
    )
    assert Reference(ad, cell_min=25, min_UMI=1).cell_type_names == [
        "SMC",
        "B cell",
        "Serpinf1",
        "adipocyte",
    ]


def _labelled_reference(labels):
    """Small reference whose types differ in one block of genes each, with the given labels."""
    import anndata

    rng = np.random.default_rng(0)
    types = sorted(set(labels), key=str)
    counts = rng.poisson(1, size=(len(labels), 10 * len(types))).astype(np.float32)
    for i, lab in enumerate(labels):
        j = types.index(lab)
        counts[i, 10 * j : 10 * j + 10] += rng.poisson(20, size=10)
    adata = anndata.AnnData(X=counts)
    adata.obs["cell_type"] = labels
    adata.obs_names = [f"c{i}" for i in range(len(labels))]
    adata.var_names = [f"g{i}" for i in range(counts.shape[1])]
    return adata


def test_integer_cell_type_labels_sort_like_r_factor():
    """Integer labels (e.g. cluster ids) are accepted, as in 0.3.9, and ordered numerically
    like R's factor() levels: 2 before 10."""
    labels = [10] * 30 + [2] * 30 + [1] * 30
    ref = Reference(_labelled_reference(labels), cell_type_col="cell_type")
    assert list(ref.cell_type_names) == [1, 2, 10]


def test_categorical_labels_drop_unused_categories():
    """A categorical keeps its category order but leaves out categories with no cells."""
    import pandas as pd

    labels = pd.Categorical(["b"] * 30 + ["a"] * 30, categories=["b", "unused", "a"])
    adata = _labelled_reference(list(labels))
    adata.obs["cell_type"] = labels
    ref = Reference(adata, cell_type_col="cell_type")
    assert list(ref.cell_type_names) == ["b", "a"]


def test_h5ad_style_categorical_gets_r_order():
    """anndata stores string labels as a categorical sorted by code point; that order carries no
    meaning, so it is replaced by R's (the CLI and every .h5ad read must match R like the API)."""
    import pandas as pd

    labels = ["T cell"] * 30 + ["adipocyte"] * 30 + ["B cell"] * 30
    adata = _labelled_reference(labels)
    adata.obs["cell_type"] = pd.Categorical(labels)  # categories sorted: B cell, T cell, adipocyte
    ref = Reference(adata, cell_type_col="cell_type")
    assert list(ref.cell_type_names) == ["adipocyte", "B cell", "T cell"]


def test_h5ad_roundtrip_with_digit_labels_gets_r_order(tmp_path):
    """anndata natsorts the categories it writes (Type_1 before Type_17); that order is not a
    deliberate one either, so a reference read back from .h5ad still gets R's order."""
    import anndata

    labels = ["Type_17_x"] * 30 + ["gamma_x"] * 30 + ["Type_1_x"] * 30
    adata = _labelled_reference(labels)
    adata.write_h5ad(tmp_path / "ref.h5ad")
    ref = Reference(anndata.read_h5ad(tmp_path / "ref.h5ad"), cell_type_col="cell_type")
    assert list(ref.cell_type_names) == ["gamma_x", "Type_1_x", "Type_17_x"]


def test_custom_categorical_order_is_kept():
    """A deliberate category order (e.g. an R factor's levels carried over) is kept."""
    import pandas as pd

    labels = ["T cell"] * 30 + ["adipocyte"] * 30 + ["B cell"] * 30
    adata = _labelled_reference(labels)
    adata.obs["cell_type"] = pd.Categorical(labels, categories=["T cell", "adipocyte", "B cell"])
    ref = Reference(adata, cell_type_col="cell_type")
    assert list(ref.cell_type_names) == ["T cell", "adipocyte", "B cell"]


def test_missing_labels_raise():
    """NaN / None labels are an error, not a silent drop or a 'None' cell type."""
    import pandas as pd

    labels = ["a"] * 30 + ["b"] * 30
    for missing in (None, float("nan")):
        adata = _labelled_reference(labels)
        lab = list(labels)
        lab[:5] = [missing] * 5
        adata.obs["cell_type"] = pd.Series(lab, index=adata.obs_names, dtype=object)
        with pytest.raises(ValueError, match="missing"):
            Reference(adata, cell_type_col="cell_type")


def test_dropped_cell_types_warn():
    """Types below cell_min are dropped (R would stop); the user is told which."""
    labels = ["a"] * 30 + ["b"] * 30 + ["rare"] * 5
    with pytest.warns(UserWarning, match="rare"):
        ref = Reference(_labelled_reference(labels), cell_type_col="cell_type")
    assert "rare" not in list(ref.cell_type_names)
