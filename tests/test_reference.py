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
        """Mitochondrial genes (mt-*) should be excluded from DE genes."""
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

        adata = anndata.AnnData(X=counts, obs={"cell_type": cell_types})
        adata.var_names = gene_names
        adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

        ref = Reference(adata, cell_type_col="cell_type")
        de_genes = ref.get_de_genes(fc_thresh=0.5, expr_thresh=1e-4)

        assert "mt-Co1" not in de_genes
        assert "MT-ND1" not in de_genes

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
