"""Tests for counts_MIN pixel filtering.

R spacexr's restrict_counts() filters pixels whose total counts across the
selected DE gene list fall below counts_MIN. This filter is applied AFTER
gene list selection in create.RCTD(), meaning a pixel can pass the UMI_min
filter (high total UMI) but be removed if it has very few counts in the
DE gene set.

rctd-py must replicate this behavior.
"""

import anndata
import numpy as np

from rctd._rctd import RCTD
from rctd._reference import Reference
from rctd._types import RCTDConfig


def _make_counts_min_test_data(seed=42):
    """Create data where some pixels pass UMI filter but fail counts_MIN on DE genes.

    Strategy:
    - 200 genes total, shared between spatial and reference
    - Reference has 5 cell types with clear marker genes (genes 0-99)
    - Spatial has 50 pixels:
      - 40 'normal' pixels: spread counts across all genes (pass both filters)
      - 10 'sparse-DE' pixels: high total UMI (>= 200) but nearly all counts
        in genes 100-199 (non-marker genes), so counts in DE genes < counts_MIN
    """
    rng = np.random.default_rng(seed)
    n_genes = 200
    n_types = 5
    n_pixels = 50
    n_normal = 40
    n_sparse = 10

    gene_names = [f"Gene_{i}" for i in range(n_genes)]

    # ── Reference: 5 types, 300 cells, markers in genes 0-99 ──
    n_ref_cells = 300
    cells_per_type = n_ref_cells // n_types
    ref_counts = np.zeros((n_ref_cells, n_genes), dtype=np.float32)
    cell_types = []
    for k in range(n_types):
        for c in range(cells_per_type):
            idx = k * cells_per_type + c
            # Marker genes for this type (genes k*20 to k*20+19)
            marker_start = k * 20
            ref_counts[idx, marker_start : marker_start + 20] = rng.poisson(50, size=20)
            # Background across all genes
            ref_counts[idx, :] += rng.poisson(1, size=n_genes)
            cell_types.append(f"Type_{k}")

    ref_adata = anndata.AnnData(
        X=ref_counts,
        obs={"cell_type": cell_types},
    )
    ref_adata.var_names = gene_names
    ref_adata.obs_names = [f"RefCell_{i}" for i in range(n_ref_cells)]

    # ── Spatial: 40 normal + 10 sparse-DE pixels ──
    spatial_counts = np.zeros((n_pixels, n_genes), dtype=np.float32)

    # Normal pixels: counts spread across genes including markers
    for i in range(n_normal):
        spatial_counts[i, :100] = rng.poisson(5, size=100)  # marker region
        spatial_counts[i, 100:] = rng.poisson(3, size=100)  # non-marker region

    # Sparse-DE pixels: high total UMI but almost all in non-marker genes
    for i in range(n_normal, n_pixels):
        spatial_counts[i, :100] = rng.poisson(0.05, size=100)  # ~5 total in DE genes
        spatial_counts[i, 100:] = rng.poisson(10, size=100)  # ~1000 total in non-DE

    spatial_adata = anndata.AnnData(
        X=spatial_counts,
        obs={"x": rng.uniform(0, 100, n_pixels), "y": rng.uniform(0, 100, n_pixels)},
    )
    spatial_adata.var_names = gene_names
    spatial_adata.obs_names = [f"Pixel_{i}" for i in range(n_pixels)]

    return spatial_adata, ref_adata, n_normal, n_sparse


class TestCountsMinFilter:
    """Test that counts_MIN filters pixels with low counts in the DE gene set."""

    def test_counts_min_reduces_pixel_count(self):
        """Pixels with high total UMI but low counts in DE genes should be filtered."""
        spatial, ref_adata, n_normal, n_sparse = _make_counts_min_test_data()

        reference = Reference(ref_adata, cell_min=10, min_UMI=10)

        # With counts_MIN=10 (default), sparse-DE pixels should be filtered
        config = RCTDConfig(UMI_min=100, counts_MIN=10)
        rctd = RCTD(spatial, reference, config)
        rctd.fit_platform_effects()

        # The sparse-DE pixels have ~5 total counts in the DE gene region,
        # so they should fail the counts_MIN=10 filter.
        # Total pixels passing UMI filter first:
        total_umi = spatial.X.sum(axis=1)
        n_pass_umi = int((total_umi >= 100).sum())

        # After counts_MIN, we should have fewer pixels
        n_after_counts_min = rctd.counts.shape[0]
        assert n_after_counts_min < n_pass_umi, (
            f"counts_MIN filter had no effect: {n_after_counts_min} pixels "
            f"(same as UMI-only filter: {n_pass_umi}). "
            f"Expected some sparse-DE pixels to be removed."
        )

    def test_counts_min_zero_disables_filter(self):
        """Setting counts_MIN=0 should keep all UMI-passing pixels."""
        spatial, ref_adata, _, _ = _make_counts_min_test_data()

        reference = Reference(ref_adata, cell_min=10, min_UMI=10)

        # counts_MIN=0 should disable the gene-list filter
        config_no_filter = RCTDConfig(UMI_min=100, counts_MIN=0)
        rctd_no = RCTD(spatial, reference, config_no_filter)
        rctd_no.fit_platform_effects()

        total_umi = spatial.X.sum(axis=1)
        n_pass_umi = int((total_umi >= 100).sum())

        assert rctd_no.counts.shape[0] == n_pass_umi, (
            f"counts_MIN=0 should keep all UMI-passing pixels, "
            f"got {rctd_no.counts.shape[0]} vs {n_pass_umi}"
        )

    def test_pixel_mask_updated_after_counts_min(self):
        """_pixel_mask should reflect counts_MIN filtering for CLI output alignment."""
        spatial, ref_adata, _, _ = _make_counts_min_test_data()

        reference = Reference(ref_adata, cell_min=10, min_UMI=10)
        config = RCTDConfig(UMI_min=100, counts_MIN=10)
        rctd = RCTD(spatial, reference, config)
        rctd.fit_platform_effects()

        # _pixel_mask should have exactly as many True values as pixels in counts
        assert rctd._pixel_mask.sum() == rctd.counts.shape[0], (
            f"_pixel_mask ({rctd._pixel_mask.sum()} True) doesn't match "
            f"counts shape ({rctd.counts.shape[0]} pixels)"
        )

        # Mask should be boolean and same length as original spatial data
        assert rctd._pixel_mask.dtype == bool
        assert len(rctd._pixel_mask) == spatial.n_obs

    def test_counts_min_1_keeps_all_nonzero(self):
        """counts_MIN=1 should only remove pixels with zero counts in DE genes."""
        spatial, ref_adata, _, _ = _make_counts_min_test_data()

        reference = Reference(ref_adata, cell_min=10, min_UMI=10)

        config_1 = RCTDConfig(UMI_min=100, counts_MIN=1)
        rctd_1 = RCTD(spatial, reference, config_1)
        rctd_1.fit_platform_effects()

        config_0 = RCTDConfig(UMI_min=100, counts_MIN=0)
        rctd_0 = RCTD(spatial, reference, config_0)
        rctd_0.fit_platform_effects()

        # counts_MIN=1 should keep almost all pixels (very few have exactly 0
        # counts across all DE genes), so it's >= counts_MIN=10 result
        assert rctd_1.counts.shape[0] >= rctd_0.counts.shape[0] - 1
        assert rctd_1.counts.shape[0] <= rctd_0.counts.shape[0]

    def test_high_counts_min_filters_more(self):
        """Higher counts_MIN should filter more pixels."""
        spatial, ref_adata, _, _ = _make_counts_min_test_data()

        reference = Reference(ref_adata, cell_min=10, min_UMI=10)

        config_10 = RCTDConfig(UMI_min=100, counts_MIN=10)
        rctd_10 = RCTD(spatial, reference, config_10)
        rctd_10.fit_platform_effects()

        config_50 = RCTDConfig(UMI_min=100, counts_MIN=50)
        rctd_50 = RCTD(spatial, reference, config_50)
        rctd_50.fit_platform_effects()

        # Higher threshold should keep fewer or equal pixels
        assert rctd_50.counts.shape[0] <= rctd_10.counts.shape[0]

    def test_numi_not_recomputed_after_filter(self):
        """nUMI should reflect total UMI from ALL genes, not just DE genes."""
        spatial, ref_adata, _, _ = _make_counts_min_test_data()

        reference = Reference(ref_adata, cell_min=10, min_UMI=10)
        config = RCTDConfig(UMI_min=100, counts_MIN=10)
        rctd = RCTD(spatial, reference, config)
        rctd.fit_platform_effects()

        # nUMI should be total across all genes for each remaining pixel
        total_umi_all_genes = np.array(spatial.X.sum(axis=1)).flatten()
        kept_umi = total_umi_all_genes[rctd._pixel_mask]
        np.testing.assert_array_almost_equal(
            rctd.nUMI,
            kept_umi,
            err_msg="nUMI was recomputed after gene restriction (should stay as total UMI)",
        )

    def test_run_rctd_with_counts_min(self):
        """run_rctd() convenience function should also apply counts_MIN."""
        from rctd._rctd import run_rctd

        spatial, ref_adata, n_normal, n_sparse = _make_counts_min_test_data()
        reference = Reference(ref_adata, cell_min=10, min_UMI=10)

        # counts_MIN=0: should get more pixels
        config_0 = RCTDConfig(UMI_min=100, counts_MIN=0)
        res_0 = run_rctd(spatial, reference, mode="full", config=config_0, batch_size=10)
        n_0 = res_0.weights.shape[0]

        # counts_MIN=10: should get fewer pixels
        config_10 = RCTDConfig(UMI_min=100, counts_MIN=10)
        res_10 = run_rctd(spatial, reference, mode="full", config=config_10, batch_size=10)
        n_10 = res_10.weights.shape[0]

        assert n_10 < n_0, (
            f"run_rctd with counts_MIN=10 ({n_10} pixels) should have fewer pixels "
            f"than counts_MIN=0 ({n_0} pixels)"
        )

    def test_counts_min_with_doublet_mode(self):
        """counts_MIN should work correctly with doublet mode too."""
        from rctd._rctd import run_rctd
        from rctd._types import DoubletResult

        spatial, ref_adata, _, _ = _make_counts_min_test_data()
        reference = Reference(ref_adata, cell_min=10, min_UMI=10)

        config = RCTDConfig(UMI_min=100, counts_MIN=10)
        res = run_rctd(spatial, reference, mode="doublet", config=config, batch_size=10)

        assert isinstance(res, DoubletResult)
        # All result arrays should have same first dimension
        assert res.weights.shape[0] == res.spot_class.shape[0]
        assert res.weights.shape[0] == res.first_type.shape[0]

    def test_cli_output_alignment_after_counts_min(self):
        """CLI _write_results_to_adata should correctly expand filtered results."""
        from rctd._rctd import RCTD
        from rctd._types import RCTDConfig
        from rctd.cli import _write_results_to_adata

        spatial, ref_adata, _, _ = _make_counts_min_test_data()
        reference = Reference(ref_adata, cell_min=10, min_UMI=10)

        config = RCTDConfig(UMI_min=100, counts_MIN=10)
        rctd = RCTD(spatial, reference, config)
        rctd.fit_platform_effects()

        from rctd._full import run_full_mode

        result = run_full_mode(
            spatial_counts=rctd.counts,
            spatial_numi=rctd.nUMI,
            norm_profiles=rctd.norm_profiles,
            cell_type_names=rctd.reference.cell_type_names,
            q_mat=rctd.q_mat,
            sq_mat=rctd.sq_mat,
            x_vals=rctd.x_vals,
            batch_size=10,
        )

        out_adata = _write_results_to_adata(
            spatial,
            result,
            "full",
            rctd._pixel_mask,
            config._asdict(),
            rctd.reference.cell_type_names,
            "test",
        )

        # Output should have same number of obs as input
        assert out_adata.n_obs == spatial.n_obs

        # Filtered pixels should have NaN weights
        n_filtered = (~rctd._pixel_mask).sum()
        nan_rows = np.isnan(out_adata.obsm["rctd_weights"]).all(axis=1)
        assert nan_rows.sum() == n_filtered, f"Expected {n_filtered} NaN rows, got {nan_rows.sum()}"

        # Non-filtered pixels should have valid weights
        valid_rows = ~nan_rows
        assert valid_rows.sum() == rctd.counts.shape[0]
        assert not np.isnan(out_adata.obsm["rctd_weights"][valid_rows]).any()
