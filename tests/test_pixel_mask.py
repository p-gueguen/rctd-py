"""Tests for pixel_mask field in result types."""

import anndata
import numpy as np
import pytest

from rctd._reference import Reference
from rctd._rctd import run_rctd
from rctd._types import RCTDConfig


def _make_test_data(n_spatial=50, n_ref=80, n_genes=100, n_types=3, seed=42):
    """Create minimal spatial + reference data for testing."""
    rng = np.random.default_rng(seed)

    # Reference
    ref_counts = rng.poisson(5, (n_ref, n_genes)).astype(np.float32)
    ref_adata = anndata.AnnData(X=ref_counts)
    ref_adata.var_names = [f"Gene{i}" for i in range(n_genes)]
    ref_adata.obs["cell_type"] = rng.choice(
        [f"Type{i}" for i in range(n_types)], n_ref
    )
    ref = Reference(ref_adata, cell_min=2)

    # Spatial — include some low-UMI pixels that will be filtered
    spatial_counts = rng.poisson(10, (n_spatial, n_genes)).astype(np.float32)
    # Make first 5 pixels very low UMI (will be filtered)
    spatial_counts[:5] = rng.poisson(0.5, (5, n_genes)).astype(np.float32)
    spatial = anndata.AnnData(X=spatial_counts)
    spatial.var_names = [f"Gene{i}" for i in range(n_genes)]

    return spatial, ref


class TestPixelMaskFull:
    """Test pixel_mask in full mode results."""

    def test_pixel_mask_exists(self):
        spatial, ref = _make_test_data()
        config = RCTDConfig(UMI_min=10, compile=False)
        result = run_rctd(spatial, ref, mode="full", config=config)

        assert hasattr(result, "pixel_mask")
        assert result.pixel_mask is not None

    def test_pixel_mask_dtype_and_shape(self):
        spatial, ref = _make_test_data()
        config = RCTDConfig(UMI_min=10, compile=False)
        result = run_rctd(spatial, ref, mode="full", config=config)

        assert result.pixel_mask.dtype == bool
        assert result.pixel_mask.shape == (spatial.n_obs,)

    def test_pixel_mask_consistency(self):
        """Sum of pixel_mask equals number of result rows."""
        spatial, ref = _make_test_data()
        config = RCTDConfig(UMI_min=10, compile=False)
        result = run_rctd(spatial, ref, mode="full", config=config)

        assert result.pixel_mask.sum() == result.weights.shape[0]

    def test_pixel_mask_filters_pixels(self):
        """pixel_mask has fewer True entries than total pixels when filtering occurs."""
        spatial, ref = _make_test_data()
        # High UMI_min to ensure some filtering
        config = RCTDConfig(UMI_min=50, compile=False)
        result = run_rctd(spatial, ref, mode="full", config=config)

        assert result.pixel_mask.sum() < spatial.n_obs

    def test_pixel_mask_barcode_mapping(self):
        """pixel_mask can map results back to original barcodes."""
        spatial, ref = _make_test_data()
        config = RCTDConfig(UMI_min=10, compile=False)
        result = run_rctd(spatial, ref, mode="full", config=config)

        selected = spatial.obs_names[result.pixel_mask]
        assert len(selected) == result.weights.shape[0]


class TestPixelMaskDoublet:
    """Test pixel_mask in doublet mode results."""

    def test_pixel_mask_doublet(self):
        spatial, ref = _make_test_data()
        config = RCTDConfig(UMI_min=10, compile=False)
        result = run_rctd(spatial, ref, mode="doublet", config=config)

        assert result.pixel_mask is not None
        assert result.pixel_mask.dtype == bool
        assert result.pixel_mask.sum() == result.weights.shape[0]


class TestPixelMaskMulti:
    """Test pixel_mask in multi mode results."""

    def test_pixel_mask_multi(self):
        spatial, ref = _make_test_data()
        config = RCTDConfig(UMI_min=10, compile=False)
        result = run_rctd(spatial, ref, mode="multi", config=config)

        assert result.pixel_mask is not None
        assert result.pixel_mask.dtype == bool
        assert result.pixel_mask.sum() == result.weights.shape[0]


class TestPixelMaskBackwardCompat:
    """Test backward compatibility — pixel_mask defaults to None."""

    def test_full_result_default_none(self):
        from rctd._types import FullResult

        result = FullResult(
            weights=np.zeros((5, 3)),
            cell_type_names=["A", "B", "C"],
            converged=np.ones(5, dtype=bool),
        )
        assert result.pixel_mask is None

    def test_doublet_result_default_none(self):
        from rctd._types import DoubletResult

        result = DoubletResult(
            weights=np.zeros((5, 3)),
            weights_doublet=np.zeros((5, 2)),
            spot_class=np.zeros(5, dtype=int),
            first_type=np.zeros(5, dtype=int),
            second_type=np.zeros(5, dtype=int),
            first_class=np.zeros(5, dtype=bool),
            second_class=np.zeros(5, dtype=bool),
            min_score=np.zeros(5),
            singlet_score=np.zeros(5),
            cell_type_names=["A", "B", "C"],
        )
        assert result.pixel_mask is None
