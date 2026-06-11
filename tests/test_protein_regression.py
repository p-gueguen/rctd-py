"""Bit-identity firewall: protein absent OR protein_weight=0.0 must reproduce
the RNA-only path byte-for-byte.

This is the regression guarantee that every protein stage must preserve. It uses
``np.testing.assert_array_equal`` (exact, not ``allclose``): the RNA-only branch is
literally the current code, so there must be no ``+0.0`` perturbation or reordered
reduction when protein is off.
"""

import numpy as np
import pytest

from rctd import RCTDConfig, Reference, run_rctd


def _baseline(spatial, reference, mode, **cfg):
    config = RCTDConfig(compile=False, **cfg)
    return run_rctd(spatial, reference, mode=mode, config=config, batch_size=10)


@pytest.mark.protein
@pytest.mark.parametrize("mode", ["full", "doublet"])
def test_protein_weight_zero_matches_rna_only(synthetic_data, mode):
    """obsm['protein'] present but protein_weight=0.0 => identical to RNA-only."""
    ref_adata = synthetic_data["reference"]
    spatial = synthetic_data["spatial"]
    reference = Reference(ref_adata, cell_min=10, min_UMI=10)

    base = _baseline(spatial, reference, mode)

    # Attach protein, but the master switch is off -> obsm must never be read.
    spatial_p = spatial.copy()
    rng = np.random.default_rng(0)
    spatial_p.obsm["protein"] = rng.standard_normal((spatial_p.n_obs, 15)).astype("f4")
    reference_p = Reference(ref_adata, cell_min=10, min_UMI=10)
    off = _baseline(spatial_p, reference_p, mode, protein_weight=0.0)

    np.testing.assert_array_equal(base.weights, off.weights)
    if mode == "doublet":
        np.testing.assert_array_equal(base.weights_doublet, off.weights_doublet)
        np.testing.assert_array_equal(base.spot_class, off.spot_class)
        np.testing.assert_array_equal(base.first_type, off.first_type)
        np.testing.assert_array_equal(base.second_type, off.second_type)


@pytest.mark.protein
@pytest.mark.parametrize("mode", ["full", "doublet"])
def test_protein_absent_with_weight_set_matches_rna_only(synthetic_data, mode):
    """No obsm['protein'] at all, but protein_weight>0 => still RNA-only (protein absent)."""
    ref_adata = synthetic_data["reference"]
    spatial = synthetic_data["spatial"]

    base = _baseline(spatial, Reference(ref_adata, cell_min=10, min_UMI=10), mode)
    # protein_weight set, but no protein matrix supplied -> nothing to fuse.
    other = _baseline(
        spatial, Reference(ref_adata, cell_min=10, min_UMI=10), mode, protein_weight=0.5
    )

    np.testing.assert_array_equal(base.weights, other.weights)
    if mode == "doublet":
        np.testing.assert_array_equal(base.weights_doublet, other.weights_doublet)
        np.testing.assert_array_equal(base.spot_class, other.spot_class)
