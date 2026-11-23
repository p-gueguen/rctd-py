import numpy as np
import pytest
import scipy.sparse as sp
import torch

from rctd._normalize import _longdouble_sum, fit_bulk
from rctd._rctd import RCTD
from rctd._reference import Reference
from rctd._types import RCTDConfig


def test_fit_bulk_runs(synthetic_data):
    """Test that fit_bulk runs without errors and produces valid outputs."""
    profiles = synthetic_data["profiles"]  # (G, K)
    spatial_adata = synthetic_data["spatial"]  # (N, G)

    spatial_counts = spatial_adata.X
    spatial_numi = np.array(spatial_counts.sum(axis=1)).flatten()

    bulk_weights, norm_profiles = fit_bulk(
        cell_type_profiles=torch.tensor(profiles),
        spatial_counts=torch.tensor(spatial_counts),
        spatial_nUMI=torch.tensor(spatial_numi),
    )

    bulk_weights_np = bulk_weights.cpu().numpy()
    norm_profiles_np = norm_profiles.cpu().numpy()

    # Weights should be finite, non-negative (R does NOT normalize to sum=1)
    assert np.all(np.isfinite(bulk_weights_np))
    assert np.all(bulk_weights_np >= 0)
    assert np.sum(bulk_weights_np) > 0

    # Normalized profiles should have the same shape
    assert norm_profiles_np.shape == profiles.shape
    assert np.all(np.isfinite(norm_profiles_np))
    assert np.all(norm_profiles_np >= 0)


def test_fit_bulk_drops_genes_below_min_obs(synthetic_data):
    """R's prepareBulkData decomposes the bulk only on genes with >= 10 total counts
    (MIN_OBS = 10), while get_norm_ref still renormalises every bulk gene. Genes below
    the cutoff must not move the bulk proportions. Missing this filter cost up to 7
    points of bulk proportion on sparse sequencing data (StrataMap, Visium)."""
    profiles = np.asarray(synthetic_data["profiles"], dtype=np.float64)  # (G, K)
    counts = np.asarray(
        synthetic_data["spatial"].X.todense()
        if hasattr(synthetic_data["spatial"].X, "todense")
        else synthetic_data["spatial"].X,
        dtype=np.float64,
    )
    numi = counts.sum(axis=1)

    # Ten extra genes that one cell type expresses strongly in the reference but that
    # are barely observed in the tissue (bulk total 0..9 counts): R ignores them.
    rng = np.random.default_rng(0)
    G, K = profiles.shape
    low_prof = np.full((10, K), 1e-5)
    low_prof[:, 0] = 0.01
    low_counts = np.zeros((counts.shape[0], 10))
    for j in range(10):
        low_counts[rng.choice(counts.shape[0], j, replace=False), j] = 1  # gene j: j counts total
    prof_all = np.vstack([profiles, low_prof])
    counts_all = np.hstack([counts, low_counts])

    w_all, norm_all = fit_bulk(torch.tensor(prof_all), torch.tensor(counts_all), torch.tensor(numi))
    w_obs, norm_obs = fit_bulk(torch.tensor(profiles), torch.tensor(counts), torch.tensor(numi))

    np.testing.assert_allclose(w_all.numpy(), w_obs.numpy(), rtol=1e-10, atol=1e-12)
    assert norm_all.shape == (G + 10, K)  # renormalised profiles still cover every gene


def test_fit_bulk_sparse_counts_match_dense(synthetic_data):
    profiles = torch.tensor(synthetic_data["profiles"])
    counts = np.asarray(synthetic_data["spatial"].X, dtype=np.float32)
    numi = torch.tensor(counts.sum(axis=1))

    w_dense, norm_dense = fit_bulk(profiles, torch.tensor(counts), numi)
    w_sparse, norm_sparse = fit_bulk(profiles, sp.csr_matrix(counts), numi)

    assert torch.equal(w_dense, w_sparse)
    assert torch.equal(norm_dense, norm_sparse)


def test_longdouble_sum_sparse_exact_above_float32_range():
    """Gene totals above 2**24 are not exact in float32; the sparse sum must still match."""
    counts = np.zeros((1_000_000, 2), dtype=np.float32)
    counts[:, 0] = 25  # total 25,000,000 > 2**24
    counts[::3, 1] = np.random.default_rng(0).poisson(40, len(counts[::3]))
    dense = _longdouble_sum(torch.tensor(counts), dim=0)
    for sparse in (sp.csr_matrix(counts), sp.csc_array(counts)):
        assert torch.equal(_longdouble_sum(sparse, dim=0), dense)
    assert dense[0].item() == 25_000_000


@pytest.mark.parametrize("counts_min", [0, 1500])
def test_platform_effects_sparse_spatial_match_dense(synthetic_data, counts_min):
    """Sparse spatial counts stay sparse until the DE genes, with identical results."""
    reference = Reference(synthetic_data["reference"])
    config = RCTDConfig(UMI_min=100, counts_MIN=counts_min)
    dense = synthetic_data["spatial"]
    sparse = dense.copy()
    sparse.X = sp.csr_matrix(dense.X)

    fits = []
    for spatial in (dense, sparse):
        rctd = RCTD(spatial, reference, config)
        assert sp.issparse(rctd.counts) == (spatial is sparse)
        rctd.fit_platform_effects()
        fits.append(rctd)

    a, b = fits
    assert isinstance(b.counts, np.ndarray) and b.counts.dtype == a.counts.dtype
    if counts_min:
        assert 0 < a.counts.shape[0] < dense.n_obs  # the counts_MIN filter removed pixels
    np.testing.assert_array_equal(a.counts, b.counts)
    np.testing.assert_array_equal(a.nUMI, b.nUMI)
    np.testing.assert_array_equal(a._pixel_mask, b._pixel_mask)
    np.testing.assert_array_equal(a.norm_profiles, b.norm_profiles)
    assert a.sigma == b.sigma
