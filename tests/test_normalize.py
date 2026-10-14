import numpy as np
import torch

from rctd._normalize import fit_bulk


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
