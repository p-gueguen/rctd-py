"""Shared test fixtures for rctd tests."""

import anndata
import numpy as np
import pytest


def _make_synthetic_reference(n_genes=200, n_cells=500, n_types=5, seed=42):
    """Create synthetic scRNA-seq reference with known cell type profiles."""
    rng = np.random.default_rng(seed)
    cell_type_names = [f"Type_{i}" for i in range(n_types)]
    cells_per_type = n_cells // n_types

    # Create distinct profiles: each type has ~20 marker genes with 10x expression
    profiles = rng.exponential(0.001, size=(n_genes, n_types))
    markers_per_type = n_genes // n_types
    for k in range(n_types):
        start = k * markers_per_type
        end = start + markers_per_type
        profiles[start:end, k] *= 10.0
    profiles = profiles / profiles.sum(axis=0, keepdims=True)

    # Generate counts from profiles
    counts = np.zeros((n_genes, n_cells), dtype=np.float32)
    cell_types = []
    nUMI = rng.integers(500, 5000, size=n_cells).astype(np.float32)
    for k in range(n_types):
        for c in range(cells_per_type):
            idx = k * cells_per_type + c
            lam = profiles[:, k] * nUMI[idx]
            counts[:, idx] = rng.poisson(lam).astype(np.float32)
            cell_types.append(cell_type_names[k])

    adata = anndata.AnnData(
        X=counts.T,  # cells x genes
        obs={"cell_type": cell_types},
    )
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
    return adata, profiles, cell_type_names


def _make_synthetic_spatial(profiles, n_pixels=100, n_types=5, seed=123):
    """Create synthetic spatial data with known mixtures."""
    rng = np.random.default_rng(seed)
    n_genes = profiles.shape[0]

    # Generate random mixtures (weights on simplex)
    raw_weights = rng.dirichlet(np.ones(n_types) * 0.5, size=n_pixels)
    true_weights = raw_weights.astype(np.float32)

    nUMI = rng.integers(200, 3000, size=n_pixels).astype(np.float32)
    counts = np.zeros((n_genes, n_pixels), dtype=np.float32)
    for i in range(n_pixels):
        lam = (profiles @ true_weights[i]) * nUMI[i]
        counts[:, i] = rng.poisson(lam).astype(np.float32)

    coords = rng.uniform(0, 100, size=(n_pixels, 2))
    adata = anndata.AnnData(
        X=counts.T,  # pixels x genes
        obs={"x": coords[:, 0], "y": coords[:, 1]},
    )
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Pixel_{i}" for i in range(n_pixels)]
    return adata, true_weights


@pytest.fixture
def synthetic_reference():
    return _make_synthetic_reference()


@pytest.fixture
def synthetic_spatial(synthetic_reference):
    _, profiles, _ = synthetic_reference
    return _make_synthetic_spatial(profiles)


@pytest.fixture
def synthetic_data(synthetic_reference, synthetic_spatial):
    ref_adata, profiles, cell_type_names = synthetic_reference
    spatial_adata, true_weights = synthetic_spatial
    return {
        "reference": ref_adata,
        "spatial": spatial_adata,
        "profiles": profiles,
        "true_weights": true_weights,
        "cell_type_names": cell_type_names,
    }
