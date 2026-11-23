from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch

from rctd._irwls import solve_irwls

# Rows per longdouble chunk in _longdouble_sum: bounds the temporary copy of a sparse matrix
_SPARSE_SUM_ROWS = 65536


def _longdouble_sum(t: torch.Tensor | sp.spmatrix, dim: int) -> torch.Tensor:
    """Sum a tensor along a dimension using longdouble accumulation.

    On x86-64, this uses 80-bit extended precision (18 significant digits),
    matching R's internal sum() which also uses long double on this platform.
    This eliminates BLAS-order-dependent floating-point divergence between
    R and Python for large reductions (e.g. summing 135k spatial pixels).

    A scipy sparse (N, G) matrix is summed over its rows without densifying it, in row
    chunks cast to longdouble (scipy's ``sum(dtype=...)`` sets only the output type and still
    accumulates in the input dtype). On integer counts, both sums are exact, so they are equal.
    """
    if sp.issparse(t):
        if dim != 0:
            raise ValueError("a sparse matrix is summed over its rows only (dim=0)")
        t = sp.csr_matrix(t)
        result = np.zeros(t.shape[1], dtype=np.longdouble)
        for start in range(0, t.shape[0], _SPARSE_SUM_ROWS):
            chunk = t[start : start + _SPARSE_SUM_ROWS].astype(np.longdouble)
            result += np.asarray(chunk.sum(axis=0)).ravel()
        return torch.as_tensor(result.astype(np.float64).astype(t.dtype))
    arr = t.detach().cpu().numpy()
    # Accumulate in longdouble without first materializing an entire
    # longdouble copy of the input array.
    result = np.sum(arr, axis=dim, dtype=np.longdouble).astype(np.float64)

    return torch.as_tensor(result, dtype=t.dtype)


def fit_bulk(
    cell_type_profiles: torch.Tensor,
    spatial_counts: torch.Tensor | sp.spmatrix,
    spatial_nUMI: torch.Tensor,
    min_change: float = 0.001,
    max_iter: int = 100,
    min_obs: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate bulk cell type composition and compute normalized reference profiles.

    Ports fitBulk from spacexr platform_effect_normalization.R.

    Args:
        cell_type_profiles: (G, K) array of mean gene expression per cell type
        spatial_counts: (N, G) observed spatial counts, a tensor or a scipy sparse matrix
        spatial_nUMI: (N,) array of total UMI per pixel
        min_change: convergence tolerance
        max_iter: max iterations for IRWLS
        min_obs: genes with fewer total counts are left out of the decomposition
            (R: prepareBulkData MIN_OBS = 10) but are still renormalised

    Returns:
        Tuple ``(bulk_weights, norm_profiles)``: the (K,) estimated cell type proportions and the
        (G, K) platform-effect normalized reference profiles.
    """
    device = cell_type_profiles.device
    dtype = cell_type_profiles.dtype

    # Reduce the complete spatial inputs on CPU, keeping their original dtype.
    bulk_Y_cpu = _longdouble_sum(spatial_counts, dim=0)
    bulk_nUMI_cpu = _longdouble_sum(spatial_nUMI, dim=0)

    # Transfer only the reduced gene vector and scalar to the compute device.
    bulk_Y = bulk_Y_cpu.to(device=device)
    bulk_nUMI = bulk_nUMI_cpu.to(device=device)

    # Scale reference profiles to bulk nUMI.
    bulk_S = cell_type_profiles * bulk_nUMI
    # Note: in R's fitBulk, bulkData is fit using the gaussian approximation method (bulk_mode=True)
    # and constrain=False. For bulk_mode=True, we don't need Q_mat etc.
    # We pass zeros/dummy values for the likelihood tables since they are ignored in bulk_mode
    dummy_Q = torch.zeros((1, 1), dtype=dtype, device=device)
    dummy_SQ = torch.zeros((1, 1), dtype=dtype, device=device)
    dummy_x = torch.zeros((1,), dtype=dtype, device=device)

    # R's prepareBulkData decomposes only genes with >= MIN_OBS (10) total counts;
    # get_norm_ref below still renormalises every bulk gene.
    obs = bulk_Y >= min_obs
    bulk_weights, converged = solve_irwls(
        S=bulk_S[obs],
        Y=bulk_Y[obs],
        nUMI=bulk_nUMI,
        Q_mat=dummy_Q,
        SQ_mat=dummy_SQ,
        x_vals=dummy_x,
        max_iter=max_iter,
        min_change=min_change,
        constrain=False,
        bulk_mode=True,
    )

    # R's fitBulk uses constrain=False: only clip negatives, do NOT
    # normalize to sum=1.  The raw magnitude matters for get_norm_ref.
    bulk_weights = torch.clamp(bulk_weights, min=0.0)

    # Compute normalized reference profiles (get_norm_ref equivalent)
    # R source (processRef.R:52-57):
    #   weight_avg = rowSums(sweep(profiles[gene_list,], 2, proportions/sum(proportions), '*'))
    #   target_means = bulk_vec[gene_list] / sum(puck@nUMI)
    #   normed = sweep(profiles[gene_list,], 1, weight_avg / target_means, '/')
    #
    # Key: R normalizes proportions to sum=1 BEFORE computing weighted average.

    # Normalize proportions to sum to 1 (matching R's proportions/sum(proportions))
    prop_sum = torch.clamp(torch.sum(bulk_weights), min=1e-10)
    prop_normalized = bulk_weights / prop_sum

    # weight_avg: expected gene expression using normalized proportions (G,)
    weight_avg = cell_type_profiles @ prop_normalized

    # target_means: observed gene frequency = bulk_Y / total_nUMI (G,)
    target_means = bulk_Y / torch.clamp(bulk_nUMI, min=1e-10)

    # gene_factor = weight_avg / target_means (G,)
    gene_factor = weight_avg / torch.clamp(target_means, min=1e-10)

    # norm_profiles = profiles / gene_factor
    norm_profiles = cell_type_profiles / torch.clamp(gene_factor[:, None], min=1e-10)

    return bulk_weights, norm_profiles
