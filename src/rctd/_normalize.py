from typing import Tuple

import torch

from rctd._irwls import solve_irwls


def fit_bulk(
    cell_type_profiles: torch.Tensor,
    spatial_counts: torch.Tensor,
    spatial_nUMI: torch.Tensor,
    min_change: float = 0.001,
    max_iter: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate bulk cell type composition and compute normalized reference profiles.

    Ports fitBulk from spacexr platform_effect_normalization.R.

    Args:
        cell_type_profiles: (G, K) array of mean gene expression per cell type
        spatial_counts: (N, G) array of observed spatial counts
        spatial_nUMI: (N,) array of total UMI per pixel
        min_change: convergence tolerance
        max_iter: max iterations for IRWLS

    Returns:
        bulk_weights: (K,) array of estimated cell type proportions
        norm_profiles: (G, K) array of platform-effect normalized reference profiles
    """
    device = cell_type_profiles.device
    dtype = cell_type_profiles.dtype

    # Sum counts across all pixels (N, G) -> (G,)
    bulk_Y = torch.sum(spatial_counts, dim=0)

    # Total UMI across all pixels
    bulk_nUMI = torch.sum(spatial_nUMI)

    # Scale reference profiles to bulk nUMI
    bulk_S = cell_type_profiles * bulk_nUMI

    # Note: in R's fitBulk, bulkData is fit using the gaussian approximation method (bulk_mode=True)
    # and constrain=False. For bulk_mode=True, we don't need Q_mat etc.
    # We pass zeros/dummy values for the likelihood tables since they are ignored in bulk_mode
    dummy_Q = torch.zeros((1, 1), dtype=dtype, device=device)
    dummy_SQ = torch.zeros((1, 1), dtype=dtype, device=device)
    dummy_x = torch.zeros((1,), dtype=dtype, device=device)

    bulk_weights, converged = solve_irwls(
        S=bulk_S,
        Y=bulk_Y,
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
