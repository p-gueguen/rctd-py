from typing import Tuple

import jax.numpy as jnp

from rctd._irwls import solve_irwls


def fit_bulk(
    cell_type_profiles: jnp.ndarray,
    spatial_counts: jnp.ndarray,
    spatial_nUMI: jnp.ndarray,
    min_change: float = 0.001,
    max_iter: int = 100,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    # Sum counts across all pixels (N, G) -> (G,)
    bulk_Y = jnp.sum(spatial_counts, axis=0)

    # Total UMI across all pixels
    bulk_nUMI = jnp.sum(spatial_nUMI)

    # Scale reference profiles to bulk nUMI
    bulk_S = cell_type_profiles * bulk_nUMI

    # Note: in R's fitBulk, bulkData is fit using the gaussian approximation method (bulk_mode=True)
    # and constrain=False. For bulk_mode=True, we don't need Q_mat etc.
    # We pass zeros/dummy values for the likelihood tables since they are ignored in bulk_mode
    dummy_Q = jnp.zeros((1, 1))
    dummy_SQ = jnp.zeros((1, 1))
    dummy_x = jnp.zeros((1,))

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
    bulk_weights = jnp.maximum(bulk_weights, 0.0)

    # Compute normalized reference profiles (get_norm_ref equivalent)
    # R source (processRef.R:52-57):
    #   weight_avg = rowSums(sweep(profiles[gene_list,], 2, proportions/sum(proportions), '*'))
    #   target_means = bulk_vec[gene_list] / sum(puck@nUMI)
    #   normed = sweep(profiles[gene_list,], 1, weight_avg / target_means, '/')
    #
    # Key: R normalizes proportions to sum=1 BEFORE computing weighted average.

    # Normalize proportions to sum to 1 (matching R's proportions/sum(proportions))
    prop_sum = jnp.maximum(jnp.sum(bulk_weights), 1e-10)
    prop_normalized = bulk_weights / prop_sum

    # weight_avg: expected gene expression using normalized proportions (G,)
    weight_avg = cell_type_profiles @ prop_normalized

    # target_means: observed gene frequency = bulk_Y / total_nUMI (G,)
    target_means = bulk_Y / jnp.maximum(bulk_nUMI, 1e-10)

    # gene_factor = weight_avg / target_means (G,)
    gene_factor = weight_avg / jnp.maximum(target_means, 1e-10)

    # norm_profiles = profiles / gene_factor
    norm_profiles = cell_type_profiles / jnp.maximum(gene_factor[:, None], 1e-10)

    return bulk_weights, norm_profiles
