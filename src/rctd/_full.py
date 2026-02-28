from typing import List

import jax.numpy as jnp
import numpy as np

from rctd._irwls import solve_irwls_batch
from rctd._types import FullResult


def run_full_mode(
    spatial_counts: np.ndarray,
    spatial_numi: np.ndarray,
    norm_profiles: np.ndarray,
    cell_type_names: List[str],
    q_mat: np.ndarray,
    sq_mat: np.ndarray,
    x_vals: np.ndarray,
    batch_size: int = 10000,
) -> FullResult:
    """Run full mode deconvolution across all spatial pixels.

    Args:
        spatial_counts: (N, G) observed count matrix
        spatial_numi: (N,) total UMI per pixel
        norm_profiles: (G, K) normalized reference profiles
        cell_type_names: List of length K
        q_mat: Likelihood table for selected sigma
        sq_mat: Likelihood spline table for selected sigma
        x_vals: Lambda grid points
        batch_size: Number of pixels to process simultaneously on GPU

    Returns:
        FullResult object containing weights per pixel.
    """
    N, G = spatial_counts.shape
    norm_profiles.shape[1]

    P_gpu = jnp.array(norm_profiles)
    Q_gpu = jnp.array(q_mat)
    SQ_gpu = jnp.array(sq_mat)
    X_gpu = jnp.array(x_vals)

    all_weights = []
    all_converged = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        batch_counts = jnp.array(spatial_counts[start:end])
        batch_numi = jnp.array(spatial_numi[start:end])

        # S_batch: (bs, G, K)
        S_batch = batch_numi[:, None, None] * P_gpu[None, :, :]

        # R's full-mode decomposition uses constrain=FALSE:
        # - Full mode (fitPixels "full"): constrain=FALSE
        # - Doublet mode initial full fit: constrain=FALSE
        # Weights are NOT constrained to simplex; they can be negative.
        weights, converged = solve_irwls_batch(
            S_batch=S_batch,
            Y_batch=batch_counts,
            nUMI_batch=batch_numi,
            Q_mat=Q_gpu,
            SQ_mat=SQ_gpu,
            x_vals=X_gpu,
            max_iter=50,
            min_change=0.001,
            constrain=False,
            bulk_mode=False,
        )

        # Get host numpy arrays
        weights_np = np.array(weights)
        converged_np = np.array(converged)

        all_weights.append(weights_np)
        all_converged.append(converged_np)

    final_weights = np.vstack(all_weights)
    final_converged = np.concatenate(all_converged)

    return FullResult(
        weights=final_weights,
        cell_type_names=cell_type_names,
        converged=final_converged,
    )
