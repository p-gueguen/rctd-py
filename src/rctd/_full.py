from typing import List

import numpy as np
import torch

from rctd._irwls import solve_irwls_batch_shared
from rctd._types import FullResult, resolve_device


def run_full_mode(
    spatial_counts: np.ndarray,
    spatial_numi: np.ndarray,
    norm_profiles: np.ndarray,
    cell_type_names: List[str],
    q_mat: np.ndarray,
    sq_mat: np.ndarray,
    x_vals: np.ndarray,
    batch_size: int = 10000,
    device: str = "auto",
    pixel_mask: np.ndarray | None = None,
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
    device = resolve_device(device)

    P_gpu = torch.tensor(norm_profiles, device=device)
    Q_gpu = torch.tensor(q_mat, device=device)
    SQ_gpu = torch.tensor(sq_mat, device=device)
    X_gpu = torch.tensor(x_vals, device=device)

    # Pre-transfer all spatial data to GPU at once (avoids per-batch numpy→CUDA copy)
    Y_gpu = torch.tensor(spatial_counts, device=device)
    nUMI_gpu = torch.tensor(spatial_numi, device=device)

    all_weights = []
    all_converged = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        # R's full-mode decomposition uses constrain=FALSE:
        # - Full mode (fitPixels "full"): constrain=FALSE
        # - Doublet mode initial full fit: constrain=FALSE
        # Weights are NOT constrained to simplex; they can be negative.
        weights, converged = solve_irwls_batch_shared(
            P=P_gpu,
            Y_batch=Y_gpu[start:end],
            nUMI_batch=nUMI_gpu[start:end],
            Q_mat=Q_gpu,
            SQ_mat=SQ_gpu,
            x_vals=X_gpu,
            max_iter=50,
            min_change=0.001,
            constrain=False,
            bulk_mode=False,
        )

        all_weights.append(weights.cpu().numpy())
        all_converged.append(converged.cpu().numpy())

    final_weights = np.vstack(all_weights)
    final_converged = np.concatenate(all_converged)

    return FullResult(
        weights=final_weights,
        cell_type_names=cell_type_names,
        converged=final_converged,
        pixel_mask=pixel_mask,
    )
