from typing import Dict

import jax.numpy as jnp
import numpy as np

from rctd._irwls import solve_irwls_batch
from rctd._likelihood import calc_log_likelihood

# The sequence of sigmas evaluated in spacexr choose_sigma
SIGMA_ALL = np.concatenate([np.arange(10, 71), np.arange(72, 201, 2)])


def choose_sigma(
    spatial_counts: np.ndarray,
    spatial_numi: np.ndarray,
    norm_profiles: np.ndarray,
    q_matrices: Dict[str, np.ndarray],
    x_vals: np.ndarray,
    sigma_init: int = 100,
    min_umi: int = 100,
    n_fit: int = 100,
    n_epoch: int = 8,
    k_val: int = 1000,
    seed: int = 42,
) -> int:
    """Estimate the optimal sigma parameter for the Poisson-Lognormal model.

    Exactly ports choose_sigma_c / chooseSigma from spacexr R:
    - Each epoch: fit IRWLS weights at *current* sigma (fixed Q_mat).
    - Then call chooseSigma: evaluate all candidate sigmas on the FIXED
      prediction (weights x profiles x nUMI), pick the sigma that minimizes
      the total negative log-likelihood.
    - Stop when sigma does not change.

    Args:
        spatial_counts: (N, G) array of observed spatial counts
        spatial_numi: (N,) array of total UMI per pixel
        norm_profiles: (G, K) array of platform-effect normalized reference profiles
        q_matrices: dictionary mapping sigma-as-string (e.g. "100") to Q_mat arrays
        x_vals: (N_X,) array of lambda grid points
        sigma_init: initial sigma value (default: 100)
        min_umi: minimum UMI count for a pixel to be used in fitting
        n_fit: number of pixels to sample for fitting
        n_epoch: maximum number of iterations
        k_val: max count value for likelihood
        seed: random seed for sampling
    """
    from rctd._likelihood import compute_spline_coefficients

    rng = np.random.default_rng(seed)

    # Filter pixels by min_umi (R: puck@nUMI > MIN_UMI)
    valid_idx = np.where(spatial_numi > min_umi)[0]
    if len(valid_idx) == 0:
        raise ValueError(f"No pixels found with UMI > {min_umi}. Try decreasing min_umi.")

    n_samples = min(n_fit, len(valid_idx))
    fit_idx = rng.choice(valid_idx, size=n_samples, replace=False)

    fit_counts = jnp.array(spatial_counts[fit_idx])  # (n_samples, G)
    fit_numi = jnp.array(spatial_numi[fit_idx])  # (n_samples,)
    P_gpu = jnp.array(norm_profiles)  # (G, K)

    sigma = sigma_init

    for epoch in range(n_epoch):
        # ── Step 1: Fit weights at current sigma (R: decompose_batch at sigma) ──
        if str(sigma) not in q_matrices:
            diffs = np.abs(SIGMA_ALL - sigma)
            sigma = int(SIGMA_ALL[np.argmin(diffs)])

        Q_cur = jnp.array(q_matrices[str(sigma)])
        SQ_cur = jnp.array(compute_spline_coefficients(np.array(Q_cur), np.array(x_vals)))
        x_j = jnp.array(x_vals)

        S_batch = fit_numi[:, None, None] * P_gpu[None, :, :]  # (n, G, K)

        weights, _ = solve_irwls_batch(
            S_batch=S_batch,
            Y_batch=fit_counts,
            nUMI_batch=fit_numi,
            Q_mat=Q_cur,
            SQ_mat=SQ_cur,
            x_vals=x_j,
            max_iter=50,
            min_change=0.001,
            constrain=False,
            bulk_mode=False,
        )
        weights = jnp.maximum(weights, 0.0)

        # ── Step 2: Fixed prediction (R: sweep(norm_profiles %*% t(weights), 2, nUMI, '*')) ──
        prediction = jnp.dot(weights, P_gpu.T) * fit_numi[:, None]  # (n, G)
        prediction = jnp.maximum(prediction, 1e-4)

        # Flatten and subsample (R: num_sample = min(1000000, length(X)))
        X = prediction.flatten()
        Y = fit_counts.flatten()
        max_samples = 1_000_000
        if len(X) > max_samples:
            sub_idx = rng.choice(len(X), size=max_samples, replace=False)
            X = X[sub_idx]
            Y = Y[sub_idx]

        # ── Step 3: chooseSigma — evaluate sigma window on fixed prediction ──
        # R: sigma_ind = c(10:70, (36:100)*2); window = ±8 around current
        try:
            si_idx = int(np.where(SIGMA_ALL == round(sigma))[0][0])
        except IndexError:
            si_idx = int(np.argmin(np.abs(SIGMA_ALL - round(sigma))))

        start_idx = max(0, si_idx - 8)
        end_idx = min(len(SIGMA_ALL), si_idx + 8 + 1)
        sigma_cands = SIGMA_ALL[start_idx:end_idx]

        # R: mult_fac_vec = (8:12)/10
        mult_fac_vec = np.arange(8, 13) / 10.0

        lowest_score = float("inf")
        best_sigma = sigma

        for cand_sigma in sigma_cands:
            cand_key = str(cand_sigma)
            if cand_key not in q_matrices:
                continue
            cand_Q = jnp.array(q_matrices[cand_key])
            cand_SQ = jnp.array(compute_spline_coefficients(np.array(cand_Q), np.array(x_vals)))

            # R: best_val = min over mult_fac of calc_log_l_vec(X*mult_fac, Y)
            best_fac_score = float("inf")
            for fac in mult_fac_vec:
                score = float(calc_log_likelihood(Y, X * fac, cand_Q, cand_SQ, x_j, k_val))
                if score < best_fac_score:
                    best_fac_score = score

            if best_fac_score < lowest_score:
                lowest_score = best_fac_score
                best_sigma = cand_sigma

        sigma_prev = sigma
        sigma = int(best_sigma)

        if sigma == sigma_prev:
            break

    return sigma
