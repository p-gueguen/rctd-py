from typing import Dict

import jax
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
    sq_matrices: Dict[str, np.ndarray] | None = None,
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
        sq_matrices: precomputed spline coefficients dict (optional, computed if None)
    """
    from rctd._likelihood import compute_spline_coefficients

    rng = np.random.default_rng(seed)

    # Filter pixels by min_umi (R: puck@nUMI > MIN_UMI)
    valid_idx = np.where(spatial_numi > min_umi)[0]
    if len(valid_idx) == 0:
        raise ValueError(f"No pixels found with UMI > {min_umi}. Try decreasing min_umi.")

    n_samples = min(n_fit, len(valid_idx))
    fit_idx = rng.choice(valid_idx, size=n_samples, replace=False)

    # ── Precompute all SQ_mats at startup (Step 2) ──
    if sq_matrices is None:
        sq_matrices = {}
        for key, q_mat in q_matrices.items():
            sq_matrices[key] = compute_spline_coefficients(q_mat, np.array(x_vals))

    # ── Pre-convert arrays to JAX once (Step 4) ──
    x_j = jnp.array(x_vals)
    P_gpu = jnp.array(norm_profiles)
    q_gpu = {k: jnp.array(v) for k, v in q_matrices.items()}
    sq_gpu = {k: jnp.array(v) for k, v in sq_matrices.items()}
    fit_counts = jnp.array(spatial_counts[fit_idx])
    fit_numi = jnp.array(spatial_numi[fit_idx])

    # ── Define vmapped evaluation kernel (Step 3) ──
    def _eval_candidates(Y, X_scaled, Q_stack, SQ_stack):
        """Evaluate all sigma candidates × scale factors in one fused kernel."""

        def score_one_sigma(Q, SQ):
            def score_one_fac(X_fac):
                return calc_log_likelihood(Y, X_fac, Q, SQ, x_j, k_val)

            return jnp.min(jax.vmap(score_one_fac)(X_scaled))

        return jax.vmap(score_one_sigma)(Q_stack, SQ_stack)

    eval_candidates_jit = jax.jit(_eval_candidates)

    sigma = sigma_init

    for epoch in range(n_epoch):
        # ── Step 1: Fit weights at current sigma (R: decompose_batch at sigma) ──
        if str(sigma) not in q_gpu:
            diffs = np.abs(SIGMA_ALL - sigma)
            sigma = int(SIGMA_ALL[np.argmin(diffs)])

        Q_cur = q_gpu[str(sigma)]
        SQ_cur = sq_gpu[str(sigma)]

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
        if X.shape[0] > max_samples:
            sub_idx = rng.choice(X.shape[0], size=max_samples, replace=False)
            X = X[sub_idx]
            Y = Y[sub_idx]

        # ── Step 3: chooseSigma — vectorized evaluation of sigma window ──
        # R: sigma_ind = c(10:70, (36:100)*2); window = ±8 around current
        try:
            si_idx = int(np.where(SIGMA_ALL == round(sigma))[0][0])
        except IndexError:
            si_idx = int(np.argmin(np.abs(SIGMA_ALL - round(sigma))))

        start_idx = max(0, si_idx - 8)
        end_idx = min(len(SIGMA_ALL), si_idx + 8 + 1)
        sigma_cands = SIGMA_ALL[start_idx:end_idx]

        # R: mult_fac_vec = (8:12)/10
        mult_fac_vec = jnp.arange(8, 13) / 10.0

        # Filter to available sigma candidates
        valid_cands = [s for s in sigma_cands if str(s) in q_gpu]
        if not valid_cands:
            break

        # Stack Q/SQ matrices for vectorized evaluation
        Q_stack = jnp.stack([q_gpu[str(s)] for s in valid_cands])
        SQ_stack = jnp.stack([sq_gpu[str(s)] for s in valid_cands])

        # Pre-scale predictions: (n_fac, G_flat)
        X_scaled = X[None, :] * mult_fac_vec[:, None]

        # Single vmapped call replaces 85 sequential calc_log_likelihood calls
        scores = eval_candidates_jit(Y, X_scaled, Q_stack, SQ_stack)
        best_idx = int(jnp.argmin(scores))
        best_sigma = valid_cands[best_idx]

        sigma_prev = sigma
        sigma = int(best_sigma)

        if sigma == sigma_prev:
            break

    return sigma
