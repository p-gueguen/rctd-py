from typing import Dict

import numpy as np
import torch

from rctd import _likelihood, _numba_kernels, _triton_kernels
from rctd._irwls import solve_irwls_batch_shared
from rctd._likelihood import calc_q_all_eager as _calc_q_all_eager

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
    device: str = "auto",
) -> int:
    """Estimate the optimal sigma parameter for the Poisson-Lognormal model.

    Ports choose_sigma_c / chooseSigma from spacexr R:

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
    from rctd._types import resolve_device

    rng = np.random.default_rng(seed)
    device = resolve_device(device)

    # Filter pixels by min_umi (R: puck@nUMI > MIN_UMI)
    valid_idx = np.where(spatial_numi > min_umi)[0]
    if len(valid_idx) == 0:
        raise ValueError(f"No pixels found with UMI > {min_umi}. Try decreasing min_umi.")

    n_samples = min(n_fit, len(valid_idx))
    fit_idx = rng.choice(valid_idx, size=n_samples, replace=False)

    # Lazily compute SQ_mats only for sigma keys visited during search.
    # If the caller supplied a non-empty sq_matrices dict, keep candidate filtering
    # restricted to its keys for compatibility with tests and partial table dicts.
    restrict_to_sq = sq_matrices is not None and len(sq_matrices) > 0
    if sq_matrices is None:
        sq_matrices = {}

    x_vals_np = np.asarray(x_vals, dtype=np.float64)

    def ensure_sq(key: str) -> np.ndarray:
        if key not in sq_matrices:
            sq_matrices[key] = compute_spline_coefficients(q_matrices[key], x_vals_np)
        return sq_matrices[key]

    # ── Pre-convert arrays to torch tensors ──
    x_j = torch.tensor(x_vals, device=device)
    P_gpu = torch.tensor(norm_profiles, device=device)
    fit_counts = torch.as_tensor(spatial_counts[fit_idx], device=device)
    fit_numi = torch.as_tensor(spatial_numi[fit_idx], device=device)

    sigma = sigma_init
    use_fast_sigma = _calc_q_all_eager is _likelihood.calc_q_all_eager
    poly_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int, float]] = {}

    def stage_sigma_tables(sigma):
        # Keep only the current Q/SQ pair on the compute device.
        key = str(sigma)

        Q = torch.as_tensor(q_matrices[key], device=device)
        SQ = torch.as_tensor(ensure_sq(key), device=device)

        return Q, SQ

    for epoch in range(n_epoch):
        # ── Step 1: Fit weights at current sigma ──
        if str(sigma) not in q_matrices:
            diffs = np.abs(SIGMA_ALL - sigma)
            sigma = int(SIGMA_ALL[np.argmin(diffs)])

        Q_cur, SQ_cur = stage_sigma_tables(sigma)
        step1_q_fn = (
            None
            if (
                use_fast_sigma
                and (
                    (_numba_kernels.ENABLED and device.type == "cpu")
                    or _triton_kernels.use_triton(P_gpu)
                )
            )
            else _calc_q_all_eager
        )

        weights, _ = solve_irwls_batch_shared(
            P=P_gpu,
            Y_batch=fit_counts,
            nUMI_batch=fit_numi,
            Q_mat=Q_cur,
            SQ_mat=SQ_cur,
            x_vals=x_j,
            max_iter=50,
            min_change=0.001,
            constrain=False,
            bulk_mode=False,
            _calc_q_fn=step1_q_fn,
        )
        weights = torch.clamp(weights, min=0.0)
        del Q_cur, SQ_cur

        # ── Step 2: Fixed prediction ──
        prediction = (weights @ P_gpu.T) * fit_numi[:, None]  # (n, G)
        prediction = torch.clamp(prediction, min=1e-4)

        # Flatten and subsample (R: num_sample = min(1000000, length(X)))
        X = prediction.reshape(-1)
        Y = fit_counts.reshape(-1)
        max_samples = 1_000_000
        if X.shape[0] > max_samples:
            sub_idx = torch.tensor(
                rng.choice(X.shape[0], size=max_samples, replace=False), device=device
            )
            X = X[sub_idx]
            Y = Y[sub_idx]

        # ── Step 3: chooseSigma — evaluate sigma window ──
        # R: sigma_ind = c(10:70, (36:100)*2); window = ±8 around current
        try:
            si_idx = int(np.where(SIGMA_ALL == round(sigma))[0][0])
        except IndexError:
            si_idx = int(np.argmin(np.abs(SIGMA_ALL - round(sigma))))

        start_idx = max(0, si_idx - 8)
        end_idx = min(len(SIGMA_ALL), si_idx + 8 + 1)
        sigma_cands = SIGMA_ALL[start_idx:end_idx]

        # R: mult_fac_vec = (8:12)/10
        mult_fac_vec = torch.arange(8, 13, dtype=X.dtype, device=device) / 10.0

        # Filter to available sigma candidates
        if restrict_to_sq:
            valid_cands = [
                s for s in sigma_cands if (str(s) in q_matrices and str(s) in sq_matrices)
            ]
        else:
            valid_cands = [s for s in sigma_cands if str(s) in q_matrices]
        if not valid_cands:
            break

        # Evaluate each sigma candidate: pick best scale factor per sigma
        if use_fast_sigma and _numba_kernels.ENABLED and device.type == "cpu":
            X_np = np.ascontiguousarray(X.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
            Y_flat = Y.reshape(-1)
            k_eff_0 = min(k_val, q_matrices[str(valid_cands[0])].shape[0] - 3)
            Y_int_0 = _numba_kernels.prepare_y_int(Y_flat, k_eff_0)

            for s in valid_cands:
                key = str(s)
                if key not in poly_cache:
                    q_s = q_matrices[key]
                    sq_s = ensure_sq(key)
                    k_s = min(k_val, q_s.shape[0] - 3)
                    poly, k_of_l, x_of_l, _, lam_max = _numba_kernels.build_poly_table(
                        q_s, sq_s, x_vals_np, K_val=k_s
                    )
                    poly_cache[key] = (poly, k_of_l, x_of_l, k_s, lam_max)

            cand_scores = np.empty(len(valid_cands), dtype=np.float64)

            def _score_cand_range(start: int, end: int) -> None:
                for idx in range(start, end):
                    poly, k_of_l, x_of_l, k_s, lam_max = poly_cache[str(valid_cands[idx])]
                    y_arr = Y_int_0 if k_s == k_eff_0 else _numba_kernels.prepare_y_int(Y_flat, k_s)
                    cand_scores[idx] = _numba_kernels.score_sigma_5_factors_cpu(
                        X_np, y_arr, poly, k_of_l, x_of_l, lam_max
                    )

            _numba_kernels.parallel_rows(_score_cand_range, len(valid_cands), 1)
            best_idx = int(np.argmin(cand_scores))
        elif use_fast_sigma and _triton_kernels.use_triton(P_gpu):
            Y_flat = Y.reshape(-1)
            k_eff_0 = min(k_val, q_matrices[str(valid_cands[0])].shape[0] - 3)
            Y_int_0 = _triton_kernels.prepare_y_int_gpu(Y_flat, k_eff_0)
            cand_scores = []
            for s in valid_cands:
                Q_s, SQ_s = stage_sigma_tables(s)
                k_s = min(k_val, Q_s.shape[0] - 3)
                y_arr = (
                    Y_int_0 if k_s == k_eff_0 else _triton_kernels.prepare_y_int_gpu(Y_flat, k_s)
                )
                poly_gpu = _triton_kernels.build_poly_table_gpu(Q_s, SQ_s, x_j, K_val=k_s)
                cand_scores.append(_triton_kernels.score_sigma_5_factors_triton(X, y_arr, poly_gpu))
                del poly_gpu, Q_s, SQ_s
            best_idx = int(np.argmin(cand_scores))
        elif use_fast_sigma and device.type == "cuda":
            scores = torch.zeros(len(valid_cands), device=device)
            for sig_idx, s in enumerate(valid_cands):
                Q_s, SQ_s = stage_sigma_tables(s)
                fac_scores = torch.zeros(len(mult_fac_vec), device=device)
                for fac_idx in range(len(mult_fac_vec)):
                    X_fac = X * mult_fac_vec[fac_idx]
                    d0 = _likelihood.calc_q_d0_eager(Y, X_fac, Q_s, SQ_s, x_j, k_val)
                    fac_scores[fac_idx] = -torch.sum(d0)
                scores[sig_idx] = fac_scores.min()
                del Q_s, SQ_s
            best_idx = int(torch.argmin(scores).item())
        else:
            scores = torch.zeros(len(valid_cands), device=device)
            for sig_idx, s in enumerate(valid_cands):
                Q_s, SQ_s = stage_sigma_tables(s)
                fac_scores = torch.zeros(len(mult_fac_vec), device=device)
                for fac_idx in range(len(mult_fac_vec)):
                    X_fac = X * mult_fac_vec[fac_idx]
                    d0, _, _ = _calc_q_all_eager(Y, X_fac, Q_s, SQ_s, x_j, k_val)
                    fac_scores[fac_idx] = -torch.sum(d0)
                scores[sig_idx] = fac_scores.min()
                del Q_s, SQ_s
            best_idx = int(torch.argmin(scores).item())

        best_sigma = valid_cands[best_idx]

        sigma_prev = sigma
        sigma = int(best_sigma)

        if sigma == sigma_prev:
            break

    return sigma
