from typing import List

import numpy as np
import torch

from rctd import _numba_kernels, _triton_kernels
from rctd._full import run_full_mode
from rctd._irwls import solve_irwls_batch
from rctd._likelihood import calc_log_likelihood_batch
from rctd._types import (
    SPOT_CLASS_DOUBLET_CERTAIN,
    SPOT_CLASS_DOUBLET_UNCERTAIN,
    SPOT_CLASS_REJECT,
    SPOT_CLASS_SINGLET,
    DoubletResult,
    RCTDConfig,
    resolve_device,
)

# Score of a missing pair or singlet; R's check_pairs_type never reads one for K >= 2
_INF = 1e18


def _lookup_converged_pair_weights(
    triples_arr: np.ndarray,
    pair_weights_all: np.ndarray,
    pair_conv_all: np.ndarray,
    final_t1: np.ndarray,
    final_t2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reuse converged Step 3 pair weights (with pair swap handling) for Step 6."""
    N = len(final_t1)
    out_dtype = pair_weights_all.dtype if len(pair_weights_all) else np.float64
    reused = np.empty((N, 2), dtype=out_dtype)
    reused_mask = np.zeros(N, dtype=bool)
    if len(triples_arr) == 0 or len(pair_conv_all) == 0:
        return reused, reused_mask

    valid = pair_conv_all & np.isfinite(pair_weights_all).all(axis=1)
    if not np.any(valid):
        return reused, reused_mask

    pix = triples_arr[valid, 0]
    t1 = triples_arr[valid, 1]
    t2 = triples_arr[valid, 2]
    w = pair_weights_all[valid]
    ft1 = final_t1[pix]
    ft2 = final_t2[pix]

    fwd = (t1 == ft1) & (t2 == ft2)
    if np.any(fwd):
        p_f = pix[fwd]
        reused[p_f] = w[fwd]
        reused_mask[p_f] = True

    rev = (~fwd) & (t1 == ft2) & (t2 == ft1)
    if np.any(rev):
        p_r = pix[rev]
        reused[p_r, 0] = w[rev, 1]
        reused[p_r, 1] = w[rev, 0]
        reused_mask[p_r] = True

    return reused, reused_mask


def run_doublet_mode(
    spatial_counts: np.ndarray,
    spatial_numi: np.ndarray,
    norm_profiles: np.ndarray,
    cell_type_names: List[str],
    q_mat: np.ndarray,
    sq_mat: np.ndarray,
    x_vals: np.ndarray,
    config: RCTDConfig,
    batch_size: int = 10000,
    device: str = "auto",
    pixel_mask: np.ndarray | None = None,
) -> DoubletResult:
    """Run doublet mode deconvolution.

    Args:
        spatial_counts: (N, G) observed count matrix
        spatial_numi: (N,) total UMI per pixel
        norm_profiles: (G, K) normalized reference profiles
        cell_type_names: List of length K
        q_mat: Likelihood table
        sq_mat: Likelihood spline table
        x_vals: Lambda grid points
        config: RCTD configuration parameters
        batch_size: Rows per GPU batch: pixels in the full-mode fit, (pixel, type, type)
            triples in pair scoring
    """
    N, G = spatial_counts.shape
    K = norm_profiles.shape[1]
    device = resolve_device(device)

    # Normalize class_df: None -> identity mapping (matches R's create.RCTD default)
    user_class_df = config.class_df
    if user_class_df is None:
        class_of_type = {ct: ct for ct in cell_type_names}
    else:
        # Validate
        missing = [ct for ct in cell_type_names if ct not in user_class_df]
        if missing:
            raise ValueError(
                f"class_df is missing entries for cell types: {missing}. "
                "Every cell type in the reference must be mapped to a class."
            )
        bad_vals = [
            k
            for k, v in user_class_df.items()
            if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v))
        ]
        if bad_vals:
            raise ValueError(f"class_df has None/NaN class values for keys: {bad_vals}")
        class_of_type = user_class_df
    # (K,) class of each type
    class_of = np.array([class_of_type[ct] for ct in cell_type_names])

    import time as _time

    # 1. Full fit to get candidates
    # In R, doublet_mode="doublet" first runs doublet_mode="full"
    # (or equivalent) in process_beads_batch
    print(f"  [doublet] Step 1/6: full-mode fit ({N} pixels, K={K})...")
    _t0 = _time.time()
    full_res = run_full_mode(
        spatial_counts=spatial_counts,
        spatial_numi=spatial_numi,
        norm_profiles=norm_profiles,
        cell_type_names=cell_type_names,
        q_mat=q_mat,
        sq_mat=sq_mat,
        x_vals=x_vals,
        batch_size=batch_size,
        device=str(device),
    )
    W_full = full_res.weights  # (N, K)
    print(f"  [doublet] Step 1 done ({_time.time() - _t0:.1f}s)")

    # 2. Candidate selection (CPU)
    candidates, n_candidates = _select_candidates(W_full)
    triples_arr, singles_arr = _candidate_pairs(candidates, n_candidates)

    use_numba_cpu = _numba_kernels.ENABLED and device.type == "cpu"
    use_triton_gpu = device.type == "cuda" and _triton_kernels.use_triton(
        torch.empty(0, device=device)
    )

    # 3. Pairwise scoring
    print(f"  [doublet] Step 3: pairwise scoring ({len(triples_arr)} triples)...")
    _t3 = _time.time()

    pair_scores = np.zeros(0, dtype=norm_profiles.dtype)
    pair_weights_all = np.zeros((0, 2), dtype=np.float64 if use_numba_cpu else norm_profiles.dtype)
    pair_conv_all = np.zeros(0, dtype=bool)

    if use_numba_cpu:
        poly, k_of_l, x_of_l, k_val, lam_max = _numba_kernels.build_poly_table(
            q_mat, sq_mat, x_vals
        )
        Y_int = _numba_kernels.prepare_y_int(spatial_counts, k_val)
        nUMI_np = np.ascontiguousarray(spatial_numi, dtype=np.float64)
        P_KT = np.ascontiguousarray(norm_profiles.T, dtype=np.float64)

        if len(triples_arr):
            pair_weights_all, pair_scores_f64, pair_conv_all = (
                _numba_kernels.fit_and_score_pairs_cpu(
                    triples_arr,
                    Y_int,
                    nUMI_np,
                    P_KT,
                    poly,
                    k_of_l,
                    x_of_l,
                    lam_max,
                    max_iter=25,
                )
            )
            pair_scores = pair_scores_f64.astype(norm_profiles.dtype, copy=False)
    else:
        P_gpu = torch.tensor(norm_profiles, device=device)
        Q_gpu = torch.tensor(q_mat, device=device)
        SQ_gpu = torch.tensor(sq_mat, device=device)
        X_gpu = torch.tensor(x_vals, device=device)

        # Pre-stage spatial data on the device once instead of re-uploading
        # spatial_counts[pix_idx] / spatial_numi[pix_idx] per batch in steps 3/4/6.
        Y_full_gpu = torch.tensor(spatial_counts, device=device)
        nUMI_full_gpu = torch.tensor(spatial_numi, device=device)

        if use_triton_gpu:
            poly_gpu = _triton_kernels.build_poly_table_gpu(Q_gpu, SQ_gpu, X_gpu)
            Y_int_gpu = _triton_kernels.prepare_y_int_gpu(Y_full_gpu, poly_gpu[6])
            del Y_full_gpu
            P_KT_gpu = P_gpu.T.contiguous()

            if len(triples_arr):
                tr_t = torch.as_tensor(triples_arr, dtype=torch.int64, device=device)
                pw_t, ps_t, pc_t = _triton_kernels.fit_and_score_pairs_triton(
                    tr_t[:, 0],
                    tr_t[:, 1],
                    tr_t[:, 2],
                    Y_int_gpu,
                    nUMI_full_gpu,
                    P_KT_gpu,
                    poly_gpu,
                    max_iter=25,
                    compute_score=True,
                )
                pair_scores = ps_t.cpu().numpy()
                pair_weights_all = pw_t.cpu().numpy()
                pair_conv_all = pc_t.cpu().numpy()
        elif len(triples_arr):
            M_total = len(triples_arr)

            all_scores_t = []
            all_weights_t = []
            all_conv_t = []

            for start in range(0, M_total, batch_size):
                end = min(start + batch_size, M_total)
                tr = triples_arr[start:end]

                pix_idx = tr[:, 0]
                t1_idx = tr[:, 1]
                t2_idx = tr[:, 2]

                pix_idx_t = torch.as_tensor(pix_idx, dtype=torch.long, device=device)
                nUMI_tr = nUMI_full_gpu[pix_idx_t]
                B_tr = Y_full_gpu[pix_idx_t]

                t1_t = torch.tensor(t1_idx, dtype=torch.long, device=device)
                t2_t = torch.tensor(t2_idx, dtype=torch.long, device=device)
                P1 = P_gpu[:, t1_t].T
                P2 = P_gpu[:, t2_t].T
                P_pair = torch.stack([P1, P2], dim=-1)  # (bs, G, 2)

                S_pair = nUMI_tr[:, None, None] * P_pair

                weights_batch, conv_batch = solve_irwls_batch(
                    S_batch=S_pair,
                    Y_batch=B_tr,
                    nUMI_batch=nUMI_tr,
                    Q_mat=Q_gpu,
                    SQ_mat=SQ_gpu,
                    x_vals=X_gpu,
                    max_iter=25,
                    min_change=0.001,
                    constrain=False,
                    bulk_mode=False,
                )

                # Score logic: R's calc_log_l_vec returns positive NLL (lower=better)
                expected_tr = torch.sum(S_pair * weights_batch[:, None, :], dim=-1)  # (bs, G)
                expected_tr = torch.clamp(expected_tr, min=1e-4)

                scores_batch = calc_log_likelihood_batch(B_tr, expected_tr, Q_gpu, SQ_gpu, X_gpu)

                all_scores_t.append(scores_batch)
                all_weights_t.append(weights_batch)
                all_conv_t.append(conv_batch)

            # Transfer all results to CPU at once
            pair_scores = torch.cat(all_scores_t).cpu().numpy()
            pair_weights_all = torch.cat(all_weights_t).cpu().numpy()
            pair_conv_all = torch.cat(all_conv_t).cpu().numpy()

    print(f"  [doublet] Step 3 done ({_time.time() - _t3:.1f}s)")

    # 4. Singlet scoring
    print(f"  [doublet] Step 4: singlet scoring ({len(singles_arr)} singles)...")
    _t4 = _time.time()
    # (N, K) singlet score per candidate type; non-candidates are never read
    singlet_scores = np.full((N, K), _INF, dtype=pair_scores.dtype)

    if len(singles_arr):
        if use_numba_cpu:
            sing_scores = _numba_kernels.fit_and_score_singles_cpu(
                singles_arr,
                Y_int,
                nUMI_np,
                P_KT,
                poly,
                k_of_l,
                x_of_l,
                lam_max,
                max_iter=25,
            )
            singlet_scores[singles_arr[:, 0], singles_arr[:, 1]] = sing_scores.astype(
                pair_scores.dtype, copy=False
            )
        elif use_triton_gpu:
            sg_t = torch.as_tensor(singles_arr, dtype=torch.int64, device=device)
            _, sing_scores_t = _triton_kernels.fit_and_score_singles_triton(
                sg_t[:, 0],
                sg_t[:, 1],
                Y_int_gpu,
                nUMI_full_gpu,
                P_KT_gpu,
                poly_gpu,
                max_iter=25,
                write_weights=False,
                compute_score=True,
            )
            singlet_scores[singles_arr[:, 0], singles_arr[:, 1]] = sing_scores_t.cpu().numpy()
        else:
            S_total = len(singles_arr)

            all_sing_scores_t = []

            for start in range(0, S_total, batch_size):
                end = min(start + batch_size, S_total)
                sg = singles_arr[start:end]

                pix_idx = sg[:, 0]
                t_idx = sg[:, 1]

                pix_idx_t = torch.as_tensor(pix_idx, dtype=torch.long, device=device)
                nUMI_sg = nUMI_full_gpu[pix_idx_t]
                B_sg = Y_full_gpu[pix_idx_t]

                # P_sg: (bs, G, 1)
                t_t = torch.tensor(t_idx, dtype=torch.long, device=device)
                P_sg = P_gpu[:, t_t].T.unsqueeze(-1)
                S_sg = nUMI_sg[:, None, None] * P_sg

                weights_batch, conv_batch = solve_irwls_batch(
                    S_batch=S_sg,
                    Y_batch=B_sg,
                    nUMI_batch=nUMI_sg,
                    Q_mat=Q_gpu,
                    SQ_mat=SQ_gpu,
                    x_vals=X_gpu,
                    max_iter=25,
                    min_change=0.001,
                    constrain=False,
                    bulk_mode=False,
                )

                expected_sg = torch.sum(S_sg * weights_batch[:, None, :], dim=-1)
                expected_sg = torch.clamp(expected_sg, min=1e-4)

                scores_batch = calc_log_likelihood_batch(B_sg, expected_sg, Q_gpu, SQ_gpu, X_gpu)
                all_sing_scores_t.append(scores_batch)

            # Transfer all results to CPU at once
            singlet_scores[singles_arr[:, 0], singles_arr[:, 1]] = (
                torch.cat(all_sing_scores_t).cpu().numpy()
            )

    print(f"  [doublet] Step 4 done ({_time.time() - _t4:.1f}s)")

    # 5. Classification — matches R's process_bead_doublet logic
    print(f"  [doublet] Step 5: classification ({N} pixels)...")
    _t5 = _time.time()
    (
        spot_class,
        first_type,
        second_type,
        first_class,
        second_class,
        min_score,
        singlet_score_res,
    ) = _classify_pixels(
        triples_arr,
        pair_scores,
        singlet_scores,
        candidates,
        np.unique(class_of, return_inverse=True)[1],
        config,
    )

    print(f"  [doublet] Step 5 done ({_time.time() - _t5:.1f}s)")

    # 6. Final doublet decomposition — matches R's line 134 in process_bead_doublet
    # R runs a FRESH decompose_sparse(first_type, second_type, score_mode=FALSE)
    # with n.iter=50, constrain=FALSE, then normalizes weights to sum=1.
    # Pairs that already converged within Step 3 (min_change=0.001, max_iter=25)
    # follow the same trajectory from (0.5, 0.5) and can be reused directly.
    print(f"  [doublet] Step 6: final decomposition ({N} pixels)...")
    _t6 = _time.time()
    final_t1 = first_type  # (N,) int
    final_t2 = second_type  # (N,) int

    w_all_np, reused_mask = _lookup_converged_pair_weights(
        triples_arr, pair_weights_all, pair_conv_all, final_t1, final_t2
    )
    rem_pix = np.flatnonzero(~reused_mask)

    if len(rem_pix):
        if use_numba_cpu:
            rem_types = np.column_stack([final_t1[rem_pix], final_t2[rem_pix]])
            w_rem = _numba_kernels.fit_subsets_weights_cpu(
                rem_pix,
                rem_types,
                Y_int,
                nUMI_np,
                P_KT,
                poly,
                k_of_l,
                x_of_l,
                lam_max,
                max_iter=50,
            )
            w_all_np[rem_pix] = w_rem.astype(w_all_np.dtype, copy=False)
        elif use_triton_gpu:
            rem_pix_t = torch.as_tensor(rem_pix, dtype=torch.int64, device=device)
            ft1_t = torch.as_tensor(final_t1[rem_pix], dtype=torch.int64, device=device)
            ft2_t = torch.as_tensor(final_t2[rem_pix], dtype=torch.int64, device=device)
            w_rem_t, _, _ = _triton_kernels.fit_and_score_pairs_triton(
                rem_pix_t,
                ft1_t,
                ft2_t,
                Y_int_gpu,
                nUMI_full_gpu,
                P_KT_gpu,
                poly_gpu,
                max_iter=50,
                compute_score=False,
            )
            w_all_np[rem_pix] = w_rem_t.cpu().numpy()
        else:
            all_rem_weights_t = []
            for start in range(0, len(rem_pix), batch_size):
                end = min(start + batch_size, len(rem_pix))
                pix_idx = rem_pix[start:end]
                pix_idx_t = torch.as_tensor(pix_idx, dtype=torch.long, device=device)
                nUMI_f = nUMI_full_gpu[pix_idx_t]
                B_f = Y_full_gpu[pix_idx_t]

                ft1_t = torch.tensor(final_t1[pix_idx], dtype=torch.long, device=device)
                ft2_t = torch.tensor(final_t2[pix_idx], dtype=torch.long, device=device)
                P1 = P_gpu[:, ft1_t].T
                P2 = P_gpu[:, ft2_t].T
                P_pair = torch.stack([P1, P2], dim=-1)  # (bs, G, 2)
                S_pair = nUMI_f[:, None, None] * P_pair

                w_final, _ = solve_irwls_batch(
                    S_batch=S_pair,
                    Y_batch=B_f,
                    nUMI_batch=nUMI_f,
                    Q_mat=Q_gpu,
                    SQ_mat=SQ_gpu,
                    x_vals=X_gpu,
                    max_iter=50,
                    min_change=0.001,
                    constrain=False,
                    bulk_mode=False,
                )
                all_rem_weights_t.append(w_final)

            w_all_np[rem_pix] = torch.cat(all_rem_weights_t).cpu().numpy()

    w_all_np = w_all_np.astype(norm_profiles.dtype, copy=False)
    # R: results$weights = results$weights / sum(results$weights)
    w_sums = w_all_np.sum(axis=1, keepdims=True)
    w_sums = np.maximum(w_sums, 1e-10)
    weights_doublet = w_all_np / w_sums

    print(f"  [doublet] Step 6 done ({_time.time() - _t6:.1f}s)")
    print(f"  [doublet] Total doublet mode: {_time.time() - _t0:.1f}s")

    # Populate class name arrays only when user provided an explicit class_df.
    # With class_df=None (internal identity), these stay None for backward compat.
    if user_class_df is not None:
        first_class_name = class_of[first_type]
        second_class_name = class_of[second_type]
    else:
        first_class_name = None
        second_class_name = None

    return DoubletResult(
        weights=W_full,
        weights_doublet=weights_doublet,
        spot_class=spot_class,
        first_type=first_type,
        second_type=second_type,
        first_class=first_class,
        second_class=second_class,
        min_score=min_score,
        singlet_score=singlet_score_res,
        cell_type_names=cell_type_names,
        pixel_mask=pixel_mask,
        first_class_name=first_class_name,
        second_class_name=second_class_name,
    )


def _select_candidates(
    W_full: np.ndarray, threshold: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate types of each pixel, in the order R lists them.

    R: ``candidates <- names(which(all_weights > initial_weight_thresh))``. With no candidate,
    R takes the first min(3, K) types; with one, it adds the first type (the second if the
    candidate is the first), after the candidate.

    Returns:
        ``(candidates, n_candidates)``: (N, C >= 2) type indices, whose row n holds the
        pixel's ``n_candidates[n]`` candidates first, padded with zeros.
    """
    N, K = W_full.shape
    passing = W_full > threshold
    n_passing = passing.sum(axis=1)
    passing[n_passing == 0, : min(3, K)] = True
    one = n_passing == 1
    appended_first = one & ~passing[:, 0]
    passing[one, 0] = True
    passing[one & ~appended_first, min(1, K - 1)] = True

    n_candidates = passing.sum(axis=1)
    rows, types = np.nonzero(passing)
    pos = np.arange(len(rows)) - (np.cumsum(n_candidates) - n_candidates)[rows]
    candidates = np.zeros((N, max(n_candidates.max(initial=0), 2)), dtype=np.int32)
    candidates[rows, pos] = types
    # The added first type goes after the candidate: [c, 0], not [0, c]
    candidates[appended_first, :2] = candidates[appended_first, 1::-1]
    return candidates, n_candidates


def _candidate_pairs(
    candidates: np.ndarray, n_candidates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The (pixel, type1, type2) pairs and (pixel, type) singles to score.

    Both are ordered by pixel, and a pixel's pairs follow ``itertools.combinations`` of its
    candidates, as R's pair loop does.
    """
    N, C = candidates.shape
    # Each pixel's pairs go at its offset; pixels with the same number of candidates at once
    n_pairs = n_candidates * (n_candidates - 1) // 2
    offsets = np.cumsum(n_pairs) - n_pairs
    triples = np.empty((n_pairs.sum(), 3), dtype=np.int32)
    for c in np.unique(n_candidates):
        rows = np.flatnonzero(n_candidates == c)
        ci, cj = np.triu_indices(c, 1)
        at = (offsets[rows, None] + np.arange(len(ci))).ravel()
        triples[at, 0] = np.repeat(rows, len(ci))
        triples[at, 1] = candidates[rows][:, ci].ravel()
        triples[at, 2] = candidates[rows][:, cj].ravel()

    pix, pos = np.nonzero(np.arange(C) < n_candidates[:, None])
    singles = np.stack([pix, candidates[pix, pos]], axis=1).astype(np.int32)
    return triples, singles


def _classify_pixels(
    triples: np.ndarray,
    pair_scores: np.ndarray,
    singlet_scores: np.ndarray,
    candidates: np.ndarray,
    type_class: np.ndarray,
    config: RCTDConfig,
) -> tuple[np.ndarray, ...]:
    """R's process_bead_doublet decision for every pixel at once.

    Args:
        triples: (M, 3) scored (pixel, type1, type2) pairs, ordered by pixel
        pair_scores: (M,) their negative log-likelihoods (lower is better)
        singlet_scores: (N, K) singlet score of each pixel's candidate types
        candidates: (N, C) candidate types, from ``_select_candidates``
        type_class: (K,) class index of each type

    Returns:
        ``(spot_class, first_type, second_type, first_class, second_class, min_score,
        singlet_score)``, each of shape (N,).
    """
    N = len(candidates)
    rows = np.arange(N)
    pix, t1, t2 = triples.T

    # The lowest-scoring pair (the first one on ties), as R's strict `<` scan from _INF finds it
    min_score = np.full(N, _INF, dtype=pair_scores.dtype)
    np.fmin.at(min_score, pix, pair_scores)
    is_min = np.flatnonzero((pair_scores == min_score[pix]) & (pair_scores < _INF))
    best_pix, first = np.unique(pix[is_min], return_index=True)
    # A pixel without a pair score below _INF keeps R's start: its first two candidates
    best_t1, best_t2 = candidates[:, 0].copy(), candidates[:, 1].copy()
    best_t1[best_pix] = t1[is_min[first]]
    best_t2[best_pix] = t2[is_min[first]]

    # Only pairs within CONFIDENCE_THRESHOLD of the best one matter from here on
    near = pair_scores < min_score[pix] + config.CONFIDENCE_THRESHOLD
    pix, t1, t2 = pix[near], t1[near], t2[near]

    def none_per_pixel(mask):
        return np.bincount(pix[mask], minlength=N) == 0

    def check_pairs_type(my_type):
        # R's check_pairs_type (spacexr/R/RCTD_helper.R:44-63):
        #   all_pairs       : my_type is in every near pair
        #   all_pairs_class : my_type's class is in every near pair
        #   singlet_score   : my_type's singlet score, or the best one among the types of its
        #                     class in near pairs if all_pairs_class but not all_pairs
        # With class_df=None (identity mapping), all_pairs_class == all_pairs and the
        # singlet_score adjustment never fires.
        m = my_type[pix]
        same1 = type_class[t1] == type_class[m]
        same2 = type_class[t2] == type_class[m]
        all_pairs = none_per_pixel((t1 != m) & (t2 != m))
        all_pairs_class = none_per_pixel(~same1 & ~same2)

        # Other types of my_type's class in near pairs. A pixel with all_pairs_class but not
        # all_pairs always has one: some near pair misses my_type but holds its class.
        other1 = same1 & (t1 != m)
        other2 = same2 & (t2 != m)
        other_pix = np.concatenate([pix[other1], pix[other2]])
        other_scores = singlet_scores[other_pix, np.concatenate([t1[other1], t2[other2]])]
        other_min = np.full(N, np.inf, dtype=singlet_scores.dtype)
        np.fmin.at(other_min, other_pix, other_scores)

        singlet_score = singlet_scores[rows, my_type]
        better = all_pairs_class & ~all_pairs & (other_min < singlet_score)
        return all_pairs, all_pairs_class, np.where(better, other_min, singlet_score)

    type1_all_pairs, type1_all_class, type1_sing = check_pairs_type(best_t1)
    type2_all_pairs, type2_all_class, type2_sing = check_pairs_type(best_t2)

    # R's 4-branch decision tree (spacexr/R/RCTD_helper.R:129-150):
    #   reject    : neither type viable even at class level
    #   uncertain : only one type viable at class level; it becomes first_type
    #   certain   : both viable at class level; the better singlet score goes first
    viable = type1_all_class | type2_all_class
    both = type1_all_class & type2_all_class
    swap = (type2_all_class & ~type1_all_class) | (both & (type2_sing < type1_sing))

    spot_class = np.where(both, SPOT_CLASS_DOUBLET_CERTAIN, SPOT_CLASS_DOUBLET_UNCERTAIN)
    spot_class[~viable] = SPOT_CLASS_REJECT
    first_class = viable & ~np.where(swap, type2_all_pairs, type1_all_pairs)
    second_class = both & ~np.where(swap, type1_all_pairs, type2_all_pairs)
    # A reject's score is arbitrary, ensuring it is not a singlet
    singlet_score = np.where(swap, type2_sing, type1_sing)
    singlet_score = np.where(viable, singlet_score, min_score + 2 * config.DOUBLET_THRESHOLD)

    # Singlet override (R lines 132-133). R relabels the spot class only: a singlet keeps the
    # first/second type (and the class flags) chosen above, not the best singlet candidate.
    spot_class[singlet_score - min_score < config.DOUBLET_THRESHOLD] = SPOT_CLASS_SINGLET

    return (
        spot_class.astype(np.int32),
        np.where(swap, best_t2, best_t1).astype(np.int32),
        np.where(swap, best_t1, best_t2).astype(np.int32),
        first_class,
        second_class,
        min_score.astype(np.float32),
        singlet_score.astype(np.float32),
    )
