from itertools import combinations
from typing import List

import numpy as np
import torch

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
)


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
        batch_size: Number of pixels to process simultaneously on GPU
    """
    N, G = spatial_counts.shape
    K = norm_profiles.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    )
    W_full = full_res.weights  # (N, K)
    print(f"  [doublet] Step 1 done ({_time.time() - _t0:.1f}s)")

    # 2. Candidate selection (CPU)
    # R: candidates <- names(which(all_weights > initial_weight_thresh))
    WEIGHT_THRESHOLD = 0.01
    candidates_list = []
    for n in range(N):
        cands = np.where(W_full[n] > WEIGHT_THRESHOLD)[0].tolist()
        if len(cands) == 0:
            # R: candidates = cell_type_info[[2]][1:min(3, n_cell_types)]
            cands = list(range(min(3, K)))
        elif len(cands) == 1:
            # R: if first type, add second; else add first
            if cands[0] == 0:
                cands.append(1)
            else:
                cands.append(0)
        candidates_list.append(cands)

    # 3. Pairwise scoring (GPU, batched)
    # Collect all (pixel_idx, type1_idx, type2_idx) triples
    triples = []
    for n, cands in enumerate(candidates_list):
        for t1, t2 in combinations(cands, 2):
            triples.append((n, t1, t2))
    print(f"  [doublet] Step 3: pairwise scoring ({len(triples)} triples)...")
    _t3 = _time.time()

    pair_log_l = {}  # (n, t1, t2) -> score
    pair_weights = {}  # (n, t1, t2) -> array([w1, w2])

    P_gpu = torch.tensor(norm_profiles, device=device)
    Q_gpu = torch.tensor(q_mat, device=device)
    SQ_gpu = torch.tensor(sq_mat, device=device)
    X_gpu = torch.tensor(x_vals, device=device)

    if triples:
        triples_arr = np.array(triples, dtype=np.int32)
        M_total = len(triples_arr)

        for start in range(0, M_total, batch_size):
            end = min(start + batch_size, M_total)
            tr = triples_arr[start:end]
            bs = tr.shape[0]

            pix_idx = tr[:, 0]
            t1_idx = tr[:, 1]
            t2_idx = tr[:, 2]

            nUMI_tr = torch.tensor(spatial_numi[pix_idx], device=device)
            B_tr = torch.tensor(spatial_counts[pix_idx], device=device)

            # P_pair: (bs, G, 2)
            # P_gpu: (G, K)
            # P_gpu[:, t1_idx]: (G, bs) -> .T: (bs, G)
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

            scores_batch = calc_log_likelihood_batch(
                B_tr, expected_tr, Q_gpu, SQ_gpu, X_gpu, config.K_val
            )

            w_np = weights_batch.cpu().numpy()
            sc_np = scores_batch.cpu().numpy()

            for i in range(bs):
                n_idx, t1, t2 = triples[start + i]
                pair_log_l[(n_idx, t1, t2)] = sc_np[i]
                pair_weights[(n_idx, t1, t2)] = w_np[i]

    print(f"  [doublet] Step 3 done ({_time.time() - _t3:.1f}s)")

    # 4. Singlet scoring
    singles = list({(n, t) for n, cands in enumerate(candidates_list) for t in cands})
    print(f"  [doublet] Step 4: singlet scoring ({len(singles)} singles)...")
    _t4 = _time.time()
    singlet_log_l = {}

    if singles:
        singles_arr = np.array(singles, dtype=np.int32)
        S_total = len(singles_arr)

        for start in range(0, S_total, batch_size):
            end = min(start + batch_size, S_total)
            sg = singles_arr[start:end]
            bs = sg.shape[0]

            pix_idx = sg[:, 0]
            t_idx = sg[:, 1]

            nUMI_sg = torch.tensor(spatial_numi[pix_idx], device=device)
            B_sg = torch.tensor(spatial_counts[pix_idx], device=device)

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

            scores_batch = calc_log_likelihood_batch(
                B_sg, expected_sg, Q_gpu, SQ_gpu, X_gpu, config.K_val
            )
            sc_np = scores_batch.cpu().numpy()

            for i in range(bs):
                n_idx, t = singles[start + i]
                singlet_log_l[(n_idx, t)] = sc_np[i]

    print(f"  [doublet] Step 4 done ({_time.time() - _t4:.1f}s)")

    # 5. Classification — matches R's process_bead_doublet logic
    print(f"  [doublet] Step 5: classification ({N} pixels)...")
    _t5 = _time.time()
    # Build score matrix per pixel for check_pairs_type
    weights_doublet = np.zeros((N, 2), dtype=np.float32)
    spot_class = np.zeros(N, dtype=np.int32)
    first_type = np.zeros(N, dtype=np.int32)
    second_type = np.zeros(N, dtype=np.int32)
    first_class = np.zeros(N, dtype=bool)
    second_class = np.zeros(N, dtype=bool)
    min_score = np.zeros(N, dtype=np.float32)
    singlet_score_res = np.zeros(N, dtype=np.float32)
    INF = 1e18

    for n in range(N):
        cands = candidates_list[n]
        C = len(cands)

        # Singlet scores per candidate type
        sing_scores = {t: singlet_log_l.get((n, t), INF) for t in cands}

        # Build score matrix (symmetric) for all pairs
        score_mat = {}
        min_p_score = INF
        best_t1, best_t2 = cands[0], cands[1] if C > 1 else cands[0]
        for i in range(C):
            for j in range(i + 1, C):
                t1, t2 = cands[i], cands[j]
                sc = pair_log_l.get((n, t1, t2), INF)
                score_mat[(t1, t2)] = sc
                score_mat[(t2, t1)] = sc
                if sc < min_p_score:
                    min_p_score = sc
                    best_t1, best_t2 = t1, t2

        # R's check_pairs_type for default class_df (each type = own class):
        # all_pairs = TRUE if no competing pair (within CONFIDENCE_THRESHOLD
        # of min_score) exists where NEITHER type is my_type.
        # i.e., my_type is essential in every competitive pair.
        def check_pairs_type(my_type):
            all_pairs = True
            for i in range(C):
                for j in range(C):
                    if i != j:
                        t1, t2 = cands[i], cands[j]
                        sc = score_mat.get((t1, t2), INF)
                        if sc < min_p_score + config.CONFIDENCE_THRESHOLD:
                            if t1 != my_type and t2 != my_type:
                                all_pairs = False
            return all_pairs, sing_scores.get(my_type, INF)

        type1_all_pairs, type1_sing = check_pairs_type(best_t1)
        type2_all_pairs, type2_sing = check_pairs_type(best_t2)

        # R classification order:
        # 1. reject if neither type is uniquely necessary
        # 2. doublet_uncertain if only one type is necessary
        # 3. doublet_certain if both types are necessary
        # 4. singlet override if singlet_score - min_score < DOUBLET_THRESHOLD
        if not type1_all_pairs and not type2_all_pairs:
            s_class = SPOT_CLASS_REJECT
            s_score = min_p_score + 2 * config.DOUBLET_THRESHOLD  # arbitrary, ensures not singlet
            f_class = False
            sc_class = False
        elif type1_all_pairs and not type2_all_pairs:
            s_class = SPOT_CLASS_DOUBLET_UNCERTAIN
            f_class = False
            sc_class = False
            s_score = type1_sing
        elif not type1_all_pairs and type2_all_pairs:
            s_class = SPOT_CLASS_DOUBLET_UNCERTAIN
            # Swap types so first_type is the confident one
            best_t1, best_t2 = best_t2, best_t1
            type1_sing, type2_sing = type2_sing, type1_sing
            f_class = False
            sc_class = False
            s_score = type1_sing
        else:
            # Both types uniquely necessary
            s_class = SPOT_CLASS_DOUBLET_CERTAIN
            s_score = min(type1_sing, type2_sing)
            f_class = False
            sc_class = False
            # R: order by singlet score (lower = more confident)
            if type2_sing < type1_sing:
                best_t1, best_t2 = best_t2, best_t1

        # Singlet override (R lines 132-133)
        if s_score - min_p_score < config.DOUBLET_THRESHOLD:
            s_class = SPOT_CLASS_SINGLET

        # Doublet weights
        pair_key = (n, best_t1, best_t2)
        # Try both orderings since we may have swapped
        if pair_key not in pair_weights:
            pair_key = (n, best_t2, best_t1)
        if pair_key in pair_weights:
            dw = pair_weights[pair_key].copy()
            # If we swapped the key ordering, swap weights too
            if pair_key[1] != best_t1:
                dw = dw[::-1]
            s = dw.sum()
            dw = dw / s if s > 0 else np.array([0.5, 0.5])
        else:
            dw = np.array([W_full[n, best_t1], W_full[n, best_t2]])
            s = dw.sum()
            dw = dw / s if s > 0 else np.array([0.5, 0.5])

        if s_class == SPOT_CLASS_SINGLET:
            np.array([1.0, 0.0])
            # For singlet, first_type = best singlet type
            best_sing_type = min(sing_scores, key=sing_scores.get)
            first_t = best_sing_type
            second_t = best_t2 if best_t1 == best_sing_type else best_t1
            f_class = False
            sc_class = False
        else:
            first_t = best_t1
            second_t = best_t2

        spot_class[n] = s_class
        first_type[n] = first_t
        second_type[n] = second_t
        min_score[n] = min_p_score
        singlet_score_res[n] = s_score
        first_class[n] = f_class
        second_class[n] = sc_class

    print(f"  [doublet] Step 5 done ({_time.time() - _t5:.1f}s)")

    # 6. Final doublet decomposition — matches R's line 134 in process_bead_doublet
    # R runs a FRESH decompose_sparse(first_type, second_type, score_mode=FALSE)
    # with n.iter=50, constrain=FALSE, then normalizes weights to sum=1.
    # This gives the final doublet weights for ALL pixels (including singlets/rejects).
    print(f"  [doublet] Step 6: final decomposition ({N} pixels)...")
    _t6 = _time.time()
    final_t1 = first_type  # (N,) int
    final_t2 = second_type  # (N,) int

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        pix_idx = np.arange(start, end)
        nUMI_f = torch.tensor(spatial_numi[pix_idx], device=device)
        B_f = torch.tensor(spatial_counts[pix_idx], device=device)

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

        w_np = w_final.cpu().numpy()
        # R: results$weights = results$weights / sum(results$weights)
        w_sums = w_np.sum(axis=1, keepdims=True)
        w_sums = np.maximum(w_sums, 1e-10)
        w_np = w_np / w_sums

        weights_doublet[start:end] = w_np

    print(f"  [doublet] Step 6 done ({_time.time() - _t6:.1f}s)")
    print(f"  [doublet] Total doublet mode: {_time.time() - _t0:.1f}s")

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
    )
