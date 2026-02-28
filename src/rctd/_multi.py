from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from rctd._full import run_full_mode
from rctd._irwls import solve_irwls_batch
from rctd._types import MultiResult, RCTDConfig


def _run_batched_scoring(
    task_triples: List[Tuple[int, List[int]]],
    spatial_numi: np.ndarray,
    spatial_counts: np.ndarray,
    P_gpu: jnp.ndarray,
    Q_gpu: jnp.ndarray,
    SQ_gpu: jnp.ndarray,
    X_gpu: jnp.ndarray,
    batch_size: int,
    K_val: int,
    n_iter: int = 25,
) -> np.ndarray:
    """Run batched IRWLS for a set of tasks all having the SAME number of cell types K_sub."""
    if not task_triples:
        return np.array([])

    M = len(task_triples)
    len(task_triples[0][1])

    all_scores = []

    # We need calc_log_likelihood to compute negative log-likelihood like R
    from rctd._likelihood import calc_log_likelihood

    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        end - start

        pix_idx = []
        type_idx = []
        for i in range(start, end):
            pix_idx.append(task_triples[i][0])
            type_idx.append(task_triples[i][1])

        pix_idx = np.array(pix_idx, dtype=np.int32)
        type_idx = np.array(type_idx, dtype=np.int32)  # (bs, K_sub)

        nUMI_tr = jnp.array(spatial_numi[pix_idx])
        B_tr = jnp.array(spatial_counts[pix_idx])

        # P_gpu is (G, K). We need (bs, G, K_sub)
        # using jax.vmap or list comprehension:
        # P_gpu[:, type_idx[i]] -> (G, K_sub)
        # To do this efficiently: type_idx is (bs, K_sub)
        # P_gpu[:, type_idx] -> shape (G, bs, K_sub). Then transpose to (bs, G, K_sub)
        # Using numpy advanced indexing:
        P_sub = P_gpu[:, type_idx].transpose((1, 0, 2))  # (bs, G, K_sub)

        S_sub = nUMI_tr[:, None, None] * P_sub

        weights_batch, conv_batch = solve_irwls_batch(
            S_batch=S_sub,
            Y_batch=B_tr,
            nUMI_batch=nUMI_tr,
            Q_mat=Q_gpu,
            SQ_mat=SQ_gpu,
            x_vals=X_gpu,
            max_iter=n_iter,
            min_change=0.001,
            constrain=False,
            bulk_mode=False,
        )

        expected_tr = jnp.sum(S_sub * weights_batch[:, None, :], axis=-1)  # (bs, G)
        expected_tr = jnp.maximum(expected_tr, 1e-4)

        batched_nll = jax.vmap(
            lambda y, lam: calc_log_likelihood(y, lam, Q_gpu, SQ_gpu, X_gpu, K_val), in_axes=(0, 0)
        )
        scores_batch = batched_nll(B_tr, expected_tr)
        all_scores.append(np.array(scores_batch))

    return np.concatenate(all_scores)


def _run_batched_weights(
    task_triples: List[Tuple[int, List[int]]],
    spatial_numi: np.ndarray,
    spatial_counts: np.ndarray,
    P_gpu: jnp.ndarray,
    Q_gpu: jnp.ndarray,
    SQ_gpu: jnp.ndarray,
    X_gpu: jnp.ndarray,
    batch_size: int,
    n_iter: int = 50,
) -> List[np.ndarray]:
    if not task_triples:
        return []

    M = len(task_triples)
    len(task_triples[0][1])

    all_weights = []

    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        end - start

        pix_idx = []
        type_idx = []
        for i in range(start, end):
            pix_idx.append(task_triples[i][0])
            type_idx.append(task_triples[i][1])

        pix_idx = np.array(pix_idx, dtype=np.int32)
        type_idx = np.array(type_idx, dtype=np.int32)

        nUMI_tr = jnp.array(spatial_numi[pix_idx])
        B_tr = jnp.array(spatial_counts[pix_idx])

        P_sub = P_gpu[:, type_idx].transpose((1, 0, 2))
        S_sub = nUMI_tr[:, None, None] * P_sub

        weights_batch, conv_batch = solve_irwls_batch(
            S_batch=S_sub,
            Y_batch=B_tr,
            nUMI_batch=nUMI_tr,
            Q_mat=Q_gpu,
            SQ_mat=SQ_gpu,
            x_vals=X_gpu,
            max_iter=n_iter,
            min_change=0.001,
            constrain=False,
            bulk_mode=False,
        )
        all_weights.append(np.array(weights_batch))

    return list(np.concatenate(all_weights))


def run_multi_mode(
    spatial_counts: np.ndarray,
    spatial_numi: np.ndarray,
    norm_profiles: np.ndarray,
    cell_type_names: List[str],
    q_mat: np.ndarray,
    sq_mat: np.ndarray,
    x_vals: np.ndarray,
    config: RCTDConfig,
    batch_size: int = 10000,
) -> MultiResult:
    """Run multi mode deconvolution.

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
    norm_profiles.shape[1]

    # 1. Full fit to get candidates
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
    W_full = full_res.weights

    # 2. Candidate selection (CPU)
    WEIGHT_THRESHOLD = 0.01
    candidates_list = []
    for n in range(N):
        cands = set(np.where(W_full[n] > WEIGHT_THRESHOLD)[0].tolist())
        if len(cands) == 0:
            best = int(np.argmax(W_full[n]))
            cands.add(best)
        candidates_list.append(cands)

    P_gpu = jnp.array(norm_profiles)
    Q_gpu = jnp.array(q_mat)
    SQ_gpu = jnp.array(sq_mat)
    X_gpu = jnp.array(x_vals)

    INF = 1e18
    current_scores = np.full(N, INF, dtype=np.float64)
    cell_type_lists = [[] for _ in range(N)]
    active_pixels = set(range(N))

    # Iterative forward selection
    for k in range(1, config.MAX_MULTI_TYPES + 1):
        if not active_pixels:
            break

        tasks = []  # list of (pixel_idx, type_list)
        task_info = []  # keep track of which candidate this task corresponds to

        for n in active_pixels:
            for cand in candidates_list[n]:
                cur_list = cell_type_lists[n] + [cand]
                tasks.append((n, cur_list))
                task_info.append((n, cand))

        # Run scoring for length k
        scores = _run_batched_scoring(
            tasks,
            spatial_numi,
            spatial_counts,
            P_gpu,
            Q_gpu,
            SQ_gpu,
            X_gpu,
            batch_size,
            config.K_val,
        )

        # Aggregate results by pixel
        best_cand_for_pixel = {}
        best_score_for_pixel = {}
        for i, (n, cand) in enumerate(task_info):
            sc = scores[i]
            if n not in best_score_for_pixel or sc < best_score_for_pixel[n]:
                best_score_for_pixel[n] = sc
                best_cand_for_pixel[n] = cand

        # Update active pixels
        new_active = set()
        for n in active_pixels:
            if n not in best_score_for_pixel:
                continue

            min_score = best_score_for_pixel[n]
            best_cand = best_cand_for_pixel[n]

            # Check improvement vs previous iteration score
            # if min_score > curr_score - doublet_like_cutoff: break
            if min_score > current_scores[n] - config.DOUBLET_THRESHOLD:
                # Does not improve enough, stop adding
                if len(cell_type_lists[n]) == 0:
                    # Must add at least 1 type
                    cell_type_lists[n].append(best_cand)
                    candidates_list[n].remove(best_cand)
                    current_scores[n] = min_score
                    # But don't continue adding to this pixel if it can't even improve
                pass  # remains inactive
            else:
                cell_type_lists[n].append(best_cand)
                candidates_list[n].remove(best_cand)
                current_scores[n] = min_score
                new_active.add(n)

        active_pixels = new_active

    # Confidence check
    # conf_list[type] = FALSE if swapping yields score < current + threshold
    conf_lists = [{} for _ in range(N)]
    for n in range(N):
        for t in cell_type_lists[n]:
            conf_lists[n][t] = True

    conf_tasks = {}  # grouped by K_sub: { k_sub: [(n, cur_list, original_t, newtype)] }
    for n in range(N):
        ct_list = cell_type_lists[n]
        if len(ct_list) == 0:
            continue

        k_sub = len(ct_list)
        if k_sub not in conf_tasks:
            conf_tasks[k_sub] = []

        for t in ct_list:
            for newtype in candidates_list[n]:
                # replace t with newtype
                cur_list = [x for x in ct_list if x != t] + [newtype]
                conf_tasks[k_sub].append((n, cur_list, t, newtype))

    for k_sub, t_list in conf_tasks.items():
        base_tasks = [(item[0], item[1]) for item in t_list]
        scores = _run_batched_scoring(
            base_tasks,
            spatial_numi,
            spatial_counts,
            P_gpu,
            Q_gpu,
            SQ_gpu,
            X_gpu,
            batch_size,
            config.K_val,
        )
        for i, (n, cur_list, t, newtype) in enumerate(t_list):
            if not conf_lists[n][t]:
                continue  # already marked false
            if scores[i] < current_scores[n] + config.CONFIDENCE_THRESHOLD:
                conf_lists[n][t] = False

    # Final weights
    final_tasks = {}
    for n in range(N):
        k_sub = len(cell_type_lists[n])
        if k_sub == 0:
            # Fallback to single best if something goes completely wrong
            k_sub = 1
            cell_type_lists[n] = [int(np.argmax(W_full[n]))]
        if k_sub not in final_tasks:
            final_tasks[k_sub] = []
        final_tasks[k_sub].append((n, cell_type_lists[n]))

    final_weights_dict = {}
    for k_sub, t_list in final_tasks.items():
        w_res = _run_batched_weights(
            t_list, spatial_numi, spatial_counts, P_gpu, Q_gpu, SQ_gpu, X_gpu, batch_size
        )
        for i, (n, _) in enumerate(t_list):
            # Normalize the weights from IRWLS
            w_norm = w_res[i]
            s = np.sum(w_norm)
            if s > 0:
                w_norm = w_norm / s
            final_weights_dict[n] = w_norm

    # Populate result format
    sub_weights = np.zeros((N, config.MAX_MULTI_TYPES), dtype=np.float32)
    cell_type_indices = np.full((N, config.MAX_MULTI_TYPES), -1, dtype=np.int32)
    n_types = np.zeros(N, dtype=np.int32)
    conf_list_arr = np.zeros((N, config.MAX_MULTI_TYPES), dtype=bool)
    min_score = np.array(current_scores, dtype=np.float32)

    for n in range(N):
        ct_list = cell_type_lists[n]
        k_sub = len(ct_list)
        n_types[n] = k_sub
        for i, t in enumerate(ct_list):
            cell_type_indices[n, i] = t
            sub_weights[n, i] = final_weights_dict[n][i]
            conf_list_arr[n, i] = conf_lists[n][t]

    return MultiResult(
        weights=W_full,
        sub_weights=sub_weights,
        cell_type_indices=cell_type_indices,
        n_types=n_types,
        conf_list=conf_list_arr,
        min_score=min_score,
        cell_type_names=cell_type_names,
    )
