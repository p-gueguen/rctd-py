from typing import List, Tuple

import numpy as np
import torch

from rctd import _numba_kernels, _triton_kernels
from rctd._full import run_full_mode
from rctd._irwls import solve_irwls_batch
from rctd._likelihood import calc_log_likelihood_batch
from rctd._types import MultiResult, RCTDConfig, resolve_device


def _build_multi_numba_ctx(
    spatial_counts,
    spatial_numi,
    P_gpu: torch.Tensor,
    Q_gpu: torch.Tensor,
    SQ_gpu: torch.Tensor,
    X_gpu: torch.Tensor,
):
    poly, k_of_l, x_of_l, k_val, lam_max = _numba_kernels.build_poly_table(Q_gpu, SQ_gpu, X_gpu)
    Y_int = _numba_kernels.prepare_y_int(spatial_counts, k_val)
    nUMI_arr = (
        spatial_numi.detach().cpu().numpy()
        if isinstance(spatial_numi, torch.Tensor)
        else spatial_numi
    )
    nUMI_np = np.ascontiguousarray(nUMI_arr, dtype=np.float64)
    P_KT = np.ascontiguousarray(P_gpu.detach().cpu().numpy().T, dtype=np.float64)
    return Y_int, nUMI_np, P_KT, poly, k_of_l, x_of_l, lam_max


def _build_multi_triton_ctx(
    spatial_counts_gpu: torch.Tensor,
    spatial_numi_gpu: torch.Tensor,
    P_gpu: torch.Tensor,
    Q_gpu: torch.Tensor,
    SQ_gpu: torch.Tensor,
    X_gpu: torch.Tensor,
):
    poly_gpu = _triton_kernels.build_poly_table_gpu(Q_gpu, SQ_gpu, X_gpu)
    Y_int_gpu = _triton_kernels.prepare_y_int_gpu(spatial_counts_gpu, poly_gpu[6])
    P_KT_gpu = P_gpu.T.contiguous()
    return Y_int_gpu, spatial_numi_gpu.contiguous(), P_KT_gpu, poly_gpu


def _run_batched_scoring(
    task_triples: List[Tuple[int, List[int]]],
    spatial_numi: np.ndarray,
    spatial_counts: np.ndarray,
    P_gpu: torch.Tensor,
    Q_gpu: torch.Tensor,
    SQ_gpu: torch.Tensor,
    X_gpu: torch.Tensor,
    batch_size: int,
    n_iter: int = 25,
    numba_ctx=None,
    triton_ctx=None,
) -> np.ndarray:
    """Run batched IRWLS for a set of tasks all having the SAME number of cell types K_sub."""
    if not task_triples:
        return np.array([])

    if _numba_kernels.use_numba(P_gpu):
        if numba_ctx is None:
            numba_ctx = _build_multi_numba_ctx(
                spatial_counts, spatial_numi, P_gpu, Q_gpu, SQ_gpu, X_gpu
            )
        Y_int, nUMI_np, P_KT, poly, k_of_l, x_of_l, lam_max = numba_ctx
        pix_idx = np.fromiter((t[0] for t in task_triples), dtype=np.int64, count=len(task_triples))
        type_matrix = np.array([t[1] for t in task_triples], dtype=np.int64)
        scores = _numba_kernels.fit_and_score_subsets_cpu(
            pix_idx,
            type_matrix,
            Y_int,
            nUMI_np,
            P_KT,
            poly,
            k_of_l,
            x_of_l,
            lam_max,
            max_iter=n_iter,
        )
        return scores.astype(
            P_gpu.numpy().dtype if P_gpu.dtype != torch.float64 else np.float64, copy=False
        )

    if _triton_kernels.use_triton(P_gpu):
        if triton_ctx is None:
            sc_t = (
                spatial_counts
                if isinstance(spatial_counts, torch.Tensor)
                else torch.as_tensor(spatial_counts, device=P_gpu.device)
            )
            sn_t = (
                spatial_numi
                if isinstance(spatial_numi, torch.Tensor)
                else torch.as_tensor(spatial_numi, device=P_gpu.device)
            )
            triton_ctx = _build_multi_triton_ctx(sc_t, sn_t, P_gpu, Q_gpu, SQ_gpu, X_gpu)
        Y_int_gpu, nUMI_gpu_c, P_KT_gpu, poly_gpu = triton_ctx
        device = P_gpu.device
        pix_idx_t = torch.tensor([t[0] for t in task_triples], dtype=torch.int64, device=device)
        type_mat_t = torch.tensor([t[1] for t in task_triples], dtype=torch.int64, device=device)
        k_sub = type_mat_t.shape[1]
        if k_sub == 1:
            _, scores_t = _triton_kernels.fit_and_score_singles_triton(
                pix_idx_t,
                type_mat_t[:, 0],
                Y_int_gpu,
                nUMI_gpu_c,
                P_KT_gpu,
                poly_gpu,
                max_iter=n_iter,
                write_weights=False,
                compute_score=True,
            )
        elif k_sub == 2:
            _, scores_t, _ = _triton_kernels.fit_and_score_pairs_triton(
                pix_idx_t,
                type_mat_t[:, 0],
                type_mat_t[:, 1],
                Y_int_gpu,
                nUMI_gpu_c,
                P_KT_gpu,
                poly_gpu,
                max_iter=n_iter,
                compute_score=True,
            )
        else:
            _, scores_t = _triton_kernels.fit_subsets_triton(
                pix_idx_t,
                type_mat_t,
                Y_int_gpu,
                nUMI_gpu_c,
                P_KT_gpu,
                poly_gpu,
                max_iter=n_iter,
                compute_score=True,
            )
        return scores_t.cpu().numpy()

    M = len(task_triples)
    device = P_gpu.device

    all_scores = []

    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)

        pix_idx = []
        type_idx = []
        for i in range(start, end):
            pix_idx.append(task_triples[i][0])
            type_idx.append(task_triples[i][1])

        pix_idx = np.array(pix_idx, dtype=np.int32)
        type_idx = np.array(type_idx, dtype=np.int32)  # (bs, K_sub)

        if isinstance(spatial_counts, torch.Tensor):
            pix_idx_t = torch.as_tensor(pix_idx, dtype=torch.long, device=device)
            nUMI_tr = spatial_numi[pix_idx_t]
            B_tr = spatial_counts[pix_idx_t]
        else:
            nUMI_tr = torch.tensor(spatial_numi[pix_idx], device=device)
            B_tr = torch.tensor(spatial_counts[pix_idx], device=device)

        # P_gpu is (G, K). We need (bs, G, K_sub)
        # type_idx is (bs, K_sub) -> convert to long tensor for indexing
        type_idx_t = torch.tensor(type_idx, dtype=torch.long, device=device)  # (bs, K_sub)
        # P_gpu[:, type_idx_t] -> shape (G, bs, K_sub). Then permute to (bs, G, K_sub)
        P_sub = P_gpu[:, type_idx_t].permute(1, 0, 2)  # (bs, G, K_sub)

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

        expected_tr = torch.sum(S_sub * weights_batch[:, None, :], dim=-1)  # (bs, G)
        expected_tr = torch.clamp(expected_tr, min=1e-4)

        scores_batch = calc_log_likelihood_batch(B_tr, expected_tr, Q_gpu, SQ_gpu, X_gpu)
        all_scores.append(scores_batch)

    return torch.cat(all_scores).cpu().numpy()


def _run_batched_weights(
    task_triples: List[Tuple[int, List[int]]],
    spatial_numi: np.ndarray,
    spatial_counts: np.ndarray,
    P_gpu: torch.Tensor,
    Q_gpu: torch.Tensor,
    SQ_gpu: torch.Tensor,
    X_gpu: torch.Tensor,
    batch_size: int,
    n_iter: int = 50,
    numba_ctx=None,
    triton_ctx=None,
) -> List[np.ndarray]:
    if not task_triples:
        return []

    if _numba_kernels.use_numba(P_gpu):
        if numba_ctx is None:
            numba_ctx = _build_multi_numba_ctx(
                spatial_counts, spatial_numi, P_gpu, Q_gpu, SQ_gpu, X_gpu
            )
        Y_int, nUMI_np, P_KT, poly, k_of_l, x_of_l, lam_max = numba_ctx
        pix_idx = np.fromiter((t[0] for t in task_triples), dtype=np.int64, count=len(task_triples))
        type_matrix = np.array([t[1] for t in task_triples], dtype=np.int64)
        w_res = _numba_kernels.fit_subsets_weights_cpu(
            pix_idx,
            type_matrix,
            Y_int,
            nUMI_np,
            P_KT,
            poly,
            k_of_l,
            x_of_l,
            lam_max,
            max_iter=n_iter,
        )
        return list(w_res.astype(P_gpu.numpy().dtype, copy=False))

    if _triton_kernels.use_triton(P_gpu):
        if triton_ctx is None:
            sc_t = (
                spatial_counts
                if isinstance(spatial_counts, torch.Tensor)
                else torch.as_tensor(spatial_counts, device=P_gpu.device)
            )
            sn_t = (
                spatial_numi
                if isinstance(spatial_numi, torch.Tensor)
                else torch.as_tensor(spatial_numi, device=P_gpu.device)
            )
            triton_ctx = _build_multi_triton_ctx(sc_t, sn_t, P_gpu, Q_gpu, SQ_gpu, X_gpu)
        Y_int_gpu, nUMI_gpu_c, P_KT_gpu, poly_gpu = triton_ctx
        device = P_gpu.device
        pix_idx_t = torch.tensor([t[0] for t in task_triples], dtype=torch.int64, device=device)
        type_mat_t = torch.tensor([t[1] for t in task_triples], dtype=torch.int64, device=device)
        k_sub = type_mat_t.shape[1]
        if k_sub == 1:
            w_t, _ = _triton_kernels.fit_and_score_singles_triton(
                pix_idx_t,
                type_mat_t[:, 0],
                Y_int_gpu,
                nUMI_gpu_c,
                P_KT_gpu,
                poly_gpu,
                max_iter=n_iter,
                write_weights=True,
                compute_score=False,
            )
            w_t = w_t.unsqueeze(1)
        elif k_sub == 2:
            w_t, _, _ = _triton_kernels.fit_and_score_pairs_triton(
                pix_idx_t,
                type_mat_t[:, 0],
                type_mat_t[:, 1],
                Y_int_gpu,
                nUMI_gpu_c,
                P_KT_gpu,
                poly_gpu,
                max_iter=n_iter,
                compute_score=False,
            )
        else:
            w_t, _ = _triton_kernels.fit_subsets_triton(
                pix_idx_t,
                type_mat_t,
                Y_int_gpu,
                nUMI_gpu_c,
                P_KT_gpu,
                poly_gpu,
                max_iter=n_iter,
                compute_score=False,
            )
        return list(w_t.cpu().numpy())

    M = len(task_triples)
    device = P_gpu.device

    all_weights = []

    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)

        pix_idx = []
        type_idx = []
        for i in range(start, end):
            pix_idx.append(task_triples[i][0])
            type_idx.append(task_triples[i][1])

        pix_idx = np.array(pix_idx, dtype=np.int32)
        type_idx = np.array(type_idx, dtype=np.int32)

        if isinstance(spatial_counts, torch.Tensor):
            pix_idx_t = torch.as_tensor(pix_idx, dtype=torch.long, device=device)
            nUMI_tr = spatial_numi[pix_idx_t]
            B_tr = spatial_counts[pix_idx_t]
        else:
            nUMI_tr = torch.tensor(spatial_numi[pix_idx], device=device)
            B_tr = torch.tensor(spatial_counts[pix_idx], device=device)

        type_idx_t = torch.tensor(type_idx, dtype=torch.long, device=device)
        P_sub = P_gpu[:, type_idx_t].permute(1, 0, 2)
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
        all_weights.append(weights_batch)

    return list(torch.cat(all_weights).cpu().numpy())


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
    device: str = "auto",
    pixel_mask: np.ndarray | None = None,
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
        batch_size: Rows per GPU batch: pixels in the full-mode fit, (pixel, candidate set)
            pairs in the selection steps
    """
    N, G = spatial_counts.shape
    device = resolve_device(device)

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
        device=str(device),
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

    P_gpu = torch.tensor(norm_profiles, device=device)
    Q_gpu = torch.tensor(q_mat, device=device)
    SQ_gpu = torch.tensor(sq_mat, device=device)
    X_gpu = torch.tensor(x_vals, device=device)

    # Pre-stage spatial data on the device once instead of re-uploading per batch
    # (same fix as _doublet.py: a pixel appears in many tasks across iterations).
    spatial_numi_gpu = torch.tensor(spatial_numi, device=device)
    spatial_counts_gpu = torch.tensor(spatial_counts, device=device)
    triton_ctx = (
        _build_multi_triton_ctx(spatial_counts_gpu, spatial_numi_gpu, P_gpu, Q_gpu, SQ_gpu, X_gpu)
        if _triton_kernels.use_triton(P_gpu)
        else None
    )
    if triton_ctx is not None:
        del spatial_counts_gpu
        spatial_counts_gpu = None
    numba_ctx = (
        _build_multi_numba_ctx(spatial_counts, spatial_numi, P_gpu, Q_gpu, SQ_gpu, X_gpu)
        if _numba_kernels.use_numba(P_gpu)
        else None
    )

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
            spatial_numi_gpu,
            spatial_counts_gpu,
            P_gpu,
            Q_gpu,
            SQ_gpu,
            X_gpu,
            batch_size,
            numba_ctx=numba_ctx,
            triton_ctx=triton_ctx,
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
            spatial_numi_gpu,
            spatial_counts_gpu,
            P_gpu,
            Q_gpu,
            SQ_gpu,
            X_gpu,
            batch_size,
            numba_ctx=numba_ctx,
            triton_ctx=triton_ctx,
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
            t_list,
            spatial_numi_gpu,
            spatial_counts_gpu,
            P_gpu,
            Q_gpu,
            SQ_gpu,
            X_gpu,
            batch_size,
            numba_ctx=numba_ctx,
            triton_ctx=triton_ctx,
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
        pixel_mask=pixel_mask,
    )
