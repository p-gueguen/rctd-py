"""Multimodal RCTD: joint cell-type deconvolution across modalities.

Extends rctd-py to deconvolve a spatial pixel using several measurement
modalities at once (e.g. RNA + protein/ADT, RNA + ATAC gene-activity, or
two gene panels) that share the *same* set of K cell types.

Mathematical basis
-------------------
RCTD's per-pixel objective is a sum over features (genes) of Poisson-
lognormal negative log-likelihoods, driven by a single shared cell-type
weight vector ``w`` (see ``_irwls.solve_irwls_batch_shared``). The IRWLS
gradient ``-(d1 * nUMI) @ P`` and Hessian ``P^T diag(-d2 * nUMI^2) P`` are
both *additive sums over features*. Concatenating features from several
modalities is therefore equivalent to summing each modality's gradient and
Hessian contribution:

    grad_total = sum_m  beta_m * grad_m
    hess_total = sum_m  beta_m * hess_m

followed by a single PSD projection -> box-constrained QP -> simplex step on
the shared ``w``. Each modality keeps its own normalized profiles ``P_m``,
counts ``Y_m``, per-pixel depth ``nUMI_m`` and its own noise model
(``Q_m``/``SQ_m`` at that modality's estimated sigma). ``beta_m`` is an
optional per-modality weight (default 1.0) that rebalances modalities whose
count scales differ by orders of magnitude.

With a single modality and ``beta = 1`` this reduces *exactly* to stock
``solve_irwls_batch_shared`` — the reduction is the primary correctness test.
"""

from __future__ import annotations

import torch

from rctd._likelihood import calc_q_all
from rctd._simplex import project_simplex_batch


def _accumulate_grad_hess(
    P_list,
    P_T_list,
    P_outer_list,
    Y_list,
    nUMI_list,
    thresh_list,
    beta_list,
    w_act,
    Q_list,
    SQ_list,
    x_vals,
    bulk_mode,
):
    """Sum per-modality IRWLS gradient and Hessian at the current weights.

    Returns (grad (n,K), hess (n,K,K)). All modalities share ``w_act``.
    """
    n_act, K = w_act.shape
    solution = torch.clamp(w_act, min=0.0)

    grad = None
    hess = None
    for m, P in enumerate(P_list):
        G = P.shape[0]
        nUMI = nUMI_list[m]
        Y = Y_list[m]
        beta = beta_list[m]

        # prediction = |nUMI * (w @ P^T)|  (matmul form, shared profiles)
        prediction = torch.abs(nUMI.unsqueeze(1) * (solution @ P_T_list[m]))
        prediction = torch.clamp(prediction, min=thresh_list[m].unsqueeze(1))

        if bulk_mode:
            d1 = -2.0 * (torch.log(prediction) - torch.log(Y + 1e-10)) / prediction
            d2 = -2.0 * (1.0 - torch.log(prediction) + torch.log(Y + 1e-10)) / prediction**2
        else:
            _, d1_flat, d2_flat = calc_q_all(
                Y.reshape(-1), prediction.reshape(-1), Q_list[m], SQ_list[m], x_vals
            )
            d1 = d1_flat.reshape(n_act, G)
            d2 = d2_flat.reshape(n_act, G)

        grad_m = -((d1 * nUMI.unsqueeze(1)) @ P)  # (n, K)
        d2_w = (-d2) * (nUMI**2).unsqueeze(1)  # (n, G)
        hess_m = (d2_w @ P_outer_list[m]).reshape(n_act, K, K)  # (n, K, K)

        # beta may be a scalar (global weight) or a (n_act,) per-pixel tensor
        # (adaptive weighting). Broadcast it onto grad (n,K) and hess (n,K,K).
        if torch.is_tensor(beta) and beta.ndim >= 1:
            bg = beta.reshape(n_act, 1)
            bh = beta.reshape(n_act, 1, 1)
        else:
            bg = bh = beta

        if grad is None:
            grad = bg * grad_m
            hess = bh * hess_m
        else:
            grad = grad + bg * grad_m
            hess = hess + bh * hess_m
    return grad, hess


@torch.no_grad()
def solve_irwls_batch_multimodal(
    P_list,
    Y_list,
    nUMI_list,
    Q_list,
    SQ_list,
    x_vals,
    beta_list=None,
    max_iter: int = 50,
    min_change: float = 0.001,
    step_size: float = 0.3,
    constrain: bool = True,
    bulk_mode: bool = False,
    return_uncertainty: bool = False,
):
    """Joint IRWLS over M modalities sharing one weight vector per pixel.

    Args:
        P_list:    list of (G_m, K) shared normalized profiles per modality.
        Y_list:    list of (N, G_m) observed counts per modality (pixel-aligned).
        nUMI_list: list of (N,) per-pixel depth per modality.
        Q_list, SQ_list: per-modality likelihood tables (at each modality's sigma).
        x_vals:    shared lambda grid (identical across modalities).
        beta_list: optional (M,) per-modality weights; default all 1.0.
        max_iter, min_change, step_size, constrain, bulk_mode: as in stock IRWLS.
        return_uncertainty: if True, also return a per-pixel standard error on
            each cell-type weight, derived from the Laplace (inverse-Hessian)
            approximation of the joint likelihood at the solution. After IRWLS
            converges, the joint (beta-weighted) Hessian is recomputed once over
            all N pixels at the final ``w`` — a single extra forward pass, not a
            reuse of the loop's active-subset Hessians — then inverted per pixel;
            its diagonal sqrt is the per-weight standard error. The extra cost is
            roughly one IRWLS iteration plus a batched KxK inverse, and it gives
            the "principled uncertainty quantification" that probabilistic
            deconvolution methods are prized for (Isik et al. 2026, Patterns).

    Returns:
        ``(weights (N,K), converged (N,) bool)`` by default, or
        ``(weights, converged, se (N,K))`` when ``return_uncertainty=True``.
    """
    from rctd._irwls import _psd_batch, _solve_box_qp_batch

    M = len(P_list)
    if beta_list is None:
        beta_list = [1.0] * M
    assert len(Y_list) == M and len(nUMI_list) == M

    K = P_list[0].shape[1]
    for P in P_list:
        assert P.shape[1] == K, "all modalities must share the same K cell types"

    dtype = P_list[0].dtype
    device = P_list[0].device
    N = Y_list[0].shape[0]

    # Per-pixel adaptive weights: a beta_list entry may be a (N,) tensor
    # instead of a scalar. Such entries are compacted alongside active pixels.
    beta_is_pp = [torch.is_tensor(b) and b.ndim >= 1 for b in beta_list]
    beta_list = [
        (b.to(dtype=dtype, device=device) if beta_is_pp[m] else b) for m, b in enumerate(beta_list)
    ]
    if N == 0:
        return (
            torch.empty(0, K, dtype=dtype, device=device),
            torch.empty(0, dtype=torch.bool, device=device),
        )

    # Clamp counts to K_val per modality (matches stock behavior)
    if not bulk_mode:
        Y_list = [torch.clamp(Y_list[m], max=Q_list[m].shape[0] - 3) for m in range(M)]

    # Precompute per-modality derived matrices
    P_T_list = [P.T.contiguous() for P in P_list]
    P_outer_list = [(P[:, :, None] * P[:, None, :]).reshape(P.shape[0], K * K) for P in P_list]
    thresh_list = [torch.clamp(nUMI_list[m] * 1e-7, min=1e-4) for m in range(M)]

    w = torch.ones(N, K, dtype=dtype, device=device) / K
    eye_K = torch.eye(K, dtype=dtype, device=device)
    converged = torch.zeros(N, dtype=torch.bool, device=device)

    # Active-pixel compaction state (mirrors solve_irwls_batch_shared)
    active_idx = torch.arange(N, device=device)
    Y_act = [Y_list[m] for m in range(M)]
    nUMI_act = [nUMI_list[m] for m in range(M)]
    thresh_act = [thresh_list[m] for m in range(M)]
    beta_act = [beta_list[m] for m in range(M)]
    w_act = w.clone()

    for _ in range(max_iter):
        if active_idx.shape[0] == 0:
            break

        grad, hess = _accumulate_grad_hess(
            P_list,
            P_T_list,
            P_outer_list,
            Y_act,
            nUMI_act,
            thresh_act,
            beta_act,
            w_act,
            Q_list,
            SQ_list,
            x_vals,
            bulk_mode,
        )

        hess, norm_factor = _psd_batch(hess)
        norm_factor = torch.clamp(norm_factor, min=1e-10)

        D = hess / norm_factor[:, None, None] + 1e-7 * eye_K.unsqueeze(0)
        d = -grad / norm_factor[:, None]

        solution = torch.clamp(w_act, min=0.0)
        delta_w = _solve_box_qp_batch(D, d, -solution)
        w_new = solution + step_size * delta_w
        if constrain:
            w_new = project_simplex_batch(w_new)

        change = torch.sum(torch.abs(w_new - w_act), dim=1)
        newly_converged = change <= min_change
        w_act = w_new

        w[active_idx] = w_act
        converged[active_idx] = converged[active_idx] | newly_converged

        if newly_converged.all():
            break
        if newly_converged.any():
            still = ~newly_converged
            active_idx = active_idx[still]
            w_act = w_act[still]
            Y_act = [Y_act[m][still] for m in range(M)]
            nUMI_act = [nUMI_act[m][still] for m in range(M)]
            thresh_act = [thresh_act[m][still] for m in range(M)]
            beta_act = [(beta_act[m][still] if beta_is_pp[m] else beta_act[m]) for m in range(M)]

    if not return_uncertainty:
        return w, converged

    # Laplace / inverse-Hessian standard errors at the converged solution.
    # Recompute the joint (beta-weighted) Hessian over ALL pixels at w, then
    # invert per-pixel; sqrt of the diagonal is the per-weight standard error.
    beta_full = [beta_list[m] for m in range(M)]
    _, hess_full = _accumulate_grad_hess(
        P_list,
        P_T_list,
        P_outer_list,
        Y_list,
        nUMI_list,
        thresh_list,
        beta_full,
        w,
        Q_list,
        SQ_list,
        x_vals,
        bulk_mode,
    )
    # Ridge-regularize for invertibility (types absent from a pixel give a
    # singular block); the ridge is tiny relative to the Fisher information.
    ridge = 1e-6 * torch.diagonal(hess_full, dim1=1, dim2=2).mean(dim=1).clamp(min=1e-12)
    hess_full = hess_full + ridge[:, None, None] * eye_K.unsqueeze(0)
    cov = torch.linalg.inv(hess_full)  # (N, K, K)
    var = torch.diagonal(cov, dim1=1, dim2=2).clamp(min=0.0)  # (N, K)
    se = torch.sqrt(var)
    return w, converged, se


def adaptive_modality_weights(nUMI_list, mode="depth_share", eps=1e-8, gamma=1.0):
    """Data-driven per-pixel modality weights for ``solve_irwls_batch_multimodal``.

    In stock multimodal RCTD each modality carries a single global scalar
    ``beta``. That is inflexible when a pixel is deep in one modality and
    shallow in another: a cell with 5 RNA UMIs but 3000 protein counts should
    lean on protein at that pixel, not everywhere equally. This helper returns
    a list of per-pixel ``(N,)`` weight tensors — the convex-solve analogue of
    SpaRank's learned fusion gate, but derived directly from measured depth
    rather than trained. Because the weights enter the additive
    gradient/Hessian accumulation multiplicatively, the joint objective stays
    convex and the box-QP/simplex machinery is unchanged.

    Parameters
    ----------
    nUMI_list : list of (N,) tensors
        Per-pixel sequencing depth for each modality (as passed to the solver).
    mode : str
        ``"depth_share"`` (default): weight_m(pixel) proportional to that
        modality's share of log-depth at the pixel, normalized to sum to M so
        the overall scale matches unweighted beta=1. ``"sqrt_depth"``: weights
        proportional to sqrt(depth) share (gentler).
    eps : float
        Numerical floor.
    gamma : float
        Sharpness exponent applied to the per-modality depth signal before
        normalizing (``gamma>1`` concentrates weight on the deeper modality,
        ``gamma<1`` flattens toward equal weighting; ``gamma=0`` recovers
        uniform beta=1).

    Returns
    -------
    list of (N,) tensors
        ``beta_list`` to hand directly to ``solve_irwls_batch_multimodal``.
    """
    M = len(nUMI_list)
    dtype = nUMI_list[0].dtype
    device = nUMI_list[0].device

    sig = []
    for n in nUMI_list:
        n = n.to(dtype=dtype, device=device)
        if mode == "depth_share":
            s = torch.log1p(torch.clamp(n, min=0.0))
        elif mode == "sqrt_depth":
            s = torch.sqrt(torch.clamp(n, min=0.0))
        else:
            raise ValueError(f"unknown mode {mode!r}")
        sig.append(s)

    stacked = torch.stack(sig, dim=0)  # (M, N)
    stacked = torch.clamp(stacked, min=eps) ** gamma
    share = stacked / stacked.sum(dim=0, keepdim=True)  # (M, N), sums to 1
    # Rescale so the M weights at each pixel sum to M (matches beta=1 baseline
    # total when the two modalities are balanced).
    weights = share * M
    return [weights[m] for m in range(M)]


def jensen_shannon_divergence(P, Q, base=2, eps=1e-12):
    """Mean per-row Jensen-Shannon divergence between two proportion matrices.

    JSD is the standard proportion-error metric in the spatial-deconvolution
    benchmarking literature (e.g. Ouologuem et al. 2025, deconvATAC). Both
    inputs are (N, K) rows that each sum to 1 (cell-type proportions). Returns
    the mean JSD over rows; with ``base=2`` the value lies in [0, 1].

    Accepts numpy arrays or torch tensors; computes in numpy.
    """
    import numpy as np

    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    P = P / np.clip(P.sum(axis=1, keepdims=True), eps, None)
    Q = Q / np.clip(Q.sum(axis=1, keepdims=True), eps, None)
    M = 0.5 * (P + Q)

    def _kl(a, b):
        a = np.clip(a, eps, None)
        b = np.clip(b, eps, None)
        return np.sum(a * np.log(a / b), axis=1)

    jsd = 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)
    if base is not None:
        jsd = jsd / np.log(base)
    return float(np.mean(jsd))


def _morans_i(values, coords, k=6, eps=1e-12):
    """Moran's I spatial autocorrelation of a per-pixel scalar field.

    values : (N,) array. coords : (N, 2) spatial coordinates. Uses a symmetric
    k-nearest-neighbour binary weight matrix (row-normalized). Returns a scalar
    in roughly [-1, 1]; higher means stronger positive spatial autocorrelation.
    """
    import numpy as np

    x = np.asarray(values, dtype=np.float64)
    N = x.shape[0]
    if N < k + 1:
        return float("nan")
    xc = x - x.mean()
    coords = np.asarray(coords, dtype=np.float64)
    # kNN via pairwise distances (fine for benchmark-sized N; chunk if large)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]
    num = 0.0
    W = 0.0
    for i in range(N):
        for j in nn[i]:
            num += xc[i] * xc[j]
            W += 1.0
    den = (xc**2).sum()
    if den < eps or W == 0:
        return float("nan")
    return float((N / W) * (num / den))


def deconvolution_metrics(W_pred, W_true, coords=None, k=6):
    """Full proportion-error metric panel for spatial deconvolution.

    The core proportion-error metrics are those Isik et al. 2026 (Patterns)
    list as the community's consensus set for cell-type deconvolution: RMSE,
    MAE, Pearson correlation, Jensen-Shannon divergence, and SSIM. Two further
    metrics are added here for convenience and are NOT from that framing:
    dominant-type accuracy (hard-call agreement), and -- when ``coords`` is
    given -- a Moran's I spatial-autocorrelation comparison between predicted
    and true proportion maps. (The review discusses Moran's I / Geary's C under
    spatial-domain coherence on domain labels, not as a deconvolution metric;
    it is used here as an optional, ground-truth-free spatial sanity check.)

    Args:
        W_pred, W_true : (N, K) predicted / ground-truth proportion matrices.
        coords : optional (N, 2) spatial coordinates enabling the spatial
            (Moran's I) metric. When None, spatial metrics are omitted.
        k : neighbours for Moran's I / SSIM neighbourhood.

    Returns:
        dict of scalar metrics.
    """
    import numpy as np

    P = np.asarray(W_pred, dtype=np.float64)
    T = np.asarray(W_true, dtype=np.float64)
    assert P.shape == T.shape, "W_pred and W_true must have the same shape"
    N, K = P.shape

    out = {}
    out["rmse"] = float(np.sqrt(((P - T) ** 2).mean()))
    out["mae"] = float(np.abs(P - T).mean())
    out["corr"] = float(np.corrcoef(P.ravel(), T.ravel())[0, 1])
    out["jsd"] = jensen_shannon_divergence(P, T)
    out["dom_acc"] = float((P.argmax(1) == T.argmax(1)).mean())

    # Per-cell-type SSIM between predicted and true proportion "images"
    # (treating the length-N proportion vector for each type as a 1-D signal).
    def _ssim(a, b, C1=1e-4, C2=1e-4):
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(), b.var()
        cov = ((a - ma) * (b - mb)).mean()
        return ((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma**2 + mb**2 + C1) * (va + vb + C2))

    ssims = [_ssim(P[:, kk], T[:, kk]) for kk in range(K)]
    out["ssim_mean"] = float(np.mean(ssims))

    if coords is not None:
        coords = np.asarray(coords, dtype=np.float64)
        mp = [_morans_i(P[:, kk], coords, k=k) for kk in range(K)]
        mt = [_morans_i(T[:, kk], coords, k=k) for kk in range(K)]
        mp = np.array(mp, dtype=np.float64)
        mt = np.array(mt, dtype=np.float64)
        good = np.isfinite(mp) & np.isfinite(mt)
        out["morans_i_pred_mean"] = float(np.nanmean(mp))
        out["morans_i_true_mean"] = float(np.nanmean(mt))
        # How well predicted spatial autocorrelation tracks the truth's
        out["morans_i_mae"] = (
            float(np.abs(mp[good] - mt[good]).mean()) if good.any() else float("nan")
        )
    return out
