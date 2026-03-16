"""Batched IRWLS solver for RCTD decomposition.

Ports solveIRWLS.weights, solveWLS, get_der_fast, psd from R spacexr.
Uses PyTorch for GPU acceleration.
"""

import torch

from rctd._likelihood import calc_q_all
from rctd._simplex import project_simplex, project_simplex_batch


def _get_derivatives(
    S: torch.Tensor,
    Y: torch.Tensor,
    prediction: torch.Tensor,
    Q_mat: torch.Tensor,
    SQ_mat: torch.Tensor,
    x_vals: torch.Tensor,
    bulk_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradient and Hessian for IRWLS step.

    Ports get_der_fast from R prob_model.R.

    Args:
        S: design matrix (G, K)
        Y: observed counts (G,)
        prediction: current predicted values (G,)
        Q_mat, SQ_mat, x_vals: likelihood tables
        bulk_mode: if True, use Gaussian approximation

    Returns:
        (grad, hess): gradient (K,) and Hessian (K, K)
    """
    if bulk_mode:
        d1_vec = -2.0 * (torch.log(prediction) - torch.log(Y + 1e-10)) / prediction
        d2_vec = -2.0 * (1.0 - torch.log(prediction) + torch.log(Y + 1e-10)) / prediction**2
    else:
        _, d1_vec, d2_vec = calc_q_all(Y, prediction, Q_mat, SQ_mat, x_vals)

    # R: grad = -d1_vec %*% S  (grad is (K,))
    grad = -(d1_vec @ S)

    # R: hess = S.T @ diag(-d2_vec) @ S
    # Vectorized: H = S.T @ ((-d2_vec)[:, None] * S)
    hess = S.T @ ((-d2_vec)[:, None] * S)

    return grad, hess


def _psd(H: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    """Project matrix to positive semi-definite.

    Ports psd from R prob_model.R.
    Uses eigendecomposition, clamping eigenvalues to >= epsilon.
    Matches R's fixed epsilon=1e-3; the box-constrained QP solver
    prevents divergence along ill-conditioned directions.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(H)
    eigenvalues = torch.clamp(eigenvalues, min=epsilon)
    return eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T


def _solve_box_qp(
    D: torch.Tensor,
    d: torch.Tensor,
    lower_bound: torch.Tensor,
    n_sweeps: int = 50,
) -> torch.Tensor:
    """Solve box-constrained QP: min 0.5 x^T D x - d^T x  s.t. x >= lb.

    Matches R's quadprog::solve.QP with only lower-bound constraints.
    Uses Gauss-Seidel coordinate descent which converges for PSD D.

    Args:
        D: PSD matrix (K, K)
        d: linear term (K,)
        lower_bound: lower bounds (K,)
        n_sweeps: number of full coordinate sweeps

    Returns:
        Optimal x (K,)
    """
    K = d.shape[0]
    # Warm start: diagonal approximation (stable for ill-conditioned D).
    # Using solve(D, d) blows up when cond(D) >> 1 (e.g. 5e6 for 45 types),
    # producing delta_w ~ 1e5 that causes unconstrained weights to diverge.
    # R's quadprog::solve.QP (active-set) doesn't have this issue.
    x = torch.maximum(d / torch.diagonal(D), lower_bound)

    for _ in range(n_sweeps):
        for i in range(K):
            # Exact minimization along coordinate i with others fixed:
            # x_i* = (d_i - sum_{j!=i} D_{ij} x_j) / D_{ii}
            residual = d[i] - torch.dot(D[i], x) + D[i, i] * x[i]
            x_i_opt = residual / D[i, i]
            x[i] = torch.max(x_i_opt, lower_bound[i])

    return x


@torch.no_grad()
def solve_irwls(
    S: torch.Tensor,
    Y: torch.Tensor,
    nUMI: float,
    Q_mat: torch.Tensor,
    SQ_mat: torch.Tensor,
    x_vals: torch.Tensor,
    max_iter: int = 50,
    min_change: float = 0.001,
    step_size: float = 0.3,
    constrain: bool = True,
    bulk_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve IRWLS for a single pixel.

    Ports solveIRWLS.weights from R IRWLS.R.

    Args:
        S: design matrix (G, K) = cell_type_profiles * nUMI
        Y: observed counts (G,)
        nUMI: total UMI for this pixel (scalar)
        Q_mat, SQ_mat, x_vals: likelihood tables
        max_iter: maximum iterations
        min_change: convergence tolerance
        step_size: damping factor (alpha=0.3 in R)
        constrain: if True, project weights onto simplex (sum=1, non-neg)
        bulk_mode: if True, use Gaussian likelihood

    Returns:
        (weights, converged): weights (K,) and convergence flag (scalar bool tensor)
    """
    K = S.shape[1]
    dtype = S.dtype
    device = S.device
    # R: if(!bulk_mode) B[B > K_val] <- K_val
    if not bulk_mode:
        k_val = Q_mat.shape[0] - 3
        Y = torch.clamp(Y, max=k_val)
    w = torch.ones(K, dtype=dtype, device=device) / K

    change = 1.0  # initial sentinel ensures at least one iteration
    for _ in range(max_iter):
        if change <= min_change:
            break
        # R: solution <- pmax(initialSol, 0)
        solution = torch.clamp(w, min=0.0)
        # R: prediction = abs(S %*% solution)
        prediction = torch.abs(S @ solution)
        # R: threshold = max(1e-4, nUMI * 1e-7)
        threshold = max(1e-4, float(nUMI) * 1e-7)
        prediction = torch.clamp(prediction, min=threshold)

        grad, hess = _get_derivatives(S, Y, prediction, Q_mat, SQ_mat, x_vals, bulk_mode)
        hess = _psd(hess)

        # R: norm_factor <- norm(D_mat, "2")
        norm_factor = torch.linalg.norm(hess, ord=2)
        norm_factor = max(norm_factor.item(), 1e-10)
        # R: D_mat <- D_mat / norm_factor
        # R: d_vec <- -derivatives$grad / norm_factor  (note: R negates grad)
        D = hess / norm_factor + 1e-7 * torch.eye(K, dtype=dtype, device=device)
        d = -grad / norm_factor

        # Solve box-constrained QP: min 0.5 x^T D x - d^T x  s.t. x >= -solution
        delta_w = _solve_box_qp(D, d, -solution)

        # Damped update (R: solution <- solution + alpha * delta_w)
        w_new = solution + step_size * delta_w

        # For constrained mode, additionally project onto simplex
        if constrain:
            w_new = project_simplex(w_new)

        # R uses norm(as.matrix(x)) which defaults to L1 norm (sum of abs)
        # for a column vector, NOT L2 norm
        change = torch.sum(torch.abs(w_new - w)).item()
        w = w_new

    converged = change <= min_change
    return w, torch.tensor(converged, dtype=torch.bool, device=device)


def _get_derivatives_batch(
    S: torch.Tensor,
    Y: torch.Tensor,
    prediction: torch.Tensor,
    Q_mat: torch.Tensor,
    SQ_mat: torch.Tensor,
    x_vals: torch.Tensor,
    bulk_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched gradient and Hessian computation.

    Args:
        S: (N, G, K) design matrices
        Y: (N, G) observed counts
        prediction: (N, G) current predicted values
        Q_mat, SQ_mat, x_vals: likelihood tables
        bulk_mode: if True, use Gaussian approximation

    Returns:
        (grad, hess): gradient (N, K) and Hessian (N, K, K)
    """
    N, G, K = S.shape
    if bulk_mode:
        d1_vec = -2.0 * (torch.log(prediction) - torch.log(Y + 1e-10)) / prediction
        d2_vec = -2.0 * (1.0 - torch.log(prediction) + torch.log(Y + 1e-10)) / prediction**2
    else:
        # Flatten (N, G) → (N*G,) for calc_q_all, then reshape back
        Y_flat = Y.reshape(-1)
        pred_flat = prediction.reshape(-1)
        _, d1_flat, d2_flat = calc_q_all(Y_flat, pred_flat, Q_mat, SQ_mat, x_vals)
        d1_vec = d1_flat.reshape(N, G)
        d2_vec = d2_flat.reshape(N, G)

    # grad[n] = -(d1[n] @ S[n]) → (N, 1, G) @ (N, G, K) → (N, 1, K) → (N, K)
    grad = -torch.bmm(d1_vec.unsqueeze(1), S).squeeze(1)

    # hess[n] = S[n].T @ diag(-d2[n]) @ S[n]
    # = (S[n] * (-d2[n])[:, None]).T @ S[n]
    # Batched: (N, K, G) @ (N, G, K) where S_weighted = S * (-d2)[:, :, None]
    S_weighted = S * (-d2_vec).unsqueeze(2)  # (N, G, K)
    hess = torch.bmm(S_weighted.transpose(1, 2), S)  # (N, K, K)

    return grad, hess


def _psd_batch(H: torch.Tensor, epsilon: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched PSD projection. H: (N, K, K) → (H_psd (N, K, K), max_eig (N,)).

    For K<=16, runs eigh directly on GPU — avoids costly CPU↔GPU transfer
    that otherwise dominates runtime. For larger K (e.g. K=45), CPU
    OpenBLAS is faster than cuSOLVER's batched syevj.
    """
    K = H.shape[-1]
    if H.device.type == "cuda" and K <= 16:
        # Small matrices: GPU eigh avoids CPU↔GPU transfer overhead
        eigenvalues, eigenvectors = torch.linalg.eigh(H)
        eigenvalues = torch.clamp(eigenvalues, min=epsilon)
        # Fused reconstruction: V * sqrt(λ) → (V*sqrt(λ)) @ (V*sqrt(λ))^T
        # Equivalent to V @ diag(λ) @ V^T but avoids diag_embed
        scaled = eigenvectors * eigenvalues.unsqueeze(-2).sqrt()  # (N, K, K)
        H_psd = scaled @ scaled.transpose(-1, -2)
        max_eig = eigenvalues[:, -1]
        return H_psd, max_eig
    else:
        # Large K or CPU: offload to CPU for eigh
        orig_device = H.device
        orig_dtype = H.dtype
        H_cpu = H if H.device.type == "cpu" else H.cpu()
        eigenvalues, eigenvectors = torch.linalg.eigh(H_cpu)
        eigenvalues = torch.clamp(eigenvalues, min=epsilon)
        H_psd = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)
        max_eig = eigenvalues[:, -1]
        if H.device.type == "cpu":
            return H_psd, max_eig
        return H_psd.to(device=orig_device, dtype=orig_dtype), max_eig.to(
            device=orig_device, dtype=orig_dtype
        )


@torch.compile(dynamic=True)
def _solve_box_qp_batch_compiled(
    D: torch.Tensor,
    d: torch.Tensor,
    lower_bound: torch.Tensor,
    n_sweeps: int = 50,
) -> torch.Tensor:
    """Compiled Gauss-Seidel coordinate descent for batched box-constrained QP.

    torch.compile with dynamic=True fuses the inner coordinate descent
    operations into optimized kernels via Triton/Inductor, reducing memory
    traffic and kernel launch overhead. Dynamic shapes avoid excessive
    recompilations when batch size N or type count K varies across calls.
    """
    K = d.shape[1]
    D_diag = torch.diagonal(D, dim1=-2, dim2=-1)  # (N, K)
    x = torch.maximum(d / D_diag, lower_bound)  # (N, K)

    for _ in range(n_sweeps):
        for i in range(K):
            Dx_i = (D[:, i, :] * x).sum(-1)  # (N,)
            residual = d[:, i] - Dx_i + D[:, i, i] * x[:, i]
            x[:, i] = torch.maximum(residual / D[:, i, i], lower_bound[:, i])

    return x


@torch.jit.script
def _solve_box_qp_batch_adaptive_jit(
    D: torch.Tensor,
    d: torch.Tensor,
    lower_bound: torch.Tensor,
    n_sweeps: int = 50,
    check_interval: int = 5,
    tol: float = 1e-4,
) -> torch.Tensor:
    """JIT-compiled Gauss-Seidel with batch-level early exit.

    Checks convergence every check_interval sweeps: if the maximum absolute
    change across all pixels and coordinates is below tol, exits early.
    """
    K = d.shape[1]
    D_diag = torch.diagonal(D, dim1=-2, dim2=-1)
    x = torch.maximum(d / D_diag, lower_bound)

    for sweep in range(n_sweeps):
        x_prev = x.clone()
        for i in range(K):
            Dx_i = (D[:, i, :] * x).sum(-1)
            residual = d[:, i] - Dx_i + D[:, i, i] * x[:, i]
            x[:, i] = torch.maximum(residual / D[:, i, i], lower_bound[:, i])

        if (sweep + 1) % check_interval == 0:
            max_change = (x - x_prev).abs().max()
            if max_change < tol:
                break

    return x


def _solve_box_qp_batch(
    D: torch.Tensor,
    d: torch.Tensor,
    lower_bound: torch.Tensor,
    n_sweeps: int = 50,
) -> torch.Tensor:
    """Batched box-constrained QP: min 0.5 x^T D x - d^T x  s.t. x >= lb.

    Gauss-Seidel coordinate descent, vectorized across pixels.
    Delegates to JIT-compiled implementation for performance.

    Args:
        D: (N, K, K) PSD matrices
        d: (N, K) linear terms
        lower_bound: (N, K) lower bounds
        n_sweeps: number of coordinate sweeps

    Returns:
        x: (N, K) optimal solutions
    """
    return _solve_box_qp_batch_compiled(D, d, lower_bound, n_sweeps)


@torch.no_grad()
def solve_irwls_batch(
    S_batch: torch.Tensor,
    Y_batch: torch.Tensor,
    nUMI_batch: torch.Tensor,
    Q_mat: torch.Tensor,
    SQ_mat: torch.Tensor,
    x_vals: torch.Tensor,
    max_iter: int = 50,
    min_change: float = 0.001,
    step_size: float = 0.3,
    constrain: bool = True,
    bulk_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve IRWLS for a batch of pixels, fully vectorized.

    All N pixels are processed simultaneously using batched matrix operations.

    Args:
        S_batch: (N, G, K) design matrices
        Y_batch: (N, G) observed counts
        nUMI_batch: (N,) total UMI per pixel
        Q_mat, SQ_mat, x_vals: shared likelihood tables
        max_iter: maximum IRWLS iterations
        min_change: convergence tolerance (L1 norm of weight change)
        step_size: damping factor
        constrain: project weights onto simplex
        bulk_mode: use Gaussian likelihood

    Returns:
        (weights, converged): (N, K) and (N,) bool arrays
    """
    N, G, K = S_batch.shape
    dtype = S_batch.dtype
    device = S_batch.device

    if not bulk_mode:
        k_val = Q_mat.shape[0] - 3
        Y_batch = torch.clamp(Y_batch, max=k_val)

    w = torch.ones(N, K, dtype=dtype, device=device) / K
    eye_K = torch.eye(K, dtype=dtype, device=device)  # shared identity

    # Per-pixel convergence tracking
    change = torch.full((N,), float("inf"), dtype=dtype, device=device)
    converged = torch.zeros(N, dtype=torch.bool, device=device)

    # Threshold per pixel: max(1e-4, nUMI * 1e-7)
    threshold = torch.clamp(nUMI_batch * 1e-7, min=1e-4)  # (N,)

    for it in range(max_iter):
        # Only process non-converged pixels
        active = ~converged
        if not active.any():
            break

        solution = torch.clamp(w, min=0.0)  # (N, K)

        # prediction = |S @ w|: (N, G, K) @ (N, K, 1) → (N, G)
        prediction = torch.abs(torch.bmm(S_batch, solution.unsqueeze(2)).squeeze(2))
        prediction = torch.clamp(prediction, min=threshold.unsqueeze(1))

        grad, hess = _get_derivatives_batch(
            S_batch, Y_batch, prediction, Q_mat, SQ_mat, x_vals, bulk_mode
        )

        hess, norm_factor = _psd_batch(hess)
        norm_factor = torch.clamp(norm_factor, min=1e-10)

        D = hess / norm_factor[:, None, None] + 1e-7 * eye_K.unsqueeze(0)
        d = -grad / norm_factor[:, None]

        delta_w = _solve_box_qp_batch(D, d, -solution)

        w_new = solution + step_size * delta_w

        if constrain:
            w_new = project_simplex_batch(w_new)

        # Per-pixel L1 change
        change = torch.sum(torch.abs(w_new - w), dim=1)  # (N,)
        newly_converged = change <= min_change

        # Freeze converged pixels: only update active ones
        w = torch.where(active.unsqueeze(1), w_new, w)
        converged = converged | newly_converged

    return w, converged


@torch.no_grad()
def solve_irwls_batch_shared(
    P: torch.Tensor,
    Y_batch: torch.Tensor,
    nUMI_batch: torch.Tensor,
    Q_mat: torch.Tensor,
    SQ_mat: torch.Tensor,
    x_vals: torch.Tensor,
    max_iter: int = 50,
    min_change: float = 0.001,
    step_size: float = 0.3,
    constrain: bool = True,
    bulk_mode: bool = False,
    _calc_q_fn=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimized IRWLS for shared reference profiles.

    When all pixels share the same profile matrix P, this replaces bmm with
    matmul for prediction, gradient, and Hessian. Active pixel compaction
    skips converged pixels entirely each iteration.

    Args:
        P: (G, K) shared reference profiles (NOT pre-scaled by nUMI)
        Y_batch: (N, G) observed counts
        nUMI_batch: (N,) total UMI per pixel
        Q_mat, SQ_mat, x_vals: shared likelihood tables
        max_iter: maximum IRWLS iterations
        min_change: convergence tolerance (L1 norm of weight change)
        step_size: damping factor
        constrain: project weights onto simplex
        bulk_mode: use Gaussian likelihood

    Returns:
        (weights, converged): (N, K) and (N,) bool arrays
    """
    _q_fn = _calc_q_fn if _calc_q_fn is not None else calc_q_all

    N = Y_batch.shape[0]
    if N == 0:
        K = P.shape[1]
        return (
            torch.empty(0, K, dtype=P.dtype, device=P.device),
            torch.empty(0, dtype=torch.bool, device=P.device),
        )

    G, K = P.shape
    dtype = P.dtype
    device = P.device

    if not bulk_mode:
        k_val = Q_mat.shape[0] - 3
        Y_batch = torch.clamp(Y_batch, max=k_val)

    w = torch.ones(N, K, dtype=dtype, device=device) / K
    eye_K = torch.eye(K, dtype=dtype, device=device)
    converged = torch.zeros(N, dtype=torch.bool, device=device)

    # Precompute frequently used derived matrices
    P_T = P.T.contiguous()  # (K, G) — avoids repeated non-contiguous transpose
    P_outer = (P[:, :, None] * P[:, None, :]).reshape(G, K * K)

    threshold = torch.clamp(nUMI_batch * 1e-7, min=1e-4)

    # Active pixel state — start with all pixels
    active_idx = torch.arange(N, device=device)
    Y_act = Y_batch
    nUMI_act = nUMI_batch
    thresh_act = threshold
    w_act = w.clone()

    for it in range(max_iter):
        n_act = active_idx.shape[0]
        if n_act == 0:
            break

        solution = torch.clamp(w_act, min=0.0)

        # Prediction: matmul replaces bmm
        # S @ w = (nUMI * P) @ w = nUMI * (w @ P.T)
        prediction = torch.abs(nUMI_act.unsqueeze(1) * (solution @ P_T))
        prediction = torch.clamp(prediction, min=thresh_act.unsqueeze(1))

        # Derivatives
        if bulk_mode:
            d1_vec = -2.0 * (torch.log(prediction) - torch.log(Y_act + 1e-10)) / prediction
            d2_vec = -2.0 * (1.0 - torch.log(prediction) + torch.log(Y_act + 1e-10)) / prediction**2
        else:
            _, d1_flat, d2_flat = _q_fn(
                Y_act.reshape(-1), prediction.reshape(-1), Q_mat, SQ_mat, x_vals
            )
            d1_vec = d1_flat.reshape(n_act, G)
            d2_vec = d2_flat.reshape(n_act, G)

        # Gradient: matmul replaces bmm
        # grad = -(d1 @ S) = -(d1 * nUMI) @ P
        grad = -((d1_vec * nUMI_act.unsqueeze(1)) @ P)

        # Hessian via P outer product: matmul replaces bmm
        # H[n,i,j] = nUMI[n]^2 * sum_g (-d2[n,g]) * P[g,i] * P[g,j]
        d2_w = (-d2_vec) * (nUMI_act**2).unsqueeze(1)  # (n_act, G)
        hess = (d2_w @ P_outer).reshape(n_act, K, K)

        hess, norm_factor = _psd_batch(hess)
        norm_factor = torch.clamp(norm_factor, min=1e-10)

        D = hess / norm_factor[:, None, None] + 1e-7 * eye_K.unsqueeze(0)
        d = -grad / norm_factor[:, None]

        delta_w = _solve_box_qp_batch(D, d, -solution)
        w_new = solution + step_size * delta_w

        if constrain:
            w_new = project_simplex_batch(w_new)

        change = torch.sum(torch.abs(w_new - w_act), dim=1)
        newly_converged = change <= min_change
        w_act = w_new

        # Scatter results back to full arrays
        w[active_idx] = w_act
        converged[active_idx] = converged[active_idx] | newly_converged

        # Compact to still-active pixels
        still_active = ~newly_converged
        if newly_converged.all():
            break
        if newly_converged.any():
            active_idx = active_idx[still_active]
            w_act = w_act[still_active]
            Y_act = Y_act[still_active]
            nUMI_act = nUMI_act[still_active]
            thresh_act = thresh_act[still_active]

    return w, converged
