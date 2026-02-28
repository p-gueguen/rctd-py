"""Batched IRWLS solver for RCTD decomposition.

Ports solveIRWLS.weights, solveWLS, get_der_fast, psd from R spacexr.
Uses JAX jit + vmap for GPU acceleration.
"""

from functools import partial

import jax
import jax.numpy as jnp

from rctd._likelihood import calc_q_all
from rctd._simplex import project_simplex


def _get_derivatives(
    S: jnp.ndarray,
    Y: jnp.ndarray,
    prediction: jnp.ndarray,
    Q_mat: jnp.ndarray,
    SQ_mat: jnp.ndarray,
    x_vals: jnp.ndarray,
    bulk_mode: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
        d1_vec = -2.0 * (jnp.log(prediction) - jnp.log(Y + 1e-10)) / prediction
        d2_vec = -2.0 * (1.0 - jnp.log(prediction) + jnp.log(Y + 1e-10)) / prediction**2
    else:
        _, d1_vec, d2_vec = calc_q_all(Y, prediction, Q_mat, SQ_mat, x_vals)

    # R: grad = -d1_vec %*% S  (grad is (K,))
    grad = -(d1_vec @ S)

    # R: hess = S.T @ diag(-d2_vec) @ S
    # Vectorized: H = S.T @ ((-d2_vec)[:, None] * S)
    hess = S.T @ ((-d2_vec)[:, None] * S)

    return grad, hess


def _psd(H: jnp.ndarray, epsilon: float = 1e-3) -> jnp.ndarray:
    """Project matrix to positive semi-definite.

    Ports psd from R prob_model.R.
    Uses eigendecomposition, clamping eigenvalues to >= epsilon.
    Matches R's fixed epsilon=1e-3; the box-constrained QP solver
    prevents divergence along ill-conditioned directions.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(H)
    eigenvalues = jnp.maximum(eigenvalues, epsilon)
    return eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T


def _solve_box_qp(
    D: jnp.ndarray,
    d: jnp.ndarray,
    lower_bound: jnp.ndarray,
    n_sweeps: int = 50,
) -> jnp.ndarray:
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
    x = jnp.maximum(d / jnp.diag(D), lower_bound)

    def sweep(x, _):
        def coord_update(i, x):
            # Exact minimization along coordinate i with others fixed:
            # x_i* = (d_i - sum_{j!=i} D_{ij} x_j) / D_{ii}
            residual = d[i] - (jnp.dot(D[i], x) - D[i, i] * x[i])
            x_i_opt = residual / D[i, i]
            return x.at[i].set(jnp.maximum(x_i_opt, lower_bound[i]))

        return jax.lax.fori_loop(0, K, coord_update, x), None

    x, _ = jax.lax.scan(sweep, x, None, length=n_sweeps)
    return x


@partial(jax.jit, static_argnames=("max_iter", "constrain", "bulk_mode"))
def solve_irwls(
    S: jnp.ndarray,
    Y: jnp.ndarray,
    nUMI: float,
    Q_mat: jnp.ndarray,
    SQ_mat: jnp.ndarray,
    x_vals: jnp.ndarray,
    max_iter: int = 50,
    min_change: float = 0.001,
    step_size: float = 0.3,
    constrain: bool = True,
    bulk_mode: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
        (weights, converged): weights (K,) and convergence flag (scalar bool)
    """
    K = S.shape[1]
    dtype = S.dtype
    # R: if(!bulk_mode) B[B > K_val] <- K_val
    if not bulk_mode:
        k_val = Q_mat.shape[0] - 3
        Y = jnp.minimum(Y, k_val)
    w = jnp.ones(K, dtype=dtype) / K

    def body_fn(state):
        w, i, change = state
        # R: solution <- pmax(initialSol, 0)
        solution = jnp.maximum(w, 0.0)
        # R: prediction = abs(S %*% solution)
        prediction = jnp.abs(S @ solution)
        # R: threshold = max(1e-4, nUMI * 1e-7)
        threshold = jnp.maximum(1e-4, nUMI * 1e-7)
        prediction = jnp.maximum(prediction, threshold)

        grad, hess = _get_derivatives(S, Y, prediction, Q_mat, SQ_mat, x_vals, bulk_mode)
        hess = _psd(hess)

        # R: norm_factor <- norm(D_mat, "2")
        norm_factor = jnp.linalg.norm(hess, ord=2)
        norm_factor = jnp.maximum(norm_factor, 1e-10)
        # R: D_mat <- D_mat / norm_factor
        # R: d_vec <- -derivatives$grad / norm_factor  (note: R negates grad)
        D = hess / norm_factor + 1e-7 * jnp.eye(K, dtype=dtype)
        d = -grad / norm_factor

        # Solve box-constrained QP: min 0.5 x^T D x - d^T x  s.t. x >= -solution
        # Matches R's quadprog::solve.QP(D_mat, d_vec, diag(K), -solution, meq=0)
        delta_w = _solve_box_qp(D, d, -solution)

        # Damped update (R: solution <- solution + alpha * delta_w)
        w_new = solution + step_size * delta_w

        # For constrained mode, additionally project onto simplex
        # For unconstrained mode, non-negativity is already ensured by box QP
        # (delta_w >= -solution, so w_new >= solution*(1-step_size) >= 0)
        w_new = jnp.where(constrain, project_simplex(w_new), w_new)

        # R uses norm(as.matrix(x)) which defaults to L1 norm (sum of abs)
        # for a column vector, NOT L2 norm
        new_change = jnp.sum(jnp.abs(w_new - w))
        return w_new, i + 1, new_change

    def cond_fn(state):
        _, i, change = state
        return (i < max_iter) & (change > min_change)

    w_final, n_iter, final_change = jax.lax.while_loop(
        cond_fn, body_fn, (w, jnp.int32(0), jnp.array(1.0, dtype=dtype))
    )

    converged = final_change <= min_change

    return w_final, converged


def solve_irwls_batch(
    S_batch: jnp.ndarray,
    Y_batch: jnp.ndarray,
    nUMI_batch: jnp.ndarray,
    Q_mat: jnp.ndarray,
    SQ_mat: jnp.ndarray,
    x_vals: jnp.ndarray,
    **kwargs,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve IRWLS for a batch of pixels using vmap.

    Args:
        S_batch: (N, G, K) design matrices
        Y_batch: (N, G) observed counts
        nUMI_batch: (N,) total UMI per pixel
        Q_mat, SQ_mat, x_vals: shared likelihood tables

    Returns:
        (weights, converged): (N, K) and (N,) arrays
    """
    vmapped = jax.vmap(
        lambda S, Y, nUMI: solve_irwls(S, Y, nUMI, Q_mat, SQ_mat, x_vals, **kwargs),
        in_axes=(0, 0, 0),
    )
    return vmapped(S_batch, Y_batch, nUMI_batch)
