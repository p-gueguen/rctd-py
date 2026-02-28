"""Poisson-Lognormal likelihood model with cubic spline interpolation.

Ports spacexr R functions: get_Q, calc_Q_mat_one, solve_sq, calc_Q_all
from prob_model.R.
"""

import urllib.request
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln
from scipy.stats import norm as normal_dist

_Q_MATRICES_URL = "https://github.com/p-gueguen/rctd-py/releases/download/v0.1.0/q_matrices.npz"


def _download_q_matrices(dest: Path) -> None:
    """Download pre-computed Q-matrices from GitHub release."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Q-matrices ({_Q_MATRICES_URL}) ...")
    urllib.request.urlretrieve(_Q_MATRICES_URL, dest)
    print(f"Saved to {dest}")


def load_cached_q_matrices(data_dir: Path | str | None = None) -> dict[str, np.ndarray]:
    """Load precomputed Q-matrices and X_vals.

    Lookup order:
    1. ``data_dir`` (or package ``data/`` directory)
    2. ``~/.cache/rctd/q_matrices.npz``
    3. Auto-download from GitHub release and cache to (2)
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    data_dir = Path(data_dir)
    npz_path = data_dir / "q_matrices.npz"

    # Fallback: user cache directory
    cache_path = Path.home() / ".cache" / "rctd" / "q_matrices.npz"

    if not npz_path.exists():
        if cache_path.exists():
            npz_path = cache_path
        else:
            _download_q_matrices(cache_path)
            npz_path = cache_path

    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def build_x_vals() -> np.ndarray:
    """Reconstruct the X_vals grid used by spacexr.

    The grid uses a non-uniform spacing defined by the index mapping:
      l = floor(sqrt(lambda / delta))   where delta = 1e-6
      m = min(l - 9, 40) + max(ceil(sqrt(max(l - 48.7499, 0) * 4)) - 2, 0)

    For each unique m value (1..438), x_vals[m-1] = l_first^2 * delta
    where l_first is the smallest l that maps to m. The final entry is 1600.0.

    Returns:
        x_vals: shape (439,) float64 array, strictly monotonically increasing.
    """
    delta = 1e-6
    # l=40000 gives x=1600.0, the maximum
    max_l = 40000
    m_to_first_l: dict[int, int] = {}
    for l in range(10, max_l + 1):
        m = min(l - 9, 40) + max(int(np.ceil(np.sqrt(max(l - 48.7499, 0) * 4))) - 2, 0)
        if m not in m_to_first_l:
            m_to_first_l[m] = l

    n_grid = 439
    x_vals = np.zeros(n_grid, dtype=np.float64)
    for m in range(1, n_grid):
        l = m_to_first_l[m]
        x_vals[m - 1] = l * l * delta
    x_vals[n_grid - 1] = 1600.0

    return x_vals


def _ht_pdf_norm(x: np.ndarray) -> np.ndarray:
    """Heavy-tailed PDF (normalized), ports ht_pdf_norm from R."""
    a = 4.0 / 9.0 * np.exp(-9.0 / 2.0) / np.sqrt(2.0 * np.pi)
    c = 7.0 / 3.0
    C = 1.0 / ((a / (3.0 - c) - normal_dist.cdf(-3.0)) * 2.0 + 1.0)
    p = np.zeros_like(x, dtype=np.float64)
    mask_small = np.abs(x) < 3.0
    mask_large = ~mask_small
    p[mask_small] = C / np.sqrt(2.0 * np.pi) * np.exp(-(x[mask_small] ** 2) / 2.0)
    p[mask_large] = C * a / (np.abs(x[mask_large]) - c) ** 2
    return p


def _get_q_single(x_vals: np.ndarray, k: int, sigma: float) -> np.ndarray:
    """Compute log P(Y=k | lambda=x) under Poisson-Lognormal for each x in x_vals.

    Ports get_Q from R prob_model.R.
    Uses numerical integration over the latent lognormal variable.
    """
    N_Y = 5000
    gamma_step = 4e-3
    N_X = len(x_vals)
    Y = np.arange(-N_Y, N_Y + 1) * gamma_step

    ht_vals = _ht_pdf_norm(Y / sigma) / sigma
    ht_vals = np.maximum(ht_vals, 1e-300)
    log_p = np.log(ht_vals)

    results = np.zeros(N_X, dtype=np.float64)
    batch = 100
    for b in range(0, N_X, batch):
        end = min(b + batch, N_X)
        curr_X = x_vals[b:end]

        # log_S[j, i] = -exp(Y[j]) * x[i] + k * Y[j] + log_p[j] - lgamma(k+1) + k * log(x[i])
        log_S = (
            np.outer(-np.exp(Y), curr_X)
            + (k * Y + log_p)[:, None]
            - gammaln(k + 1)
            + k * np.log(curr_X)[None, :]
        )
        S = np.exp(log_S)
        results[b:end] = np.sum(S, axis=0) * gamma_step

    return results


def compute_q_matrix(sigma: float, x_vals: np.ndarray, K_val: int = 100) -> np.ndarray:
    """Compute full Q-matrix for a given sigma.

    Q_mat[k, i] = log P(Y=k | lambda=x_vals[i]) for k in 0..K_val+2.

    Ports calc_Q_par / calc_Q_mat_one from R.

    Args:
        sigma: noise parameter (sigma_c / 100 in R convention).
        x_vals: lambda grid points, shape (N_X,).
        K_val: maximum count value (default 100).

    Returns:
        Q_mat of shape (K_val + 3, len(x_vals)), containing log-probabilities.
    """
    n_rows = K_val + 3
    N_X = len(x_vals)
    Q_mat = np.zeros((n_rows, N_X), dtype=np.float64)

    for k in range(n_rows):
        raw_probs = _get_q_single(x_vals, k, sigma)
        # Store as log-probabilities (matching R's convention)
        Q_mat[k, :] = np.log(np.maximum(raw_probs, 1e-300))

    return Q_mat


def compute_spline_coefficients(Q_mat: np.ndarray, x_vals: np.ndarray) -> np.ndarray:
    """Compute cubic spline second-derivative coefficients.

    Ports solve_sq from R prob_model.R.

    Args:
        Q_mat: likelihood table, shape (K_val+3, N_X).
        x_vals: lambda grid points, shape (N_X,).

    Returns:
        SQ_mat: spline coefficients, shape (K_val+3, N_X).
    """
    n = len(x_vals) - 1
    delta = np.diff(x_vals)

    # Build tridiagonal matrix M of size (n-1, n-1)
    M = np.zeros((n - 1, n - 1))
    np.fill_diagonal(M, 2.0 * (delta[:-1] + delta[1:]))
    for i in range(n - 2):
        M[i + 1, i] = delta[i + 1]
        M[i, i + 1] = delta[i + 1]

    MI = np.linalg.inv(M)

    # Forward differences: fB = diff(t(Q_mat)) / del in R
    # In R: fB <- sweep(diff(t(Q_mat)), 1, del, '/') — shape (n, K_val+3)
    # diff(t(Q_mat)) diffs along columns of transposed Q_mat = diffs along x-axis
    # In numpy: np.diff(Q_mat, axis=1) gives (K_val+3, n), each row diffed along x
    fB = np.diff(Q_mat, axis=1) / delta[None, :]  # (K_val+3, n)
    fBD = 6.0 * np.diff(fB, axis=1)  # (K_val+3, n-1)

    # R: SQ_mat <- t(MI %*% fBD)
    # fBD in R is (n-1, K_val+3), MI is (n-1, n-1)
    # In our case fBD is (K_val+3, n-1), so: SQ_mat = (MI @ fBD.T).T
    SQ_mat = (MI @ fBD.T).T  # (K_val+3, n-1)

    # Pad with zeros at boundaries (natural spline)
    SQ_mat = np.concatenate(
        [
            np.zeros((SQ_mat.shape[0], 1)),
            SQ_mat,
            np.zeros((SQ_mat.shape[0], 1)),
        ],
        axis=1,
    )

    return SQ_mat


def calc_q_all(
    Y: jnp.ndarray,
    lam: jnp.ndarray,
    Q_mat: jnp.ndarray,
    SQ_mat: jnp.ndarray,
    x_vals: jnp.ndarray,
    K_val: int = -1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Cubic spline interpolation of Poisson-Lognormal likelihood.

    Ports calc_Q_all from R prob_model.R lines 125-145.
    Returns (d0, d1, d2): value, first derivative, second derivative
    of the log-likelihood (note: d0 = log P, so it's negative).

    Args:
        Y: observed counts, shape (G,), clipped to [0, K_val].
        lam: predicted intensities, shape (G,).
        Q_mat: precomputed log-likelihood table, shape (K_val+3, N_X).
        SQ_mat: spline coefficients, shape (K_val+3, N_X).
        x_vals: lambda grid points, shape (N_X,).
        K_val: maximum count value.

    Returns:
        (d0_vec, d1_vec, d2_vec): each shape (G,).
    """
    # Infer K_val from Q_mat shape if not explicitly provided
    # R: K_val = dim(Q_mat)[1] - 3, set globally by set_likelihood_vars
    if K_val < 0:
        K_val = Q_mat.shape[0] - 3
    Y = jnp.minimum(Y, K_val)
    epsilon = 1e-4
    X_max = x_vals[-1]
    lam = jnp.clip(lam, epsilon, X_max - epsilon)

    delta = 1e-6
    l = jnp.floor(jnp.sqrt(lam / delta)).astype(jnp.int32)
    # R: m <- pmin(l - 9, 40) + pmax(ceiling(sqrt(pmax(l - 48.7499, 0) * 4)) - 2, 0)
    m = jnp.minimum(l - 9, 40) + jnp.maximum(
        jnp.ceil(jnp.sqrt(jnp.maximum(l.astype(jnp.float32) - 48.7499, 0.0) * 4.0)).astype(
            jnp.int32
        )
        - 2,
        0,
    )

    # R is 1-indexed: ti1 <- X_vals[m]; ti <- X_vals[m+1]
    # Python 0-indexed: x_vals[m-1] and x_vals[m]
    ti1 = x_vals[m - 1]
    ti = x_vals[m]
    hi = ti - ti1

    # R: Q_mat[Y+1, m] (1-indexed) -> Python: Q_mat[Y, m-1] (0-indexed)
    Y_idx = Y.astype(jnp.int32)
    fti1 = Q_mat[Y_idx, m - 1]
    fti = Q_mat[Y_idx, m]
    zi1 = SQ_mat[Y_idx, m - 1]
    zi = SQ_mat[Y_idx, m]

    diff1 = lam - ti1
    diff2 = ti - lam
    diff3 = fti / hi - zi * hi / 6.0
    diff4 = fti1 / hi - zi1 * hi / 6.0
    zdi = zi / hi
    zdi1 = zi1 / hi

    # Cubic spline interpolation
    d0_vec = zdi * diff1**3 / 6.0 + zdi1 * diff2**3 / 6.0 + diff3 * diff1 + diff4 * diff2
    d1_vec = zdi * diff1**2 / 2.0 - zdi1 * diff2**2 / 2.0 + diff3 - diff4
    d2_vec = zdi * diff1 + zdi1 * diff2

    return d0_vec, d1_vec, d2_vec


def calc_log_likelihood(
    Y: jnp.ndarray,
    lam: jnp.ndarray,
    Q_mat: jnp.ndarray,
    SQ_mat: jnp.ndarray,
    x_vals: jnp.ndarray,
    K_val: int = -1,
) -> jnp.ndarray:
    """Compute total negative log-likelihood.

    Ports calc_log_l_vec from R: sum of -d0_vec.

    Args:
        Y: observed counts, shape (G,).
        lam: predicted intensities, shape (G,).
        Q_mat: precomputed log-likelihood table, shape (K_val+3, N_X).
        SQ_mat: spline coefficients, shape (K_val+3, N_X).
        x_vals: lambda grid points, shape (N_X,).
        K_val: maximum count value.

    Returns:
        Scalar negative log-likelihood (lower = better fit).
    """
    d0, _, _ = calc_q_all(Y, lam, Q_mat, SQ_mat, x_vals, K_val)
    return -jnp.sum(d0)
