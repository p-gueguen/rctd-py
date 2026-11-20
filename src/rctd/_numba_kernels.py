"""Numba kernels for the likelihood and IRWLS derivatives on CPU.

On a CPU, torch evaluates the likelihood spline (``_calc_q_all_impl``) as some thirty elementwise
ops, each streaming a pixels x genes temporary through memory, and ``solve_irwls_batch`` adds three
batched matmuls over the (pixels, genes, types) design matrices on top. These kernels do the same
arithmetic in a single pass over the genes, in parallel over pixels:

- ``calc_q_all``: ``_calc_q_all_impl`` elementwise (bitwise equal to it in float64);
- ``irwls_derivatives``: prediction, likelihood derivatives, gradient and Hessian of one
  ``solve_irwls_batch`` iteration, skipping the pixels that have already converged.

They compute in float64 whatever the fit's dtype, and round their outputs to it. The PSD
projection and the box QP stay in torch, shared with the GPU path.

The kernels release the GIL and run on chunks of rows in a thread pool of their own, not on
numba's parallel threading layer: its OpenMP layer competes with torch's OpenMP threads (a 2-10x
slowdown, worse on a loaded machine), and its other layers would mean changing numba's
process-wide configuration. The pool has as many threads as torch (``torch.get_num_threads()``,
which ``torch.set_num_threads`` and ``OMP_NUM_THREADS`` set), at most ``NUMBA_NUM_THREADS``.

numba is optional (``pip install rctd-py[cpu]``). Without it, or with ``RCTD_NUMBA=0``, the CPU
runs the torch code instead.
"""

import functools
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

try:
    import numba
except ImportError:  # numba is optional: the CPU then runs the torch code
    numba = None

ENABLED = numba is not None and os.environ.get("RCTD_NUMBA", "1") != "0"


def use_numba(t: torch.Tensor) -> bool:
    """Whether a fit on ``t``'s device runs on these kernels (it is on the CPU)."""
    return ENABLED and t.device.type == "cpu"


if numba is not None:

    @numba.njit(cache=True)
    def _spline(y, lam, Q, SQ, x, m_of_l, lam_max):
        """d0, d1, d2 of ``_calc_q_all_impl`` for one count row ``y`` and one lambda."""
        if lam < 1e-4:  # torch.clamp, which keeps NaN
            lam = 1e-4
        if lam > lam_max:
            lam = lam_max
        r = np.floor(np.sqrt(lam / 1e-6))
        m = m_of_l[int(r) if r < m_of_l.size else m_of_l.size - 1]  # a NaN lambda stays in range
        ti1 = np.float64(x[m - 1])
        ti = np.float64(x[m])
        hi = ti - ti1
        fti1 = np.float64(Q[y, m - 1])
        fti = np.float64(Q[y, m])
        zi1 = np.float64(SQ[y, m - 1])
        zi = np.float64(SQ[y, m])
        diff1 = lam - ti1
        diff2 = ti - lam
        diff3 = fti / hi - zi * hi / 6.0
        diff4 = fti1 / hi - zi1 * hi / 6.0
        zdi = zi / hi
        zdi1 = zi1 / hi
        d0 = (
            zdi * (diff1 * diff1 * diff1) / 6.0
            + zdi1 * (diff2 * diff2 * diff2) / 6.0
            + diff3 * diff1
            + diff4 * diff2
        )
        d1 = zdi * (diff1 * diff1) / 2.0 - zdi1 * (diff2 * diff2) / 2.0 + diff3 - diff4
        d2 = zdi * diff1 + zdi1 * diff2
        return d0, d1, d2

    @numba.njit(inline="always")
    def _count(y, k_val):
        """Row of the likelihood tables for count ``y``: min(y, k_val), like torch's clamp."""
        c = int(min(y, k_val))
        if c < 0:  # not a count: torch wraps it or fails, numba would read outside the tables
            raise ValueError("counts must be non-negative")
        return c

    @numba.njit(nogil=True, cache=True)
    def _calc_q_kernel(start, end, Y, lam, Q, SQ, x, m_of_l, k_val, lam_max, d0, d1, d2):
        for j in range(start, end):
            d0[j], d1[j], d2[j] = _spline(
                _count(Y[j], k_val), np.float64(lam[j]), Q, SQ, x, m_of_l, lam_max
            )

    @numba.njit(nogil=True, cache=True)
    def _derivatives_kernel(
        start, end, S, Y, w, thresh, active, Q, SQ, x, m_of_l, k_val, lam_max, grad, hess
    ):
        _, G, K = S.shape
        for n in range(start, end):
            if not active[n]:
                continue
            g_acc = np.zeros(K)
            h_acc = np.zeros((K, K))
            for g in range(G):
                pred = 0.0
                for i in range(K):
                    pred += S[n, g, i] * w[n, i]
                pred = abs(pred)
                if pred < thresh[n]:
                    pred = np.float64(thresh[n])
                _, d1, d2 = _spline(_count(Y[n, g], k_val), pred, Q, SQ, x, m_of_l, lam_max)
                for i in range(K):
                    g_acc[i] += d1 * S[n, g, i]
                    s = -d2 * S[n, g, i]
                    for j in range(i, K):
                        h_acc[i, j] += s * S[n, g, j]
            for i in range(K):
                grad[n, i] = -g_acc[i]
                for j in range(i, K):
                    hess[n, i, j] = h_acc[i, j]
                    hess[n, j, i] = h_acc[i, j]


_pool: ThreadPoolExecutor | None = None
_pool_threads = 0


def num_threads() -> int:
    """Threads the kernels run on: torch's, at most ``NUMBA_NUM_THREADS``.

    ``NUMBA_NUM_THREADS`` defaults to every CPU the process may run on, which on a node that does
    not pin a job's CPUs is the whole node; torch follows ``OMP_NUM_THREADS`` and
    ``torch.set_num_threads``, which job schedulers and users set.
    """
    return max(1, min(torch.get_num_threads(), numba.config.NUMBA_NUM_THREADS))


def _run(kernel, n: int, min_chunk: int, *args) -> None:
    """``kernel(start, end, *args)`` over rows [0, n), in one chunk per thread."""
    global _pool, _pool_threads
    threads = num_threads()
    chunk = max(min_chunk, -(-n // threads))
    if n <= chunk:
        kernel(0, n, *args)
        return
    if _pool_threads != threads:  # a new pool when torch's thread count changes
        if _pool is not None:
            _pool.shutdown(wait=False)
        _pool = ThreadPoolExecutor(threads, thread_name_prefix="rctd-numba")
        _pool_threads = threads
    for f in [_pool.submit(kernel, s, min(s + chunk, n), *args) for s in range(0, n, chunk)]:
        f.result()


@functools.lru_cache
def _interval_table(x_max: float, n_x: int) -> np.ndarray:
    """The spline interval m of every l = floor(sqrt(lambda / 1e-6)) a clamped lambda can reach,
    computed by ``_calc_q_all_impl``'s own code."""
    from rctd._likelihood import _spline_interval  # (imports this module)

    l = torch.arange(int(np.sqrt(x_max / 1e-6)) + 2)
    return torch.clamp(_spline_interval(l), 1, n_x - 1).numpy()


def _tables(Q_mat, SQ_mat, x_vals, K_val):
    """The likelihood tables and constants the kernels take after the data."""
    k_val = Q_mat.shape[0] - 3 if K_val < 0 else K_val
    x_max = float(x_vals[-1])
    m_of_l = _interval_table(x_max, x_vals.shape[0])
    return Q_mat.numpy(), SQ_mat.numpy(), x_vals.numpy(), m_of_l, k_val, x_max - 1e-4


def calc_q_all(Y, lam, Q_mat, SQ_mat, x_vals, K_val=-1):
    """``_calc_q_all_impl`` on CPU tensors."""
    out = [torch.empty(lam.shape, dtype=lam.dtype) for _ in range(3)]
    _run(
        _calc_q_kernel,
        lam.numel(),
        1 << 14,
        Y.contiguous().numpy().reshape(-1),
        lam.contiguous().numpy().reshape(-1),
        *_tables(Q_mat, SQ_mat, x_vals, K_val),
        *(o.numpy().reshape(-1) for o in out),
    )
    return tuple(out)


def irwls_derivatives(S, Y, w, threshold, active, Q_mat, SQ_mat, x_vals):
    """Gradient (N, K) and Hessian (N, K, K) of one ``solve_irwls_batch`` step at weights ``w``.

    The same as ``_get_derivatives_batch`` at prediction ``max(|S @ w|, threshold)``, for the
    ``active`` pixels; the others get zeros.
    """
    N, _, K = S.shape
    grad = torch.zeros(N, K, dtype=S.dtype)
    hess = torch.zeros(N, K, K, dtype=S.dtype)
    _run(
        _derivatives_kernel,
        N,
        16,
        S.numpy(),
        Y.numpy(),
        w.contiguous().numpy(),
        threshold.contiguous().numpy(),
        active.contiguous().numpy(),
        *_tables(Q_mat, SQ_mat, x_vals, -1),
        grad.numpy(),
        hess.numpy(),
    )
    return grad, hess
