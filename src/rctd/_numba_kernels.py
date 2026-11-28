"""Numba kernels for the likelihood and IRWLS derivatives on CPU.

On a CPU, torch evaluates the likelihood spline (``_calc_q_all_impl``) as some thirty elementwise
ops, each streaming a pixels x genes temporary through memory, and ``solve_irwls_batch`` adds three
batched matmuls over the (pixels, genes, types) design matrices on top. These kernels do the same
arithmetic in a single pass over the genes, in parallel over pixels:

- ``calc_q_all``: ``_calc_q_all_impl`` elementwise (bitwise equal to it in float64);
- ``irwls_derivatives``: prediction, likelihood derivatives, gradient and Hessian of one
  ``solve_irwls_batch`` iteration, skipping the pixels that have already converged;
- ``box_qp_sweeps``: the Gauss-Seidel sweeps of the box QP (K != 2), per pixel. torch.compile
  would otherwise spend ~50 s compiling them in every new process.

They compute in float64 whatever the fit's dtype, and round their outputs to it. The PSD
projection (its eigendecompositions spread over the same threads) and the box QP's exact finish
stay in torch, shared with the GPU path.

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
import scipy.sparse as sp
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

    @numba.njit(inline="always")
    def _maximum(a, b):
        """torch.maximum: NaN if either is."""
        if a != a or b != b:
            return a + b
        return a if a > b else b

    # error_model="numpy": a zero diagonal gives inf/NaN as in torch, not ZeroDivisionError
    @numba.njit(nogil=True, cache=True, error_model="numpy")
    def _box_qp_kernel(start, end, D, d, lb, x, n_sweeps):
        K = d.shape[1]
        for r in range(start, end):
            for i in range(K):
                x[r, i] = _maximum(d[r, i] / D[r, i, i], lb[r, i])
            for _ in range(n_sweeps):
                for i in range(K):
                    acc = 0.0
                    for j in range(K):
                        acc += D[r, i, j] * x[r, j]
                    resid = d[r, i] - acc + D[r, i, i] * x[r, i]
                    x[r, i] = _maximum(resid / D[r, i, i], lb[r, i])

    @numba.njit(nogil=True, cache=True)
    def _calc_d1_d2_scaled_kernel(
        start, end, Y_int, pred, nUMI, poly, k_of_l, x_of_l, lam_max, d1_out, d2_out
    ):
        """Fused Horner (d1 * nUMI, -d2 * nUMI^2) spline evaluation for shared IRWLS."""
        G = Y_int.shape[1]
        k_size = k_of_l.size
        poly_0 = poly[0]
        for n in range(start, end):
            numi = np.float64(nUMI[n])
            numi_sq = numi * numi
            for g in range(G):
                p = np.float64(pred[n, g])
                if p > lam_max:
                    p = lam_max
                elif p < 1e-4:
                    p = 1e-4
                r = int(np.sqrt(p * 1e6))
                if r >= k_size:
                    r = k_size - 1
                k = k_of_l[r]
                delta = p - x_of_l[r]
                y_c = Y_int[n, g]
                if y_c == 0:
                    c1 = poly_0[k, 1]
                    z0 = poly_0[k, 2]
                    s6 = poly_0[k, 3]
                else:
                    c1 = poly[y_c, k, 1]
                    z0 = poly[y_c, k, 2]
                    s6 = poly[y_c, k, 3]
                s6_delta = s6 * delta
                d2_val = s6_delta + z0
                d1_val = (0.5 * s6_delta + z0) * delta + c1
                d1_out[n, g] = d1_val * numi
                d2_out[n, g] = -d2_val * numi_sq

    @numba.njit(nogil=True, cache=True)
    def _unpack_sym_hess_kernel(start, end, H_tri, K, H_out):
        """Unpack packed upper-triangle Hessian rows into symmetric (N, K, K) matrices."""
        for n in range(start, end):
            p = 0
            for i in range(K):
                for j in range(i, K):
                    v = H_tri[n, p]
                    H_out[n, i, j] = v
                    H_out[n, j, i] = v
                    p += 1

    @numba.njit(inline="always")
    def _solve_free_submatrix(D_r, g, free_idx, m, delta_x, A, b):
        """Solve D[free, free] * delta_x = -g[free] via partial-pivoting Gaussian elimination."""
        for i in range(m):
            fi = free_idx[i]
            b[i] = -g[fi]
            for j in range(m):
                A[i, j] = np.float64(D_r[fi, free_idx[j]])

        for i in range(m):
            max_val = abs(A[i, i])
            max_row = i
            for k in range(i + 1, m):
                val = abs(A[k, i])
                if val > max_val:
                    max_val = val
                    max_row = k
            if not np.isfinite(max_val) or max_val < 1e-12:
                return False
            if max_row != i:
                for j in range(i, m):
                    tmp = A[i, j]
                    A[i, j] = A[max_row, j]
                    A[max_row, j] = tmp
                tmp_b = b[i]
                b[i] = b[max_row]
                b[max_row] = tmp_b

            pivot = A[i, i]
            for k in range(i + 1, m):
                factor = A[k, i] / pivot
                for j in range(i + 1, m):
                    A[k, j] -= factor * A[i, j]
                b[k] -= factor * b[i]

        for i in range(m - 1, -1, -1):
            sum_ax = 0.0
            for j in range(i + 1, m):
                sum_ax += A[i, j] * delta_x[j]
            delta_x[i] = (b[i] - sum_ax) / A[i, i]
        return True

    @numba.njit(nogil=True, cache=True, error_model="numpy")
    def _solve_box_qp_fused_kernel(start, end, D, d, lb, x_out, done_out, n_sweeps, kkt_tol):
        """Residual-maintained coordinate descent + reduced |F|x|F| active-set polish."""
        K = d.shape[1]
        x_buf = np.empty(K, dtype=np.float64)
        g = np.empty(K, dtype=np.float64)
        inv_D = np.empty(K, dtype=np.float64)
        free_idx = np.empty(K, dtype=np.int64)
        delta_x = np.empty(K, dtype=np.float64)
        A_buf = np.empty((K, K), dtype=np.float64)
        b_buf = np.empty(K, dtype=np.float64)
        check_tol = min(kkt_tol, 1e-7) if kkt_tol < 1e-5 else kkt_tol
        free_eps = 1e-11 if kkt_tol < 1e-6 else 1e-7

        for r in range(start, end):
            bad_row = False
            for i in range(K):
                d_ii = np.float64(D[r, i, i])
                d_i = np.float64(d[r, i])
                lb_i = np.float64(lb[r, i])
                if not (
                    np.isfinite(d_ii) and d_ii > 1e-12 and np.isfinite(d_i) and np.isfinite(lb_i)
                ):
                    bad_row = True
                    break
                inv_D[i] = 1.0 / d_ii
                init_val = d_i * inv_D[i]
                x_buf[i] = init_val if init_val > lb_i else lb_i

            if bad_row:
                for i in range(K):
                    x_out[r, i] = np.nan
                done_out[r] = False
                continue

            for i in range(K):
                acc = -np.float64(d[r, i])
                for j in range(K):
                    acc += np.float64(D[r, i, j]) * x_buf[j]
                g[i] = acc

            for _ in range(n_sweeps):
                max_viol = 0.0
                for i in range(K):
                    g_i = g[i]
                    x_i = x_buf[i]
                    lb_i = np.float64(lb[r, i])
                    viol = abs(g_i) if x_i > lb_i + free_eps else (-g_i if -g_i > 0.0 else 0.0)
                    if viol > max_viol:
                        max_viol = viol
                    x_cand = x_i - g_i * inv_D[i]
                    x_new = x_cand if x_cand > lb_i else lb_i
                    dx = x_new - x_i
                    if dx != 0.0:
                        x_buf[i] = x_new
                        for j in range(K):
                            g[j] += np.float64(D[r, i, j]) * dx
                if max_viol <= check_tol:
                    break

            max_viol = 0.0
            for i in range(K):
                g_i = g[i]
                x_i = x_buf[i]
                lb_i = np.float64(lb[r, i])
                viol = abs(g_i) if x_i > lb_i + free_eps else (-g_i if -g_i > 0.0 else 0.0)
                if viol > max_viol:
                    max_viol = viol

            if max_viol > kkt_tol:
                for _ in range(6):
                    m = 0
                    for i in range(K):
                        if x_buf[i] > np.float64(lb[r, i]) + free_eps:
                            free_idx[m] = i
                            m += 1
                    if m == 0:
                        break
                    if not _solve_free_submatrix(D[r], g, free_idx, m, delta_x, A_buf, b_buf):
                        break

                    alpha = 1.0
                    for i in range(m):
                        dx_i = delta_x[i]
                        if dx_i < -1e-12:
                            fi = free_idx[i]
                            ratio = (np.float64(lb[r, fi]) - x_buf[fi]) / dx_i
                            if ratio < alpha:
                                alpha = ratio
                    if alpha < 0.0:
                        alpha = 0.0
                    elif alpha > 1.0:
                        alpha = 1.0

                    for i in range(m):
                        fi = free_idx[i]
                        lb_fi = np.float64(lb[r, fi])
                        dx_step = alpha * delta_x[i]
                        x_cand = x_buf[fi] + dx_step
                        x_new = x_cand if x_cand > lb_fi else lb_fi
                        actual_dx = x_new - x_buf[fi]
                        x_buf[fi] = x_new
                        if actual_dx != 0.0:
                            for j in range(K):
                                g[j] += np.float64(D[r, j, fi]) * actual_dx

                    max_viol = 0.0
                    for i in range(K):
                        g_i = g[i]
                        x_i = x_buf[i]
                        lb_i = np.float64(lb[r, i])
                        viol = abs(g_i) if x_i > lb_i + free_eps else (-g_i if -g_i > 0.0 else 0.0)
                        if viol > max_viol:
                            max_viol = viol
                    if max_viol <= kkt_tol:
                        break

                    for _ in range(5):
                        for i in range(K):
                            g_i = g[i]
                            x_i = x_buf[i]
                            lb_i = np.float64(lb[r, i])
                            x_cand = x_i - g_i * inv_D[i]
                            x_new = x_cand if x_cand > lb_i else lb_i
                            dx = x_new - x_i
                            if dx != 0.0:
                                x_buf[i] = x_new
                                for j in range(K):
                                    g[j] += np.float64(D[r, i, j]) * dx

                max_viol = 0.0
                for i in range(K):
                    g_i = g[i]
                    x_i = x_buf[i]
                    lb_i = np.float64(lb[r, i])
                    viol = abs(g_i) if x_i > lb_i + free_eps else (-g_i if -g_i > 0.0 else 0.0)
                    if viol > max_viol:
                        max_viol = viol

            if np.isfinite(max_viol) and max_viol <= kkt_tol:
                for i in range(K):
                    lb_i = np.float64(lb[r, i])
                    x_out[r, i] = lb_i if x_buf[i] < lb_i else x_buf[i]
                done_out[r] = True
            else:
                done_out[r] = False

    @numba.njit(inline="always")
    def _psd_small_jacobi(H, K, eps):
        """Closed-form K=1/2 or Jacobi symmetric eigensolve + eigenvalue clamp >= eps."""
        if K == 1:
            lam = H[0, 0] if H[0, 0] >= eps else eps
            H_out = np.empty((1, 1), dtype=np.float64)
            H_out[0, 0] = lam
            return H_out, lam
        if K == 2:
            h00 = H[0, 0]
            h01 = H[0, 1]
            h11 = H[1, 1]
            half_trace = (h00 + h11) * 0.5
            disc = np.sqrt(((h00 - h11) * 0.5) ** 2 + h01 * h01)
            min_eig = half_trace - disc
            max_eig = half_trace + disc
            H_out = np.empty((2, 2), dtype=np.float64)
            if min_eig >= eps:
                H_out[0, 0] = h00
                H_out[0, 1] = h01
                H_out[1, 0] = h01
                H_out[1, 1] = h11
                return H_out, max_eig
            lam1 = eps
            lam2 = max_eig if max_eig >= eps else eps
            safe_disc = disc if disc >= 1e-30 else 1e-30
            cos2t = (h00 - h11) * 0.5 / safe_disc
            sin2t = h01 / safe_disc
            half_sum = (lam1 + lam2) * 0.5
            half_diff = (lam2 - lam1) * 0.5
            H_out[0, 0] = half_sum + half_diff * cos2t
            H_out[0, 1] = half_diff * sin2t
            H_out[1, 0] = H_out[0, 1]
            H_out[1, 1] = half_sum - half_diff * cos2t
            return H_out, lam2

        A = np.empty((K, K), dtype=np.float64)
        V = np.zeros((K, K), dtype=np.float64)
        for i in range(K):
            V[i, i] = 1.0
            for j in range(K):
                A[i, j] = H[i, j]

        for _ in range(15):
            max_off = 0.0
            for p in range(K):
                for q in range(p + 1, K):
                    apq = abs(A[p, q])
                    if apq > max_off:
                        max_off = apq
            if max_off < 1e-14:
                break
            for p in range(K):
                for q in range(p + 1, K):
                    if abs(A[p, q]) < 1e-15:
                        continue
                    tau = (A[q, q] - A[p, p]) / (2.0 * A[p, q])
                    if tau >= 0.0:
                        t = 1.0 / (tau + np.sqrt(1.0 + tau * tau))
                    else:
                        t = -1.0 / (-tau + np.sqrt(1.0 + tau * tau))
                    c = 1.0 / np.sqrt(1.0 + t * t)
                    s = t * c
                    app = A[p, p]
                    aqq = A[q, q]
                    apq = A[p, q]
                    A[p, p] = app - t * apq
                    A[q, q] = aqq + t * apq
                    A[p, q] = 0.0
                    A[q, p] = 0.0
                    for i in range(K):
                        if i != p and i != q:
                            aip = A[i, p]
                            aiq = A[i, q]
                            A[i, p] = c * aip - s * aiq
                            A[p, i] = A[i, p]
                            A[i, q] = s * aip + c * aiq
                            A[q, i] = A[i, q]
                        vip = V[i, p]
                        viq = V[i, q]
                        V[i, p] = c * vip - s * viq
                        V[i, q] = s * vip + c * viq

        max_eig = eps
        H_out = np.zeros((K, K), dtype=np.float64)
        for i in range(K):
            lam = A[i, i] if A[i, i] >= eps else eps
            if lam > max_eig:
                max_eig = lam
            for r in range(K):
                for c in range(K):
                    H_out[r, c] += lam * V[r, i] * V[c, i]
        return H_out, max_eig

    @numba.njit(inline="always")
    def _box_qp_small(D, d, lb, K, n_sweeps=25):
        """Exact K=1, analytical K=2, or K>=3 coordinate descent on a stack buffer."""
        x = np.empty(K, dtype=np.float64)
        if K == 1:
            x0 = d[0] / D[0, 0]
            x[0] = x0 if x0 > lb[0] else lb[0]
            return x
        if K == 2:
            det = D[0, 0] * D[1, 1] - D[0, 1] * D[0, 1]
            det_abs = abs(det)
            det_safe = (det_abs if det_abs >= 1e-30 else 1e-30) * (1.0 if det >= 0.0 else -1.0)
            x0 = (D[1, 1] * d[0] - D[0, 1] * d[1]) / det_safe
            x1 = (D[0, 0] * d[1] - D[0, 1] * d[0]) / det_safe
            x0 = x0 if x0 > lb[0] else lb[0]
            x1 = x1 if x1 > lb[1] else lb[1]
            x0c = (d[0] - D[0, 1] * x1) / D[0, 0]
            x0 = x0c if x0c > lb[0] else lb[0]
            x1c = (d[1] - D[0, 1] * x0) / D[1, 1]
            x1 = x1c if x1c > lb[1] else lb[1]
            x[0] = x0
            x[1] = x1
            return x

        for i in range(K):
            xi = d[i] / D[i, i]
            x[i] = xi if xi > lb[i] else lb[i]
        for _ in range(n_sweeps):
            for i in range(K):
                acc = 0.0
                for j in range(K):
                    acc += D[i, j] * x[j]
                resid = d[i] - acc + D[i, i] * x[i]
                xi = resid / D[i, i]
                x[i] = xi if xi > lb[i] else lb[i]

        # Short active-set polish when coordinate descent stops above 1e-10
        g = np.empty(K, dtype=np.float64)
        max_viol = 0.0
        for i in range(K):
            acc = -d[i]
            for j in range(K):
                acc += D[i, j] * x[j]
            g[i] = acc
            viol = abs(acc) if x[i] > lb[i] + 1e-11 else (-acc if -acc > 0.0 else 0.0)
            if viol > max_viol:
                max_viol = viol
        if max_viol > 1e-10:
            free_idx = np.empty(K, dtype=np.int64)
            delta_x = np.empty(K, dtype=np.float64)
            A_buf = np.empty((K, K), dtype=np.float64)
            b_buf = np.empty(K, dtype=np.float64)
            for _ in range(4):
                m = 0
                for i in range(K):
                    if x[i] > lb[i] + 1e-11:
                        free_idx[m] = i
                        m += 1
                if m == 0 or not _solve_free_submatrix(D, g, free_idx, m, delta_x, A_buf, b_buf):
                    break
                alpha = 1.0
                for i in range(m):
                    dx_i = delta_x[i]
                    if dx_i < -1e-12:
                        fi = free_idx[i]
                        ratio = (lb[fi] - x[fi]) / dx_i
                        if ratio < alpha:
                            alpha = ratio
                if alpha < 0.0:
                    alpha = 0.0
                elif alpha > 1.0:
                    alpha = 1.0
                for i in range(m):
                    fi = free_idx[i]
                    x_cand = x[fi] + alpha * delta_x[i]
                    x_new = x_cand if x_cand > lb[fi] else lb[fi]
                    actual_dx = x_new - x[fi]
                    x[fi] = x_new
                    if actual_dx != 0.0:
                        for j in range(K):
                            g[j] += D[j, fi] * actual_dx
                max_viol = 0.0
                for i in range(K):
                    gi = g[i]
                    viol = abs(gi) if x[i] > lb[i] + 1e-11 else (-gi if -gi > 0.0 else 0.0)
                    if viol > max_viol:
                        max_viol = viol
                if max_viol <= 1e-10:
                    break
                for i in range(K):
                    xi_new = x[i] - g[i] / D[i, i]
                    xi_clamped = xi_new if xi_new > lb[i] else lb[i]
                    dx = xi_clamped - x[i]
                    if dx != 0.0:
                        x[i] = xi_clamped
                        for j in range(K):
                            g[j] += D[i, j] * dx
        return x

    @numba.njit(nogil=True, cache=True)
    def _fit_and_score_pairs_kernel(
        start,
        end,
        triples,
        Y_int,
        nUMI_full,
        P_KT,
        poly,
        k_of_l,
        x_of_l,
        lam_max,
        max_iter,
        out_weights,
        out_scores,
        out_converged,
    ):
        """Fused K=2 IRWLS + final NLL evaluation without materializing S_pair."""
        G = P_KT.shape[1]
        k_size = k_of_l.size
        poly_0 = poly[0]
        for task_idx in range(start, end):
            n = triples[task_idx, 0]
            t1 = triples[task_idx, 1]
            t2 = triples[task_idx, 2]

            numi = nUMI_full[n]
            numi_sq = numi * numi
            thresh = max(numi * 1e-7, 1e-4)

            p0_vec = P_KT[t1]
            p1_vec = P_KT[t2]
            y_vec = Y_int[n]

            w0 = 0.5
            w1 = 0.5
            conv = False

            for _ in range(max_iter):
                sol0 = w0 if w0 > 0.0 else 0.0
                sol1 = w1 if w1 > 0.0 else 0.0
                s0_numi = sol0 * numi
                s1_numi = sol1 * numi

                g0 = 0.0
                g1 = 0.0
                h00 = 0.0
                h01 = 0.0
                h11 = 0.0

                for g in range(G):
                    p0 = p0_vec[g]
                    p1 = p1_vec[g]
                    if p0 == 0.0 and p1 == 0.0:
                        continue
                    pred = abs(s0_numi * p0 + s1_numi * p1)
                    if pred < thresh:
                        pred = thresh
                    elif pred > lam_max:
                        pred = lam_max

                    y_c = y_vec[g]
                    r = int(np.sqrt(pred * 1e6))
                    if r >= k_size:
                        r = k_size - 1
                    k = k_of_l[r]
                    delta = pred - x_of_l[r]

                    if y_c == 0:
                        c1 = poly_0[k, 1]
                        z0 = poly_0[k, 2]
                        s6 = poly_0[k, 3]
                    else:
                        c1 = poly[y_c, k, 1]
                        z0 = poly[y_c, k, 2]
                        s6 = poly[y_c, k, 3]

                    s6_delta = s6 * delta
                    d2 = s6_delta + z0
                    d1 = (0.5 * s6_delta + z0) * delta + c1

                    g0 += d1 * p0
                    g1 += d1 * p1
                    s = -d2
                    sp0 = s * p0
                    h00 += sp0 * p0
                    h01 += sp0 * p1
                    h11 += s * p1 * p1

                grad0 = -g0 * numi
                grad1 = -g1 * numi
                h00 *= numi_sq
                h01 *= numi_sq
                h11 *= numi_sq

                half_trace = (h00 + h11) * 0.5
                disc = np.sqrt(((h00 - h11) * 0.5) ** 2 + h01 * h01)
                min_eig = half_trace - disc
                max_eig = half_trace + disc

                if min_eig >= 1e-3:
                    H00 = h00
                    H01 = h01
                    H11 = h11
                    lam2 = max_eig
                else:
                    lam1 = 1e-3
                    lam2 = max_eig if max_eig >= 1e-3 else 1e-3
                    safe_disc = disc if disc >= 1e-30 else 1e-30
                    cos2t = (h00 - h11) * 0.5 / safe_disc
                    sin2t = h01 / safe_disc
                    half_sum = (lam1 + lam2) * 0.5
                    half_diff = (lam2 - lam1) * 0.5
                    H00 = half_sum + half_diff * cos2t
                    H01 = half_diff * sin2t
                    H11 = half_sum - half_diff * cos2t

                norm_factor = lam2 if lam2 >= 1e-10 else 1e-10
                D00 = H00 / norm_factor + 1e-7
                D01 = H01 / norm_factor
                D11 = H11 / norm_factor + 1e-7
                d0_qp = -grad0 / norm_factor
                d1_qp = -grad1 / norm_factor

                det = D00 * D11 - D01 * D01
                det_abs = abs(det)
                det_safe = (det_abs if det_abs >= 1e-30 else 1e-30) * (1.0 if det >= 0.0 else -1.0)
                x0 = (D11 * d0_qp - D01 * d1_qp) / det_safe
                x1 = (D00 * d1_qp - D01 * d0_qp) / det_safe
                x0 = x0 if x0 > -sol0 else -sol0
                x1 = x1 if x1 > -sol1 else -sol1
                x0c = (d0_qp - D01 * x1) / D00
                x0 = x0c if x0c > -sol0 else -sol0
                x1c = (d1_qp - D01 * x0) / D11
                x1 = x1c if x1c > -sol1 else -sol1

                w0_new = sol0 + 0.3 * x0
                w1_new = sol1 + 0.3 * x1

                change = abs(w0_new - w0) + abs(w1_new - w1)
                w0 = w0_new
                w1 = w1_new
                if change <= 0.001:
                    conv = True
                    break

            out_weights[task_idx, 0] = w0
            out_weights[task_idx, 1] = w1
            out_converged[task_idx] = conv

            w0_numi = w0 * numi
            w1_numi = w1 * numi
            nll = 0.0
            for g in range(G):
                p0 = p0_vec[g]
                p1 = p1_vec[g]
                pred = w0_numi * p0 + w1_numi * p1
                if pred < 1e-4:
                    pred = 1e-4
                elif pred > lam_max:
                    pred = lam_max
                y_c = y_vec[g]
                r = int(np.sqrt(pred * 1e6))
                if r >= k_size:
                    r = k_size - 1
                k = k_of_l[r]
                delta = pred - x_of_l[r]

                if y_c == 0:
                    c0 = poly_0[k, 0]
                    c1 = poly_0[k, 1]
                    z0 = poly_0[k, 2]
                    s6 = poly_0[k, 3]
                else:
                    c0 = poly[y_c, k, 0]
                    c1 = poly[y_c, k, 1]
                    z0 = poly[y_c, k, 2]
                    s6 = poly[y_c, k, 3]

                s6_delta = s6 * delta
                d0_val = ((s6_delta * (1.0 / 6.0) + 0.5 * z0) * delta + c1) * delta + c0
                nll -= d0_val
            out_scores[task_idx] = nll

    @numba.njit(nogil=True, cache=True)
    def _fit_and_score_singles_kernel(
        start,
        end,
        singles,
        Y_int,
        nUMI_full,
        P_KT,
        poly,
        k_of_l,
        x_of_l,
        lam_max,
        max_iter,
        out_scores,
    ):
        """Fused K=1 IRWLS + final NLL evaluation without materializing S_sg."""
        G = P_KT.shape[1]
        k_size = k_of_l.size
        poly_0 = poly[0]
        for task_idx in range(start, end):
            n = singles[task_idx, 0]
            t = singles[task_idx, 1]

            numi = nUMI_full[n]
            numi_sq = numi * numi
            thresh = max(numi * 1e-7, 1e-4)

            p_vec = P_KT[t]
            y_vec = Y_int[n]

            w = 1.0
            for _ in range(max_iter):
                sol = w if w > 0.0 else 0.0
                s_numi = sol * numi
                g_acc = 0.0
                h_acc = 0.0
                for g in range(G):
                    p = p_vec[g]
                    if p == 0.0:
                        continue
                    pred = abs(s_numi * p)
                    if pred < thresh:
                        pred = thresh
                    elif pred > lam_max:
                        pred = lam_max

                    y_c = y_vec[g]
                    r = int(np.sqrt(pred * 1e6))
                    if r >= k_size:
                        r = k_size - 1
                    k = k_of_l[r]
                    delta = pred - x_of_l[r]

                    if y_c == 0:
                        c1 = poly_0[k, 1]
                        z0 = poly_0[k, 2]
                        s6 = poly_0[k, 3]
                    else:
                        c1 = poly[y_c, k, 1]
                        z0 = poly[y_c, k, 2]
                        s6 = poly[y_c, k, 3]

                    s6_delta = s6 * delta
                    d2 = s6_delta + z0
                    d1 = (0.5 * s6_delta + z0) * delta + c1

                    g_acc += d1 * p
                    h_acc += -d2 * p * p

                grad = -g_acc * numi
                hess_val = h_acc * numi_sq
                H_psd = hess_val if hess_val >= 1e-3 else 1e-3
                norm_factor = H_psd if H_psd >= 1e-10 else 1e-10
                D = H_psd / norm_factor + 1e-7
                d_qp = -grad / norm_factor

                dq = d_qp / D
                delta_w = dq if dq > -sol else -sol
                w_new = sol + 0.3 * delta_w
                change = abs(w_new - w)
                w = w_new
                if change <= 0.001:
                    break

            w_numi = w * numi
            nll = 0.0
            for g in range(G):
                p = p_vec[g]
                pred = w_numi * p
                if pred < 1e-4:
                    pred = 1e-4
                elif pred > lam_max:
                    pred = lam_max
                y_c = y_vec[g]
                r = int(np.sqrt(pred * 1e6))
                if r >= k_size:
                    r = k_size - 1
                k = k_of_l[r]
                delta = pred - x_of_l[r]

                if y_c == 0:
                    c0 = poly_0[k, 0]
                    c1 = poly_0[k, 1]
                    z0 = poly_0[k, 2]
                    s6 = poly_0[k, 3]
                else:
                    c0 = poly[y_c, k, 0]
                    c1 = poly[y_c, k, 1]
                    z0 = poly[y_c, k, 2]
                    s6 = poly[y_c, k, 3]

                s6_delta = s6 * delta
                d0_val = ((s6_delta * (1.0 / 6.0) + 0.5 * z0) * delta + c1) * delta + c0
                nll -= d0_val
            out_scores[task_idx] = nll

    @numba.njit(nogil=True, cache=True)
    def _fit_subsets_kernel(
        start,
        end,
        pix_indices,
        type_matrix,
        k_sub,
        Y_int,
        nUMI_full,
        P_KT,
        poly,
        k_of_l,
        x_of_l,
        lam_max,
        max_iter,
        out_weights,
        out_scores,
        compute_score,
    ):
        """Fused K_sub IRWLS (arbitrary K_sub >= 1) and optional NLL scoring without S_sub."""
        G = P_KT.shape[1]
        K = k_sub
        k_size = k_of_l.size
        poly_0 = poly[0]
        w = np.empty(K, dtype=np.float64)
        sol = np.empty(K, dtype=np.float64)
        s_numi = np.empty(K, dtype=np.float64)
        grad = np.empty(K, dtype=np.float64)
        hess = np.empty((K, K), dtype=np.float64)
        D = np.empty((K, K), dtype=np.float64)
        d_qp = np.empty(K, dtype=np.float64)
        lb = np.empty(K, dtype=np.float64)
        types = np.empty(K, dtype=np.int64)
        p_vec = np.empty(K, dtype=np.float64)

        for task_idx in range(start, end):
            n = pix_indices[task_idx]
            for i in range(K):
                types[i] = type_matrix[task_idx, i]
                w[i] = 1.0 / K

            numi = nUMI_full[n]
            numi_sq = numi * numi
            thresh = max(numi * 1e-7, 1e-4)
            y_vec = Y_int[n]

            for _ in range(max_iter):
                for i in range(K):
                    sol[i] = w[i] if w[i] > 0.0 else 0.0
                    s_numi[i] = sol[i] * numi
                    grad[i] = 0.0
                    for j in range(K):
                        hess[i, j] = 0.0

                for g in range(G):
                    pred = 0.0
                    all_zero = True
                    for i in range(K):
                        p_val = P_KT[types[i], g]
                        p_vec[i] = p_val
                        if p_val != 0.0:
                            all_zero = False
                        pred += s_numi[i] * p_val
                    if all_zero:
                        continue
                    pred = abs(pred)
                    if pred < thresh:
                        pred = thresh
                    elif pred > lam_max:
                        pred = lam_max

                    y_c = y_vec[g]
                    r = int(np.sqrt(pred * 1e6))
                    if r >= k_size:
                        r = k_size - 1
                    k = k_of_l[r]
                    delta = pred - x_of_l[r]

                    if y_c == 0:
                        c1 = poly_0[k, 1]
                        z0 = poly_0[k, 2]
                        s6 = poly_0[k, 3]
                    else:
                        c1 = poly[y_c, k, 1]
                        z0 = poly[y_c, k, 2]
                        s6 = poly[y_c, k, 3]

                    s6_delta = s6 * delta
                    d2 = s6_delta + z0
                    d1 = (0.5 * s6_delta + z0) * delta + c1
                    s_neg_d2 = -d2
                    for i in range(K):
                        grad[i] -= d1 * p_vec[i]
                        sp_i = s_neg_d2 * p_vec[i]
                        for j in range(i, K):
                            hess[i, j] += sp_i * p_vec[j]

                for i in range(K):
                    grad[i] *= numi
                    for j in range(i, K):
                        hess[i, j] *= numi_sq
                        hess[j, i] = hess[i, j]

                H_psd, max_eig = _psd_small_jacobi(hess, K, 1e-3)
                norm_factor = max_eig if max_eig >= 1e-10 else 1e-10

                for i in range(K):
                    lb[i] = -sol[i]
                    d_qp[i] = -grad[i] / norm_factor
                    for j in range(K):
                        D[i, j] = H_psd[i, j] / norm_factor
                    D[i, i] += 1e-7

                delta_w = _box_qp_small(D, d_qp, lb, K, 50)

                change = 0.0
                for i in range(K):
                    w_new = sol[i] + 0.3 * delta_w[i]
                    change += abs(w_new - w[i])
                    w[i] = w_new
                if change <= 0.001:
                    break

            for i in range(K):
                out_weights[task_idx, i] = w[i]

            if compute_score:
                nll = 0.0
                for g in range(G):
                    p_sum = 0.0
                    for i in range(K):
                        p_sum += w[i] * P_KT[types[i], g]
                    pred = numi * p_sum
                    if pred < 1e-4:
                        pred = 1e-4
                    elif pred > lam_max:
                        pred = lam_max
                    y_c = y_vec[g]
                    r = int(np.sqrt(pred * 1e6))
                    if r >= k_size:
                        r = k_size - 1
                    k = k_of_l[r]
                    delta = pred - x_of_l[r]

                    if y_c == 0:
                        c0 = poly_0[k, 0]
                        c1 = poly_0[k, 1]
                        z0 = poly_0[k, 2]
                        s6 = poly_0[k, 3]
                    else:
                        c0 = poly[y_c, k, 0]
                        c1 = poly[y_c, k, 1]
                        z0 = poly[y_c, k, 2]
                        s6 = poly[y_c, k, 3]

                    s6_delta = s6 * delta
                    d0_val = ((s6_delta * (1.0 / 6.0) + 0.5 * z0) * delta + c1) * delta + c0
                    nll -= d0_val
                out_scores[task_idx] = nll

    @numba.njit(nogil=True, cache=True)
    def _score_sigma_5_factors_kernel(X, Y_int, poly, k_of_l, x_of_l, lam_max):
        """Evaluate min NLL across (0.8, 0.9, 1.0, 1.1, 1.2) in one fused d0-only pass."""
        N = X.shape[0]
        k_size = k_of_l.size
        poly_0 = poly[0]
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0

        for j in range(N):
            x = X[j]
            y_c = Y_int[j]

            p0 = x * 0.8
            if p0 < 1e-4:
                p0 = 1e-4
            elif p0 > lam_max:
                p0 = lam_max
            r0 = int(np.sqrt(p0 * 1e6))
            if r0 >= k_size:
                r0 = k_size - 1
            k0 = k_of_l[r0]
            d0_val = p0 - x_of_l[r0]

            p1 = x * 0.9
            if p1 < 1e-4:
                p1 = 1e-4
            elif p1 > lam_max:
                p1 = lam_max
            r1 = int(np.sqrt(p1 * 1e6))
            if r1 >= k_size:
                r1 = k_size - 1
            k1 = k_of_l[r1]
            d1_val = p1 - x_of_l[r1]

            p2 = x
            if p2 < 1e-4:
                p2 = 1e-4
            elif p2 > lam_max:
                p2 = lam_max
            r2 = int(np.sqrt(p2 * 1e6))
            if r2 >= k_size:
                r2 = k_size - 1
            k2 = k_of_l[r2]
            d2_val = p2 - x_of_l[r2]

            p3 = x * 1.1
            if p3 < 1e-4:
                p3 = 1e-4
            elif p3 > lam_max:
                p3 = lam_max
            r3 = int(np.sqrt(p3 * 1e6))
            if r3 >= k_size:
                r3 = k_size - 1
            k3 = k_of_l[r3]
            d3_val = p3 - x_of_l[r3]

            p4 = x * 1.2
            if p4 < 1e-4:
                p4 = 1e-4
            elif p4 > lam_max:
                p4 = lam_max
            r4 = int(np.sqrt(p4 * 1e6))
            if r4 >= k_size:
                r4 = k_size - 1
            k4 = k_of_l[r4]
            d4_val = p4 - x_of_l[r4]

            if y_c == 0:
                sd0 = poly_0[k0, 3] * d0_val
                s0 -= (
                    (sd0 * (1.0 / 6.0) + 0.5 * poly_0[k0, 2]) * d0_val + poly_0[k0, 1]
                ) * d0_val + poly_0[k0, 0]
                sd1 = poly_0[k1, 3] * d1_val
                s1 -= (
                    (sd1 * (1.0 / 6.0) + 0.5 * poly_0[k1, 2]) * d1_val + poly_0[k1, 1]
                ) * d1_val + poly_0[k1, 0]
                sd2 = poly_0[k2, 3] * d2_val
                s2 -= (
                    (sd2 * (1.0 / 6.0) + 0.5 * poly_0[k2, 2]) * d2_val + poly_0[k2, 1]
                ) * d2_val + poly_0[k2, 0]
                sd3 = poly_0[k3, 3] * d3_val
                s3 -= (
                    (sd3 * (1.0 / 6.0) + 0.5 * poly_0[k3, 2]) * d3_val + poly_0[k3, 1]
                ) * d3_val + poly_0[k3, 0]
                sd4 = poly_0[k4, 3] * d4_val
                s4 -= (
                    (sd4 * (1.0 / 6.0) + 0.5 * poly_0[k4, 2]) * d4_val + poly_0[k4, 1]
                ) * d4_val + poly_0[k4, 0]
            else:
                sd0 = poly[y_c, k0, 3] * d0_val
                s0 -= (
                    (sd0 * (1.0 / 6.0) + 0.5 * poly[y_c, k0, 2]) * d0_val + poly[y_c, k0, 1]
                ) * d0_val + poly[y_c, k0, 0]
                sd1 = poly[y_c, k1, 3] * d1_val
                s1 -= (
                    (sd1 * (1.0 / 6.0) + 0.5 * poly[y_c, k1, 2]) * d1_val + poly[y_c, k1, 1]
                ) * d1_val + poly[y_c, k1, 0]
                sd2 = poly[y_c, k2, 3] * d2_val
                s2 -= (
                    (sd2 * (1.0 / 6.0) + 0.5 * poly[y_c, k2, 2]) * d2_val + poly[y_c, k2, 1]
                ) * d2_val + poly[y_c, k2, 0]
                sd3 = poly[y_c, k3, 3] * d3_val
                s3 -= (
                    (sd3 * (1.0 / 6.0) + 0.5 * poly[y_c, k3, 2]) * d3_val + poly[y_c, k3, 1]
                ) * d3_val + poly[y_c, k3, 0]
                sd4 = poly[y_c, k4, 3] * d4_val
                s4 -= (
                    (sd4 * (1.0 / 6.0) + 0.5 * poly[y_c, k4, 2]) * d4_val + poly[y_c, k4, 1]
                ) * d4_val + poly[y_c, k4, 0]

        min_s = s0
        if s1 < min_s:
            min_s = s1
        if s2 < min_s:
            min_s = s2
        if s3 < min_s:
            min_s = s3
        if s4 < min_s:
            min_s = s4
        return min_s


_pool: ThreadPoolExecutor | None = None
_pool_threads = 0


def num_threads() -> int:
    """Threads the kernels run on: torch's, at most ``NUMBA_NUM_THREADS``.

    ``NUMBA_NUM_THREADS`` defaults to every CPU the process may run on, which on a node that does
    not pin a job's CPUs is the whole node; torch follows ``OMP_NUM_THREADS`` and
    ``torch.set_num_threads``, which job schedulers and users set.
    """
    return max(1, min(torch.get_num_threads(), numba.config.NUMBA_NUM_THREADS))


def parallel_rows(fn, n: int, min_chunk: int, *args, threads: int | None = None, align: int = 1):
    """``fn(start, end, *args)`` over rows [0, n) on the thread pool, in one chunk per thread
    (``fn`` must release the GIL to run in parallel: a numba kernel, a LAPACK call).

    ``threads`` defaults to ``num_threads()``; every chunk starts at a multiple of ``align``.
    """
    global _pool, _pool_threads
    threads = threads or num_threads()
    chunk = max(min_chunk, -(-n // threads))
    chunk = -(-chunk // align) * align
    if n <= chunk:
        fn(0, n, *args)
        return
    if _pool_threads != threads:  # a new pool when torch's thread count changes
        if _pool is not None:
            _pool.shutdown(wait=False)
        _pool = ThreadPoolExecutor(threads, thread_name_prefix="rctd-numba")
        _pool_threads = threads
    for f in [_pool.submit(fn, s, min(s + chunk, n), *args) for s in range(0, n, chunk)]:
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
    parallel_rows(
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
    parallel_rows(
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


def box_qp_sweeps(D, d, lower_bound, n_sweeps=50):
    """The Gauss-Seidel sweeps of ``_solve_box_qp_batch`` (min 0.5 x'Dx - d'x, x >= lb).

    All ``n_sweeps`` of them, as the compiled ``_solve_box_qp_batch_impl`` does (the TorchScript
    fallback stops once the whole batch has converged, which makes a pixel's result depend on its
    batch). Stopping each pixel on its own instead was as exact in float64 after
    ``_finish_box_qp``, but in float32, whose finish tolerance is far looser, it moved weights.
    """
    x = torch.empty_like(d)
    parallel_rows(
        _box_qp_kernel,
        d.shape[0],
        16,
        D.contiguous().numpy(),
        d.contiguous().numpy(),
        lower_bound.contiguous().numpy(),
        x.numpy(),
        n_sweeps,
    )
    return x


def prepare_y_int(Y, k_val: int, chunk_rows: int = 32_768) -> np.ndarray:
    """Validate non-negative finite counts and clip to ``[0, k_val]`` as ``int32``."""
    if sp.issparse(Y):
        src = Y if isinstance(Y, sp.csr_matrix) else Y.tocsr()
        out = np.empty(src.shape, dtype=np.int32)
        if src.shape[0] == 0 or src.shape[1] == 0:
            return out
        is_float = np.issubdtype(src.dtype, np.floating)
        row_bytes = max(src.shape[1] * max(np.dtype(src.dtype).itemsize, 1), 1)
        step = max(1, min(chunk_rows, (256 * 1024 * 1024) // row_bytes))
        n_rows = src.shape[0]
        for start in range(0, n_rows, step):
            end = min(start + step, n_rows)
            chunk = src[start:end].toarray()
            if is_float and not np.isfinite(chunk).all():
                raise ValueError("counts must be finite")
            if (chunk < 0).any():
                raise ValueError("counts must be non-negative")
            out[start:end] = np.clip(chunk, 0, k_val).astype(np.int32, copy=False)
        return out
    Y_np = Y.detach().cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
    if np.issubdtype(Y_np.dtype, np.floating) and not np.isfinite(Y_np).all():
        raise ValueError("counts must be finite")
    if Y_np.size > 0 and (Y_np < 0).any():
        raise ValueError("counts must be non-negative")
    return np.ascontiguousarray(np.clip(Y_np, 0, k_val), dtype=np.int32)


def build_poly_table(
    Q_mat, SQ_mat, x_vals, K_val: int = -1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Build Horner cubic-spline table ``(poly, k_of_l, x_of_l, k_val, lam_max)``."""
    Q_np = Q_mat.detach().cpu().numpy() if isinstance(Q_mat, torch.Tensor) else np.asarray(Q_mat)
    SQ_np = (
        SQ_mat.detach().cpu().numpy() if isinstance(SQ_mat, torch.Tensor) else np.asarray(SQ_mat)
    )
    x_np = x_vals.detach().cpu().numpy() if isinstance(x_vals, torch.Tensor) else np.asarray(x_vals)
    Q_np = np.asarray(Q_np, dtype=np.float64)
    SQ_np = np.asarray(SQ_np, dtype=np.float64)
    x_np = np.asarray(x_np, dtype=np.float64)

    k_val = int(Q_np.shape[0] - 3 if K_val < 0 else K_val)
    x_max = float(x_np[-1])
    lam_max = x_max - 1e-4
    m_of_l = _interval_table(x_max, x_np.shape[0])
    k_of_l = np.ascontiguousarray(m_of_l - 1, dtype=np.int64)
    x_of_l = np.ascontiguousarray(x_np[k_of_l], dtype=np.float64)

    Q_sub = Q_np[: k_val + 1]
    SQ_sub = SQ_np[: k_val + 1]
    hi = np.diff(x_np)[None, :]
    fti1 = Q_sub[:, :-1]
    fti = Q_sub[:, 1:]
    zi1 = SQ_sub[:, :-1]
    zi = SQ_sub[:, 1:]

    poly = np.empty((k_val + 1, x_np.shape[0] - 1, 4), dtype=np.float64)
    poly[:, :, 0] = fti1
    poly[:, :, 1] = (fti - fti1) / hi - hi * (zi + 2.0 * zi1) * (1.0 / 6.0)
    poly[:, :, 2] = zi1
    poly[:, :, 3] = (zi - zi1) / hi
    return np.ascontiguousarray(poly), k_of_l, x_of_l, k_val, lam_max


def irwls_shared_d1_d2(
    Y_int: np.ndarray,
    pred: torch.Tensor,
    nUMI: torch.Tensor,
    poly: np.ndarray,
    k_of_l: np.ndarray,
    x_of_l: np.ndarray,
    lam_max: float,
    d1_dtype: torch.dtype = torch.float64,
    d2_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled spline derivatives ``(d1 * nUMI, -d2 * nUMI^2)`` on CPU."""
    N, G = Y_int.shape
    d1_out = torch.empty((N, G), dtype=d1_dtype)
    d2_out = torch.empty((N, G), dtype=d2_dtype)
    if N > 0:
        parallel_rows(
            _calc_d1_d2_scaled_kernel,
            N,
            4,
            np.ascontiguousarray(Y_int, dtype=np.int32),
            pred.contiguous().numpy(),
            nUMI.contiguous().numpy(),
            poly,
            k_of_l,
            x_of_l,
            float(lam_max),
            d1_out.numpy(),
            d2_out.numpy(),
        )
    return d1_out, d2_out


def unpack_sym_hess(H_tri: torch.Tensor, K: int) -> torch.Tensor:
    """Unpack upper-triangle ``(N, K*(K+1)//2)`` Hessian rows into ``(N, K, K)``."""
    N = H_tri.shape[0]
    H_out = torch.empty((N, K, K), dtype=H_tri.dtype)
    if N > 0:
        parallel_rows(
            _unpack_sym_hess_kernel,
            N,
            4,
            H_tri.contiguous().numpy(),
            int(K),
            H_out.numpy(),
        )
    return H_out


def solve_box_qp_cpu(
    D: torch.Tensor,
    d: torch.Tensor,
    lower_bound: torch.Tensor,
    n_sweeps: int = 50,
    kkt_tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused CPU box-QP coordinate descent + active-set polish, returning ``(x, done)``."""
    N = d.shape[0]
    x = torch.empty_like(d)
    done = torch.empty(N, dtype=torch.bool)
    if N > 0:
        parallel_rows(
            _solve_box_qp_fused_kernel,
            N,
            4,
            D.contiguous().numpy(),
            d.contiguous().numpy(),
            lower_bound.contiguous().numpy(),
            x.numpy(),
            done.numpy(),
            int(n_sweeps),
            float(kkt_tol),
        )
    return x, done


def fit_and_score_pairs_cpu(
    triples_arr: np.ndarray,
    Y_int: np.ndarray,
    nUMI_np: np.ndarray,
    P_KT: np.ndarray,
    poly: np.ndarray,
    k_of_l: np.ndarray,
    x_of_l: np.ndarray,
    lam_max: float,
    max_iter: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fused doublet pair fitting + NLL scoring on CPU."""
    M = triples_arr.shape[0]
    out_weights = np.empty((M, 2), dtype=np.float64)
    out_scores = np.empty(M, dtype=np.float64)
    out_converged = np.empty(M, dtype=np.bool_)
    if M > 0:
        parallel_rows(
            _fit_and_score_pairs_kernel,
            M,
            4,
            np.ascontiguousarray(triples_arr, dtype=np.int64),
            np.ascontiguousarray(Y_int, dtype=np.int32),
            np.ascontiguousarray(nUMI_np, dtype=np.float64),
            np.ascontiguousarray(P_KT, dtype=np.float64),
            poly,
            k_of_l,
            x_of_l,
            float(lam_max),
            int(max_iter),
            out_weights,
            out_scores,
            out_converged,
        )
    return out_weights, out_scores, out_converged


def fit_and_score_singles_cpu(
    singles_arr: np.ndarray,
    Y_int: np.ndarray,
    nUMI_np: np.ndarray,
    P_KT: np.ndarray,
    poly: np.ndarray,
    k_of_l: np.ndarray,
    x_of_l: np.ndarray,
    lam_max: float,
    max_iter: int = 25,
) -> np.ndarray:
    """Fused singlet fitting + NLL scoring on CPU."""
    M = singles_arr.shape[0]
    out_scores = np.empty(M, dtype=np.float64)
    if M > 0:
        parallel_rows(
            _fit_and_score_singles_kernel,
            M,
            8,
            np.ascontiguousarray(singles_arr, dtype=np.int64),
            np.ascontiguousarray(Y_int, dtype=np.int32),
            np.ascontiguousarray(nUMI_np, dtype=np.float64),
            np.ascontiguousarray(P_KT, dtype=np.float64),
            poly,
            k_of_l,
            x_of_l,
            float(lam_max),
            int(max_iter),
            out_scores,
        )
    return out_scores


def fit_and_score_subsets_cpu(
    pix_indices: np.ndarray,
    type_matrix: np.ndarray,
    Y_int: np.ndarray,
    nUMI_np: np.ndarray,
    P_KT: np.ndarray,
    poly: np.ndarray,
    k_of_l: np.ndarray,
    x_of_l: np.ndarray,
    lam_max: float,
    max_iter: int = 25,
) -> np.ndarray:
    """Fused subset IRWLS + NLL scoring (arbitrary ``K_sub >= 1``) on CPU."""
    M, k_sub = type_matrix.shape
    out_weights = np.empty((M, k_sub), dtype=np.float64)
    out_scores = np.empty(M, dtype=np.float64)
    if M > 0:
        parallel_rows(
            _fit_subsets_kernel,
            M,
            4,
            np.ascontiguousarray(pix_indices, dtype=np.int64),
            np.ascontiguousarray(type_matrix, dtype=np.int64),
            int(k_sub),
            np.ascontiguousarray(Y_int, dtype=np.int32),
            np.ascontiguousarray(nUMI_np, dtype=np.float64),
            np.ascontiguousarray(P_KT, dtype=np.float64),
            poly,
            k_of_l,
            x_of_l,
            float(lam_max),
            int(max_iter),
            out_weights,
            out_scores,
            True,
        )
    return out_scores


def fit_subsets_weights_cpu(
    pix_indices: np.ndarray,
    type_matrix: np.ndarray,
    Y_int: np.ndarray,
    nUMI_np: np.ndarray,
    P_KT: np.ndarray,
    poly: np.ndarray,
    k_of_l: np.ndarray,
    x_of_l: np.ndarray,
    lam_max: float,
    max_iter: int = 50,
) -> np.ndarray:
    """Fused subset IRWLS returning fitted weights ``(M, K_sub)`` on CPU."""
    M, k_sub = type_matrix.shape
    out_weights = np.empty((M, k_sub), dtype=np.float64)
    out_scores = np.empty(1, dtype=np.float64)
    if M > 0:
        parallel_rows(
            _fit_subsets_kernel,
            M,
            4,
            np.ascontiguousarray(pix_indices, dtype=np.int64),
            np.ascontiguousarray(type_matrix, dtype=np.int64),
            int(k_sub),
            np.ascontiguousarray(Y_int, dtype=np.int32),
            np.ascontiguousarray(nUMI_np, dtype=np.float64),
            np.ascontiguousarray(P_KT, dtype=np.float64),
            poly,
            k_of_l,
            x_of_l,
            float(lam_max),
            int(max_iter),
            out_weights,
            out_scores,
            False,
        )
    return out_weights


def score_sigma_5_factors_cpu(
    X_np: np.ndarray,
    Y_int: np.ndarray,
    poly: np.ndarray,
    k_of_l: np.ndarray,
    x_of_l: np.ndarray,
    lam_max: float,
) -> float:
    """Evaluate min NLL across factors (0.8, 0.9, 1.0, 1.1, 1.2) on CPU."""
    X_arr = np.ascontiguousarray(X_np, dtype=np.float64).reshape(-1)
    Y_arr = np.ascontiguousarray(Y_int, dtype=np.int32).reshape(-1)
    if X_arr.size == 0:
        return 0.0
    return float(_score_sigma_5_factors_kernel(X_arr, Y_arr, poly, k_of_l, x_of_l, float(lam_max)))
