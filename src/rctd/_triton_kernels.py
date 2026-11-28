"""Fused Triton GPU kernels for likelihood splines, IRWLS, doublet/multi, and sigma estimation.

Eliminates intermediate ``(N, G)`` / ``(M, G, K)`` PyTorch tensor allocations and dozens of
elementwise GPU kernel launches per IRWLS iteration without incurring ``torch.compile`` warmup.
``_calc_q_all_impl`` / ``_solve_box_qp_batch`` PyTorch paths remain active when Triton is
unavailable, when ``RCTD_TRITON=0``, or when a caller overrides ``_calc_q_fn``.
"""

from __future__ import annotations

import functools
import math
import os
import warnings

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

ENABLED = triton is not None and os.environ.get("RCTD_TRITON", "1") != "0"

# Set to True if the runtime JIT probe fails (e.g. missing cuda.h/libnvrtc on a broken stack).
_RUNTIME_DISABLED = False

# Cap elements per 1D grid slab so 32-bit CUDA grid limits are never exceeded.
_MAX_PAIR_SLAB = 524_288


if triton is not None:

    @triton.jit
    def _triton_probe_kernel(out_ptr):
        tl.store(out_ptr, 1)


def _run_triton_probe(device_index: int) -> None:
    """Compile and launch a 1-element kernel to verify the CUDA/Triton JIT stack works."""
    dev = torch.device("cuda", device_index)
    with torch.cuda.device(dev):
        buf = torch.empty((1,), dtype=torch.int32, device=dev)
        _triton_probe_kernel[(1,)](buf)


@functools.lru_cache(maxsize=8)
def _cuda_triton_supported(device_index: int) -> bool:
    """Return True when the GPU is Volta+ (sm_70+) and Triton JIT compiles/launches cleanly."""
    global _RUNTIME_DISABLED
    if _RUNTIME_DISABLED or triton is None:
        return False
    try:
        major, _ = torch.cuda.get_device_capability(device_index)
    except Exception:
        return False
    if major < 7:
        return False
    try:
        _run_triton_probe(device_index)
    except Exception as exc:
        _RUNTIME_DISABLED = True
        warnings.warn(
            f"Triton GPU kernel probe failed on cuda:{device_index} ({exc}); falling back to "
            "PyTorch GPU kernels. Set RCTD_TRITON=0 to skip this check.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


def use_triton(t: torch.Tensor) -> bool:
    """Return True when Triton kernels are enabled, supported on ``t``'s CUDA arch, and healthy."""
    if _RUNTIME_DISABLED or not ENABLED or t.device.type != "cuda":
        return False
    if os.environ.get("RCTD_TRITON", "1") == "0":
        return False
    idx = int(t.device.index) if t.device.index is not None else int(torch.cuda.current_device())
    return _cuda_triton_supported(idx)


def _int_count_dtype(k_val: int) -> tuple[torch.dtype, np.dtype]:
    """Use compact int16 for standard Q_mat ceilings (k_val <= 32767) and int32 otherwise."""
    if k_val <= 32_767:
        return torch.int16, np.dtype(np.int16)
    return torch.int32, np.dtype(np.int32)


@functools.lru_cache(maxsize=16)
def _interval_table_cpu(x_max: float, n_x: int) -> np.ndarray:
    """Spline segment index m in [1, n_x-1] for each l = floor(sqrt(lambda / 1e-6))."""
    max_l = int(math.sqrt(x_max / 1e-6)) + 2
    l = torch.arange(max_l, dtype=torch.long)
    m = torch.clamp(l - 9, max=40) + torch.clamp(
        torch.ceil(torch.sqrt(torch.clamp(l.float() - 48.7499, min=0.0) * 4.0)).long() - 2,
        min=0,
    )
    return torch.clamp(m, 1, n_x - 1).numpy()


def prepare_y_int_gpu(Y: torch.Tensor, k_val: int) -> torch.Tensor:
    """Validate finite non-negative counts on GPU and clamp to ``[0, k_val]`` as int16/int32."""
    if Y.is_floating_point() and not torch.isfinite(Y).all():
        raise ValueError("counts must be finite")
    if Y.numel() > 0 and (Y < 0).any():
        raise ValueError("counts must be non-negative")
    t_dtype, _ = _int_count_dtype(k_val)
    return Y.clamp(0, k_val).to(t_dtype).contiguous()


def prepare_y_int_from_cpu(
    spatial_counts: np.ndarray,
    k_val: int,
    device: torch.device,
    chunk_rows: int = 32_768,
) -> torch.Tensor:
    """Validate, clamp, and stream CPU counts into a compact ``int16``/``int32`` GPU tensor.

    Avoids staging the full ``float64``/``float32`` count matrix on GPU before conversion.
    """
    t_dtype, np_dtype = _int_count_dtype(k_val)
    arr = np.asarray(spatial_counts)
    out = torch.empty(arr.shape, dtype=t_dtype, device=device)
    if arr.size == 0:
        return out
    is_float = np.issubdtype(arr.dtype, np.floating)
    n_rows = arr.shape[0]
    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        chunk = arr[start:end]
        if is_float and not np.isfinite(chunk).all():
            raise ValueError("counts must be finite")
        if (chunk < 0).any():
            raise ValueError("counts must be non-negative")
        clamped = np.ascontiguousarray(np.clip(chunk, 0, k_val).astype(np_dtype, copy=False))
        out[start:end].copy_(torch.from_numpy(clamped), non_blocking=False)
    return out


def prepare_y_int_to_device(
    counts: np.ndarray | torch.Tensor,
    k_val: int,
    device: torch.device,
) -> torch.Tensor:
    """Validate and stage CPU or GPU counts as compact clamped ``int16``/``int32`` on ``device``."""
    if isinstance(counts, torch.Tensor):
        t_dev = counts if counts.device == device else counts.to(device)
        return prepare_y_int_gpu(t_dev, k_val)
    return prepare_y_int_from_cpu(counts, k_val, device)


def build_poly_table_gpu(
    Q_mat: torch.Tensor,
    SQ_mat: torch.Tensor,
    x_vals: torch.Tensor,
    K_val: int = -1,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, float
]:
    """Build GPU Horner cubic-spline tables ``(c0, c1, z0, s6, k_of_l, x_of_l, k_val, lam_max)``."""
    device = Q_mat.device
    target_dtype = Q_mat.dtype
    k_val = int(Q_mat.shape[0] - 3 if K_val < 0 else K_val)

    x_f64 = x_vals.to(device=device, dtype=torch.float64)
    x_max = float(x_f64[-1].item())
    lam_max = x_max - 1e-4

    m_of_l = _interval_table_cpu(x_max, int(x_f64.shape[0]))
    k_of_l = torch.as_tensor(m_of_l - 1, device=device, dtype=torch.int32).contiguous()
    x_of_l = x_vals.to(device=device, dtype=target_dtype)[k_of_l.long()].contiguous()

    Q_sub = Q_mat[: k_val + 1].to(device=device, dtype=target_dtype)
    SQ_sub = SQ_mat[: k_val + 1].to(device=device, dtype=target_dtype)
    x_t = x_vals.to(device=device, dtype=target_dtype)

    hi = (x_t[1:] - x_t[:-1]).unsqueeze(0)
    fti1 = Q_sub[:, :-1].contiguous()
    fti = Q_sub[:, 1:]
    zi1 = SQ_sub[:, :-1].contiguous()
    zi = SQ_sub[:, 1:]

    c0 = fti1
    c1 = ((fti - fti1) / hi - hi * (zi + 2.0 * zi1) * (1.0 / 6.0)).contiguous()
    z0 = zi1
    s6 = ((zi - zi1) / hi).contiguous()
    return c0, c1, z0, s6, k_of_l, x_of_l, k_val, lam_max


if triton is not None:

    @triton.jit
    def _max_nan_prop(a, b):
        """Exact torch.maximum semantics: propagate NaN when either operand is NaN."""
        nan_mask = (a != a) | (b != b)
        res = tl.where(a >= b, a, b)
        return tl.where(nan_mask, a + b, res)

    @triton.jit
    def _irwls_shared_d1_d2_kernel(
        Y_ptr,
        pred_raw_ptr,
        nUMI_ptr,
        thresh_ptr,
        c1_ptr,
        z0_ptr,
        s6_ptr,
        k_of_l_ptr,
        x_of_l_ptr,
        d1_out_ptr,
        d2_out_ptr,
        N,
        G,
        n_seg,
        k_size,
        lam_max,
        stride_yn,
        stride_yg,
        stride_pn,
        stride_pg,
        stride_d1n,
        stride_d1g,
        stride_d2n,
        stride_d2g,
        USE_FP64: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        n_i64 = tl.full([], N, dtype=tl.int64)
        g_i64 = tl.full([], G, dtype=tl.int64)
        total = n_i64 * g_i64
        mask = offs < total

        n_idx = offs // g_i64
        g_idx = offs - n_idx * g_i64

        compute_t = tl.float64 if USE_FP64 else tl.float32
        numi = tl.load(nUMI_ptr + n_idx, mask=mask, other=1.0).to(compute_t)
        thresh = tl.load(thresh_ptr + n_idx, mask=mask, other=1e-4).to(compute_t)
        raw = tl.load(
            pred_raw_ptr + n_idx * stride_pn + g_idx * stride_pg, mask=mask, other=0.0
        ).to(compute_t)
        p = tl.abs(numi * raw)
        lam_max_t = tl.full([BLOCK_SIZE], lam_max, dtype=compute_t)
        p = tl.minimum(tl.maximum(p, thresh), lam_max_t)

        r_raw = tl.floor(tl.sqrt(p.to(tl.float64) / 1e-6)).to(tl.int32)
        r = tl.minimum(tl.maximum(r_raw, 0), k_size - 1).to(tl.int64)
        k = tl.load(k_of_l_ptr + r, mask=mask, other=0).to(tl.int64)
        x_l = tl.load(x_of_l_ptr + r, mask=mask, other=0.0).to(compute_t)
        delta = p - x_l

        y_c = tl.load(Y_ptr + n_idx * stride_yn + g_idx * stride_yg, mask=mask, other=0).to(
            tl.int64
        )
        idx = y_c * n_seg + k
        c1 = tl.load(c1_ptr + idx, mask=mask, other=0.0).to(compute_t)
        z0 = tl.load(z0_ptr + idx, mask=mask, other=0.0).to(compute_t)
        s6 = tl.load(s6_ptr + idx, mask=mask, other=0.0).to(compute_t)

        s6_delta = s6 * delta
        d2_val = s6_delta + z0
        d1_val = (0.5 * s6_delta + z0) * delta + c1

        numi_sq = numi * numi
        d1_scaled = tl.where(mask, d1_val * numi, 0.0)
        d2_scaled = tl.where(mask, -d2_val * numi_sq, 0.0)
        tl.store(d1_out_ptr + n_idx * stride_d1n + g_idx * stride_d1g, d1_scaled, mask=mask)
        tl.store(d2_out_ptr + n_idx * stride_d2n + g_idx * stride_d2g, d2_scaled, mask=mask)

    @triton.jit
    def _unpack_sym_hess_kernel(
        H_tri_ptr,
        H_out_ptr,
        row_tri_ptr,
        row_i_ptr,
        row_j_ptr,
        N,
        P_len,
        K,
        BLOCK_P: tl.constexpr,
    ):
        n = tl.program_id(0).to(tl.int64)
        if n >= N:
            return
        offs = tl.arange(0, BLOCK_P).to(tl.int64)
        mask = offs < P_len
        p_idx = tl.load(row_tri_ptr + offs, mask=mask, other=0).to(tl.int64)
        i_idx = tl.load(row_i_ptr + offs, mask=mask, other=0).to(tl.int64)
        j_idx = tl.load(row_j_ptr + offs, mask=mask, other=0).to(tl.int64)
        vals = tl.load(H_tri_ptr + n * P_len + p_idx, mask=mask, other=0.0)

        base = n * K * K
        tl.store(H_out_ptr + base + i_idx * K + j_idx, vals, mask=mask)
        tl.store(H_out_ptr + base + j_idx * K + i_idx, vals, mask=mask)

    @triton.jit
    def _solve_box_qp_kernel(
        D_ptr,
        d_ptr,
        lb_ptr,
        x_ptr,
        N,
        K,
        n_sweeps,
        tol,
        BLOCK_K: tl.constexpr,
        USE_FP64: tl.constexpr,
    ):
        r = tl.program_id(0).to(tl.int64)
        if r >= N:
            return
        compute_t = tl.float64 if USE_FP64 else tl.float32
        k_i64 = tl.full([], K, dtype=tl.int64)
        offs = tl.arange(0, BLOCK_K).to(tl.int64)
        mask = offs < k_i64

        d_vec = tl.load(d_ptr + r * k_i64 + offs, mask=mask, other=0.0).to(compute_t)
        lb_vec = tl.load(lb_ptr + r * k_i64 + offs, mask=mask, other=0.0).to(compute_t)
        row_base = r * k_i64 * k_i64
        D_diag = tl.load(D_ptr + row_base + offs * k_i64 + offs, mask=mask, other=1.0).to(compute_t)
        x = _max_nan_prop(d_vec / D_diag, lb_vec)

        sweep = 0
        done = tl.full([], 0, dtype=tl.int32)
        while (sweep < n_sweeps) & (done == 0):
            x_prev = x
            for i in range(K):
                row_i = tl.load(D_ptr + row_base + i * k_i64 + offs, mask=mask, other=0.0).to(
                    compute_t
                )
                off_diag = mask & (offs != i)
                Dx_off = tl.sum(tl.where(off_diag, row_i * x, 0.0), axis=0)
                D_ii = tl.load(D_ptr + row_base + i * k_i64 + i).to(compute_t)
                d_i = tl.load(d_ptr + r * k_i64 + i).to(compute_t)
                lb_i = tl.load(lb_ptr + r * k_i64 + i).to(compute_t)
                resid = d_i - Dx_off
                x_i_new = _max_nan_prop(resid / D_ii, lb_i)
                x = tl.where(offs == i, x_i_new, x)
            if tol > 0.0:
                max_diff = tl.max(tl.where(mask, tl.abs(x - x_prev), 0.0), axis=0)
                if (max_diff < tol) & (max_diff == max_diff):
                    done = 1
            sweep += 1

        tl.store(x_ptr + r * k_i64 + offs, x, mask=mask)

    @triton.jit
    def _fit_and_score_pairs_kernel(
        pix_ptr,
        t1_ptr,
        t2_ptr,
        Y_int_ptr,
        nUMI_ptr,
        P_KT_ptr,
        c0_ptr,
        c1_ptr,
        z0_ptr,
        s6_ptr,
        k_of_l_ptr,
        x_of_l_ptr,
        out_w_ptr,
        out_conv_ptr,
        out_score_ptr,
        M,
        G,
        n_seg,
        k_size,
        lam_max,
        max_iter,
        USE_FP64: tl.constexpr,
        COMPUTE_SCORE: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        tid = tl.program_id(0).to(tl.int64)
        if tid >= M:
            return

        n = tl.load(pix_ptr + tid).to(tl.int64)
        t1 = tl.load(t1_ptr + tid).to(tl.int64)
        t2 = tl.load(t2_ptr + tid).to(tl.int64)

        val_t = tl.float64 if USE_FP64 else tl.float32
        numi = tl.load(nUMI_ptr + n).to(tl.float64)
        numi_sq = numi * numi
        thresh = tl.maximum(numi * 1e-7, 1e-4)
        lam_max_f64 = tl.full([1], lam_max, dtype=tl.float64)
        lam_max_v = lam_max_f64.to(val_t)

        y_row = Y_int_ptr + n * G
        p0_row = P_KT_ptr + t1 * G
        p1_row = P_KT_ptr + t2 * G

        w0 = tl.full([], 0.5, dtype=tl.float64)
        w1 = tl.full([], 0.5, dtype=tl.float64)
        conv = tl.full([], 0, dtype=tl.int32)
        it = 0

        while (it < max_iter) & (conv == 0):
            sol0 = tl.maximum(w0, 0.0)
            sol1 = tl.maximum(w1, 0.0)
            s0_numi = (sol0 * numi).to(val_t)
            s1_numi = (sol1 * numi).to(val_t)
            thresh_v = thresh.to(val_t)

            g0_acc = tl.zeros([BLOCK_G], dtype=tl.float64)
            g1_acc = tl.zeros([BLOCK_G], dtype=tl.float64)
            h00_acc = tl.zeros([BLOCK_G], dtype=tl.float64)
            h01_acc = tl.zeros([BLOCK_G], dtype=tl.float64)
            h11_acc = tl.zeros([BLOCK_G], dtype=tl.float64)

            for g_start in range(0, G, BLOCK_G):
                g_offs = g_start + tl.arange(0, BLOCK_G).to(tl.int64)
                in_bounds = g_offs < G
                p0 = tl.load(p0_row + g_offs, mask=in_bounds, other=0.0).to(val_t)
                p1 = tl.load(p1_row + g_offs, mask=in_bounds, other=0.0).to(val_t)
                nz_mask = in_bounds & ((p0 != 0.0) | (p1 != 0.0))

                pred = tl.abs(s0_numi * p0 + s1_numi * p1)
                pred = tl.minimum(tl.maximum(pred, thresh_v), lam_max_v)

                r_raw = tl.floor(tl.sqrt(pred.to(tl.float64) / 1e-6)).to(tl.int32)
                r = tl.minimum(tl.maximum(r_raw, 0), k_size - 1).to(tl.int64)
                k = tl.load(k_of_l_ptr + r, mask=nz_mask, other=0).to(tl.int64)
                x_l = tl.load(x_of_l_ptr + r, mask=nz_mask, other=0.0).to(val_t)
                delta = pred - x_l

                y_c = tl.load(y_row + g_offs, mask=nz_mask, other=0).to(tl.int64)
                idx = y_c * n_seg + k
                c1 = tl.load(c1_ptr + idx, mask=nz_mask, other=0.0).to(val_t)
                z0 = tl.load(z0_ptr + idx, mask=nz_mask, other=0.0).to(val_t)
                s6 = tl.load(s6_ptr + idx, mask=nz_mask, other=0.0).to(val_t)

                s6_delta = s6 * delta
                d2 = s6_delta + z0
                d1 = (0.5 * s6_delta + z0) * delta + c1

                d1_f = d1.to(tl.float64)
                s_f = (-d2).to(tl.float64)
                p0_f = p0.to(tl.float64)
                p1_f = p1.to(tl.float64)
                nz_f = nz_mask.to(tl.float64)

                sp0 = s_f * p0_f
                g0_acc += (d1_f * p0_f) * nz_f
                g1_acc += (d1_f * p1_f) * nz_f
                h00_acc += (sp0 * p0_f) * nz_f
                h01_acc += (sp0 * p1_f) * nz_f
                h11_acc += (s_f * p1_f * p1_f) * nz_f

            grad0 = -tl.sum(g0_acc, axis=0) * numi
            grad1 = -tl.sum(g1_acc, axis=0) * numi
            h00 = tl.sum(h00_acc, axis=0) * numi_sq
            h01 = tl.sum(h01_acc, axis=0) * numi_sq
            h11 = tl.sum(h11_acc, axis=0) * numi_sq

            half_trace = (h00 + h11) * 0.5
            diff_ht = (h00 - h11) * 0.5
            disc = tl.sqrt(diff_ht * diff_ht + h01 * h01)
            min_eig = half_trace - disc
            max_eig = half_trace + disc

            lam1_c = 1e-3
            lam2_c = tl.maximum(max_eig, 1e-3)
            safe_disc = tl.maximum(disc, 1e-30)
            cos2t = diff_ht / safe_disc
            sin2t = h01 / safe_disc
            half_sum = (lam1_c + lam2_c) * 0.5
            half_diff = (lam2_c - lam1_c) * 0.5

            psd_needed = min_eig < 1e-3
            H00 = tl.where(psd_needed, half_sum + half_diff * cos2t, h00)
            H01 = tl.where(psd_needed, half_diff * sin2t, h01)
            H11 = tl.where(psd_needed, half_sum - half_diff * cos2t, h11)
            lam2 = tl.where(psd_needed, lam2_c, max_eig)

            norm_factor = tl.maximum(lam2, 1e-10)
            D00 = H00 / norm_factor + 1e-7
            D01 = H01 / norm_factor
            D11 = H11 / norm_factor + 1e-7
            d0_qp = -grad0 / norm_factor
            d1_qp = -grad1 / norm_factor

            det = D00 * D11 - D01 * D01
            det_sign = tl.where(det >= 0.0, 1.0, -1.0)
            det_safe = tl.maximum(tl.abs(det), 1e-30) * det_sign

            x0 = (D11 * d0_qp - D01 * d1_qp) / det_safe
            x1 = (D00 * d1_qp - D01 * d0_qp) / det_safe
            x0 = tl.maximum(x0, -sol0)
            x1 = tl.maximum(x1, -sol1)
            x0 = tl.maximum((d0_qp - D01 * x1) / D00, -sol0)
            x1 = tl.maximum((d1_qp - D01 * x0) / D11, -sol1)

            w0_new = sol0 + 0.3 * x0
            w1_new = sol1 + 0.3 * x1

            change = tl.abs(w0_new - w0) + tl.abs(w1_new - w1)
            w0 = w0_new
            w1 = w1_new
            if change <= 0.001:
                conv = 1
            it += 1

        tl.store(out_w_ptr + tid * 2, w0)
        tl.store(out_w_ptr + tid * 2 + 1, w1)
        tl.store(out_conv_ptr + tid, conv != 0)

        if COMPUTE_SCORE:
            w0_numi = (w0 * numi).to(val_t)
            w1_numi = (w1 * numi).to(val_t)
            min_eps = tl.full([BLOCK_G], 1e-4, dtype=val_t)
            nll_acc = tl.zeros([BLOCK_G], dtype=tl.float64)

            for g_start in range(0, G, BLOCK_G):
                g_offs = g_start + tl.arange(0, BLOCK_G).to(tl.int64)
                in_bounds = g_offs < G
                p0 = tl.load(p0_row + g_offs, mask=in_bounds, other=0.0).to(val_t)
                p1 = tl.load(p1_row + g_offs, mask=in_bounds, other=0.0).to(val_t)

                pred = w0_numi * p0 + w1_numi * p1
                pred = tl.minimum(tl.maximum(pred, min_eps), lam_max_v)

                r_raw = tl.floor(tl.sqrt(pred.to(tl.float64) / 1e-6)).to(tl.int32)
                r = tl.minimum(tl.maximum(r_raw, 0), k_size - 1).to(tl.int64)
                k = tl.load(k_of_l_ptr + r, mask=in_bounds, other=0).to(tl.int64)
                x_l = tl.load(x_of_l_ptr + r, mask=in_bounds, other=0.0).to(val_t)
                delta = pred - x_l

                y_c = tl.load(y_row + g_offs, mask=in_bounds, other=0).to(tl.int64)
                idx = y_c * n_seg + k
                c0 = tl.load(c0_ptr + idx, mask=in_bounds, other=0.0).to(val_t)
                c1 = tl.load(c1_ptr + idx, mask=in_bounds, other=0.0).to(val_t)
                z0 = tl.load(z0_ptr + idx, mask=in_bounds, other=0.0).to(val_t)
                s6 = tl.load(s6_ptr + idx, mask=in_bounds, other=0.0).to(val_t)

                s6_delta = s6 * delta
                d0_val = ((s6_delta * (1.0 / 6.0) + 0.5 * z0) * delta + c1) * delta + c0
                nll_acc -= tl.where(in_bounds, d0_val.to(tl.float64), 0.0)

            tl.store(out_score_ptr + tid, tl.sum(nll_acc, axis=0))

    @triton.jit
    def _fit_and_score_singles_kernel(
        pix_ptr,
        t_ptr,
        Y_int_ptr,
        nUMI_ptr,
        P_KT_ptr,
        c0_ptr,
        c1_ptr,
        z0_ptr,
        s6_ptr,
        k_of_l_ptr,
        x_of_l_ptr,
        out_w_ptr,
        out_score_ptr,
        M,
        G,
        n_seg,
        k_size,
        lam_max,
        max_iter,
        USE_FP64: tl.constexpr,
        WRITE_WEIGHTS: tl.constexpr,
        COMPUTE_SCORE: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        tid = tl.program_id(0).to(tl.int64)
        if tid >= M:
            return

        n = tl.load(pix_ptr + tid).to(tl.int64)
        t = tl.load(t_ptr + tid).to(tl.int64)

        val_t = tl.float64 if USE_FP64 else tl.float32
        numi = tl.load(nUMI_ptr + n).to(tl.float64)
        numi_sq = numi * numi
        thresh = tl.maximum(numi * 1e-7, 1e-4)
        lam_max_f64 = tl.full([1], lam_max, dtype=tl.float64)
        lam_max_v = lam_max_f64.to(val_t)

        y_row = Y_int_ptr + n * G
        p_row = P_KT_ptr + t * G

        w = tl.full([], 1.0, dtype=tl.float64)
        conv = tl.full([], 0, dtype=tl.int32)
        it = 0

        while (it < max_iter) & (conv == 0):
            sol = tl.maximum(w, 0.0)
            s_numi = (sol * numi).to(val_t)
            thresh_v = thresh.to(val_t)

            g_acc = tl.zeros([BLOCK_G], dtype=tl.float64)
            h_acc = tl.zeros([BLOCK_G], dtype=tl.float64)

            for g_start in range(0, G, BLOCK_G):
                g_offs = g_start + tl.arange(0, BLOCK_G).to(tl.int64)
                in_bounds = g_offs < G
                p = tl.load(p_row + g_offs, mask=in_bounds, other=0.0).to(val_t)
                nz_mask = in_bounds & (p != 0.0)

                pred = tl.abs(s_numi * p)
                pred = tl.minimum(tl.maximum(pred, thresh_v), lam_max_v)

                r_raw = tl.floor(tl.sqrt(pred.to(tl.float64) / 1e-6)).to(tl.int32)
                r = tl.minimum(tl.maximum(r_raw, 0), k_size - 1).to(tl.int64)
                k = tl.load(k_of_l_ptr + r, mask=nz_mask, other=0).to(tl.int64)
                x_l = tl.load(x_of_l_ptr + r, mask=nz_mask, other=0.0).to(val_t)
                delta = pred - x_l

                y_c = tl.load(y_row + g_offs, mask=nz_mask, other=0).to(tl.int64)
                idx = y_c * n_seg + k
                c1 = tl.load(c1_ptr + idx, mask=nz_mask, other=0.0).to(val_t)
                z0 = tl.load(z0_ptr + idx, mask=nz_mask, other=0.0).to(val_t)
                s6 = tl.load(s6_ptr + idx, mask=nz_mask, other=0.0).to(val_t)

                s6_delta = s6 * delta
                d2 = s6_delta + z0
                d1 = (0.5 * s6_delta + z0) * delta + c1

                d1_f = d1.to(tl.float64)
                d2_f = d2.to(tl.float64)
                p_f = p.to(tl.float64)
                nz_f = nz_mask.to(tl.float64)

                g_acc += (d1_f * p_f) * nz_f
                h_acc += (-d2_f * p_f * p_f) * nz_f

            grad = -tl.sum(g_acc, axis=0) * numi
            hess_val = tl.sum(h_acc, axis=0) * numi_sq
            H_psd = tl.maximum(hess_val, 1e-3)
            norm_factor = tl.maximum(H_psd, 1e-10)
            D = H_psd / norm_factor + 1e-7
            d_qp = -grad / norm_factor

            dq = d_qp / D
            delta_w = tl.maximum(dq, -sol)
            w_new = sol + 0.3 * delta_w
            change = tl.abs(w_new - w)
            w = w_new
            if change <= 0.001:
                conv = 1
            it += 1

        if WRITE_WEIGHTS:
            tl.store(out_w_ptr + tid, w)

        if COMPUTE_SCORE:
            w_numi = (w * numi).to(val_t)
            min_eps = tl.full([BLOCK_G], 1e-4, dtype=val_t)
            nll_acc = tl.zeros([BLOCK_G], dtype=tl.float64)

            for g_start in range(0, G, BLOCK_G):
                g_offs = g_start + tl.arange(0, BLOCK_G).to(tl.int64)
                in_bounds = g_offs < G
                p = tl.load(p_row + g_offs, mask=in_bounds, other=0.0).to(val_t)

                pred = w_numi * p
                pred = tl.minimum(tl.maximum(pred, min_eps), lam_max_v)

                r_raw = tl.floor(tl.sqrt(pred.to(tl.float64) / 1e-6)).to(tl.int32)
                r = tl.minimum(tl.maximum(r_raw, 0), k_size - 1).to(tl.int64)
                k = tl.load(k_of_l_ptr + r, mask=in_bounds, other=0).to(tl.int64)
                x_l = tl.load(x_of_l_ptr + r, mask=in_bounds, other=0.0).to(val_t)
                delta = pred - x_l

                y_c = tl.load(y_row + g_offs, mask=in_bounds, other=0).to(tl.int64)
                idx = y_c * n_seg + k
                c0 = tl.load(c0_ptr + idx, mask=in_bounds, other=0.0).to(val_t)
                c1 = tl.load(c1_ptr + idx, mask=in_bounds, other=0.0).to(val_t)
                z0 = tl.load(z0_ptr + idx, mask=in_bounds, other=0.0).to(val_t)
                s6 = tl.load(s6_ptr + idx, mask=in_bounds, other=0.0).to(val_t)

                s6_delta = s6 * delta
                d0_val = ((s6_delta * (1.0 / 6.0) + 0.5 * z0) * delta + c1) * delta + c0
                nll_acc -= tl.where(in_bounds, d0_val.to(tl.float64), 0.0)

            tl.store(out_score_ptr + tid, tl.sum(nll_acc, axis=0))

    @triton.jit
    def _irwls_subsets_deriv_generic_kernel(
        pix_ptr,
        types_ptr,
        sol_ptr,
        Y_int_ptr,
        nUMI_ptr,
        P_KT_ptr,
        c1_ptr,
        z0_ptr,
        s6_ptr,
        k_of_l_ptr,
        x_of_l_ptr,
        grad_out_ptr,
        hess_out_ptr,
        M,
        G,
        K_sub,
        BLOCK_K: tl.constexpr,
        n_seg,
        k_size,
        lam_max,
        USE_FP64: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        """Fallback for padded BLOCK_K = next_power_of_2(K_sub) when K_sub > 8 or non-pow2."""
        tid = tl.program_id(0).to(tl.int64)
        if tid >= M:
            return

        val_t = tl.float64 if USE_FP64 else tl.float32
        n = tl.load(pix_ptr + tid).to(tl.int64)
        numi = tl.load(nUMI_ptr + n).to(tl.float64)
        numi_sq = numi * numi
        thresh_v = tl.maximum(numi * 1e-7, 1e-4).to(val_t)
        lam_max_v = tl.full([1], lam_max, dtype=tl.float64).to(val_t)

        k_offs = tl.arange(0, BLOCK_K).to(tl.int64)
        k_mask = k_offs < K_sub
        types = tl.load(types_ptr + tid * K_sub + k_offs, mask=k_mask, other=0).to(tl.int64)
        sol = tl.load(sol_ptr + tid * K_sub + k_offs, mask=k_mask, other=0.0).to(tl.float64)
        s_numi = (sol * numi).to(val_t)

        g_vec = tl.zeros([BLOCK_K], dtype=tl.float64)
        h_mat = tl.zeros([BLOCK_K, BLOCK_K], dtype=tl.float64)
        y_row = Y_int_ptr + n * G

        for g_start in range(0, G, BLOCK_G):
            g_offs = g_start + tl.arange(0, BLOCK_G).to(tl.int64)
            in_bounds = g_offs < G
            p_mask = k_mask[:, None] & in_bounds[None, :]
            p_blk = tl.load(
                P_KT_ptr + types[:, None] * G + g_offs[None, :], mask=p_mask, other=0.0
            ).to(val_t)

            pred = tl.sum(p_blk * s_numi[:, None], axis=0)
            nz_mask = in_bounds & (tl.sum((p_blk != 0.0).to(tl.int32), axis=0) > 0)

            pred = tl.minimum(tl.maximum(tl.abs(pred), thresh_v), lam_max_v)
            r_raw = tl.floor(tl.sqrt(pred.to(tl.float64) / 1e-6)).to(tl.int32)
            r = tl.minimum(tl.maximum(r_raw, 0), k_size - 1).to(tl.int64)
            k = tl.load(k_of_l_ptr + r, mask=nz_mask, other=0).to(tl.int64)
            x_l = tl.load(x_of_l_ptr + r, mask=nz_mask, other=0.0).to(val_t)
            delta = pred - x_l

            y_c = tl.load(y_row + g_offs, mask=nz_mask, other=0).to(tl.int64)
            idx = y_c * n_seg + k
            c1 = tl.load(c1_ptr + idx, mask=nz_mask, other=0.0).to(val_t)
            z0 = tl.load(z0_ptr + idx, mask=nz_mask, other=0.0).to(val_t)
            s6 = tl.load(s6_ptr + idx, mask=nz_mask, other=0.0).to(val_t)

            s6_delta = s6 * delta
            d2 = s6_delta + z0
            d1 = (0.5 * s6_delta + z0) * delta + c1

            nz_f = nz_mask.to(tl.float64)
            d1_f = d1.to(tl.float64) * nz_f
            s_f = (-d2).to(tl.float64) * nz_f
            p_f = p_blk.to(tl.float64)

            g_vec += tl.sum(p_f * d1_f[None, :], axis=1)
            # Manual outer-sum accumulation (BLOCK_K <= 16 avoids tl.dot minimum-dim requirements)
            sp = p_f * s_f[None, :]
            h_mat += tl.sum(sp[:, None, :] * p_f[None, :, :], axis=2)

        grad_out = (-numi) * g_vec
        hess_out = numi_sq * h_mat
        tl.store(grad_out_ptr + tid * K_sub + k_offs, grad_out, mask=k_mask)
        hm = k_mask[:, None] & k_mask[None, :]
        h_offs = tid * K_sub * K_sub + k_offs[:, None] * K_sub + k_offs[None, :]
        tl.store(hess_out_ptr + h_offs, hess_out, mask=hm)

    @triton.jit
    def _score_subsets_nll_kernel(
        pix_ptr,
        types_ptr,
        w_ptr,
        Y_int_ptr,
        nUMI_ptr,
        P_KT_ptr,
        c0_ptr,
        c1_ptr,
        z0_ptr,
        s6_ptr,
        k_of_l_ptr,
        x_of_l_ptr,
        out_score_ptr,
        M,
        G,
        K_sub,
        BLOCK_K: tl.constexpr,
        n_seg,
        k_size,
        lam_max,
        USE_FP64: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        tid = tl.program_id(0).to(tl.int64)
        if tid >= M:
            return

        val_t = tl.float64 if USE_FP64 else tl.float32
        n = tl.load(pix_ptr + tid).to(tl.int64)
        numi = tl.load(nUMI_ptr + n).to(tl.float64)
        lam_max_v = tl.full([BLOCK_G], lam_max, dtype=tl.float64).to(val_t)
        min_eps = tl.full([BLOCK_G], 1e-4, dtype=val_t)

        k_offs = tl.arange(0, BLOCK_K).to(tl.int64)
        k_mask = k_offs < K_sub
        types = tl.load(types_ptr + tid * K_sub + k_offs, mask=k_mask, other=0).to(tl.int64)
        w = tl.load(w_ptr + tid * K_sub + k_offs, mask=k_mask, other=0.0).to(tl.float64)
        w_numi = (w * numi).to(val_t)

        y_row = Y_int_ptr + n * G
        nll_acc = tl.zeros([BLOCK_G], dtype=tl.float64)

        for g_start in range(0, G, BLOCK_G):
            g_offs = g_start + tl.arange(0, BLOCK_G).to(tl.int64)
            in_bounds = g_offs < G
            p_mask = k_mask[:, None] & in_bounds[None, :]
            p_blk = tl.load(
                P_KT_ptr + types[:, None] * G + g_offs[None, :], mask=p_mask, other=0.0
            ).to(val_t)
            pred = tl.sum(p_blk * w_numi[:, None], axis=0)
            pred = tl.minimum(tl.maximum(pred, min_eps), lam_max_v)

            r_raw = tl.floor(tl.sqrt(pred.to(tl.float64) / 1e-6)).to(tl.int32)
            r = tl.minimum(tl.maximum(r_raw, 0), k_size - 1).to(tl.int64)
            k = tl.load(k_of_l_ptr + r, mask=in_bounds, other=0).to(tl.int64)
            x_l = tl.load(x_of_l_ptr + r, mask=in_bounds, other=0.0).to(val_t)
            delta = pred - x_l

            y_c = tl.load(y_row + g_offs, mask=in_bounds, other=0).to(tl.int64)
            idx = y_c * n_seg + k
            c0 = tl.load(c0_ptr + idx, mask=in_bounds, other=0.0).to(val_t)
            c1 = tl.load(c1_ptr + idx, mask=in_bounds, other=0.0).to(val_t)
            z0 = tl.load(z0_ptr + idx, mask=in_bounds, other=0.0).to(val_t)
            s6 = tl.load(s6_ptr + idx, mask=in_bounds, other=0.0).to(val_t)

            s6_delta = s6 * delta
            d0_val = ((s6_delta * (1.0 / 6.0) + 0.5 * z0) * delta + c1) * delta + c0
            nll_acc -= tl.where(in_bounds, d0_val.to(tl.float64), 0.0)

        tl.store(out_score_ptr + tid, tl.sum(nll_acc, axis=0))

    @triton.jit
    def _score_sigma_5_factors_kernel(
        X_ptr,
        Y_int_ptr,
        c0_ptr,
        c1_ptr,
        z0_ptr,
        s6_ptr,
        k_of_l_ptr,
        x_of_l_ptr,
        partial_ptr,
        N,
        n_seg,
        k_size,
        lam_max,
        USE_FP64: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offs < N

        val_t = tl.float64 if USE_FP64 else tl.float32
        x = tl.load(X_ptr + offs, mask=mask, other=1e-4).to(val_t)
        y_c = tl.load(Y_int_ptr + offs, mask=mask, other=0).to(tl.int64)
        lam_max_v = tl.full([BLOCK_SIZE], lam_max, dtype=tl.float64).to(val_t)
        min_eps = tl.full([BLOCK_SIZE], 1e-4, dtype=val_t)

        # Evaluate 5 scale factors (0.8, 0.9, 1.0, 1.1, 1.2) sharing X and y_c in registers.
        for f_idx in range(5):
            fac = 0.8 + 0.1 * f_idx
            p = tl.minimum(tl.maximum(x * fac, min_eps), lam_max_v)
            r_raw = tl.floor(tl.sqrt(p.to(tl.float64) / 1e-6)).to(tl.int32)
            r = tl.minimum(tl.maximum(r_raw, 0), k_size - 1).to(tl.int64)
            k = tl.load(k_of_l_ptr + r, mask=mask, other=0).to(tl.int64)
            x_l = tl.load(x_of_l_ptr + r, mask=mask, other=0.0).to(val_t)
            delta = p - x_l

            idx = y_c * n_seg + k
            c0 = tl.load(c0_ptr + idx, mask=mask, other=0.0).to(val_t)
            c1 = tl.load(c1_ptr + idx, mask=mask, other=0.0).to(val_t)
            z0 = tl.load(z0_ptr + idx, mask=mask, other=0.0).to(val_t)
            s6 = tl.load(s6_ptr + idx, mask=mask, other=0.0).to(val_t)

            s6_delta = s6 * delta
            d0_val = ((s6_delta * (1.0 / 6.0) + 0.5 * z0) * delta + c1) * delta + c0
            nll = -tl.sum(tl.where(mask, d0_val.to(tl.float64), 0.0), axis=0)
            tl.store(partial_ptr + pid * 5 + f_idx, nll)


def _pairs_block_g(G: int) -> int:
    return 128 if G <= 512 else 256


@functools.lru_cache(maxsize=32)
def _tri_unpack_maps(K: int, device_str: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = torch.device(device_str)
    tri_idx = torch.triu_indices(K, K, device=dev)
    p_len = tri_idx.shape[1]
    return (
        torch.arange(p_len, device=dev, dtype=torch.int32),
        tri_idx[0].to(torch.int32).contiguous(),
        tri_idx[1].to(torch.int32).contiguous(),
    )


def irwls_shared_d1_d2_triton(
    Y_int: torch.Tensor,
    pred_raw: torch.Tensor,
    nUMI: torch.Tensor,
    thresh: torch.Tensor,
    poly_gpu: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        float,
    ],
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled spline derivatives ``(d1 * nUMI, -d2 * nUMI^2)`` on GPU in one pass."""
    _, c1, z0, s6, k_of_l, x_of_l, _, lam_max = poly_gpu
    N, G = Y_int.shape
    d1_out = torch.empty((N, G), dtype=out_dtype, device=Y_int.device)
    d2_out = torch.empty((N, G), dtype=out_dtype, device=Y_int.device)
    total = N * G
    if total == 0:
        return d1_out, d2_out

    nUMI_c = nUMI.contiguous()
    thresh_c = thresh.contiguous()
    BLOCK_SIZE = 256
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _irwls_shared_d1_d2_kernel[grid](
        Y_int,
        pred_raw,
        nUMI_c,
        thresh_c,
        c1,
        z0,
        s6,
        k_of_l,
        x_of_l,
        d1_out,
        d2_out,
        N,
        G,
        c1.shape[1],
        k_of_l.shape[0],
        float(lam_max),
        Y_int.stride(0),
        Y_int.stride(1),
        pred_raw.stride(0),
        pred_raw.stride(1),
        d1_out.stride(0),
        d1_out.stride(1),
        d2_out.stride(0),
        d2_out.stride(1),
        USE_FP64=(out_dtype == torch.float64),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return d1_out, d2_out


def unpack_sym_hess_triton(H_tri: torch.Tensor, K: int) -> torch.Tensor:
    """Unpack upper-triangle ``(N, K*(K+1)//2)`` Hessian rows into symmetric ``(N, K, K)``."""
    N = H_tri.shape[0]
    H_out = torch.empty((N, K, K), dtype=H_tri.dtype, device=H_tri.device)
    if N == 0:
        return H_out
    H_tri_c = H_tri.contiguous()
    p_len = K * (K + 1) // 2
    p_map, i_map, j_map = _tri_unpack_maps(K, str(H_tri.device))
    block_p = triton.next_power_of_2(max(p_len, 16))
    if block_p <= 8192:
        _unpack_sym_hess_kernel[(N,)](
            H_tri_c,
            H_out,
            p_map,
            i_map,
            j_map,
            N,
            p_len,
            K,
            BLOCK_P=block_p,
            num_warps=4 if block_p <= 1024 else 8,
        )
    else:
        H_out[:, i_map.long(), j_map.long()] = H_tri_c
        H_out[:, j_map.long(), i_map.long()] = H_tri_c
    return H_out


def solve_box_qp_triton(
    D: torch.Tensor,
    d: torch.Tensor,
    lower_bound: torch.Tensor,
    n_sweeps: int = 50,
    tol: float = 0.0,
) -> torch.Tensor:
    """Run up to ``n_sweeps`` of Gauss-Seidel box-QP coordinate descent in one Triton launch."""
    N, K = d.shape
    x = torch.empty_like(d)
    if N == 0:
        return x
    block_k = triton.next_power_of_2(K)
    _solve_box_qp_kernel[(N,)](
        D.contiguous(),
        d.contiguous(),
        lower_bound.contiguous(),
        x,
        N,
        K=K,
        BLOCK_K=block_k,
        n_sweeps=n_sweeps,
        tol=float(tol),
        USE_FP64=(d.dtype == torch.float64),
        num_warps=1 if block_k <= 128 else 2,
    )
    return x


def fit_and_score_pairs_triton(
    pix_idx: torch.Tensor,
    t1_idx: torch.Tensor,
    t2_idx: torch.Tensor,
    Y_int: torch.Tensor,
    nUMI: torch.Tensor,
    P_KT: torch.Tensor,
    poly_gpu: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        float,
    ],
    max_iter: int = 25,
    compute_score: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused K=2 doublet IRWLS (+ optional NLL scoring) on GPU."""
    c0, c1, z0, s6, k_of_l, x_of_l, _, lam_max = poly_gpu
    M = int(pix_idx.shape[0])
    device = P_KT.device
    w_dtype = P_KT.dtype
    out_w = torch.empty((M, 2), dtype=torch.float64, device=device)
    out_conv = torch.empty((M,), dtype=torch.bool, device=device)
    out_scores = torch.empty((M if compute_score else 0,), dtype=torch.float64, device=device)
    if M == 0:
        return out_w.to(w_dtype), out_scores.to(w_dtype), out_conv

    G = int(P_KT.shape[1])
    pix_c = pix_idx.to(device=device, dtype=torch.int64).contiguous()
    t1_c = t1_idx.to(device=device, dtype=torch.int64).contiguous()
    t2_c = t2_idx.to(device=device, dtype=torch.int64).contiguous()
    nUMI_c = nUMI.contiguous()
    block_g = _pairs_block_g(G)
    use_fp64 = P_KT.dtype == torch.float64

    for start in range(0, M, _MAX_PAIR_SLAB):
        end = min(start + _MAX_PAIR_SLAB, M)
        m_chunk = end - start
        _fit_and_score_pairs_kernel[(m_chunk,)](
            pix_c[start:end],
            t1_c[start:end],
            t2_c[start:end],
            Y_int,
            nUMI_c,
            P_KT,
            c0,
            c1,
            z0,
            s6,
            k_of_l,
            x_of_l,
            out_w[start:end],
            out_conv[start:end],
            out_scores[start:end] if compute_score else out_w,
            m_chunk,
            G,
            c1.shape[1],
            k_of_l.shape[0],
            float(lam_max),
            int(max_iter),
            USE_FP64=use_fp64,
            COMPUTE_SCORE=compute_score,
            BLOCK_G=block_g,
            num_warps=4,
        )

    return out_w.to(w_dtype), out_scores.to(w_dtype), out_conv


def fit_and_score_singles_triton(
    pix_idx: torch.Tensor,
    t_idx: torch.Tensor,
    Y_int: torch.Tensor,
    nUMI: torch.Tensor,
    P_KT: torch.Tensor,
    poly_gpu: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        float,
    ],
    max_iter: int = 25,
    write_weights: bool = False,
    compute_score: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused K=1 singlet IRWLS (+ optional NLL scoring) on GPU."""
    c0, c1, z0, s6, k_of_l, x_of_l, _, lam_max = poly_gpu
    M = int(pix_idx.shape[0])
    device = P_KT.device
    w_dtype = P_KT.dtype
    out_w = torch.empty((M if write_weights else 0,), dtype=torch.float64, device=device)
    out_scores = torch.empty((M if compute_score else 0,), dtype=torch.float64, device=device)
    if M == 0:
        return out_w.to(w_dtype), out_scores.to(w_dtype)

    G = int(P_KT.shape[1])
    pix_c = pix_idx.to(device=device, dtype=torch.int64).contiguous()
    t_c = t_idx.to(device=device, dtype=torch.int64).contiguous()
    nUMI_c = nUMI.contiguous()
    block_g = _pairs_block_g(G)
    use_fp64 = P_KT.dtype == torch.float64

    for start in range(0, M, _MAX_PAIR_SLAB):
        end = min(start + _MAX_PAIR_SLAB, M)
        m_chunk = end - start
        _fit_and_score_singles_kernel[(m_chunk,)](
            pix_c[start:end],
            t_c[start:end],
            Y_int,
            nUMI_c,
            P_KT,
            c0,
            c1,
            z0,
            s6,
            k_of_l,
            x_of_l,
            out_w[start:end] if write_weights else out_scores,
            out_scores[start:end] if compute_score else out_w,
            m_chunk,
            G,
            c1.shape[1],
            k_of_l.shape[0],
            float(lam_max),
            int(max_iter),
            USE_FP64=use_fp64,
            WRITE_WEIGHTS=write_weights,
            COMPUTE_SCORE=compute_score,
            BLOCK_G=block_g,
            num_warps=4,
        )

    return out_w.to(w_dtype), out_scores.to(w_dtype)


def fit_subsets_triton(
    pix_idx: torch.Tensor,
    type_matrix: torch.Tensor,
    Y_int: torch.Tensor,
    nUMI: torch.Tensor,
    P_KT: torch.Tensor,
    poly_gpu: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        float,
    ],
    max_iter: int = 25,
    compute_score: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused K_sub IRWLS derivatives + batched GPU PSD/QP + fused NLL for Multi mode."""
    from rctd._irwls import _psd_batch, _solve_box_qp_batch

    c0, c1, z0, s6, k_of_l, x_of_l, _, lam_max = poly_gpu
    M, K_sub = type_matrix.shape
    device = P_KT.device
    w_dtype = P_KT.dtype
    w = torch.full((M, K_sub), 1.0 / K_sub, dtype=w_dtype, device=device)
    if M == 0:
        return w, torch.empty((0,), dtype=w_dtype, device=device)

    G = int(P_KT.shape[1])
    pix_c = pix_idx.to(device=device, dtype=torch.int64).contiguous()
    types_c = type_matrix.to(device=device, dtype=torch.int64).contiguous()
    nUMI_c = nUMI.contiguous()
    block_g = 128 if G <= 512 else 256
    block_k = max(2, triton.next_power_of_2(K_sub))
    use_fp64 = w_dtype == torch.float64
    eye_K = torch.eye(K_sub, dtype=w_dtype, device=device)

    act_idx = torch.arange(M, device=device, dtype=torch.int64)
    act_pix = pix_c
    act_types = types_c
    w_act = w.clone()

    for _ in range(max_iter):
        n_act = int(act_idx.shape[0])
        if n_act == 0:
            break
        solution = torch.clamp(w_act, min=0.0)
        grad = torch.empty((n_act, K_sub), dtype=torch.float64, device=device)
        hess = torch.empty((n_act, K_sub, K_sub), dtype=torch.float64, device=device)

        _irwls_subsets_deriv_generic_kernel[(n_act,)](
            act_pix,
            act_types,
            solution,
            Y_int,
            nUMI_c,
            P_KT,
            c1,
            z0,
            s6,
            k_of_l,
            x_of_l,
            grad,
            hess,
            n_act,
            G,
            K_sub,
            BLOCK_K=block_k,
            n_seg=c1.shape[1],
            k_size=k_of_l.shape[0],
            lam_max=float(lam_max),
            USE_FP64=use_fp64,
            BLOCK_G=block_g,
            num_warps=4,
        )

        hess_t = hess.to(w_dtype)
        grad_t = grad.to(w_dtype)
        hess_t, norm_factor = _psd_batch(hess_t)
        norm_factor = torch.clamp(norm_factor, min=1e-10)
        D = hess_t / norm_factor[:, None, None] + 1e-7 * eye_K.unsqueeze(0)
        d = -grad_t / norm_factor[:, None]
        delta_w = _solve_box_qp_batch(D, d, -solution)
        w_new = solution + 0.3 * delta_w

        change = torch.sum(torch.abs(w_new - w_act), dim=1)
        newly_conv = change <= 0.001
        w[act_idx] = w_new
        if newly_conv.all():
            break
        if newly_conv.any():
            still = ~newly_conv
            act_idx = act_idx[still]
            act_pix = act_pix[still]
            act_types = act_types[still]
            w_act = w_new[still]
        else:
            w_act = w_new

    scores = torch.empty((M if compute_score else 0,), dtype=torch.float64, device=device)
    if compute_score:
        _score_subsets_nll_kernel[(M,)](
            pix_c,
            types_c,
            w,
            Y_int,
            nUMI_c,
            P_KT,
            c0,
            c1,
            z0,
            s6,
            k_of_l,
            x_of_l,
            scores,
            M,
            G,
            K_sub,
            BLOCK_K=block_k,
            n_seg=c1.shape[1],
            k_size=k_of_l.shape[0],
            lam_max=float(lam_max),
            USE_FP64=use_fp64,
            BLOCK_G=block_g,
            num_warps=4,
        )
    return w, scores.to(w_dtype)


def score_sigma_5_factors_triton(
    X: torch.Tensor,
    Y_int: torch.Tensor,
    poly_gpu: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        float,
    ],
) -> float:
    """Evaluate min NLL across factors (0.8, 0.9, 1.0, 1.1, 1.2) on GPU in one pass."""
    c0, c1, z0, s6, k_of_l, x_of_l, _, lam_max = poly_gpu
    X_c = X.contiguous().reshape(-1)
    Y_c = Y_int.contiguous().reshape(-1)
    N = int(X_c.numel())
    if N == 0:
        return 0.0

    BLOCK_SIZE = 256
    n_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    partial = torch.empty((n_blocks, 5), dtype=torch.float64, device=X_c.device)
    _score_sigma_5_factors_kernel[(n_blocks,)](
        X_c,
        Y_c,
        c0,
        c1,
        z0,
        s6,
        k_of_l,
        x_of_l,
        partial,
        N,
        c1.shape[1],
        k_of_l.shape[0],
        float(lam_max),
        USE_FP64=(X_c.dtype == torch.float64),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return float(partial.sum(dim=0).min().item())
