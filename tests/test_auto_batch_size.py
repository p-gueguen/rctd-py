"""auto_batch_size must size batches per mode: doublet and multi batch rows carry their own (G, k)
design matrix, and every mode but streamed full mode keeps the dense (N, G) counts on the GPU.
Before the fix, doublet and multi mode on a 44 GB L40S (39k Visium HD cells, G=4,747, K=40,
float64) got 200,000 rows per batch and ran out of memory.
"""

import numpy as np
import pytest
import torch

import rctd._irwls as irwls
from rctd._types import _batch_bytes, _eigh_terms, _row_values, auto_batch_size

GIB = 2**30


def _fake_gpu(monkeypatch, free_bytes, eigh_threshold=16):
    # the eigh threshold is pinned so results do not depend on the host GPU or on test order
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (free_bytes, free_bytes))
    monkeypatch.setattr(irwls, "_cuda_eigh_threshold", lambda dev: eigh_threshold)


def test_issue49_doublet_no_longer_hits_cap(monkeypatch):
    _fake_gpu(monkeypatch, 44 * GIB)
    bs = auto_batch_size(4747, 40, dtype_bytes=8, mode="doublet", n_resident=38990)
    # measured cost on an L40S: ~21.5 x G values per row
    assert bs * 21.5 * 4747 * 8 + 38990 * 4747 * 8 < 44 * GIB
    assert bs < 50_000


def test_resident_counts_shrink_the_budget(monkeypatch):
    _fake_gpu(monkeypatch, 44 * GIB)
    G = 4638
    without = auto_batch_size(G, 40, dtype_bytes=8, mode="doublet")
    with_res = auto_batch_size(G, 40, dtype_bytes=8, mode="doublet", n_resident=557_300)
    resident = 557_300 * G * 8  # ~19 GiB of counts
    assert with_res < without
    assert with_res * _row_values("doublet", G, 40, 4) * 8 + resident < 44 * GIB


def test_resident_bytes_overrides_dtype_bytes(monkeypatch):
    """Triton int16 resident counts (2 B/elem) allow larger batches than float64 (8 B/elem)."""
    _fake_gpu(monkeypatch, 12 * GIB)
    G, K, n_res = 4000, 30, 200_000
    bs_f64 = auto_batch_size(G, K, dtype_bytes=8, mode="doublet", n_resident=n_res)
    bs_i16 = auto_batch_size(
        G, K, dtype_bytes=8, mode="doublet", n_resident=n_res, resident_bytes=2
    )
    assert bs_i16 > bs_f64
    resident = n_res * G * 2
    assert bs_i16 * _row_values("doublet", G, K, 4) * 8 + resident < 12 * GIB


def test_multi_cost_scales_with_max_types(monkeypatch):
    _fake_gpu(monkeypatch, 44 * GIB)
    four = auto_batch_size(18000, 40, dtype_bytes=8, mode="multi", max_multi_types=4)
    eight = auto_batch_size(18000, 40, dtype_bytes=8, mode="multi", max_multi_types=8)
    assert eight < four


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown mode"):
        auto_batch_size(4747, 40, mode="doublets")


def test_warns_when_minimum_batch_does_not_fit(monkeypatch):
    _fake_gpu(monkeypatch, 2 * GIB)
    with pytest.warns(UserWarning, match="may run out of memory"):
        bs = auto_batch_size(18000, 40, dtype_bytes=8, mode="doublet")
    assert bs == 1024


def test_nonpositive_shape_returns_default(monkeypatch):
    _fake_gpu(monkeypatch, 44 * GIB)
    assert auto_batch_size(0, 0, mode="doublet") == 10000


def test_p_outer_shrinks_the_budget(monkeypatch):
    # P_outer (G x K^2) of the full-mode fit is batch-independent: ~0.9 GiB at G=5,000, K=150
    _fake_gpu(monkeypatch, 4 * GIB)
    G, K = 5000, 150
    bs = auto_batch_size(G, K, dtype_bytes=8, mode="doublet", n_resident=6000)
    fixed = (6000 * G + G * K * K) * 8
    assert bs * _row_values("doublet", G, K, 4) * 8 + fixed <= 4 * GIB


def test_cpu_falls_back_to_default(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert auto_batch_size(4747, 40, mode="doublet") == 10000


def test_cpu_device_ignores_gpu_memory(monkeypatch):
    _fake_gpu(monkeypatch, 1 * GIB)  # a busy GPU must not matter for a CPU run
    assert auto_batch_size(4747, 40, mode="doublet", n_resident=10**6, device="cpu") == 10000


def test_warns_when_counts_alone_do_not_fit(monkeypatch):
    _fake_gpu(monkeypatch, 10 * GIB)
    with pytest.warns(UserWarning, match="count matrix.* do not fit"):
        auto_batch_size(12000, 40, dtype_bytes=8, mode="doublet", n_resident=557_300)


@pytest.mark.parametrize("mode", ["doublet", "multi"])
@pytest.mark.parametrize("G,K", [(1500, 100), (40_000, 20), (4747, 40)])
def test_first_full_fit_is_covered(mode, G, K):
    # doublet and multi mode start with a full-mode fit at the same batch size
    assert _row_values(mode, G, K, 4) >= _row_values("full", G, K, 4)


def _thr(monkeypatch, value):
    monkeypatch.setattr(irwls, "_cuda_eigh_threshold", lambda dev: value)


@pytest.mark.parametrize("dtype_bytes", [4, 8])
def test_gpu_eigh_term_switches_with_threshold(monkeypatch, dtype_bytes):
    # K=38: CPU eigh below the threshold of 16 (cc < 9), GPU eigh at 128 (cc >= 9)
    _fake_gpu(monkeypatch, 44 * GIB)
    _thr(monkeypatch, 16)
    cpu = auto_batch_size(5007, 38, dtype_bytes=dtype_bytes, mode="full", n_resident=11326)
    _thr(monkeypatch, 128)
    gpu = auto_batch_size(5007, 38, dtype_bytes=dtype_bytes, mode="full", n_resident=11326)
    assert gpu < cpu


def test_eigh_terms_cases(monkeypatch):
    _thr(monkeypatch, 128)
    dev = torch.device("cuda")
    assert _eigh_terms(2, 4, dev) == (0, 0)  # K=2: analytic, no cuSOLVER
    row, fixed = _eigh_terms(16, 8, dev)  # float64 K <= 32: chunked at 1,024 matrices
    assert row == 0 and fixed > 0
    row, fixed = _eigh_terms(16, 4, dev)  # float32: never chunked below 25,000 rows
    assert row > 0 and fixed == 0
    _thr(monkeypatch, 16)
    assert _eigh_terms(38, 4, dev) == (0, 0)  # above the threshold: CPU eigh


def test_multi_float64_counts_its_own_chunked_eigh(monkeypatch):
    # K=40 on cc < 9: full fit on CPU eigh; multi's 3- and 4-type fits run chunked GPU eigh
    _thr(monkeypatch, 16)
    dev = torch.device("cuda")
    _, fixed_multi = _batch_bytes("multi", 4747, 40, 4, 8, dev)
    _, fixed_doublet = _batch_bytes("doublet", 4747, 40, 4, 8, dev)
    assert fixed_multi > fixed_doublet == 4747 * 40 * 40 * 8


@pytest.mark.parametrize(
    "G,K,dtype_bytes,measured",
    [(5007, 38, 4, 452_742), (3000, 40, 8, 996_785)],  # PR #50 comment, RTX 5070, cu130
)
def test_blackwell_report_is_covered(monkeypatch, G, K, dtype_bytes, measured):
    _thr(monkeypatch, 128)  # compute capability 12.0
    row, _ = _batch_bytes("full", G, K, 4, dtype_bytes, torch.device("cuda"))
    assert row >= measured


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["full", "doublet", "multi"])
# K=20/40 run CPU eigh on cc < 9 (L40S); K=8 runs GPU eigh there; threshold 128 forces GPU eigh
# as on cc >= 9 (K=38 float32 per row; K=40 float64 per row; K=16 float64 chunked = fixed).
# G=18,000: multi's 4-type stage, where an older multi constant fell short.
@pytest.mark.parametrize(
    "G,K,dtype,eigh_thr",
    [
        (3000, 20, "float32", None),
        (3000, 40, "float64", None),
        (18000, 20, "float32", None),
        (3000, 8, "float32", None),
        (3000, 38, "float32", 128),
        (3000, 40, "float64", 128),
        (3000, 16, "float64", 128),
    ],
)
def test_measured_peak_per_row_within_estimate(monkeypatch, mode, G, K, dtype, eigh_thr):
    """Run each mode at two batch sizes on synthetic data and check the measured peak GPU memory
    per batch row (the slope), and the absolute peak, stay below the estimate auto_batch_size
    uses."""
    if eigh_thr is not None:
        monkeypatch.setattr(irwls, "_EIGH_THRESHOLD_OVERRIDE", eigh_thr)
    from rctd._doublet import run_doublet_mode
    from rctd._full import run_full_mode
    from rctd._likelihood import compute_spline_coefficients, load_cached_q_matrices
    from rctd._multi import run_multi_mode
    from rctd._types import RCTDConfig

    N = 4000
    npdt = np.float32 if dtype == "float32" else np.float64
    rng = np.random.default_rng(0)
    prof = rng.gamma(0.3, 1.0, size=(G, K))
    prof /= prof.sum(0, keepdims=True)
    W = np.zeros((N, K))
    for i in range(N):  # 4-type mixtures: multi mode reaches its 4-type stage with many rows
        W[i, rng.choice(K, 4, replace=False)] = rng.dirichlet([1, 1, 1, 1])
    counts = rng.poisson(2000 * W @ prof.T).astype(npdt)
    cache = load_cached_q_matrices()
    x_vals = cache.pop("X_vals")
    q = cache["Q_100"]
    sq = compute_spline_coefficients(q, x_vals)
    args = dict(
        spatial_counts=counts,
        spatial_numi=counts.sum(1),
        norm_profiles=prof.astype(npdt),
        cell_type_names=[f"t{i}" for i in range(K)],
        q_mat=q.astype(npdt),
        sq_mat=sq.astype(npdt),
        x_vals=x_vals.astype(npdt),
        device="cuda",
    )
    cfg = RCTDConfig(device="cuda", dtype=dtype)

    def peak(bs):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        if mode == "full":
            run_full_mode(**args, batch_size=bs)
        elif mode == "doublet":
            run_doublet_mode(**args, config=cfg, batch_size=bs)
        else:
            run_multi_mode(**args, config=cfg, batch_size=bs)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    db = np.dtype(npdt).itemsize
    p1, p3 = peak(1000), peak(3000)
    per_row = (p3 - p1) / 2000
    est_row, est_fixed = _batch_bytes(mode, G, K, cfg.MAX_MULTI_TYPES, db, torch.device("cuda"))
    assert per_row > 0
    assert per_row <= est_row, f"{mode}: measured {per_row:.0f} B/row > estimate {est_row} B/row"
    # absolute peak: rows + fixed + resident counts, plus 64 MiB for Q/SQ tables and lists
    bound = 3000 * est_row + est_fixed + N * G * db + 64 * 2**20
    assert p3 <= bound, f"{mode}: peak {p3 / 2**30:.2f} GiB > bound {bound / 2**30:.2f} GiB"
