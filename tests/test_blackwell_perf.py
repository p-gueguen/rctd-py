"""Regression tests for the Blackwell perf fixes (v0.3.2).

The two fixes wired through ``_solve_box_qp_batch`` and ``_psd_batch`` change
which code path executes at K>16, but the *numerical output* must remain
equivalent (within IRWLS tolerance) to the previous eager / CPU-offload paths.
These tests pin that equivalence and verify the arch-gated rollout behaves
correctly across CPU, Ampere, Hopper, and Blackwell.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from rctd._irwls import (
    _cuda_eigh_threshold,
    _psd_batch,
    _solve_box_qp_batch,
    _solve_box_qp_batch_adaptive_jit,
    _solve_box_qp_batch_impl,
)


def _make_psd_batch(N: int, K: int, seed: int = 0, ill_conditioned: bool = False) -> torch.Tensor:
    """Random PSD batch: (A @ A.T) + eps*I. Deterministic per seed."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(N, K, K, generator=g, dtype=torch.float64)
    H = A @ A.transpose(-1, -2) + 1e-3 * torch.eye(K, dtype=torch.float64)
    if ill_conditioned:
        # Inject heavy condition-number variance: scale eigenvectors of last
        # half of batch by 1e6 to mimic real IRWLS Hessians from sparse data.
        scale = torch.ones(N, K, dtype=torch.float64)
        scale[N // 2 :, : K // 2] = 1e3
        H = scale.unsqueeze(-1) * H * scale.unsqueeze(-2)
    return H


# ─── Arch-detection unit tests ──────────────────────────────────────────


def test_cuda_eigh_threshold_cpu_returns_zero():
    """CPU device → threshold 0 (no CUDA dispatch path possible)."""
    assert _cuda_eigh_threshold(torch.device("cpu")) == 0


@pytest.mark.parametrize(
    "capability,expected",
    [
        ((7, 0), 16),  # Volta
        ((7, 5), 16),  # Turing
        ((8, 0), 16),  # Ampere A100
        ((8, 6), 16),  # Ampere RTX 30xx
        ((8, 9), 16),  # Ada / L40S — historical benchmark target
        ((9, 0), 128),  # Hopper H100
        ((10, 0), 128),  # Blackwell B200
        ((12, 0), 128),  # Blackwell RTX PRO 6000 (tjli's hardware)
    ],
)
def test_cuda_eigh_threshold_arch_gating(capability, expected):
    """Threshold must be 128 only for sm_90+; older arches keep K<=16
    (where original L40S/Ampere benchmarks showed CPU OpenBLAS won)."""
    fake_cuda = torch.device("cuda")
    with patch("torch.cuda.get_device_capability", return_value=capability):
        assert _cuda_eigh_threshold(fake_cuda) == expected


def test_cuda_eigh_threshold_falls_back_safely_on_capability_error():
    """If CUDA introspection raises (driver mismatch, etc.), default to the
    conservative K<=16 threshold rather than crashing IRWLS."""
    fake_cuda = torch.device("cuda")
    with patch("torch.cuda.get_device_capability", side_effect=RuntimeError("nope")):
        assert _cuda_eigh_threshold(fake_cuda) == 16


# ─── _psd_batch CPU path preservation ───────────────────────────────────


def test_psd_batch_cpu_path_unchanged():
    """On CPU, _psd_batch must still go through the CPU LAPACK eigh path
    regardless of K. Behavior on CPU-only deployments must not regress."""
    K = 78
    N = 32
    H = _make_psd_batch(N, K, seed=11)
    H_psd, max_eig = _psd_batch(H)

    # Reference: explicit CPU eigh
    eigenvalues, eigenvectors = torch.linalg.eigh(H)
    eigenvalues = torch.clamp(eigenvalues, min=1e-3)
    H_ref = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)
    eig_ref = eigenvalues[:, -1]

    np.testing.assert_allclose(H_psd.numpy(), H_ref.numpy(), atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(max_eig.numpy(), eig_ref.numpy(), atol=1e-9, rtol=1e-9)


# ─── Box-QP equivalence: eager vs JIT ───────────────────────────────────


@pytest.mark.parametrize("K", [3, 8, 16, 32, 78, 100])
def test_solve_box_qp_jit_matches_eager_across_K(K):
    """Eager Python loop and JIT-script loop must agree across K within
    IRWLS-meaningful tolerance. The JIT path has batch-level early exit
    (eager runs all 50 sweeps); both should converge to the same answer."""
    N = 32
    D = _make_psd_batch(N, K, seed=42)
    d = torch.randn(N, K, generator=torch.Generator().manual_seed(43), dtype=torch.float64)
    lb = -torch.ones(N, K, dtype=torch.float64)

    x_eager = _solve_box_qp_batch_impl(D, d, lb, n_sweeps=50)
    x_jit = _solve_box_qp_batch_adaptive_jit(D, d, lb, n_sweeps=50)

    # IRWLS converges to within min_change=0.001 in production; 1e-4 is
    # tighter than that and well within the noise floor of the outer loop.
    np.testing.assert_allclose(x_eager.numpy(), x_jit.numpy(), atol=1e-4, rtol=1e-4)


def test_solve_box_qp_jit_matches_eager_ill_conditioned():
    """Stress test: ill-conditioned D matrices (cond ~ 1e6, mimics real IRWLS
    Hessians on sparse data). JIT and eager must agree even when convergence
    is slow and the early exit fires at very different sweep counts."""
    K = 45
    N = 64
    D = _make_psd_batch(N, K, seed=999, ill_conditioned=True)
    d = torch.randn(N, K, generator=torch.Generator().manual_seed(998), dtype=torch.float64)
    lb = -torch.ones(N, K, dtype=torch.float64)

    x_eager = _solve_box_qp_batch_impl(D, d, lb, n_sweeps=50)
    x_jit = _solve_box_qp_batch_adaptive_jit(D, d, lb, n_sweeps=50)

    # Slightly looser tolerance for ill-conditioned: convergence-noise dominated
    np.testing.assert_allclose(x_eager.numpy(), x_jit.numpy(), atol=1e-3, rtol=1e-3)


def test_solve_box_qp_jit_matches_eager_at_active_lower_bound():
    """When the lower-bound constraint is actually binding for many coords,
    the clamp logic must produce identical results in both paths."""
    K = 32
    N = 16
    D = _make_psd_batch(N, K, seed=111)
    # Bias d toward negative so optimum is below lb = 0
    d = torch.randn(N, K, generator=torch.Generator().manual_seed(112), dtype=torch.float64) - 2.0
    lb = torch.zeros(N, K, dtype=torch.float64)

    x_eager = _solve_box_qp_batch_impl(D, d, lb, n_sweeps=50)
    x_jit = _solve_box_qp_batch_adaptive_jit(D, d, lb, n_sweeps=50)

    np.testing.assert_allclose(x_eager.numpy(), x_jit.numpy(), atol=1e-4, rtol=1e-4)
    # Sanity: many coords should be at the bound
    assert (x_jit == 0).any()


def test_solve_box_qp_dispatch_uses_jit_when_compile_disabled():
    """With _USE_COMPILE=False, _solve_box_qp_batch must return values matching
    the JIT path (post-fix), not the previous eager implementation. This pins
    the dispatch change."""
    import rctd._irwls as _irwls

    K = 32  # any K>2 exercises the dispatch logic
    N = 16
    D = _make_psd_batch(N, K, seed=1)
    d = torch.randn(N, K, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    lb = -torch.ones(N, K, dtype=torch.float64)

    saved = _irwls._USE_COMPILE
    try:
        _irwls._USE_COMPILE = False
        x_dispatch = _solve_box_qp_batch(D, d, lb, n_sweeps=50)
        x_jit = _solve_box_qp_batch_adaptive_jit(D, d, lb, n_sweeps=50)
        np.testing.assert_allclose(x_dispatch.numpy(), x_jit.numpy(), atol=1e-6)
    finally:
        _irwls._USE_COMPILE = saved


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_psd_batch_gpu_path_at_k78_matches_cpu_offload_within_tol():
    """On a CUDA-capable GPU, _psd_batch at K=78 should now stay on GPU
    (when arch >= sm_90) and produce values within float-eigh tolerance of the
    CPU offload path. We verify this by forcing both paths and comparing."""
    device = torch.device("cuda")
    K = 78
    N = 32
    H_gpu = _make_psd_batch(N, K, seed=7).to(device=device, dtype=torch.float64)
    H_cpu_copy = H_gpu.cpu().clone()

    H_psd_via_dispatch, eig_via_dispatch = _psd_batch(H_gpu)

    # Reference: explicit CPU offload (the v0.3.0 large-K path)
    eigenvalues, eigenvectors = torch.linalg.eigh(H_cpu_copy)
    eigenvalues = torch.clamp(eigenvalues, min=1e-3)
    H_ref = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)
    eig_ref = eigenvalues[:, -1]

    # GPU eigh and CPU eigh agree to ~1e-6 in float64; allow some headroom
    np.testing.assert_allclose(
        H_psd_via_dispatch.cpu().numpy(), H_ref.numpy(), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        eig_via_dispatch.cpu().numpy(), eig_ref.numpy(), atol=1e-5, rtol=1e-5
    )


# ─── CPU perf neutrality ─────────────────────────────────────────────────


def test_solve_box_qp_jit_no_cpu_perf_regression():
    """JIT path on CPU must not be dramatically slower than the eager path.
    Users without GPUs (or with `compile=False`) on older PyTorch versions
    must not see a regression. Allow up to 4x slower (TorchScript dispatch
    overhead is real but bounded) — anything worse signals a problem."""
    import time

    K = 32
    N = 100
    D = _make_psd_batch(N, K, seed=2025)
    d = torch.randn(N, K, generator=torch.Generator().manual_seed(2026), dtype=torch.float64)
    lb = -torch.ones(N, K, dtype=torch.float64)

    # Warmup JIT (first call compiles the script)
    _solve_box_qp_batch_adaptive_jit(D, d, lb, n_sweeps=50)

    n_runs = 3
    t_eager = float("inf")
    t_jit = float("inf")
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _solve_box_qp_batch_impl(D, d, lb, n_sweeps=50)
        t_eager = min(t_eager, time.perf_counter() - t0)
        t0 = time.perf_counter()
        _solve_box_qp_batch_adaptive_jit(D, d, lb, n_sweeps=50)
        t_jit = min(t_jit, time.perf_counter() - t0)

    # On CPU TorchScript may be slightly slower (dispatch overhead) or faster
    # (early exit). 4x ceiling is generous — typical observation is <1.5x.
    assert t_jit < 4.0 * t_eager, (
        f"JIT path 4x+ slower than eager on CPU: jit={t_jit:.3f}s eager={t_eager:.3f}s"
    )


# ─── End-to-end regression: full IRWLS pipeline ──────────────────────────


def test_full_mode_results_unchanged(synthetic_data):
    """Full-mode RCTD on the synthetic fixture must produce weights stable
    against the dispatch change (within IRWLS convergence tolerance).

    Compares two configurations on the same data:
      A) compile=False (now uses JIT box-QP per the v0.3.2 dispatch fix)
      B) explicitly-eager box-QP (forcibly re-dispatches to the old eager
         implementation by monkey-patching the dispatch function)
    Both must agree within tolerance.
    """
    from rctd import RCTDConfig, Reference, run_rctd
    from rctd._irwls import _solve_box_qp_batch_impl

    spatial = synthetic_data["spatial"]
    # Build a Reference adata from raw counts so RCTD wires it correctly
    ref_adata = synthetic_data["reference"]
    ref = Reference(ref_adata, cell_type_col="cell_type")

    config = RCTDConfig(UMI_min=10, device="cpu", dtype="float64", compile=False)

    # Run A: post-fix dispatch (JIT path)
    res_a = run_rctd(spatial, ref, mode="full", config=config, batch_size=50)

    # Run B: forcibly use eager Python path
    import rctd._irwls as _irwls

    saved = _irwls._solve_box_qp_batch_adaptive_jit
    try:
        # Monkey-patch so dispatch reaches eager when compile=False
        _irwls._solve_box_qp_batch_adaptive_jit = _solve_box_qp_batch_impl
        res_b = run_rctd(spatial, ref, mode="full", config=config, batch_size=50)
    finally:
        _irwls._solve_box_qp_batch_adaptive_jit = saved

    # Both runs must produce numerically equivalent weights
    np.testing.assert_allclose(res_a.weights, res_b.weights, atol=1e-3, rtol=1e-3)
    np.testing.assert_array_equal(res_a.cell_type_names, res_b.cell_type_names)


def test_doublet_mode_results_unchanged(synthetic_data):
    """Doublet mode classification (spot_class, first_type, second_type) must
    be stable against the dispatch change. Weights match within IRWLS tol."""
    from rctd import RCTDConfig, Reference, run_rctd
    from rctd._irwls import _solve_box_qp_batch_impl

    spatial = synthetic_data["spatial"]
    ref = Reference(synthetic_data["reference"], cell_type_col="cell_type")

    config = RCTDConfig(UMI_min=10, device="cpu", dtype="float64", compile=False)

    res_a = run_rctd(spatial, ref, mode="doublet", config=config, batch_size=10)

    import rctd._irwls as _irwls

    saved = _irwls._solve_box_qp_batch_adaptive_jit
    try:
        _irwls._solve_box_qp_batch_adaptive_jit = _solve_box_qp_batch_impl
        res_b = run_rctd(spatial, ref, mode="doublet", config=config, batch_size=10)
    finally:
        _irwls._solve_box_qp_batch_adaptive_jit = saved

    # Hard discrete labels must match exactly
    np.testing.assert_array_equal(res_a.spot_class, res_b.spot_class)
    np.testing.assert_array_equal(res_a.first_type, res_b.first_type)
    np.testing.assert_array_equal(res_a.second_type, res_b.second_type)
    # Continuous weights match within IRWLS tol
    np.testing.assert_allclose(res_a.weights_doublet, res_b.weights_doublet, atol=1e-3, rtol=1e-3)
