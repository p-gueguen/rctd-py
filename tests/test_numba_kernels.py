"""The numba CPU kernels against the torch code they replace (``RCTD_NUMBA=0``)."""

import numpy as np
import pytest
import torch

from rctd import _irwls, _likelihood, _numba_kernels
from rctd._doublet import run_doublet_mode
from rctd._likelihood import compute_spline_coefficients, load_cached_q_matrices
from rctd._multi import run_multi_mode
from rctd._types import RCTDConfig

pytestmark = pytest.mark.skipif(not _numba_kernels.ENABLED, reason="numba kernels disabled")


@pytest.fixture(scope="module")
def tables():
    cache = load_cached_q_matrices()
    x_vals = cache.pop("X_vals")
    q_mat = cache["Q_100"]
    return q_mat, compute_spline_coefficients(q_mat, x_vals), x_vals


def _t(tables, dtype):
    return [torch.tensor(a, dtype=dtype) for a in tables]


@pytest.fixture
def torch_path(monkeypatch):
    """Call to switch the CPU back to the torch code, as RCTD_NUMBA=0 does."""
    monkeypatch.setattr(_irwls, "_USE_COMPILE", False)
    monkeypatch.setattr(_likelihood, "_CALC_Q_USE_COMPILE", False)
    return lambda enabled=False: monkeypatch.setattr(_numba_kernels, "ENABLED", enabled)


def _likelihood_inputs(dtype, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    Y = rng.poisson(rng.gamma(0.5, 4, n)).clip(0, 1500)  # beyond K_val = 1000 too
    lam = np.exp(rng.uniform(np.log(1e-6), np.log(3000), n))
    # the clamps' edges, grid points and a lambda past the last grid point
    lam[:6] = [0.0, 1e-4, 1.2e-4, 1.0, 1600.0, 5e3]
    return torch.tensor(Y, dtype=dtype), torch.tensor(lam, dtype=dtype)


@pytest.mark.parametrize("K_val", [-1, 200])
def test_calc_q_all_bitwise_in_float64(tables, K_val):
    Y, lam = _likelihood_inputs(torch.float64)
    Q, SQ, x = _t(tables, torch.float64)
    expected = _likelihood._calc_q_all_impl(Y, lam, Q, SQ, x, K_val)
    got = _likelihood.calc_q_all_eager(Y, lam, Q, SQ, x, K_val)
    for e, g in zip(expected, got):
        assert torch.equal(e, g)


def test_calc_q_all_float32_is_float64_rounded(tables):
    """In float32 the kernel computes in float64 on the same inputs and rounds once at the end
    (torch's float32 path rounds after every op)."""
    Y, lam = _likelihood_inputs(torch.float32)
    tables32 = _t(tables, torch.float32)
    expected = _likelihood._calc_q_all_impl(
        Y.double(), lam.double(), *(t.double() for t in tables32)
    )
    got = _likelihood.calc_q_all_eager(Y, lam, *tables32)
    for e, g in zip(expected, got):
        assert g.dtype == torch.float32
        assert torch.equal(g, e.float())


def _subset_problem(dtype, K, N=400, G=150, K_total=8, seed=1):
    rng = np.random.default_rng(seed)
    P = rng.gamma(0.3, 1.0, (G, K_total))
    P /= P.sum(0)
    types = np.stack([rng.choice(K_total, K, replace=False) for _ in range(N)])
    numi = rng.uniform(100, 3000, N)
    Y = rng.poisson(numi[:, None] * rng.dirichlet(np.full(K_total, 0.3), N) @ P.T)
    S = numi[:, None, None] * P[:, types].transpose(1, 0, 2)
    return [torch.tensor(a, dtype=dtype) for a in (S, Y, numi)]


def test_irwls_derivatives_match_torch(tables, torch_path):
    S, Y, numi = _subset_problem(torch.float64, K=3)
    Q, SQ, x = _t(tables, torch.float64)
    w = torch.rand(S.shape[0], 3, dtype=torch.float64)
    threshold = torch.clamp(numi * 1e-7, min=1e-4)
    active = torch.rand(S.shape[0]) < 0.7
    grad, hess = _numba_kernels.irwls_derivatives(S, Y, w, threshold, active, Q, SQ, x)
    pred = torch.clamp(torch.abs(torch.bmm(S, w.unsqueeze(2)).squeeze(2)), min=threshold[:, None])
    torch_path()
    g_ref, h_ref = _irwls._get_derivatives_batch(S, Y, pred, Q, SQ, x)
    torch.testing.assert_close(grad[active], g_ref[active], rtol=1e-12, atol=1e-9)
    torch.testing.assert_close(hess[active], h_ref[active], rtol=1e-12, atol=1e-9)
    assert not grad[~active].any() and not hess[~active].any()


def _fit(tables, dtype, K):
    S, Y, numi = _subset_problem(dtype, K)
    Q, SQ, x = _t(tables, dtype)
    return _irwls.solve_irwls_batch(
        S, Y, numi, Q_mat=Q, SQ_mat=SQ, x_vals=x, max_iter=25, constrain=False
    )


@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_solve_irwls_batch_matches_torch(tables, torch_path, K):
    w, conv = _fit(tables, torch.float64, K)
    torch_path()
    w_ref, conv_ref = _fit(tables, torch.float64, K)
    # numba runs all 50 QP sweeps, the TorchScript QP stops per batch; both finish to KKT 1e-10
    torch.testing.assert_close(w, w_ref, rtol=0, atol=1e-12 if K <= 2 else 1e-7)
    assert (conv == conv_ref).float().mean() > 0.99


@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_solve_irwls_batch_float32(tables, torch_path, K):
    """float32 fits are no further from the float64 fit than torch's float32 ones.

    Not pixel for pixel against torch's: a fit stops once its step moves the weights by less than
    min_change (1e-3), which can leave a pixel ~1e-3 from the optimum, and which pixels depends on
    rounding (it differs between machines, and between the two paths)."""
    w64, _ = _fit(tables, torch.float64, K)
    w32, _ = _fit(tables, torch.float32, K)
    torch_path()
    w32_ref, _ = _fit(tables, torch.float32, K)
    err, err_ref = ((w.double() - w64).abs() for w in (w32, w32_ref))
    assert err.max() <= max(err_ref.max(), 1e-3)  # 1e-3: solve_irwls_batch's min_change
    assert err.quantile(0.99) <= 2 * err_ref.quantile(0.99)


def _mode_inputs(synthetic_data, tables):
    q_mat, sq_mat, x_vals = tables
    X = synthetic_data["spatial"].X
    counts = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float64)
    return dict(
        spatial_counts=counts,
        spatial_numi=counts.sum(1),
        norm_profiles=synthetic_data["profiles"],
        cell_type_names=synthetic_data["cell_type_names"],
        q_mat=q_mat,
        sq_mat=sq_mat,
        x_vals=x_vals,
        batch_size=37,
        device="cpu",
    )


@pytest.mark.parametrize("run", [run_doublet_mode, run_multi_mode])
def test_modes_match_torch(synthetic_data, tables, torch_path, run):
    """Doublet and multi mode make the same calls, with the same weights, on either path."""
    inputs = _mode_inputs(synthetic_data, tables)
    config = RCTDConfig(compile=False)
    new = run(config=config, **inputs)
    torch_path()
    ref = run(config=config, **inputs)
    for name, a in ref._asdict().items():
        if not isinstance(a, np.ndarray):
            continue
        b = getattr(new, name)
        if a.dtype.kind == "f":
            np.testing.assert_allclose(b, a, rtol=0, atol=1e-9, err_msg=name)
        else:
            np.testing.assert_array_equal(b, a, err_msg=name)


def test_gpu_tensors_take_the_torch_path():
    assert _numba_kernels.use_numba(torch.zeros(1))
    assert not _numba_kernels.use_numba(torch.zeros(1, device="meta"))


def test_negative_counts_raise(tables):
    """A negative count would index outside the likelihood tables."""
    q_mat, sq_mat, x_vals = _t(tables, torch.float64)
    Y = torch.tensor([3.0, -1.0, 0.0])
    with pytest.raises(ValueError, match="non-negative"):
        _numba_kernels.calc_q_all(Y, torch.ones(3), q_mat, sq_mat, x_vals)
    S = torch.ones(1, 3, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="non-negative"):
        _numba_kernels.irwls_derivatives(
            S,
            Y[None],
            torch.ones(1, 2),
            torch.full((1,), 1e-4),
            torch.ones(1, dtype=bool),
            q_mat,
            sq_mat,
            x_vals,
        )


def test_threads_follow_torch(monkeypatch):
    """The pool takes torch's thread count, capped by NUMBA_NUM_THREADS (all CPUs by default)."""
    monkeypatch.setattr(_numba_kernels.numba.config, "NUMBA_NUM_THREADS", 64)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    assert _numba_kernels.num_threads() == 8
    monkeypatch.setattr(_numba_kernels.numba.config, "NUMBA_NUM_THREADS", 4)
    assert _numba_kernels.num_threads() == 4


def _qp_problems(K, n=500, seed=2):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, K, K, generator=g, dtype=torch.float64)
    D = A @ A.mT + 1e-3 * torch.eye(K, dtype=torch.float64)
    D = D / torch.linalg.matrix_norm(D, ord=2)[:, None, None]  # as solve_irwls normalises it
    return D, torch.randn(n, K, generator=g, dtype=torch.float64), -torch.rand(n, K, generator=g)


@pytest.mark.parametrize("K", [1, 3, 4, 16, 31])
def test_box_qp_matches_torch(torch_path, K):
    """Per-pixel Gauss-Seidel stops, then the exact finish: the same optimum as the torch QP."""
    D, d, lb = _qp_problems(K)
    lb = lb.double()
    x = _irwls._solve_box_qp_batch(D, d, lb)
    torch_path()
    x_ref = _irwls._solve_box_qp_batch(D, d, lb)
    torch.testing.assert_close(x, x_ref, rtol=0, atol=1e-6)
    # KKT: free coordinates have zero gradient, bound ones a non-negative one
    g = torch.bmm(D, x.unsqueeze(2)).squeeze(2) - d
    viol = torch.where(x > lb, g.abs(), torch.clamp(-g, min=0.0)).amax(1)
    assert viol.max() < 1e-8


def test_box_qp_does_not_depend_on_the_batch():
    D, d, lb = _qp_problems(5, n=300)
    lb = lb.double()
    whole = _numba_kernels.box_qp_sweeps(D, d, lb)
    parts = torch.cat(
        [
            _numba_kernels.box_qp_sweeps(D[i : i + 7], d[i : i + 7], lb[i : i + 7])
            for i in range(0, 300, 7)
        ]
    )
    assert torch.equal(whole, parts)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("threads", [3, 8, 28])
@pytest.mark.parametrize("K", [3, 5, 31])
def test_psd_batch_threads_give_identical_results(monkeypatch, K, threads, dtype):
    """The CPU eigendecompositions run in chunks across threads; the result is bitwise that of one
    call over the batch, which RCTD_NUMBA=0 makes (n=3000 on 8 threads gave 375-row chunks, whose
    odd starts changed MKL's results for odd K)."""
    D, _, _ = _qp_problems(K, n=3000)
    H = (D - 0.3 * torch.eye(K, dtype=torch.float64)).to(dtype)  # negative eigenvalues to clamp
    H[5] = float("nan")  # the non-finite guard still applies
    monkeypatch.setattr(_numba_kernels, "num_threads", lambda: threads)
    prev = torch.get_num_threads()
    H_psd, max_eig = _irwls._psd_batch(H)
    assert torch.get_num_threads() == prev
    monkeypatch.setattr(_numba_kernels, "ENABLED", False)
    H_ref, max_ref = _irwls._psd_batch(H)
    assert torch.equal(H_psd, H_ref) and torch.equal(max_eig, max_ref)


def test_psd_batch_workers_run_eigh_single_threaded(monkeypatch):
    """A pool thread keeps the torch thread count it had when it first ran torch code, so one that
    ran torch at 8 threads would otherwise run every later eigh chunk on 8 threads."""
    prev = torch.get_num_threads()
    torch.set_num_threads(8)
    try:
        monkeypatch.setattr(_numba_kernels, "num_threads", lambda: 8)

        def touch(start, end):
            torch.ones(1024, 1024, dtype=torch.float64).sum()  # large enough to run threaded

        _numba_kernels._pool_threads = None  # fresh threads, which first run torch at 8 threads
        _numba_kernels.parallel_rows(touch, 8 * 64, 64, threads=8)
        seen = []
        eigh = _irwls._eigh_or_jitter

        def spy(H, epsilon):
            seen.append(torch.get_num_threads())
            return eigh(H, epsilon)

        monkeypatch.setattr(_irwls, "_eigh_or_jitter", spy)
        g = torch.Generator().manual_seed(3)
        S = torch.rand(3000, 60, 31, generator=g, dtype=torch.float64)
        H = S.mT @ S
        H_psd, max_eig = _irwls._psd_batch(H)
        assert set(seen) == {1}
        monkeypatch.setattr(_numba_kernels, "ENABLED", False)
        H_ref, max_ref = _irwls._psd_batch(H)
        assert torch.equal(H_psd, H_ref) and torch.equal(max_eig, max_ref)
    finally:
        torch.set_num_threads(prev)


def test_psd_batch_without_numba_skips_the_pool(monkeypatch):
    """RCTD_NUMBA=0, a missing numba and the GPU's CPU eigh all make one eigh call, as main."""
    monkeypatch.setattr(_numba_kernels, "ENABLED", False)

    def fail(*args, **kwargs):
        raise AssertionError("thread pool used")

    monkeypatch.setattr(_numba_kernels, "parallel_rows", fail)
    D, _, _ = _qp_problems(5, n=300)
    _irwls._psd_batch(D)


def test_box_qp_zero_diagonal_matches_torch(torch_path):
    """A zero diagonal divides by zero: inf/NaN as in torch, not ZeroDivisionError."""
    D, d, lb = _qp_problems(4, n=20)
    lb = lb.double()
    D[3, 1, 1] = 0.0
    x = _numba_kernels.box_qp_sweeps(D, d, lb)
    ref = _irwls._solve_box_qp_batch_impl(D, d, lb)
    torch.testing.assert_close(x, ref, equal_nan=True, rtol=0, atol=1e-12)
    assert not torch.isfinite(x[3]).all()
