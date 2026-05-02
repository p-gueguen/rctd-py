"""Tests for torch.compile fallback behavior."""

import warnings

import numpy as np
import pytest
import torch

from rctd import _irwls, _likelihood


def _make_small_qp_problem(N=10, K=3, device="cpu"):
    """Create a small box-QP problem for testing."""
    # Random PSD matrices
    A = torch.randn(N, K, K, device=device, dtype=torch.float64)
    D = A @ A.transpose(-1, -2) + 0.1 * torch.eye(K, device=device, dtype=torch.float64)
    d = torch.randn(N, K, device=device, dtype=torch.float64)
    lower_bound = -torch.ones(N, K, device=device, dtype=torch.float64)
    return D, d, lower_bound


class TestBoxQPFallback:
    """Tests for _solve_box_qp_batch compile fallback."""

    def test_eager_matches_impl(self):
        """Eager implementation produces correct results."""
        D, d, lb = _make_small_qp_problem()
        result = _irwls._solve_box_qp_batch_impl(D, d, lb)
        assert result.shape == (10, 3)
        assert torch.isfinite(result).all()

    def test_compile_false_uses_jit(self, monkeypatch):
        """When _USE_COMPILE is False, dispatch uses the TorchScript JIT path
        (not eager Python). The new contract from v0.3.2 — JIT and eager agree
        within IRWLS tolerance but differ at exact-equality (early-exit order).
        See tests/test_blackwell_perf.py for the full dispatch contract."""
        monkeypatch.setattr(_irwls, "_USE_COMPILE", False)
        D, d, lb = _make_small_qp_problem()
        result = _irwls._solve_box_qp_batch(D, d, lb)
        expected = _irwls._solve_box_qp_batch_adaptive_jit(D, d, lb)
        torch.testing.assert_close(result, expected)

    def test_fallback_on_runtime_error(self, monkeypatch):
        """Auto-fallback to eager when compiled version raises RuntimeError."""
        monkeypatch.setattr(_irwls, "_USE_COMPILE", None)

        def raise_on_call(*args, **kwargs):
            raise RuntimeError("Simulated torch.compile failure")

        monkeypatch.setattr(_irwls, "_solve_box_qp_batch_compiled", raise_on_call)

        D, d, lb = _make_small_qp_problem()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _irwls._solve_box_qp_batch(D, d, lb)
            assert len(w) == 1
            assert "torch.compile failed" in str(w[0].message)

        # Verify result is correct (matches eager)
        expected = _irwls._solve_box_qp_batch_impl(D, d, lb)
        torch.testing.assert_close(result, expected)

        # Subsequent calls should use eager without warning
        assert _irwls._USE_COMPILE is False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _irwls._solve_box_qp_batch(D, d, lb)
            assert len(w) == 0

    def test_k2_bypasses_compile(self):
        """K=2 uses analytical solver, not compiled path."""
        D, d, lb = _make_small_qp_problem(K=2)
        result = _irwls._solve_box_qp_batch(D, d, lb)
        assert result.shape == (10, 2)


class TestCalcQAllFallback:
    """Tests for calc_q_all compile fallback."""

    @pytest.fixture
    def q_inputs(self):
        """Create minimal inputs for calc_q_all."""
        from rctd._likelihood import load_cached_q_matrices

        cache = load_cached_q_matrices()
        x_vals = cache.pop("X_vals")
        # Use first Q matrix
        key = next(k for k in cache if k.startswith("Q_"))
        Q_mat = cache[key]
        SQ_mat = Q_mat**2  # Simplified
        Q_t = torch.tensor(Q_mat, dtype=torch.float64)
        SQ_t = torch.tensor(SQ_mat, dtype=torch.float64)
        x_t = torch.tensor(x_vals, dtype=torch.float64)

        Y = torch.randint(0, 50, (100,), dtype=torch.float64)
        lam = torch.rand(100, dtype=torch.float64) * 10 + 1

        return Y, lam, Q_t, SQ_t, x_t

    def test_compile_false_uses_eager(self, monkeypatch, q_inputs):
        """When _CALC_Q_USE_COMPILE is False, eager path is used."""
        monkeypatch.setattr(_likelihood, "_CALC_Q_USE_COMPILE", False)
        Y, lam, Q, SQ, x = q_inputs
        result = _likelihood.calc_q_all(Y, lam, Q, SQ, x)
        expected = _likelihood.calc_q_all_eager(Y, lam, Q, SQ, x)
        for r, e in zip(result, expected):
            torch.testing.assert_close(r, e)

    def test_fallback_on_runtime_error(self, monkeypatch, q_inputs):
        """Auto-fallback to eager when compiled raises RuntimeError."""
        monkeypatch.setattr(_likelihood, "_CALC_Q_USE_COMPILE", None)

        def raise_on_call(*args, **kwargs):
            raise RuntimeError("Simulated torch.compile failure")

        monkeypatch.setattr(_likelihood, "_calc_q_all_compiled", raise_on_call)

        Y, lam, Q, SQ, x = q_inputs
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _likelihood.calc_q_all(Y, lam, Q, SQ, x)
            assert len(w) == 1
            assert "torch.compile failed" in str(w[0].message)

        expected = _likelihood.calc_q_all_eager(Y, lam, Q, SQ, x)
        for r, e in zip(result, expected):
            torch.testing.assert_close(r, e)

        assert _likelihood._CALC_Q_USE_COMPILE is False


class TestConfigCompileFlag:
    """Test that RCTDConfig(compile=False) disables compilation."""

    def test_config_compile_default_true(self):
        from rctd._types import RCTDConfig

        assert RCTDConfig().compile is True

    def test_config_compile_false(self):
        from rctd._types import RCTDConfig

        assert RCTDConfig(compile=False).compile is False

    def test_rctd_init_disables_compile(self, monkeypatch):
        """RCTD(config=RCTDConfig(compile=False)) sets module globals."""
        import anndata

        from rctd._rctd import RCTD
        from rctd._types import RCTDConfig

        # Reset globals
        monkeypatch.setattr(_irwls, "_USE_COMPILE", None)
        monkeypatch.setattr(_likelihood, "_CALC_Q_USE_COMPILE", None)

        # Create minimal data
        n_cells, n_genes = 50, 100
        ref_adata = anndata.AnnData(
            X=np.random.poisson(5, (n_cells, n_genes)).astype(np.float32),
            var={"_index": [f"Gene{i}" for i in range(n_genes)]},
        )
        ref_adata.var_names = [f"Gene{i}" for i in range(n_genes)]
        ref_adata.obs["cell_type"] = np.random.choice(["A", "B", "C"], n_cells)

        from rctd._reference import Reference

        ref = Reference(ref_adata, cell_min=2)

        spatial = anndata.AnnData(
            X=np.random.poisson(10, (30, n_genes)).astype(np.float32),
        )
        spatial.var_names = [f"Gene{i}" for i in range(n_genes)]

        config = RCTDConfig(compile=False, UMI_min=1)
        RCTD(spatial, ref, config)

        assert _irwls._USE_COMPILE is False
        assert _likelihood._CALC_Q_USE_COMPILE is False
