"""Correctness and memory regression tests for full-mode batch transfers."""

import numpy as np
import pytest
import torch

from rctd import _full, _rctd
from rctd._irwls import solve_irwls_batch_shared
from rctd._likelihood import (
    _calc_q_all_impl,
    compute_spline_coefficients,
    load_cached_q_matrices,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_stream_gpu_matches_resident_path(monkeypatch, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    # Stub ONLY the IRWLS solver to isolate the transfer/batching behavior.
    def fake_solver(*, P, Y_batch, nUMI_batch, **_):
        k = P.shape[1]
        return (
            Y_batch[:, :k] / nUMI_batch[:, None],
            torch.ones(Y_batch.shape[0], device=Y_batch.device, dtype=torch.bool),
        )

    monkeypatch.setattr(_full, "solve_irwls_batch_shared", fake_solver)
    rng = np.random.default_rng(123)
    n, g, k = 19, 13, 3
    Y = rng.integers(1, 9, size=(n, g)).astype(dtype)
    u = Y.sum(axis=1)
    P = rng.random((g, k)).astype(dtype)
    args = dict(
        spatial_counts=Y,
        spatial_numi=u,
        norm_profiles=P,
        cell_type_names=["A", "B", "C"],
        q_mat=np.zeros((3, 4), dtype=np.float64),
        sq_mat=np.zeros((3, 4), dtype=np.float64),
        x_vals=np.arange(4, dtype=dtype),
        batch_size=7,
        device=device,
    )
    baseline = _full.run_full_mode(**args, stream_gpu=False)
    streamed = _full.run_full_mode(**args, stream_gpu=True)
    np.testing.assert_array_equal(streamed.weights, baseline.weights)
    np.testing.assert_array_equal(streamed.converged, baseline.converged)
    assert streamed.weights.shape == (n, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_stream_gpu_avoids_full_counts_allocation(monkeypatch):
    def fake_solver(*, P, Y_batch, nUMI_batch, **_):
        k = P.shape[1]
        return (
            torch.zeros((Y_batch.shape[0], k), device=Y_batch.device, dtype=P.dtype),
            torch.ones(Y_batch.shape[0], device=Y_batch.device, dtype=torch.bool),
        )

    monkeypatch.setattr(_full, "solve_irwls_batch_shared", fake_solver)
    n, g, k = 8192, 512, 3
    Y = np.ones((n, g), dtype=np.float64)
    u = np.full(n, g, dtype=np.float64)
    P = np.full((g, k), 1.0 / g, dtype=np.float64)
    args = dict(
        spatial_counts=Y,
        spatial_numi=u,
        norm_profiles=P,
        cell_type_names=["A", "B", "C"],
        q_mat=np.zeros((3, 4)),
        sq_mat=np.zeros((3, 4)),
        x_vals=np.arange(4, dtype=np.float64),
        batch_size=256,
        device="cuda",
    )
    peaks = {}
    for stream in [False, True]:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        result = _full.run_full_mode(**args, stream_gpu=stream)
        torch.cuda.synchronize()
        peaks[stream] = torch.cuda.max_memory_allocated()
        del result
    # 8192 x 512 x 8 bytes = 32 MiB; streaming input is only 1 MiB.
    assert peaks[False] - peaks[True] > 16 * 1024 * 1024, peaks


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_stream_gpu_matches_resident_with_irwls(monkeypatch, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    def eager_solver(**kwargs):
        return solve_irwls_batch_shared(**kwargs, _calc_q_fn=_calc_q_all_impl)

    monkeypatch.setattr(_full, "solve_irwls_batch_shared", eager_solver)
    cache = load_cached_q_matrices()
    x_vals = cache["X_vals"]
    q_mat = cache["Q_100"]
    sq_mat = compute_spline_coefficients(q_mat, x_vals)
    rng = np.random.default_rng(123)
    n, g, k = 19, 13, 3
    counts = rng.integers(1, 9, size=(n, g)).astype(dtype)
    profiles = rng.random((g, k)).astype(dtype)
    profiles /= profiles.sum(axis=0, keepdims=True)
    args = dict(
        spatial_counts=counts,
        spatial_numi=counts.sum(axis=1),
        norm_profiles=profiles,
        cell_type_names=["A", "B", "C"],
        q_mat=q_mat.astype(dtype),
        sq_mat=sq_mat.astype(dtype),
        x_vals=x_vals.astype(dtype),
        batch_size=7,
        device=device,
        pixel_mask=np.ones(n, dtype=bool),
    )
    original_counts = counts.copy()
    original_profiles = profiles.copy()
    resident = _full.run_full_mode(**args)
    streamed = _full.run_full_mode(**args, stream_gpu=True)
    np.testing.assert_array_equal(streamed.weights, resident.weights)
    np.testing.assert_array_equal(streamed.converged, resident.converged)
    np.testing.assert_array_equal(streamed.pixel_mask, resident.pixel_mask)
    np.testing.assert_array_equal(counts, original_counts)
    np.testing.assert_array_equal(profiles, original_profiles)


@pytest.mark.parametrize("stream_gpu", [False, True])
def test_run_rctd_forwards_streaming_without_changing_auto_batch(monkeypatch, stream_gpu):
    captured = {}

    class FakeRCTD:
        def __init__(self, *_):
            self.config = type("Config", (), {"dtype": "float64", "device": "cpu"})()
            self.reference = type("Reference", (), {"cell_type_names": ["A", "B"]})()
            self.counts = np.ones((5, 3))
            self.nUMI = np.ones(5)
            self.norm_profiles = np.ones((3, 2))
            self.q_mat = np.ones((4, 5))
            self.sq_mat = np.ones((4, 5))
            self.x_vals = np.arange(5.0)
            self._pixel_mask = np.ones(5, dtype=bool)

        def fit_platform_effects(self, sigma_override=None):
            return None

    def fake_auto_batch_size(G, K, dtype_bytes):
        captured["auto_args"] = (G, K, dtype_bytes)
        return 77

    def fake_full(**kwargs):
        captured.update(kwargs)
        return "result"

    monkeypatch.setattr(_rctd, "RCTD", FakeRCTD)
    monkeypatch.setattr(_rctd, "auto_batch_size", fake_auto_batch_size)
    monkeypatch.setattr(_rctd, "run_full_mode", fake_full)
    result = _rctd.run_rctd(None, None, mode="full", batch_size="auto", stream_gpu=stream_gpu)
    assert result == "result"
    assert captured["auto_args"] == (3, 2, 8)
    assert captured["batch_size"] == 77
    assert captured["stream_gpu"] is stream_gpu


@pytest.mark.parametrize("mode", ["doublet", "multi"])
def test_run_rctd_rejects_streaming_in_other_modes(mode):
    with pytest.raises(ValueError, match="stream_gpu currently supports only mode='full'"):
        _rctd.run_rctd(None, None, mode=mode, stream_gpu=True)
