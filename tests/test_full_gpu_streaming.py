"""Correctness and memory regression tests for P0.1's batch transfer path."""

import numpy as np
import pytest
import torch

from rctd import _full, _rctd, _types


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stream_gpu_matches_resident_path(monkeypatch, device):
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
    Y = rng.integers(1, 9, size=(n, g)).astype(np.float64)
    u = Y.sum(axis=1)
    P = rng.random((g, k)).astype(np.float64)
    args = dict(
        spatial_counts=Y,
        spatial_numi=u,
        norm_profiles=P,
        cell_type_names=["A", "B", "C"],
        q_mat=np.zeros((3, 4), dtype=np.float64),
        sq_mat=np.zeros((3, 4), dtype=np.float64),
        x_vals=np.arange(4, dtype=np.float64),
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


def test_auto_batch_size_respects_streaming_cap(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (12 * 2**30, 12 * 2**30))

    unbounded = _types.auto_batch_size(G=5007, K=38, dtype_bytes=8)
    bounded = _types.auto_batch_size(
        G=5007,
        K=38,
        dtype_bytes=8,
        max_batch_size=_types.STREAM_GPU_AUTO_BATCH_MAX,
    )

    assert unbounded > _types.STREAM_GPU_AUTO_BATCH_MAX
    assert bounded == _types.STREAM_GPU_AUTO_BATCH_MAX


def test_run_rctd_uses_bounded_auto_batch_for_streaming(monkeypatch):
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

    def fake_auto_batch_size(*_, **kwargs):
        captured["max_batch_size"] = kwargs["max_batch_size"]
        return 77

    def fake_full(**kwargs):
        captured.update(kwargs)
        return "result"

    monkeypatch.setattr(_rctd, "RCTD", FakeRCTD)
    monkeypatch.setattr(_rctd, "auto_batch_size", fake_auto_batch_size)
    monkeypatch.setattr(_rctd, "run_full_mode", fake_full)

    result = _rctd.run_rctd(None, None, mode="full", batch_size="auto", stream_gpu=True)

    assert result == "result"
    assert captured["max_batch_size"] == _types.STREAM_GPU_AUTO_BATCH_MAX
    assert captured["batch_size"] == 77
    assert captured["stream_gpu"] is True
