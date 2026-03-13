"""Performance regression tests for runtime and memory usage.

These tests are intentionally opt-in because they are heavier and can be noisy on
shared CI runners. Run them with:

    pytest -m performance tests/test_performance.py

Each execution appends measured runtime/memory metrics to a CSV file.
Default path: tests/performance_metrics.csv
Override path with: RCTD_PERF_CSV=/path/to/metrics.csv
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import platform
import resource
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import anndata
import numpy as np
import pytest
import torch
from scipy import sparse

from rctd._rctd import run_rctd
from rctd._reference import Reference
from rctd._types import RCTDConfig


_PERF_CSV_PATH = Path(
    os.environ.get(
        "RCTD_PERF_CSV",
        str(Path(__file__).resolve().parent / "performance_metrics.csv"),
    )
)
_RUN_ID = os.environ.get("RCTD_PERF_RUN_ID", uuid.uuid4().hex)
_FIXED_BATCH_SIZE = 10000
_PERF_FIELDNAMES = [
    "run_id",
    "timestamp_utc",
    "status",
    "error",
    "test_name",
    "dtype",
    "device",
    "batch_size",
    "n_genes",
    "n_cells",
    "n_types",
    "n_pixels",
    "elapsed_s",
    "maxrss_kb",
    "maxrss_mb",
    "cuda_peak_bytes",
    "cuda_peak_mb",
    "python",
    "torch",
    "platform",
]


def _ensure_perf_csv_schema() -> None:
    """Ensure output CSV has the expected header; rotate legacy schema if needed."""
    if not _PERF_CSV_PATH.exists() or _PERF_CSV_PATH.stat().st_size == 0:
        return

    with _PERF_CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    if header == _PERF_FIELDNAMES:
        return

    legacy_path = _PERF_CSV_PATH.with_name(
        f"{_PERF_CSV_PATH.stem}.legacy.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{_PERF_CSV_PATH.suffix}"
    )
    _PERF_CSV_PATH.rename(legacy_path)


def _append_metric_row(
    test_name: str,
    dtype: str,
    device: str,
    stats: dict | None = None,
    status: str = "ok",
    error: str = "",
) -> None:
    """Append one performance measurement row to a CSV file."""
    _PERF_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ensure_perf_csv_schema()
    stats = stats or {}

    row = {
        "run_id": _RUN_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "error": error,
        "test_name": test_name,
        "dtype": dtype,
        "device": device,
        "batch_size": int(stats.get("batch_size", 0)),
        "n_genes": int(stats.get("n_genes", 0)),
        "n_cells": int(stats.get("n_cells", 0)),
        "n_types": int(stats.get("n_types", 0)),
        "elapsed_s": f"{float(stats.get('elapsed_s', 0.0)):.6f}",
        "maxrss_kb": int(stats.get("maxrss_kb", 0)),
        "maxrss_mb": f"{float(stats.get('maxrss_kb', 0)) / 1024.0:.3f}",
        "cuda_peak_bytes": int(stats.get("cuda_peak_bytes", 0)),
        "cuda_peak_mb": f"{float(stats.get('cuda_peak_bytes', 0)) / (1024.0 * 1024.0):.3f}",
        "n_pixels": int(stats.get("n_pixels", 0)),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }

    fieldnames = _PERF_FIELDNAMES
    write_header = not _PERF_CSV_PATH.exists() or _PERF_CSV_PATH.stat().st_size == 0

    with _PERF_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _make_reference_and_spatial(
    n_genes: int = 500,
    n_cells: int = 50000,
    n_types: int = 12,
    n_pixels: int = 50000,
    seed: int = 1234,
    sparse_inputs: bool = False,
):
    """Create deterministic synthetic reference + spatial data for perf tests."""
    rng = np.random.default_rng(seed)

    # Reference profiles: each type has a stronger marker block.
    profiles = rng.exponential(0.0015, size=(n_genes, n_types)).astype(np.float64)
    block = max(1, n_genes // n_types)
    for k in range(n_types):
        start = k * block
        stop = min(n_genes, start + block)
        profiles[start:stop, k] *= 8.0
    profiles /= profiles.sum(axis=0, keepdims=True)

    base_cells_per_type, remainder = divmod(n_cells, n_types)
    ref_counts = np.zeros((n_cells, n_genes), dtype=np.float32)
    cell_types = []
    row = 0
    for k in range(n_types):
        n_cells_k = base_cells_per_type + (1 if k < remainder else 0)
        for _ in range(n_cells_k):
            n_umi = int(rng.integers(1200, 5000))
            lam = profiles[:, k] * n_umi
            ref_counts[row, :] = rng.poisson(lam).astype(np.float32)
            cell_types.append(f"Type_{k}")
            row += 1

    ref_x = sparse.csr_matrix(ref_counts) if sparse_inputs else ref_counts
    ref = anndata.AnnData(X=ref_x, obs={"cell_type": cell_types})
    ref.var_names = [f"Gene_{i}" for i in range(n_genes)]

    weights = rng.dirichlet(alpha=np.ones(n_types) * 0.7, size=n_pixels)
    spatial_counts = np.zeros((n_pixels, n_genes), dtype=np.float32)
    for i in range(n_pixels):
        n_umi = int(rng.integers(800, 3500))
        lam = (profiles @ weights[i]) * n_umi
        spatial_counts[i, :] = rng.poisson(lam).astype(np.float32)

    spatial_x = sparse.csr_matrix(spatial_counts) if sparse_inputs else spatial_counts
    spatial = anndata.AnnData(
        X=spatial_x,
        obs={
            "x": rng.uniform(0.0, 100.0, size=n_pixels),
            "y": rng.uniform(0.0, 100.0, size=n_pixels),
        },
    )
    spatial.var_names = ref.var_names.copy()

    return ref, spatial


_GPU_DATASETS = [
    (
        "genes_500",
        {
            "n_genes": 500,
            "n_cells": 50000,
            "n_types": 12,
            "n_pixels": 50000,
            "seed": 2234,
        },
    ),
    (
        "genes_5000",
        {
            "n_genes": 5000,
            "n_cells": 50000,
            "n_types": 12,
            "n_pixels": 50000,
            "seed": 3234,
        },
    ),
]

_TARGETED_DATASETS = {
    "sparse_full": {
        "n_genes": 5000,
        "n_cells": 50000,
        "n_types": 12,
        "n_pixels": 50000,
        "seed": 4234,
    },
    "sigma_calibration": {
        "n_genes": 1200,
        "n_cells": 50000,
        "n_types": 12,
        "n_pixels": 50000,
        "seed": 5234,
    },
    "multi_mode": {
        "n_genes": 1000,
        "n_cells": 50000,
        "n_types": 12,
        "n_pixels": 50000,
        "seed": 6234,
    },
    "doublet_mode": {
        "n_genes": 1000,
        "n_cells": 50000,
        "n_types": 12,
        "n_pixels": 50000,
        "seed": 7234,
    },
}


def _child_run(
    dtype: str,
    device: str,
    queue,
    dataset_kwargs: dict | None = None,
    batch_size: int = 128,
    mode: str = "full",
    sigma_override: int | None = 100,
    sparse_inputs: bool = False,
    config_overrides: dict | None = None,
):
    """Run RCTD in a subprocess and return timing + memory stats."""
    try:
        torch.set_num_threads(1)
        if device == "cuda":
            if not torch.cuda.is_available():
                queue.put({"error": "CUDA requested but unavailable"})
                return
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        dataset_kwargs = dataset_kwargs or {}
        data_kwargs = {
            "n_genes": int(dataset_kwargs.get("n_genes", 700)),
            "n_cells": int(dataset_kwargs.get("n_cells", 50000)),
            "n_types": int(dataset_kwargs.get("n_types", 7)),
            "n_pixels": int(dataset_kwargs.get("n_pixels", 50000)),
            "seed": int(dataset_kwargs.get("seed", 1234)),
        }
        ref_adata, spatial_adata = _make_reference_and_spatial(
            **data_kwargs,
            sparse_inputs=bool(sparse_inputs),
        )
        reference = Reference(ref_adata, cell_min=10, min_UMI=10)

        cfg = {
            "dtype": dtype,
            "device": device,
            "N_fit": 60,
            "N_epoch": 4,
            "UMI_min": 20,
            "UMI_min_sigma": 50,
        }
        if config_overrides:
            cfg.update(config_overrides)
        config = RCTDConfig(**cfg)

        t0 = time.perf_counter()
        res = run_rctd(
            spatial=spatial_adata,
            reference=reference,
            mode=mode,
            config=config,
            batch_size=batch_size,
            sigma_override=sigma_override,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        cuda_peak = 0
        if device == "cuda":
            torch.cuda.synchronize()
            cuda_peak = int(torch.cuda.max_memory_allocated())

        queue.put(
            {
                "elapsed_s": float(elapsed),
                "maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "cuda_peak_bytes": cuda_peak,
                "n_pixels": int(res.weights.shape[0]),
                "batch_size": int(batch_size),
                "n_genes": int(data_kwargs["n_genes"]),
                "n_cells": int(data_kwargs["n_cells"]),
                "n_types": int(data_kwargs["n_types"]),
            }
        )
    except Exception:
        queue.put({"error": traceback.format_exc()})


def _run_case(
    test_name: str,
    dtype: str,
    device: str,
    timeout_s: int = 240,
    dataset_kwargs: dict | None = None,
    batch_size: int | None = None,
    mode: str = "full",
    sigma_override: int | None = 100,
    sparse_inputs: bool = False,
    config_overrides: dict | None = None,
):
    """Execute one perf case in an isolated process and append a CSV row."""
    dataset_kwargs = dataset_kwargs or {}
    if batch_size is None:
        batch_size = _FIXED_BATCH_SIZE

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_child_run,
        args=(
            dtype,
            device,
            queue,
            dataset_kwargs,
            batch_size,
            mode,
            sigma_override,
            sparse_inputs,
            config_overrides,
        ),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.kill()
        proc.join()
        _append_metric_row(
            test_name,
            dtype,
            device,
            stats={
                "batch_size": batch_size,
                "n_genes": dataset_kwargs.get("n_genes", 0),
                "n_cells": dataset_kwargs.get("n_cells", 0),
                "n_types": dataset_kwargs.get("n_types", 0),
                "n_pixels": dataset_kwargs.get("n_pixels", 0),
            },
            status="timeout",
            error=f"timed out after {timeout_s}s",
        )
        pytest.fail(f"Performance case timed out for dtype={dtype}, device={device}")

    if queue.empty():
        _append_metric_row(
            test_name,
            dtype,
            device,
            stats={
                "batch_size": batch_size,
                "n_genes": dataset_kwargs.get("n_genes", 0),
                "n_cells": dataset_kwargs.get("n_cells", 0),
                "n_types": dataset_kwargs.get("n_types", 0),
                "n_pixels": dataset_kwargs.get("n_pixels", 0),
            },
            status="error",
            error=f"no result returned; exitcode={proc.exitcode}",
        )
        pytest.fail(
            f"No performance result returned for dtype={dtype}, device={device}; "
            f"exitcode={proc.exitcode}"
        )

    out = queue.get()
    if "error" in out:
        _append_metric_row(
            test_name,
            dtype,
            device,
            stats={
                "batch_size": batch_size,
                "n_genes": dataset_kwargs.get("n_genes", 0),
                "n_cells": dataset_kwargs.get("n_cells", 0),
                "n_types": dataset_kwargs.get("n_types", 0),
                "n_pixels": dataset_kwargs.get("n_pixels", 0),
            },
            status="error",
            error=str(out["error"]),
        )
        pytest.fail(out["error"])

    _append_metric_row(test_name, dtype, device, out)

    return out


@pytest.mark.performance
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "dataset_name,dataset_kwargs",
    _GPU_DATASETS,
    ids=[name for name, _ in _GPU_DATASETS],
)
def test_full_mode_gpu_float32_fixed_batch(dataset_name: str, dataset_kwargs: dict):
    """Run full-mode RCTD on CUDA only, with float32 and fixed batch size."""
    stats = _run_case(
        test_name=f"test_full_mode_gpu_float32_fixed_batch[{dataset_name}]",
        dtype="float32",
        device="cuda",
        timeout_s=1200,
        dataset_kwargs=dataset_kwargs,
        batch_size=_FIXED_BATCH_SIZE,
    )

    assert stats["n_pixels"] == dataset_kwargs["n_pixels"]
    assert stats["elapsed_s"] > 0.0
    assert stats["cuda_peak_bytes"] > 0


@pytest.mark.performance
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_full_mode_gpu_float32_sparse_inputs_fixed_batch():
    """Run full mode with sparse reference/spatial inputs to stress sparse memory path."""
    dataset_kwargs = _TARGETED_DATASETS["sparse_full"]
    stats = _run_case(
        test_name="test_full_mode_gpu_float32_sparse_inputs_fixed_batch",
        dtype="float32",
        device="cuda",
        timeout_s=1200,
        dataset_kwargs=dataset_kwargs,
        batch_size=_FIXED_BATCH_SIZE,
        mode="full",
        sigma_override=100,
        sparse_inputs=True,
    )

    assert stats["n_pixels"] == dataset_kwargs["n_pixels"]
    assert stats["elapsed_s"] > 0.0
    assert stats["cuda_peak_bytes"] > 0


@pytest.mark.performance
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_full_mode_gpu_float32_sigma_calibration():
    """Run full mode without sigma override to benchmark choose_sigma calibration."""
    dataset_kwargs = _TARGETED_DATASETS["sigma_calibration"]
    stats = _run_case(
        test_name="test_full_mode_gpu_float32_sigma_calibration",
        dtype="float32",
        device="cuda",
        timeout_s=1800,
        dataset_kwargs=dataset_kwargs,
        batch_size=_FIXED_BATCH_SIZE,
        mode="full",
        sigma_override=None,
        config_overrides={"N_fit": 80, "N_epoch": 6},
    )

    assert stats["n_pixels"] == dataset_kwargs["n_pixels"]
    assert stats["elapsed_s"] > 0.0
    assert stats["cuda_peak_bytes"] > 0


@pytest.mark.performance
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multi_mode_gpu_float32_fixed_batch():
    """Run multi mode to benchmark batched multi scoring path."""
    dataset_kwargs = _TARGETED_DATASETS["multi_mode"]
    stats = _run_case(
        test_name="test_multi_mode_gpu_float32_fixed_batch",
        dtype="float32",
        device="cuda",
        timeout_s=1800,
        dataset_kwargs=dataset_kwargs,
        batch_size=_FIXED_BATCH_SIZE,
        mode="multi",
        sigma_override=100,
    )

    assert stats["n_pixels"] == dataset_kwargs["n_pixels"]
    assert stats["elapsed_s"] > 0.0
    assert stats["cuda_peak_bytes"] > 0


@pytest.mark.performance
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_doublet_mode_gpu_float32_fixed_batch():
    """Run doublet mode to benchmark pair bookkeeping and scoring path."""
    dataset_kwargs = _TARGETED_DATASETS["doublet_mode"]
    stats = _run_case(
        test_name="test_doublet_mode_gpu_float32_fixed_batch",
        dtype="float32",
        device="cuda",
        timeout_s=1800,
        dataset_kwargs=dataset_kwargs,
        batch_size=_FIXED_BATCH_SIZE,
        mode="doublet",
        sigma_override=100,
    )

    assert stats["n_pixels"] == dataset_kwargs["n_pixels"]
    assert stats["elapsed_s"] > 0.0
    assert stats["cuda_peak_bytes"] > 0
