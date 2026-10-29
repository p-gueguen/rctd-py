"""Verify provenance and fail-closed behavior without running large GPU jobs."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def harness(monkeypatch):
    directory = Path(__file__).parents[1] / "benchmarks/regression"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("regression_bench", directory / "bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_snapshot_keeps_dirty_deleted_and_untracked_state(tmp_path, monkeypatch, harness):
    repo = tmp_path / "repo"
    package = repo / "src/rctd"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n")
    (package / "deleted.py").write_text("VALUE = 2\n")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])

    git("init", "-q")
    git("add", ".")
    git(
        "-c",
        "user.name=Harness Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    (package / "__init__.py").write_text("VALUE = 3\n")
    (package / "deleted.py").unlink()
    (package / "untracked.py").write_text("VALUE = 4\n")
    before = git("status", "--porcelain")
    monkeypatch.setattr(harness, "ROOT", repo)
    original = harness.snapshot("HEAD", tmp_path / "baseline")
    current = harness.snapshot("working", tmp_path / "candidate")
    assert original["code_sha256"] != current["code_sha256"]
    assert (tmp_path / "baseline/src/rctd/__init__.py").read_text() == "VALUE = 1\n"
    assert (tmp_path / "candidate/src/rctd/__init__.py").read_text() == "VALUE = 3\n"
    assert (tmp_path / "candidate/src/rctd/untracked.py").exists()
    assert not (tmp_path / "candidate/src/rctd/deleted.py").exists()
    assert git("status", "--porcelain") == before


def test_changed_data_or_case_is_rejected(tmp_path, harness):
    case = tmp_path / "case"
    case.mkdir()
    data = tmp_path / "counts.bin"
    data.write_bytes(b"original counts")
    (case / "settings.json").write_text("{}")
    manifest = {
        "external_files": {str(data): harness.digest(data)},
        "files": harness.file_manifest(case),
    }
    manifest["case_id"] = harness.object_hash(manifest)
    harness.write_json(case / "manifest.json", manifest)
    assert harness.verify_case(case)["case_id"] == manifest["case_id"]
    data.write_bytes(b"changed counts")
    with pytest.raises(ValueError, match="Input data changed"):
        harness.verify_case(case)
    data.write_bytes(b"original counts")
    (case / "settings.json").write_text('{"atol": 1}')
    with pytest.raises(ValueError, match="Frozen case changed"):
        harness.verify_case(case)


def test_timeout_retains_failure_report(tmp_path, harness):
    scripts = tmp_path / "harness"
    scripts.mkdir()
    (scripts / "worker.py").write_text("import time\ntime.sleep(30)\n")
    result = harness.launch(
        tmp_path, tmp_path, tmp_path / "run", scripts, {"seed": 42, "threads": 1, "timeout_s": 0.2}
    )
    assert result["status"] == "timeout"
    assert json.loads((tmp_path / "run/metrics.json").read_text())["status"] == "timeout"
    assert (tmp_path / "run/worker.log").exists()


def test_incomplete_run_cannot_be_compared(tmp_path, harness):
    harness.write_json(tmp_path / "run.json", {"stages": {}})
    with pytest.raises(ValueError, match="input-integrity"):
        harness.load_run(tmp_path)


def test_cpu_normalize_input_shares_frozen_arrays(harness):
    import torch
    from worker import bulk_inputs

    frozen = {
        "cell_type_profiles": np.ones((4, 2), dtype=np.float64),
        "spatial_counts": np.arange(20, dtype=np.float32).reshape(5, 4),
        "spatial_nUMI": np.arange(5, dtype=np.float32),
    }
    inputs = bulk_inputs(frozen, "cpu", cpu_input=True)
    for key in ["spatial_counts", "spatial_nUMI"]:
        assert inputs[key].device.type == "cpu"
        assert inputs[key].data_ptr() == torch.from_numpy(frozen[key]).data_ptr()
        np.testing.assert_array_equal(inputs[key].numpy(), frozen[key])


def _saved_run(path, harness, weight=1.0, environment="same"):
    (path / "source").mkdir(parents=True)
    (path / "source/code.py").write_text("# code snapshot\n")
    (path / "harness").mkdir()
    (path / "harness/worker.py").write_text("# evaluator snapshot\n")
    (path / "full").mkdir()
    np.savez(path / "full/output_000.npz", weights=np.array([[weight]]), spot_ids=np.array(["a"]))
    sample = {
        "warmup": False,
        "elapsed_s": 1.0,
        "cuda_peak_allocated_bytes": 100,
        "cuda_peak_reserved_bytes": 200,
        "rss_sampled_peak_bytes": 300,
        "process_lifetime_peak_rss_bytes": 400,
    }
    info = {
        "case_id": "fixed",
        "settings": {"atol": 0.0, "rtol": 0.0},
        "source": {"files": harness.file_manifest(path / "source")},
        "harness": harness.file_manifest(path / "harness"),
        "input_integrity_verified_after_run": True,
        "artifact_files": {"full/output_000.npz": harness.digest(path / "full/output_000.npz")},
        "stages": {
            "full": {
                "status": "ok",
                "environment": environment,
                "repeatability": [{"pass": True}],
                "measurements": [sample] * 3,
            }
        },
    }
    harness.write_json(path / "run.json", info)


def test_comparison_emits_failure_and_preserves_raw_difference(tmp_path, harness):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    _saved_run(baseline, harness)
    _saved_run(candidate, harness, weight=1.1)
    out = tmp_path / "comparison"
    code = harness.compare(SimpleNamespace(baseline=baseline, candidate=candidate, out=out))
    assert code == 2
    result = harness.read_json(out / "comparison.json")
    assert not result["pass"]
    assert result["stages"]["full"]["comparisons"][0]["arrays"]["weights"]["max_abs"] == (
        pytest.approx(0.1)
    )
    assert "FAIL" in (out / "report.md").read_text()


def test_different_environments_are_not_comparable(tmp_path, harness):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    _saved_run(baseline, harness)
    _saved_run(candidate, harness, environment="different GPU")
    with pytest.raises(ValueError, match="Incomparable environments"):
        harness.compare(
            SimpleNamespace(baseline=baseline, candidate=candidate, out=tmp_path / "comparison")
        )
