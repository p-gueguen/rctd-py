#!/usr/bin/env python3
"""Freeze a case, benchmark a source snapshot, and compare saved runs (Linux/WSL)."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from metrics import compare_outputs, timing_comparison

ROOT = Path(__file__).resolve().parents[2]
HARNESS = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_manifest(directory):
    return {
        str(p.relative_to(directory)): digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def snapshot(source, destination):
    destination.mkdir(parents=True)
    working = source == "working"
    commit = git("rev-parse", "HEAD" if working else source + "^{commit}").decode().strip()
    if working:
        names = git("ls-files", "-co", "--exclude-standard", "--", "src/rctd").decode().splitlines()
    else:
        names = git("ls-tree", "-r", "--name-only", commit, "src/rctd").decode().splitlines()
    for name in sorted(set(names)):
        if "/data/" in name or "__pycache__" in name:
            continue
        if working and not (ROOT / name).is_file():
            continue
        content = (ROOT / name).read_bytes() if working else git("show", f"{commit}:{name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    if not (destination / "src/rctd/__init__.py").exists():
        raise ValueError("Source must contain src/rctd")
    files = file_manifest(destination)
    result = {
        "requested_source": source,
        "commit": commit,
        "files": files,
        "code_sha256": object_hash(files),
        "working_tree": working,
    }
    if working:
        result["git_status"] = git("status", "--porcelain").decode()
        (destination.parent / "working.patch").write_bytes(git("diff", "HEAD", "--", "src/rctd"))
    return result


def copy_harness(destination):
    destination.mkdir()
    for p in HARNESS.glob("*.py"):
        shutil.copy2(p, destination / p.name)
    return file_manifest(destination)


def launch(
    case,
    source,
    destination,
    harness,
    settings,
    action="run",
    stage="pipeline",
    stream_gpu=False,
    bulk_input_cpu=False,
):
    destination.mkdir(parents=True)
    command = [
        sys.executable,
        "-B",
        str(harness / "worker.py"),
        "--case",
        str(case),
        "--source",
        str(source),
        "--out",
        str(destination),
        "--action",
        action,
        "--stage",
        stage,
    ]
    if stream_gpu:
        command.append("--stream-gpu")
    if bulk_input_cpu:
        command.append("--bulk-input-cpu")
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED=str(settings["seed"]),
        OMP_NUM_THREADS=str(settings["threads"]),
        MKL_NUM_THREADS=str(settings["threads"]),
    )
    # Freeze compilation cache per run. Disk cache is shared by warm repeats, not by sources.
    env["TORCHINDUCTOR_CACHE_DIR"] = str(destination / "compile_cache")
    env["TRITON_CACHE_DIR"] = str(destination / "triton_cache")
    with (destination / "worker.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            code = process.wait(timeout=settings["timeout_s"])
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = -1
    metrics = destination / "metrics.json"
    result = read_json(metrics) if metrics.exists() else {}
    if code != 0:
        result["status"] = "timeout" if code == -1 else result.get("status", "error")
        if result["status"] in ("running", "ok"):
            result["status"] = "error"
        result["exit_code"] = code
        write_json(metrics, result)
    print(f"{stage}: {result.get('status', 'error')} ({destination / 'worker.log'})", flush=True)
    return result


def configure(path):
    settings = {
        "seed": 2025,
        "max_spots": None,
        "batch_size": 256,
        "threads": 4,
        "warmup": 1,
        "repeats": 5,
        "timeout_s": 3600,
        "sigma_override": None,
        "atol": 0.0,
        "rtol": 0.0,
        "reference_options": {"cell_type_col": "cell_type"},
        "rctd_config": {"device": "cuda", "dtype": "float64", "compile": False},
        "annotation_index": "spot_id",
        "mode": "full",
    }
    settings.update(read_json(path))
    if settings["mode"] not in ("full", "doublet", "multi"):
        raise ValueError("mode must be full, doublet or multi")
    if settings["warmup"] < 1 or settings["repeats"] < 1:
        raise ValueError("At least one warmup and one measured repeat are required")
    for k in ["batch_size", "threads", "timeout_s"]:
        if settings[k] <= 0:
            raise ValueError(f"{k} must be positive")
    if any(not np.isfinite(settings[k]) or settings[k] < 0 for k in ["atol", "rtol"]):
        raise ValueError("Tolerances must be finite and nonnegative")
    # No automatic download: a specific local Q file is part of the case identity.
    for key in ["spatial", "reference", "q_matrices", "annotation"]:
        if settings.get(key):
            p = Path(settings[key]).expanduser().resolve(strict=True)
            if not p.is_file():
                raise ValueError(f"{key} must be a file")
            settings[key] = str(p)
        elif key != "annotation":
            raise ValueError(f"Missing {key}")
    return settings


def prepare(args):
    case = Path(args.case).expanduser().resolve()
    case.mkdir(parents=True, exist_ok=False)
    settings = configure(args.config)
    write_json(case / "settings.json", settings)
    external = {
        settings[k]: digest(settings[k])
        for k in ["spatial", "reference", "q_matrices", "annotation"]
        if settings.get(k)
    }
    source = snapshot(args.source, case / "baseline_source")
    copy_harness(case / "harness")
    result = launch(
        case,
        case / "baseline_source",
        case / "preparation",
        case / "harness",
        settings,
        action="prepare",
    )
    if result["status"] != "ok":
        raise RuntimeError(f"Case preparation failed; inspect {case / 'preparation/worker.log'}")
    settings["rctd_config"] = result["resolved_rctd_config"]
    write_json(case / "settings.json", settings)
    if any(digest(path) != value for path, value in external.items()):
        raise RuntimeError("An input file changed during preparation")
    manifest = {
        "schema": 1,
        "source": source,
        "external_files": external,
        "files": file_manifest(case),
    }
    manifest["case_id"] = object_hash(manifest)
    write_json(case / "manifest.json", manifest)
    print(f"Frozen case: {case}\ncase_id: {manifest['case_id']}")


def verify_case(case):
    manifest = read_json(case / "manifest.json")
    identity = {k: v for k, v in manifest.items() if k != "case_id"}
    if object_hash(identity) != manifest["case_id"]:
        raise ValueError("Case manifest identity is invalid")
    for name, expected in manifest["files"].items():
        if digest(case / name) != expected:
            raise ValueError(f"Frozen case changed: {name}")
    for name, expected in manifest["external_files"].items():
        if digest(name) != expected:
            raise ValueError(f"Input data changed: {name}")
    return manifest


def run(args):
    case = Path(args.case).expanduser().resolve()
    manifest = verify_case(case)
    settings = read_json(case / "settings.json")
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.source == "baseline":
        shutil.copytree(case / "baseline_source", out / "source")
        source = manifest["source"]
    else:
        source = snapshot(args.source, out / "source")
    harness = copy_harness(out / "harness")
    info = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": manifest["case_id"],
        "settings": settings,
        "source": source,
        "harness": harness,
        "stream_gpu": args.stream_gpu,
        "bulk_input_cpu": args.bulk_input_cpu,
        "stages": {},
    }
    write_json(out / "run.json", info)
    for stage in args.stages:
        info["stages"][stage] = launch(
            case,
            out / "source",
            out / stage,
            out / "harness",
            settings,
            stage=stage,
            stream_gpu=args.stream_gpu,
            bulk_input_cpu=args.bulk_input_cpu,
        )
        write_json(out / "run.json", info)
    verify_case(case)
    info["input_integrity_verified_after_run"] = True
    info["artifact_files"] = {
        str(p.relative_to(out)): digest(p)
        for stage in args.stages
        for p in (out / stage).glob("output_*.npz")
    }
    write_json(out / "run.json", info)
    return 0 if all(s["status"] == "ok" for s in info["stages"].values()) else 2


def load_run(directory):
    info = read_json(directory / "run.json")
    if not info.get("input_integrity_verified_after_run"):
        raise ValueError("Run did not complete its final input-integrity check")
    if file_manifest(directory / "source") != info["source"]["files"]:
        raise ValueError("Run source snapshot was changed")
    if file_manifest(directory / "harness") != info["harness"]:
        raise ValueError("Run evaluation harness was changed")
    for name, expected in info["artifact_files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Run output was changed: {name}")
    return info


def compare(args):
    baseline, candidate = Path(args.baseline).resolve(), Path(args.candidate).resolve()
    a, b = load_run(baseline), load_run(candidate)
    for key in ["case_id", "settings", "harness"]:
        if a[key] != b[key]:
            raise ValueError(f"Incomparable runs: {key} differs")
    if a["stages"].keys() != b["stages"].keys():
        raise ValueError("Incomparable runs: stage sets differ")
    report = {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "case_id": a["case_id"],
        "stages": {},
        "pass": True,
    }
    lines = [
        "# RCTD 回归评估",
        "",
        f"基线：`{baseline}`",
        f"候选：`{candidate}`",
        "",
        "数值通过只表示本数据、本设置下与基线一致；不等同于生物学准确。",
        "",
        "| 阶段 | 数值 | 基线中位秒 | 候选中位秒 | 倍速 | 时间判定 | 显存 allocated MiB（前→后） |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for stage in a["stages"]:
        old, new = a["stages"][stage], b["stages"][stage]
        result = {"baseline_status": old["status"], "candidate_status": new["status"]}
        report["stages"][stage] = result
        if old["status"] != "ok" or new["status"] != "ok":
            result.update(pass_=False, verdict="not_evaluated")
            report["pass"] = False
            lines.append(
                f"| {stage} | 未完成：{old['status']} / {new['status']} | — | — | — | — | — |"
            )
            continue
        if old["environment"] != new["environment"]:
            raise ValueError(f"Incomparable environments in {stage}")
        with np.load(baseline / stage / "output_000.npz", allow_pickle=False) as z:
            expected = dict(z)
        comparisons = []
        for path in sorted((candidate / stage).glob("output_*.npz")):
            with np.load(path, allow_pickle=False) as actual:
                comparisons.append(
                    compare_outputs(expected, actual, a["settings"]["atol"], a["settings"]["rtol"])
                )
        passed = bool(comparisons) and all(x["pass"] for x in comparisons)
        passed = passed and all(x["pass"] for x in old["repeatability"] + new["repeatability"])
        result["numerical_pass"] = passed
        result["comparisons"] = comparisons
        report["pass"] &= passed
        am = [x for x in old["measurements"] if not x["warmup"]]
        bm = [x for x in new["measurements"] if not x["warmup"]]
        timing = timing_comparison([x["elapsed_s"] for x in am], [x["elapsed_s"] for x in bm])
        result["timing"] = timing
        for key in [
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
            "rss_sampled_peak_bytes",
            "process_lifetime_peak_rss_bytes",
        ]:
            before, after = max(x[key] for x in am), max(x[key] for x in bm)
            result[key] = {"baseline": before, "candidate": after, "delta": after - before}
        result["annotation"] = {
            "baseline": old.get("annotation"),
            "candidate": new.get("annotation"),
        }
        memory = result["cuda_peak_allocated_bytes"]
        lines.append(
            f"| {stage} | {'PASS' if passed else 'FAIL'} | "
            f"{timing['reference_median_s']:.4f} | {timing['candidate_median_s']:.4f} | "
            f"{timing['speedup']:.3f} | {timing['verdict']} | "
            f"{memory['baseline'] / 2**20:.1f} → {memory['candidate'] / 2**20:.1f} |"
        )
    lines += ["", "## 数值差异", ""]
    for stage, result in report["stages"].items():
        for key, value in (result.get("comparisons") or [{"arrays": {}}])[0]["arrays"].items():
            if not value["pass"]:
                lines.append(f"- {stage}.{key}: `{json.dumps(value, ensure_ascii=False)}`")
    lines += [
        "",
        "## 所提供标注的一致程度",
        "",
        "预测按行归一化；标注细胞计数按行归一化。RNA 权重与细胞数比例不必相等。",
        "",
        "| 阶段/版本 | 有效 spots | RMSE | MAE | 宏平均 Pearson | JSD (base 2) | 主导类型准确率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, result in report["stages"].items():
        for label, metric in result.get("annotation", {}).items():
            if metric and metric["status"] == "ok":
                lines.append(
                    f"| {stage}/{label} | {metric['n_evaluated']} | {metric['rmse']:.6g} | "
                    f"{metric['mae']:.6g} | {metric['pearson_macro']} | "
                    f"{metric['jsd_base2_mean']:.6g} | "
                    f"{metric['dominant_type_accuracy']:.4f} |"
                )
    lines += [
        "",
        "完整逐类型指标、每次测量、最差差异位置见 comparison.json 和 run.json。",
        "当前 Full 模式不提供 doublet 置信分类；该官方指标标记为不适用。",
        "RSS 为 10 ms 采样峰值，另存进程生命周期峰值；CUDA 为 PyTorch allocated/reserved，",
        "不代表 nvidia-smi 的整卡占用。时间判定是保守描述，不是统计显著性检验。",
        "",
    ]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "comparison.json", report)
    (out / "report.md").write_text("\n".join(lines))
    print(f"Numerical {'PASS' if report['pass'] else 'FAIL'}; {out / 'report.md'}")
    return 0 if report["pass"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="freeze dataset, settings and baseline calibration")
    p.add_argument("--config", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--source", default="HEAD", help="git ref, or working")
    p = commands.add_parser("run", help="benchmark without modifying algorithm files")
    p.add_argument("--case", required=True)
    p.add_argument("--source", default="working", help="baseline, working, or git ref")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--stages",
        nargs="+",
        choices=["pipeline", "normalize", "full"],
        default=["normalize", "full", "pipeline"],
    )
    p.add_argument("--stream-gpu", action="store_true")
    p.add_argument(
        "--bulk-input-cpu",
        action="store_true",
        help="only affects isolated normalize; pipeline uses its actual caller",
    )
    p = commands.add_parser("compare", help="compare two saved, compatible runs")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run inside Linux/WSL with the repository Python environment")
    try:
        return {"prepare": prepare, "run": run, "compare": compare}[args.command](args) or 0
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
