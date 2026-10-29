"""Fresh-process worker. Imports RCTD only from the requested frozen source."""

import argparse
import gc
import inspect
import json
import os
import platform
import random
import resource
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from metrics import annotation_metrics, compare_outputs, doublet_call_metrics


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_inputs(settings):
    import anndata as ad

    spatial = ad.read_h5ad(settings["spatial"])
    reference = ad.read_h5ad(settings["reference"])
    for name, data in [("spatial", spatial), ("reference", reference)]:
        layer = settings.get(f"{name}_layer")
        if layer:
            data.X = data.layers[layer]
        column = settings.get("gene_id_column")
        if column:
            data.var_names = data.var[column].astype(str)
        if not data.var_names.is_unique or not data.obs_names.is_unique:
            raise ValueError(f"{name}: gene and observation IDs must be unique")
        values = data.X.data if hasattr(data.X, "tocsr") else np.asarray(data.X).ravel()
        # Bounded temporary memory even for large raw-count matrices.
        for start in range(0, values.size, 1_000_000):
            part = values[start : start + 1_000_000]
            if not np.isfinite(part).all() or (part < 0).any() or (part != np.floor(part)).any():
                raise ValueError(f"{name}: selected X/layer must contain finite raw counts")
    n = settings.get("max_spots")
    if n is not None:
        if not 0 < n <= spatial.n_obs:
            raise ValueError("max_spots must be positive and no larger than the dataset")
        indices = np.sort(
            np.random.default_rng(settings["seed"]).choice(spatial.n_obs, n, replace=False)
        )
        spatial = spatial[indices].copy()
    return spatial, reference


def environment(torch, device):
    import anndata
    import scipy

    details = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "scipy": scipy.__version__,
        "anndata": anndata.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "env": {
            k: os.environ.get(k)
            for k in ["CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "CUBLAS_WORKSPACE_CONFIG"]
        },
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        details.update(
            gpu=props.name,
            gpu_total_bytes=props.total_memory,
            capability=list(torch.cuda.get_device_capability(device)),
        )
        try:
            import subprocess

            details["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            details["driver"] = None
    return details


def rss_bytes():
    # Linux/WSL only; RSS sample is not a CUDA driver / whole-system measurement.
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def measure(fn, torch, device):
    gc.collect()
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    before = rss_bytes()
    samples = [before]
    stop = threading.Event()

    def poll():
        while not stop.wait(0.01):
            samples.append(rss_bytes())

    sampler = threading.Thread(target=poll, daemon=True)
    sampler.start()
    t0 = time.perf_counter()
    try:
        output = fn()
        if cuda:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
    finally:
        stop.set()
        sampler.join()
    samples.append(rss_bytes())
    stats = {
        "elapsed_s": elapsed,
        "rss_start_bytes": before,
        "rss_sampled_peak_bytes": max(samples),
        "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else 0,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else 0,
    }
    return output, stats


def save_arrays(directory, arrays):
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(directory / f"{name}.npy", np.asarray(value), allow_pickle=False)


def load_arrays(directory):
    # Eager CPU load is intentional; this framework initially evaluates RAM -> GPU.
    return {p.stem: np.load(p, allow_pickle=False) for p in directory.glob("*.npy")}


def bulk_inputs(frozen, device, cpu_input):
    """Match the caller: CPU inputs are NumPy views, not extra matrix copies."""
    import torch

    factory = torch.as_tensor if cpu_input else torch.tensor
    input_device = "cpu" if cpu_input else device
    return {
        "cell_type_profiles": torch.tensor(frozen["cell_type_profiles"], device=device),
        "spatial_counts": factory(frozen["spatial_counts"], device=input_device),
        "spatial_nUMI": factory(frozen["spatial_nUMI"], device=input_device),
    }


def worker(args):
    case, destination = Path(args.case), Path(args.out)
    settings = json.loads((case / "settings.json").read_text())
    sys.path.insert(0, str(Path(args.source) / "src"))
    os.environ["RCTD_Q_MATRICES"] = settings["q_matrices"]
    import torch

    import rctd
    from rctd import _irwls, _likelihood, _rctd
    from rctd._types import resolve_device

    if not Path(rctd.__file__).resolve().is_relative_to(Path(args.source).resolve()):
        raise RuntimeError("RCTD imported from the wrong source snapshot")
    torch.set_num_threads(settings["threads"])
    device = resolve_device(settings["rctd_config"]["device"])
    if device.type == "cuda":
        device = torch.device("cuda", device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
    config = rctd.RCTDConfig(**settings["rctd_config"])
    if not config.compile:
        _irwls._USE_COMPILE = False
        _likelihood._CALC_Q_USE_COMPILE = False
    else:
        _irwls._USE_COMPILE = None
        _likelihood._CALC_Q_USE_COMPILE = None
    info = {
        "status": "running",
        "environment": environment(torch, device),
        "imported_rctd": rctd.__file__,
        "measurements": [],
    }
    write_json(destination / "metrics.json", info)
    stage_times, captured = {}, {}
    original_bulk, original_sigma = _rctd.fit_bulk, _rctd.choose_sigma
    original_platform = _rctd.RCTD.fit_platform_effects
    original_full = _rctd.run_full_mode

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def timed(name, fn, *a, **kw):
        sync()
        start = time.perf_counter()
        result = fn(*a, **kw)
        sync()
        stage_times[name] = time.perf_counter() - start
        return result

    def observe_bulk(*a, **kw):
        if args.action == "prepare":
            bound = inspect.signature(original_bulk).bind(*a, **kw)
            save_arrays(
                case / "normalize",
                {
                    k: bound.arguments[k].detach().cpu().numpy()
                    for k in ["cell_type_profiles", "spatial_counts", "spatial_nUMI"]
                },
            )
        result = timed("fit_bulk_s", original_bulk, *a, **kw)
        captured["bulk_weights"] = result[0].detach().cpu().numpy()
        return result

    def observe_platform(obj, *a, **kw):
        result = timed("platform_total_s", original_platform, obj, *a, **kw)
        captured["object"] = obj
        return result

    _rctd.fit_bulk = observe_bulk
    _rctd.choose_sigma = lambda *a, **kw: timed("choose_sigma_s", original_sigma, *a, **kw)
    _rctd.RCTD.fit_platform_effects = observe_platform
    _rctd.run_full_mode = lambda *a, **kw: timed("full_s", original_full, *a, **kw)
    if args.action == "prepare" or args.stage == "pipeline":
        start = time.perf_counter()
        spatial, reference_data = read_inputs(settings)
        info["input_load_validate_s"] = time.perf_counter() - start
    if args.action == "prepare":
        ref = rctd.Reference(reference_data, **settings["reference_options"])
        obj = rctd.RCTD(spatial, ref, config)
        obj.fit_platform_effects(sigma_override=settings["sigma_override"])
        save_arrays(
            case / "full",
            {
                "spatial_counts": obj.counts,
                "spatial_numi": obj.nUMI,
                "norm_profiles": obj.norm_profiles,
                "cell_type_names": np.asarray(ref.cell_type_names),
                "q_mat": obj.q_mat,
                "sq_mat": obj.sq_mat,
                "x_vals": obj.x_vals,
                "pixel_mask": obj._pixel_mask,
                "spot_ids": np.asarray(spatial.obs_names[obj._pixel_mask], dtype=str),
                "genes": np.asarray(obj._gene_list_reg),
                "sigma": obj.sigma,
            },
        )
        save_arrays(
            case / "normalize",
            {
                "cell_type_names": np.asarray(ref.cell_type_names),
                "genes": np.asarray(obj._gene_list_bulk),
            },
        )
        save_arrays(case / "selection", {"spot_ids": np.asarray(spatial.obs_names, dtype=str)})
        info.update(
            status="ok",
            n_input=spatial.n_obs,
            n_filtered=len(obj.counts),
            n_genes=obj.counts.shape[1],
            n_types=ref.n_types,
            sigma=int(obj.sigma),
            resolved_rctd_config=config._asdict(),
            stage_times=stage_times,
        )
        write_json(destination / "metrics.json", info)
        return

    if args.stage in ("full", "normalize"):
        frozen = load_arrays(case / args.stage)
    if args.stage == "pipeline":
        expected_ids = np.load(case / "selection/spot_ids.npy")
        if not np.array_equal(np.asarray(spatial.obs_names), expected_ids):
            raise ValueError("Input spot selection drifted from frozen case")

    def stream_option(fn):
        if args.stream_gpu:
            if "stream_gpu" not in inspect.signature(fn).parameters:
                raise ValueError("Selected source does not support stream_gpu")
            return {"stream_gpu": True}
        return {}

    def run_once():
        if args.stage == "normalize":
            bw, profiles = original_bulk(
                **bulk_inputs(frozen, device, args.bulk_input_cpu),
                min_change=config.MIN_CHANGE_BULK,
            )
            return {
                "bulk_weights": bw.detach().cpu().numpy(),
                "norm_profiles": profiles.detach().cpu().numpy(),
                "genes": frozen["genes"],
                "cell_type_names": frozen["cell_type_names"],
            }
        if args.stage == "full":
            kwargs = {
                k: frozen[k]
                for k in [
                    "spatial_counts",
                    "spatial_numi",
                    "norm_profiles",
                    "q_mat",
                    "sq_mat",
                    "x_vals",
                    "pixel_mask",
                ]
            }
            result = original_full(
                **kwargs,
                cell_type_names=frozen["cell_type_names"].tolist(),
                batch_size=settings["batch_size"],
                device=str(device),
                **stream_option(original_full),
            )
            meta = {k: frozen[k] for k in ["spot_ids", "genes", "sigma"]}
        else:
            start = time.perf_counter()
            ref = rctd.Reference(reference_data, **settings["reference_options"])
            stage_times["reference_s"] = time.perf_counter() - start
            result = _rctd.run_rctd(
                spatial,
                ref,
                mode=settings.get("mode", "full"),
                config=config,
                batch_size=settings["batch_size"],
                sigma_override=settings["sigma_override"],
                **stream_option(_rctd.run_rctd),
            )
            obj = captured["object"]
            meta = {
                "spot_ids": np.asarray(spatial.obs_names[obj._pixel_mask], dtype=str),
                "genes": np.asarray(obj._gene_list_reg),
                "sigma": np.asarray(obj.sigma),
                "bulk_genes": np.asarray(obj._gene_list_bulk),
                "norm_profiles": obj.norm_profiles,
                "bulk_weights": captured["bulk_weights"],
            }
        arrays = {
            "weights": result.weights,
            "cell_type_names": np.asarray(result.cell_type_names),
            "pixel_mask": result.pixel_mask,
            **meta,
        }
        for key, value in result._asdict().items():
            if value is not None and key not in arrays:
                arrays[key] = np.asarray(value)
        if hasattr(result, "weights_doublet"):
            pred = np.zeros_like(result.weights)
            for i in range(len(pred)):
                if result.spot_class[i] == 1:
                    pred[i, result.first_type[i]] = 1
                elif result.spot_class[i] in (2, 3):
                    pred[i, result.first_type[i]] += result.weights_doublet[i, 0]
                    pred[i, result.second_type[i]] += result.weights_doublet[i, 1]
            arrays["prediction_weights"] = pred
        elif hasattr(result, "sub_weights"):
            pred = np.zeros_like(result.weights)
            for i in range(len(pred)):
                for j in range(result.n_types[i]):
                    pred[i, result.cell_type_indices[i, j]] += result.sub_weights[i, j]
            arrays["prediction_weights"] = pred
        return arrays

    truth = None
    if settings.get("annotation") and args.stage != "normalize":
        import pandas as pd

        truth = pd.read_csv(settings["annotation"], index_col=settings["annotation_index"])
        if not truth.index.is_unique or not truth.columns.is_unique:
            raise ValueError("Annotation IDs/types must be unique")
    repeat_reference = None
    info["repeatability"] = []
    for index in range(settings["warmup"] + settings["repeats"]):
        captured.clear()
        stage_times.clear()
        random.seed(settings["seed"])
        np.random.seed(settings["seed"])
        torch.manual_seed(settings["seed"])
        output, stats = measure(run_once, torch, device)
        warmup = index < settings["warmup"]
        stats.update(warmup=warmup, stage_times=dict(stage_times))
        if "weights" in output:
            stats.update(
                n_spots=len(output["weights"]),
                throughput_spots_s=len(output["weights"]) / stats["elapsed_s"],
            )
            if "converged" in output:
                stats["converged_fraction"] = float(output["converged"].mean())
        info["measurements"].append(stats)
        if not warmup:
            number = index - settings["warmup"]
            np.savez(destination / f"output_{number:03d}.npz", **output)
            if repeat_reference is None:
                repeat_reference = output
                if truth is not None:
                    names, ids = output["cell_type_names"].tolist(), output["spot_ids"].tolist()
                    if set(names) != set(truth.columns) or not set(ids) <= set(truth.index):
                        raise ValueError("Annotation types/spots do not match; no silent dropping")
                    aligned_truth = truth.loc[ids, names].to_numpy()
                    info["annotation"] = annotation_metrics(
                        output.get("prediction_weights", output["weights"]), aligned_truth, names
                    )
                    if "spot_class" in output:
                        info["annotation"]["official_doublet_calls"] = doublet_call_metrics(
                            output["spot_class"],
                            output["first_type"],
                            output["second_type"],
                            aligned_truth,
                            names,
                            config.class_df,
                        )
                        missing = info["annotation"]["official_not_applicable"]
                        missing.pop("singlet_doublet_classification", None)
                        missing.pop("confident_call_cell_type_accuracy", None)
            info["repeatability"].append(
                compare_outputs(repeat_reference, output, settings["atol"], settings["rtol"])
            )
        print(json.dumps({"stage": args.stage, "iteration": index, **stats}), flush=True)
        write_json(destination / "metrics.json", info)
        del output
    info["status"] = "ok"
    info["first_call_s"] = info["measurements"][0]["elapsed_s"]
    write_json(destination / "metrics.json", info)


def main():
    parser = argparse.ArgumentParser()
    for name in ["case", "source", "out"]:
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--action", choices=["prepare", "run"], default="run")
    parser.add_argument("--stage", choices=["pipeline", "normalize", "full"], default="pipeline")
    parser.add_argument("--stream-gpu", action="store_true")
    parser.add_argument("--bulk-input-cpu", action="store_true")
    args = parser.parse_args()
    try:
        worker(args)
    except Exception as exc:
        path = Path(args.out) / "metrics.json"
        info = json.loads(path.read_text()) if path.exists() else {}
        oom = isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()
        info.update(
            status="oom" if oom else "error", error=str(exc), traceback=traceback.format_exc()
        )
        write_json(path, info)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
