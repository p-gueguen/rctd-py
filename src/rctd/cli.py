"""Command-line interface for rctd-py."""

from __future__ import annotations

import json
import sys

import click


@click.group()
@click.version_option(package_name="rctd-py")
def main():
    """rctd-py: GPU-accelerated cell type deconvolution for spatial transcriptomics."""


@main.command()
@click.option("--json", "use_json", is_flag=True, help="Output as JSON.")
def info(use_json):
    """Show version, device, and environment information."""
    import torch

    from rctd import __version__

    data = {
        "rctd_version": __version__,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_devices": [],
    }
    try:
        import anndata

        data["anndata_version"] = anndata.__version__
    except ImportError:
        data["anndata_version"] = "not installed"
    try:
        import numpy

        data["numpy_version"] = numpy.__version__
    except ImportError:
        data["numpy_version"] = "not installed"
    try:
        import scipy

        data["scipy_version"] = scipy.__version__
    except ImportError:
        data["scipy_version"] = "not installed"

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            data["cuda_devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "vram_mb": getattr(props, "total_memory", getattr(props, "total_mem", 0)) // (1024 * 1024),
                }
            )

    if use_json:
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(f"rctd-py {data['rctd_version']}")
        click.echo(f"Python  {data['python_version']}")
        click.echo(f"PyTorch {data['torch_version']}")
        if data["cuda_available"]:
            for dev in data["cuda_devices"]:
                click.echo(f"CUDA    {dev['name']} ({dev['vram_mb']} MB)")
        else:
            click.echo("CUDA    not available")
        click.echo(f"anndata {data['anndata_version']}")
        click.echo(f"numpy   {data['numpy_version']}")
        click.echo(f"scipy   {data['scipy_version']}")


@main.command()
@click.argument("spatial", type=click.Path(exists=True, dir_okay=False))
@click.argument("reference", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--cell-type-col",
    default="cell_type",
    show_default=True,
    help="Column in reference .obs for cell type labels.",
)
@click.option("--umi-min", default=100, show_default=True, help="Minimum UMI count per pixel.")
@click.option(
    "--cell-min", default=25, show_default=True, help="Minimum cells per cell type in reference."
)
@click.option("--json", "use_json", is_flag=True, help="Output as JSON.")
def validate(spatial, reference, cell_type_col, umi_min, cell_min, use_json):
    """Validate inputs before running RCTD (fast, no GPU needed)."""
    import anndata
    import numpy as np
    from scipy import sparse

    checks = {}
    warnings = []

    # 1. Read spatial
    try:
        sp = anndata.read_h5ad(spatial)
        checks["spatial_readable"] = {
            "pass": True,
            "detail": f"{sp.n_obs} pixels, {sp.n_vars} genes",
        }
    except Exception as e:
        checks["spatial_readable"] = {"pass": False, "detail": str(e)}
        sp = None

    # 2. Read reference
    try:
        ref = anndata.read_h5ad(reference)
        checks["reference_readable"] = {
            "pass": True,
            "detail": f"{ref.n_obs} cells, {ref.n_vars} genes",
        }
    except Exception as e:
        checks["reference_readable"] = {"pass": False, "detail": str(e)}
        ref = None

    # 3. Check cell_type_col
    if ref is not None:
        if cell_type_col in ref.obs.columns:
            types = ref.obs[cell_type_col].unique()
            checks["cell_type_col_exists"] = {
                "pass": True,
                "detail": f"{cell_type_col}, {len(types)} types",
            }
        else:
            available = ", ".join(ref.obs.columns.tolist()[:10])
            checks["cell_type_col_exists"] = {
                "pass": False,
                "detail": f"'{cell_type_col}' not found. Available: [{available}]",
            }

    # 4. Check cell type counts after cell_min filter
    if ref is not None and checks.get("cell_type_col_exists", {}).get("pass"):
        from collections import Counter

        type_counts = Counter(ref.obs[cell_type_col].values)
        valid_types = [ct for ct, n in type_counts.items() if n >= cell_min]
        low_types = [ct for ct, n in type_counts.items() if n < cell_min]
        if len(valid_types) >= 2:
            checks["min_cell_types"] = {
                "pass": True,
                "detail": f"{len(valid_types)} types >= {cell_min} cells",
            }
        else:
            checks["min_cell_types"] = {
                "pass": False,
                "detail": f"Only {len(valid_types)} types have >= {cell_min} cells",
            }
        if low_types:
            warnings.append(
                f"{len(low_types)} cell types have < {cell_min} cells: "
                f"{low_types[:5]}{'...' if len(low_types) > 5 else ''}"
            )

    # 5. Gene overlap
    if sp is not None and ref is not None:
        common = set(sp.var_names) & set(ref.var_names)
        n_common = len(common)
        pct = (n_common / sp.n_vars * 100) if sp.n_vars > 0 else 0
        if n_common >= 50:
            checks["gene_overlap"] = {
                "pass": True,
                "detail": f"{n_common} common genes ({pct:.1f}% of spatial)",
            }
        else:
            checks["gene_overlap"] = {
                "pass": False,
                "detail": f"{n_common} common genes ({pct:.1f}% of spatial) — need >= 50",
            }

    # 6. Pixel count after UMI filter
    numi = None
    if sp is not None:
        X = sp.X
        if sparse.issparse(X):
            numi = np.array(X.sum(axis=1)).flatten()
        else:
            numi = np.array(X.sum(axis=1)).flatten()
        n_pass = int((numi >= umi_min).sum())
        if n_pass > 0:
            checks["pixel_count_after_filter"] = {
                "pass": True,
                "detail": f"{n_pass}/{sp.n_obs} pass UMI >= {umi_min}",
            }
        else:
            checks["pixel_count_after_filter"] = {
                "pass": False,
                "detail": f"0/{sp.n_obs} pass UMI >= {umi_min}. Min UMI: {numi.min():.0f}",
            }
        if n_pass < sp.n_obs * 0.5:
            warnings.append(
                f"Over half of pixels filtered: {n_pass}/{sp.n_obs} pass UMI >= {umi_min}"
            )

    # Summary
    all_pass = all(c["pass"] for c in checks.values())
    status = "pass" if all_pass else "fail"

    # Estimates
    estimates = {}
    if sp is not None and ref is not None and checks.get("gene_overlap", {}).get("pass"):
        n_common = len(set(sp.var_names) & set(ref.var_names))
        n_types_est = (
            len(ref.obs[cell_type_col].unique())
            if checks.get("cell_type_col_exists", {}).get("pass")
            else 0
        )
        n_pix = int((numi >= umi_min).sum()) if numi is not None else sp.n_obs
        estimates = {
            "n_pixels": n_pix,
            "n_genes": n_common,
            "n_cell_types": n_types_est,
        }

    output = {"status": status, "checks": checks, "warnings": warnings, "estimates": estimates}

    if use_json:
        click.echo(json.dumps(output, indent=2))
    else:
        label = "PASS" if all_pass else "FAIL"
        click.echo(f"Validation: {label}")
        for name, check in checks.items():
            icon = "+" if check["pass"] else "x"
            click.echo(f"  [{icon}] {name}: {check['detail']}")
        for w in warnings:
            click.echo(f"  [!] {w}")
        if estimates:
            click.echo(
                f"  Pixels: {estimates['n_pixels']}, "
                f"Genes: {estimates['n_genes']}, "
                f"Types: {estimates['n_cell_types']}"
            )


def _build_summary(result, mode, cell_type_names):
    """Build mode-specific summary dict for JSON output."""
    from rctd._types import SPOT_CLASS_NAMES

    summary = {}
    if mode == "full":
        converged = result.converged
        summary["n_converged"] = int(converged.sum())
        summary["convergence_rate"] = float(converged.mean())
        dominant_idx = result.weights.argmax(axis=1)
        dominant_names = [cell_type_names[i] for i in dominant_idx]
        from collections import Counter

        summary["dominant_type_counts"] = dict(Counter(dominant_names).most_common())
    elif mode == "doublet":
        from collections import Counter

        class_names = [SPOT_CLASS_NAMES[i] for i in result.spot_class]
        summary["spot_class_counts"] = dict(Counter(class_names))
        first_names = [cell_type_names[i] for i in result.first_type]
        summary["top_singlet_types"] = dict(
            Counter(n for n, c in zip(first_names, result.spot_class) if c == 1).most_common(10)
        )
    elif mode == "multi":
        from collections import Counter

        n_types = result.n_types
        summary["n_types_distribution"] = dict(Counter(int(n) for n in n_types))
        summary["mean_n_types"] = float(n_types.mean())
        dominant_idx = result.weights.argmax(axis=1)
        dominant_names = [cell_type_names[i] for i in dominant_idx]
        summary["dominant_type_counts"] = dict(Counter(dominant_names).most_common())
    return summary


def _write_results_to_adata(
    spatial_adata, result, mode, pixel_mask, config_dict, cell_type_names, version
):
    """Expand results to full AnnData shape and write slots."""
    import numpy as np
    import pandas as pd

    from rctd._types import SPOT_CLASS_NAMES

    adata = spatial_adata.copy()
    n_total = adata.n_obs
    n_types = len(cell_type_names)

    # Weights (all modes)
    full_weights = np.full((n_total, n_types), np.nan, dtype=np.float32)
    full_weights[pixel_mask] = result.weights
    adata.obsm["rctd_weights"] = full_weights

    # Dominant type (all modes)
    dominant = np.full(n_total, "filtered", dtype=object)
    dominant_idx = result.weights.argmax(axis=1)
    dominant[pixel_mask] = [cell_type_names[i] for i in dominant_idx]
    adata.obs["rctd_dominant_type"] = pd.Categorical(dominant)

    if mode == "full":
        converged = np.full(n_total, False)
        converged[pixel_mask] = result.converged
        adata.obs["rctd_converged"] = converged

    elif mode == "doublet":
        # Spot class
        spot_class = np.full(n_total, "filtered", dtype=object)
        spot_class[pixel_mask] = [SPOT_CLASS_NAMES[i] for i in result.spot_class]
        adata.obs["rctd_spot_class"] = pd.Categorical(spot_class)

        # First/second type
        first_type = np.full(n_total, "filtered", dtype=object)
        first_type[pixel_mask] = [cell_type_names[i] for i in result.first_type]
        adata.obs["rctd_first_type"] = pd.Categorical(first_type)

        second_type = np.full(n_total, "filtered", dtype=object)
        second_type[pixel_mask] = [cell_type_names[i] for i in result.second_type]
        adata.obs["rctd_second_type"] = pd.Categorical(second_type)

        # Doublet weights
        full_wt_doublet = np.full((n_total, 2), np.nan, dtype=np.float32)
        full_wt_doublet[pixel_mask] = result.weights_doublet
        adata.obsm["rctd_weights_doublet"] = full_wt_doublet

    elif mode == "multi":
        n_types_per_pixel = np.zeros(n_total, dtype=np.int32)
        n_types_per_pixel[pixel_mask] = result.n_types
        adata.obs["rctd_n_types"] = n_types_per_pixel

        max_multi = result.sub_weights.shape[1]
        full_sub_wt = np.full((n_total, max_multi), np.nan, dtype=np.float32)
        full_sub_wt[pixel_mask] = result.sub_weights
        adata.obsm["rctd_sub_weights"] = full_sub_wt

        full_ct_idx = np.full((n_total, max_multi), -1, dtype=np.int32)
        full_ct_idx[pixel_mask] = result.cell_type_indices
        adata.obsm["rctd_cell_type_indices"] = full_ct_idx

    # Metadata
    adata.uns["rctd_mode"] = mode
    adata.uns["rctd_config"] = config_dict
    adata.uns["rctd_version"] = version
    adata.uns["rctd_cell_type_names"] = cell_type_names

    return adata


@main.command()
@click.argument("spatial", type=click.Path(exists=True, dir_okay=False))
@click.argument("reference", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--cell-type-col",
    default="cell_type",
    show_default=True,
    help="Column in reference .obs for cell type labels.",
)
@click.option(
    "--mode",
    type=click.Choice(["full", "doublet", "multi"]),
    default="doublet",
    show_default=True,
    help="Deconvolution mode.",
)
@click.option(
    "--output", "-o", default=None, help="Output h5ad path. [default: <spatial_stem>_rctd.h5ad]"
)
@click.option("--json", "use_json", is_flag=True, help="Print structured JSON summary to stdout.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress messages.")
# RCTDConfig parameters
@click.option("--gene-cutoff", default=0.000125, show_default=True, type=float)
@click.option("--fc-cutoff", default=0.5, show_default=True, type=float)
@click.option("--gene-cutoff-reg", default=0.0002, show_default=True, type=float)
@click.option("--fc-cutoff-reg", default=0.75, show_default=True, type=float)
@click.option(
    "--umi-min", default=100, show_default=True, type=int, help="Minimum UMI count per pixel."
)
@click.option("--umi-max", default=20_000_000, show_default=True, type=int)
@click.option("--umi-min-sigma", default=300, show_default=True, type=int)
@click.option("--max-multi-types", default=4, show_default=True, type=int)
@click.option("--confidence-threshold", default=5.0, show_default=True, type=float)
@click.option("--doublet-threshold", default=20.0, show_default=True, type=float)
@click.option(
    "--dtype", type=click.Choice(["float32", "float64"]), default="float64", show_default=True
)
@click.option(
    "--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", show_default=True
)
# Performance
@click.option(
    "--batch-size",
    default=10000,
    show_default=True,
    type=int,
    help="GPU batch size for pixel processing.",
)
@click.option(
    "--sigma-override",
    default=None,
    type=int,
    help="Skip sigma estimation, use this value directly.",
)
# Reference construction
@click.option(
    "--cell-min",
    default=25,
    show_default=True,
    type=int,
    help="Minimum cells per type in reference.",
)
@click.option(
    "--n-max-cells",
    default=10000,
    show_default=True,
    type=int,
    help="Max cells per type for downsampling.",
)
@click.option(
    "--min-umi-ref",
    default=100,
    show_default=True,
    type=int,
    help="Minimum UMI for reference cells.",
)
def run(
    spatial,
    reference,
    cell_type_col,
    mode,
    output,
    use_json,
    quiet,
    gene_cutoff,
    fc_cutoff,
    gene_cutoff_reg,
    fc_cutoff_reg,
    umi_min,
    umi_max,
    umi_min_sigma,
    max_multi_types,
    confidence_threshold,
    doublet_threshold,
    dtype,
    device,
    batch_size,
    sigma_override,
    cell_min,
    n_max_cells,
    min_umi_ref,
):
    """Run RCTD deconvolution on spatial transcriptomics data."""
    import contextlib
    import time
    import traceback
    from pathlib import Path

    import anndata

    from rctd import __version__
    from rctd._doublet import run_doublet_mode
    from rctd._full import run_full_mode
    from rctd._multi import run_multi_mode
    from rctd._rctd import RCTD
    from rctd._reference import Reference
    from rctd._types import RCTDConfig

    # Default output path
    if output is None:
        sp = Path(spatial)
        output = str(sp.parent / f"{sp.stem}_rctd.h5ad")

    # Build config
    config = RCTDConfig(
        gene_cutoff=gene_cutoff,
        fc_cutoff=fc_cutoff,
        gene_cutoff_reg=gene_cutoff_reg,
        fc_cutoff_reg=fc_cutoff_reg,
        UMI_min=umi_min,
        UMI_max=umi_max,
        UMI_min_sigma=umi_min_sigma,
        MAX_MULTI_TYPES=max_multi_types,
        CONFIDENCE_THRESHOLD=confidence_threshold,
        DOUBLET_THRESHOLD=doublet_threshold,
        dtype=dtype,
        device=device,
    )
    config_dict = config._asdict()

    try:
        # Redirect stdout to stderr when --json or --quiet
        if use_json or quiet:
            redirect = contextlib.redirect_stdout(sys.stderr)
        else:
            redirect = contextlib.nullcontext()

        with redirect:
            t_start = time.time()

            # Load data
            if not quiet:
                click.echo("Loading spatial data...", err=True)
            spatial_adata = anndata.read_h5ad(spatial)
            if not quiet:
                click.echo("Loading reference...", err=True)
            ref_adata = anndata.read_h5ad(reference)
            ref_obj = Reference(
                ref_adata,
                cell_type_col=cell_type_col,
                cell_min=cell_min,
                n_max_cells=n_max_cells,
                min_UMI=min_umi_ref,
            )

            # Run RCTD
            rctd_obj = RCTD(spatial_adata, ref_obj, config)
            rctd_obj.fit_platform_effects(sigma_override=sigma_override)

            cell_type_names = rctd_obj.reference.cell_type_names
            n_pixels_total = spatial_adata.n_obs
            n_pixels_filtered = int(rctd_obj._pixel_mask.sum())
            n_genes_common = len(rctd_obj.common_genes)

            print(f"Running in {mode} mode...")

            kwargs = {
                "spatial_counts": rctd_obj.counts,
                "spatial_numi": rctd_obj.nUMI,
                "norm_profiles": rctd_obj.norm_profiles,
                "cell_type_names": cell_type_names,
                "q_mat": rctd_obj.q_mat,
                "sq_mat": rctd_obj.sq_mat,
                "x_vals": rctd_obj.x_vals,
                "batch_size": batch_size,
                "device": config.device,
            }

            if mode == "full":
                result = run_full_mode(**kwargs)
            elif mode == "doublet":
                result = run_doublet_mode(**kwargs, config=config)
            elif mode == "multi":
                result = run_multi_mode(**kwargs, config=config)

            elapsed = time.time() - t_start

            # Write output h5ad
            out_adata = _write_results_to_adata(
                spatial_adata,
                result,
                mode,
                rctd_obj._pixel_mask,
                config_dict,
                cell_type_names,
                __version__,
            )
            out_adata.write_h5ad(output)
            print(f"Results written to {output}")

        # JSON output to real stdout
        if use_json:
            import torch

            device_used = "cpu"
            if torch.cuda.is_available() and config.device != "cpu":
                device_used = torch.cuda.get_device_name(0)

            summary = _build_summary(result, mode, cell_type_names)
            json_output = {
                "status": "success",
                "version": __version__,
                "mode": mode,
                "output_path": str(Path(output).resolve()),
                "input": {
                    "spatial_path": str(Path(spatial).resolve()),
                    "reference_path": str(Path(reference).resolve()),
                    "n_pixels_total": n_pixels_total,
                    "n_pixels_after_filter": n_pixels_filtered,
                    "n_genes_common": n_genes_common,
                    "n_cell_types": len(cell_type_names),
                    "cell_type_names": cell_type_names,
                },
                "config": config_dict,
                "results": {
                    "sigma": rctd_obj.sigma,
                    "elapsed_seconds": round(elapsed, 1),
                    "device_used": device_used,
                },
                "summary": summary,
            }
            click.echo(json.dumps(json_output, indent=2, default=str))

    except Exception as e:
        if use_json:
            error_output = {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            click.echo(json.dumps(error_output, indent=2))
            sys.exit(1)
        else:
            raise


if __name__ == "__main__":
    main()
