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
            data["cuda_devices"].append({
                "index": i,
                "name": props.name,
                "vram_mb": props.total_mem // (1024 * 1024),
            })

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


if __name__ == "__main__":
    main()
