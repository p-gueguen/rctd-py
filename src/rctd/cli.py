"""Command-line interface for rctd-py."""

from __future__ import annotations

import click


@click.group()
@click.version_option(package_name="rctd-py")
def main():
    """rctd-py: GPU-accelerated cell type deconvolution for spatial transcriptomics."""


if __name__ == "__main__":
    main()
