"""Preprocessing, scverse style: build the single-cell reference from an AnnData."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rctd._anndata import _with_counts_from
from rctd._reference import Reference

if TYPE_CHECKING:
    import anndata


def reference(
    adata: anndata.AnnData,
    cell_type_key: str = "cell_type",
    *,
    layer: str | None = None,
    cell_min: int = 25,
    n_max_cells: int = 10000,
    min_UMI: int = 100,
) -> Reference:
    """Build the reference for :func:`rctd.tl.deconvolve` from a single-cell AnnData.

    The same as ``Reference(adata, cell_type_col=cell_type_key, ...)``.

    Args:
        adata: Single-cell data, cells x genes, raw counts in ``X`` (or in ``layers[layer]``).
        cell_type_key: Column of ``adata.obs`` with the cell type labels.
        layer: Read the counts from ``adata.layers[layer]`` instead of ``adata.X``.
        cell_min: Cell types with fewer cells are dropped.
        n_max_cells: Cells kept per type, sampled at random (seed 42) above this.
        min_UMI: Cells with fewer total UMIs are dropped.

    Returns:
        The reference, to pass to :func:`rctd.tl.deconvolve` or :func:`rctd.run_rctd`.
    """
    return Reference(
        _with_counts_from(adata, layer),
        cell_type_col=cell_type_key,
        cell_min=cell_min,
        n_max_cells=n_max_cells,
        min_UMI=min_UMI,
    )
