"""Result types for RCTD."""

from typing import NamedTuple

import numpy as np
import torch


class RCTDConfig(NamedTuple):
    """RCTD configuration parameters.

    Field names keep R spacexr's spelling. Defaults match R's ``create.RCTD()`` and its internal
    config, except ``N_fit`` (100 here, 1000 in R).
    """

    #: Min normalised expression for the bulk gene list (``gene_cutoff`` in R).
    gene_cutoff: float = 0.000125
    #: Min log fold change for the bulk gene list (``fc_cutoff`` in R).
    fc_cutoff: float = 0.5
    #: Min normalised expression for the per-pixel DE gene list (``gene_cutoff_reg`` in R).
    gene_cutoff_reg: float = 0.0002
    #: Min log fold change for the per-pixel DE gene list (``fc_cutoff_reg`` in R).
    fc_cutoff_reg: float = 0.75
    #: Min total UMI for a spatial pixel to be fitted.
    UMI_min: int = 100
    #: Max total UMI for a spatial pixel to be fitted.
    UMI_max: int = 20_000_000
    #: Min counts a pixel needs on the bulk gene list.
    counts_MIN: int = 10
    #: Min UMI for a pixel to be sampled for sigma estimation.
    UMI_min_sigma: int = 300
    #: Max epochs of the sigma estimation loop.
    N_epoch: int = 8
    #: Number of pixels sampled for sigma estimation.
    N_fit: int = 100
    #: Convergence threshold of the bulk platform-effect fit.
    MIN_CHANGE_BULK: float = 0.0001
    #: Convergence threshold of the per-pixel fits. Not read yet; the solvers use a fixed 0.001.
    MIN_CHANGE_REG: float = 0.001
    #: Min observations for a gene in DE gene selection.
    MIN_OBS: int = 3
    #: Max cell types per pixel in multi mode.
    MAX_MULTI_TYPES: int = 4
    #: Score margin within which two fits count as a tie (doublet and multi mode).
    CONFIDENCE_THRESHOLD: float = 5.0
    #: Score gain needed to call a doublet (doublet mode) or to add another type (multi mode).
    DOUBLET_THRESHOLD: float = 20.0
    #: Max IRWLS iterations per pixel. Not read yet; the solvers use a fixed 50 (25 for scoring).
    max_iter: int = 50
    #: IRWLS step size. Not read yet; the solvers use a fixed 0.3.
    step_size: float = 0.3
    #: Largest count value in the likelihood tables.
    K_val: int = 1000
    #: ``"float64"`` (R arithmetic) or ``"float32"`` (1.2-2x faster and half the VRAM on GPUs
    #: with limited fp64 throughput; measured on an L40S).
    dtype: str = "float64"
    #: ``"auto"`` (GPU if available), ``"cpu"`` or ``"cuda"``.
    device: str = "auto"
    #: Use ``torch.compile``; ``False`` forces eager mode.
    compile: bool = True
    #: ``{cell_type -> class}`` mapping for hierarchical fallback in doublet mode;
    #: ``None`` makes each type its own class.
    class_df: dict[str, str] | None = None
    #: Override the per-architecture K cutoff for GPU ``eigh`` (default ``None``, meaning K <= 16 on
    #: sm < 9 and K <= 128 on sm >= 9). Raise it when the CPU offload is the bottleneck (issue #22).
    #: CPU runs ignore it.
    eigh_threshold: int | None = None


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve device string to torch.device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def auto_batch_size(
    G: int,
    K: int,
    dtype_bytes: int = 4,
    budget_fraction: float = 0.7,
    default: int = 10000,
) -> int:
    """Calculate optimal batch size based on available GPU VRAM.

    Estimates per-pixel memory footprint from all IRWLS intermediates
    and scales batch size to use ``budget_fraction`` of free VRAM.
    Falls back to ``default`` on CPU or if CUDA info is unavailable.
    """
    if not torch.cuda.is_available():
        return default
    try:
        free_mem, _ = torch.cuda.mem_get_info()
    except Exception:
        return default

    budget = int(free_mem * budget_fraction)

    # Per-pixel intermediates in solve_irwls_batch_shared:
    #   Y_batch(G) + prediction(G) + d1/d2(G) + solution(K) + grad(K)
    #   + hess(K*K) + D(K*K) + d_vec(K) + delta_w(K) + w_new(K)
    bytes_per_pixel = (3 * G + 5 * K + 2 * K * K) * dtype_bytes

    if bytes_per_pixel <= 0:
        return default

    computed = budget // bytes_per_pixel
    return max(1024, min(computed, 200000))


class FullResult(NamedTuple):
    """Result from full mode RCTD.

    Arrays cover only the pixels that passed the UMI and ``counts_MIN`` filters;
    ``spatial[result.pixel_mask]`` lines up row for row with them.
    """

    #: (N, K) unnormalised cell type weights.
    weights: np.ndarray
    #: Cell type names; column order of ``weights``.
    cell_type_names: list[str]
    #: (N,) whether the per-pixel fit converged.
    converged: np.ndarray
    #: (N_total,) which input pixels passed filtering.
    pixel_mask: np.ndarray | None = None


class DoubletResult(NamedTuple):
    """Result from doublet mode RCTD.

    Arrays cover only the pixels that passed the UMI and ``counts_MIN`` filters;
    ``spatial[result.pixel_mask]`` lines up row for row with them.
    """

    #: (N, K) full-mode weights fitted before the doublet step.
    weights: np.ndarray
    #: (N, 2) normalised weights of ``first_type`` and ``second_type``.
    weights_doublet: np.ndarray
    #: (N,) 0 = reject, 1 = singlet, 2 = doublet_certain, 3 = doublet_uncertain
    #: (see ``SPOT_CLASS_NAMES``).
    spot_class: np.ndarray
    #: (N,) index into ``cell_type_names`` of the first type.
    first_type: np.ndarray
    #: (N,) index into ``cell_type_names`` of the second type.
    second_type: np.ndarray
    #: (N,) ``True`` when the first type is only confident at the class level (``class_df``).
    first_class: np.ndarray
    #: (N,) ``True`` when the second type is only confident at the class level (``class_df``).
    second_class: np.ndarray
    #: (N,) score of the best pair of types (negative log-likelihood).
    min_score: np.ndarray
    #: (N,) score of the best single-type fit.
    singlet_score: np.ndarray
    #: Cell type names; column order of ``weights``.
    cell_type_names: list[str]
    #: (N_total,) which input pixels passed filtering.
    pixel_mask: np.ndarray | None = None
    #: (N,) class of ``first_type`` when ``class_df`` was given, else ``None``.
    first_class_name: np.ndarray | None = None
    #: (N,) class of ``second_type`` when ``class_df`` was given, else ``None``.
    second_class_name: np.ndarray | None = None


class MultiResult(NamedTuple):
    """Result from multi mode RCTD.

    Arrays cover only the pixels that passed the UMI and ``counts_MIN`` filters;
    ``spatial[result.pixel_mask]`` lines up row for row with them.
    """

    #: (N, K) full-mode weights fitted before type selection.
    weights: np.ndarray
    #: (N, MAX_MULTI_TYPES) weights of the selected types, padded with 0.
    sub_weights: np.ndarray
    #: (N, MAX_MULTI_TYPES) indices of the selected types, padded with -1.
    cell_type_indices: np.ndarray
    #: (N,) number of selected types per pixel.
    n_types: np.ndarray
    #: (N, MAX_MULTI_TYPES) whether each selected type is confident.
    conf_list: np.ndarray
    #: (N,) score of the selected set of types.
    min_score: np.ndarray
    #: Cell type names; indexed by ``cell_type_indices``.
    cell_type_names: list[str]
    #: (N_total,) which input pixels passed filtering.
    pixel_mask: np.ndarray | None = None


# Spot class encoding
SPOT_CLASS_REJECT = 0
SPOT_CLASS_SINGLET = 1
SPOT_CLASS_DOUBLET_CERTAIN = 2
SPOT_CLASS_DOUBLET_UNCERTAIN = 3
SPOT_CLASS_NAMES = ["reject", "singlet", "doublet_certain", "doublet_uncertain"]
