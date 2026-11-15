"""Result types for RCTD."""

import warnings
from typing import NamedTuple

import numpy as np
import torch


class RCTDConfig(NamedTuple):
    """RCTD configuration parameters.

    Field names keep R spacexr's spelling. Defaults match R's ``create.RCTD()`` and its internal
    config.
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
    #: Number of pixels sampled for sigma estimation (R's default, 1000).
    N_fit: int = 1000
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
    #: ``"float64"`` (R arithmetic) or ``"float32"`` (on GPUs with limited fp64 throughput:
    #: 1.3-1.9x faster in doublet mode and about half the VRAM on whole-transcriptome data;
    #: measured on an L40S).
    dtype: str = "float64"
    #: ``"auto"`` (GPU if available), ``"cpu"`` or ``"cuda"``.
    device: str = "auto"
    #: Use ``torch.compile``. The first run in a new environment compiles kernels: 10-20 min with
    #: torch 2.14, up to hours with torch 2.4-2.7 (torch 2.4 recompiles in every new process).
    #: ``False`` skips it at about 15% slower runs (the box-QP solver then uses TorchScript, the
    #: rest runs eagerly); also for environments without CUDA development headers.
    compile: bool = True
    #: ``{cell_type -> class}`` mapping for hierarchical fallback in doublet mode;
    #: ``None`` makes each type its own class.
    class_df: dict[str, str] | None = None
    #: Override the per-architecture K cutoff for GPU ``eigh`` (default ``None``, meaning K <= 16 on
    #: sm < 9 and K <= 128 on sm >= 9). Raising it helps only with recent CUDA: on an
    #: L40S, GPU eigh was 1.7-3.4x faster at K=45-78 with CUDA 13, 3-7x slower with CUDA 11.8/12.1.
    #: CPU runs ignore it.
    eigh_threshold: int | None = None


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve device string to torch.device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# Peak GPU memory of one batch row, in values. A batch row is a pixel in full mode, but a
# (pixel, type, type) triple in doublet mode and a candidate type set in multi mode, each carrying
# its own (G, k) design matrix, so those modes need far more per row. Measured with
# torch.cuda.max_memory_allocated on an L40S (G 1,500-25,000, K 10-150, float32 and float64; CPU
# eigh except for K <= 16, so the upper end of full includes some eigh workspace): full
# 7.5-14 x G, doublet 20-21.5 x G, multi (MAX_MULTI_TYPES=4) up to 18.4 x G at G >= 18,000.
# The constants below sit above every measurement (measured / estimate 0.44-0.76);
# budget_fraction adds the remaining margin. The GPU eigh workspace of the full-mode fit is added
# on top (`_eigh_terms`). Batch-independent memory (the resident counts, the G x K^2 P_outer of
# the full-mode fit, chunked float64 eigh) is subtracted from free memory instead.
_MIN_BATCH = 1024
_MAX_BATCH = 200_000


def _row_values(mode: str, G: int, K: int, max_multi_types: int) -> int:
    """Peak values per batch row. Doublet and multi mode start with a full-mode fit at the same
    batch size, so their cost is the larger of the two stages."""
    full = 16 * G + 5 * K + 2 * K * K  # + per-pixel K x K Hessian and QP terms
    if mode == "full":
        return full
    if mode == "doublet":
        return max(full, 24 * G)
    if mode == "multi":
        return max(full, 6 * max_multi_types * G + 80_000)
    raise ValueError(f"Unknown mode: {mode!r} (expected 'full', 'doublet' or 'multi')")


def _eigh_workspace_bytes(K: int, dtype_bytes: int) -> int:
    """cuSOLVER batched-eigh workspace per K x K matrix, in bytes.

    Measured as the full-mode peak per batch row with GPU eigh minus CPU eigh (L40S, G=3,000,
    torch 2.10+cu128 and 2.14.1+cu130; PR #50 follow-up). float32, torch 2.14.1+cu130:
    199 KB at K=3, 242 KB at K=16, 298 KB at K=32, 453 KB at K=64, 787 KB at K=128; torch
    2.10+cu128 matches up to K=32 and has none above (66 KB at K=128). The fit sits above every
    point (K=3: 209 KB vs 199 KB measured); the model
    assumes the larger (cu130) workspace whatever the stack.
    """
    f32 = 200_000 + 3_000 * K + 20 * K * K
    return f32 if dtype_bytes <= 4 else int(_EIGH_F64_FACTOR * f32)


# float64 workspace relative to float32: 2.0x on the L40S (K=38: 655 KB vs 329 KB), and an RTX
# 5070 (Blackwell, torch 2.14.1+cu130) measured ~778 KB at K=40, above 2 x the float32 fit
# (684 KB); 2.4 covers both.
_EIGH_F64_FACTOR = 2.4


def _eigh_terms(K: int, dtype_bytes: int, dev: torch.device) -> tuple[int, int]:
    """(bytes per batch row, batch-independent bytes) added by GPU eigh in the full-mode fit.

    Eigh runs on the GPU when 3 <= K <= the per-arch threshold (`_cuda_eigh_threshold`: 16 on
    compute capability < 9, 128 on >= 9, or RCTDConfig.eigh_threshold); K <= 2 is analytic and
    larger K goes to CPU. float64 with K <= 32 is chunked at 1,024 matrices, so its workspace
    does not grow with the batch.
    """
    from rctd._irwls import _EIGH_SMALL_K_MAX, _MAX_EIGH_BATCH_SMALL_K, _cuda_eigh_threshold

    if K < 3 or K > _cuda_eigh_threshold(dev):
        return 0, 0
    ws = _eigh_workspace_bytes(K, dtype_bytes)
    if dtype_bytes == 8 and K <= _EIGH_SMALL_K_MAX:
        return 0, _MAX_EIGH_BATCH_SMALL_K * ws
    return ws, 0


def _batch_bytes(
    mode: str, G: int, K: int, max_multi_types: int, dtype_bytes: int, dev: torch.device
) -> tuple[int, int]:
    """(peak bytes per batch row, batch-independent bytes besides the resident counts).

    Every mode starts with a full-mode fit, which runs GPU eigh for 3 <= K <= threshold, so the
    eigh workspace enters every mode through the full-mode row.
    """
    eigh_row, eigh_fixed = _eigh_terms(K, dtype_bytes, dev)
    if mode == "multi":
        # Multi's own fits (k = 3..max_multi_types types) also run eigh, on the GPU at the default
        # thresholds even when the full fit does not. Their float32 workspace is inside the
        # measured multi row; float64 is chunked (fixed). Not live together with the full fit's.
        eigh_fixed = max(eigh_fixed, _eigh_terms(max_multi_types, dtype_bytes, dev)[1])
    full_row = _row_values("full", G, K, max_multi_types) * dtype_bytes + eigh_row
    row = max(full_row, _row_values(mode, G, K, max_multi_types) * dtype_bytes)
    return row, G * K * K * dtype_bytes + eigh_fixed  # P_outer + chunked eigh workspace


def auto_batch_size(
    G: int,
    K: int,
    dtype_bytes: int = 4,
    budget_fraction: float = 0.7,
    default: int = 10000,
    mode: str = "full",
    n_resident: int = 0,
    max_multi_types: int = 4,
    device: str | torch.device = "auto",
) -> int:
    """Calculate a batch size that fits in the free memory of the GPU the run uses.

    Estimates the peak memory of one batch row in ``mode`` and scales the batch to use
    ``budget_fraction`` of the VRAM left free once the ``n_resident`` x ``G`` count matrix the
    mode keeps on the GPU, and the G x K^2 ``P_outer`` table of the full-mode fit every mode runs,
    are in place, plus the cuSOLVER eigh workspace when the full-mode fit runs eigh on this GPU
    (see `_eigh_terms`). Returns ``default`` when ``device`` is not a CUDA device, CUDA info is
    unavailable, or ``G`` or ``K`` is not positive.
    """
    if mode not in ("full", "doublet", "multi"):
        raise ValueError(f"Unknown mode: {mode!r} (expected 'full', 'doublet' or 'multi')")
    if G <= 0 or K <= 0:
        return default
    dev = resolve_device(device) if isinstance(device, str) else device
    if dev.type != "cuda" or not torch.cuda.is_available():
        return default
    try:
        free_mem, _ = torch.cuda.mem_get_info(dev)
    except Exception:
        return default

    bytes_per_row, fixed = _batch_bytes(mode, G, K, max_multi_types, dtype_bytes, dev)
    resident = n_resident * G * dtype_bytes + fixed  # counts + P_outer + chunked eigh
    if resident >= free_mem:
        warnings.warn(
            f"The count matrix, reference tables and eigh workspace ({resident / 2**30:.1f} GiB) "
            f"do not fit in the free GPU memory ({free_mem / 2**30:.1f} GiB) in {mode} mode, "
            "whatever the batch size. Use dtype='float32', fewer pixels per run, or "
            "stream_gpu=True in full mode.",
            stacklevel=2,
        )
        return _MIN_BATCH
    computed = int((free_mem - resident) * budget_fraction) // bytes_per_row
    if computed < _MIN_BATCH:
        warnings.warn(
            f"Only about {computed} rows fit in the free GPU memory ({free_mem / 2**30:.1f} GiB) "
            f"in {mode} mode; using batch_size={_MIN_BATCH}, which may run out of memory. Pass a "
            "smaller batch_size, use dtype='float32', or free the GPU.",
            stacklevel=2,
        )
    return max(_MIN_BATCH, min(computed, _MAX_BATCH))


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
    #: (N,) singlet score used for the singlet / doublet call (R's ``results_df$singlet_score``).
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
    #: (N, MAX_MULTI_TYPES) weights of the selected types (summing to 1), padded with 0.
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
