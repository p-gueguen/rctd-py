"""Result types for RCTD."""

from typing import NamedTuple

import numpy as np
import torch


class RCTDConfig(NamedTuple):
    """RCTD configuration parameters."""

    gene_cutoff: float = 0.000125
    fc_cutoff: float = 0.5
    gene_cutoff_reg: float = 0.0002
    fc_cutoff_reg: float = 0.75
    UMI_min: int = 100
    UMI_max: int = 20_000_000
    counts_MIN: int = 10
    UMI_min_sigma: int = 300
    N_epoch: int = 8
    N_fit: int = 100
    MIN_CHANGE_BULK: float = 0.0001
    MIN_CHANGE_REG: float = 0.001
    MIN_OBS: int = 3
    MAX_MULTI_TYPES: int = 4
    CONFIDENCE_THRESHOLD: float = 5.0
    DOUBLET_THRESHOLD: float = 20.0
    max_iter: int = 50
    step_size: float = 0.3
    K_val: int = 1000
    dtype: str = "float64"  # "float32" or "float64"; float32 saves GPU memory
    device: str = "auto"  # "auto", "cpu", or "cuda"; auto uses GPU if available
    compile: bool = True  # Use torch.compile; False forces eager mode
    class_df: dict[str, str] | None = None  # Doublet mode: cell_type -> class; None = identity
    # Override the per-arch K cutoff for staying on GPU eigh in _psd_batch.
    # None = use arch-based default (K<=16 on sm_<9, K<=128 on sm_>=9). Set
    # higher to force GPU eigh on older arches when the CPU offload is the
    # bottleneck (issue #22 — K=38 on V100/L20). Ignored on CPU runs.
    eigh_threshold: int | None = None

    # ── Multi-modal (protein) parameters ──────────────────────────────────
    # An OPTIONAL second modality (e.g. protein immunofluorescence) added as a
    # Gaussian/WLS likelihood block sharing the per-pixel cell-type weights.
    # protein_weight is the master switch: 0.0 (default) => pure RNA, the path
    # is byte-for-byte identical to the RNA-only solver. "auto" balances the
    # protein gradient magnitude against RNA at init; a float sets lambda
    # directly. See _protein.py and the protein blocks in _irwls.py.
    # lambda; 0.0 = RNA-only (default), a fixed float, "auto" (per-feature gradient
    # balance at init — underweights a deep, informative panel), or "landmark"
    # (measured: sweep protein_lambda_grid and keep the best macro-F1 against
    # held-out protein-gated landmark cells).
    protein_weight: float | str = 0.0
    protein_obsm_key: str = "protein"  # spatial.obsm key for the (N, M) intensity matrix
    protein_norm: str = "arcsinh_robust"  # "arcsinh_robust" (default) | "clr"
    protein_profile_source: str = "bootstrap"  # "bootstrap" | "curated" (needs protein_signatures)
    # Optional curated signed gates: {cell_type: {"positive":[markers], "negative":[markers]}}.
    # When set, these OVERRIDE the per-type profile (negative markers let protein actively reject a
    # contaminant - e.g. NK = CD16+ / CD3E-/CD8A-, which the bootstrap cannot encode). With
    # source="bootstrap" only listed types are overridden (hybrid: bootstrap the rest); with
    # "curated" unlisted types get a neutral (zero) profile. Build from scGate models via
    # _protein.scgate_signatures. Overriding a wrong RNA call needs a strong protein_weight
    # (e.g. 4), not the balanced "auto" (which only sharpens).
    protein_signatures: dict | None = None
    # +/- z written for positive/negative curated markers. "calibrated" measures a
    # per-marker level from landmark cells instead (see _protein.calibrate_signed_levels):
    # one global magnitude that understates a real separation has to be paid for with a
    # larger lambda, which over-weights the whole panel rather than that one marker.
    protein_signature_magnitude: float | str = 1.5
    protein_var_model: str = "wls_pooled"  # "wls_pooled" (1/tau_m) | "unit"
    protein_arcsinh_cofactor: float = 5.0  # arcsinh cofactor for IF intensity
    protein_singlet_purity: float = 0.8  # confident-singlet weight gate for the bootstrap
    protein_tau_floor: float = 1e-3  # floor on per-marker tau to bound 1/tau^2
    # Per-cell protein reliability. "neighbour_ratio" downweights a cell whose
    # protein signal does not stand above the same markers in its spatial
    # neighbours (segmentation bleed-through) instead of trusting it fully.
    # Needs spatial.obsm["spatial"]; None = every cell with protein weighs 1.
    protein_reliability: str | None = None  # None | "neighbour_ratio"
    protein_reliability_k: int = 6  # neighbours per cell for the ratio
    protein_reliability_floor: float = 0.05  # never strip a cell's protein term entirely
    # Landmark cells (protein-gated, used when protein_weight="landmark" and by
    # protein_signature_magnitude="calibrated").
    protein_landmark_min_cells: int = 20  # per-type target for the gating ladder
    protein_landmark_folds: int = 2  # marker folds: gate truth with one, fit without it
    # Gate and score landmarks at a coarser level than the reference types:
    # {cell_type: class}. A 27-plex panel can adjudicate T vs B vs myeloid vs
    # epithelium; it cannot adjudicate CD4 vs Treg, and gating 25 fine types on
    # it left 72 landmarks with 10 types empty (renal, 2026-08). Types absent
    # from the map never match a landmark class (a call there scores as wrong).
    # The landmark gates are then read from protein_landmark_signatures (class-
    # keyed); protein_signatures stays per-type for the curated PROFILE.
    protein_landmark_classes: dict | None = None
    protein_landmark_signatures: dict | None = None  # default: protein_signatures
    protein_lambda_grid: tuple = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)  # swept by "landmark"
    protein_lambda_max_pixels: int = 10000  # subsample for the sweep; final fit uses all
    # Permutation-null realisations the measured gain must beat (0 = skip, and
    # accept any gain over lambda=0). Costs (n+1)x the sweep. One is NOT enough:
    # the null gain is centred at zero but spreads as wide as a real effect when
    # landmarks are few, so a single shuffled sweep let pure noise win lambda=4.
    protein_lambda_nulls: int = 6


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
    """Result from full mode RCTD."""

    weights: np.ndarray  # (N, K) float32
    cell_type_names: list[str]
    converged: np.ndarray  # (N,) bool
    pixel_mask: np.ndarray | None = None  # (N_total,) bool — which pixels passed filtering


class DoubletResult(NamedTuple):
    """Result from doublet mode RCTD."""

    weights: np.ndarray  # (N, K) float32
    weights_doublet: np.ndarray  # (N, 2) float32
    spot_class: np.ndarray  # (N,) int: 0=reject, 1=singlet, 2=doublet_certain, 3=doublet_uncertain
    first_type: np.ndarray  # (N,) int indices into cell_type_names
    second_type: np.ndarray  # (N,) int indices into cell_type_names
    first_class: np.ndarray  # (N,) bool
    second_class: np.ndarray  # (N,) bool
    min_score: np.ndarray  # (N,) float
    singlet_score: np.ndarray  # (N,) float
    cell_type_names: list[str]
    pixel_mask: np.ndarray | None = None  # (N_total,) bool — which pixels passed filtering
    # (N,) str, class of first/second_type when class_df was provided; else None
    first_class_name: np.ndarray | None = None
    second_class_name: np.ndarray | None = None
    # ── Modality disagreement, populated only when the protein modality is on ──
    # Which type each modality would pick alone, read off the joint single-type
    # scores (no extra fit). -1 = the pixel has no protein, or protein does not
    # discriminate between its candidates.
    rna_first_type: np.ndarray | None = None  # (N,) int
    prot_first_type: np.ndarray | None = None  # (N,) int
    modality_conflict: np.ndarray | None = None  # (N,) bool: the two disagree
    modality_confusion: np.ndarray | None = None  # (K, K) counts, rows RNA / cols protein


class MultiResult(NamedTuple):
    """Result from multi mode RCTD."""

    weights: np.ndarray  # (N, K) float32
    sub_weights: np.ndarray  # (N, MAX_MULTI_TYPES) float32, padded
    cell_type_indices: np.ndarray  # (N, MAX_MULTI_TYPES) int, padded with -1
    n_types: np.ndarray  # (N,) int, number of types per pixel
    conf_list: np.ndarray  # (N, MAX_MULTI_TYPES) bool
    min_score: np.ndarray  # (N,) float
    cell_type_names: list[str]
    pixel_mask: np.ndarray | None = None  # (N_total,) bool — which pixels passed filtering


# Spot class encoding
SPOT_CLASS_REJECT = 0
SPOT_CLASS_SINGLET = 1
SPOT_CLASS_DOUBLET_CERTAIN = 2
SPOT_CLASS_DOUBLET_UNCERTAIN = 3
SPOT_CLASS_NAMES = ["reject", "singlet", "doublet_certain", "doublet_uncertain"]
