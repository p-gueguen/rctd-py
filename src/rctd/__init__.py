from rctd._doublet import run_doublet_mode
from rctd._full import run_full_mode
from rctd._multi import run_multi_mode
from rctd._multimodal import (
    adaptive_modality_weights,
    deconvolution_metrics,
    jensen_shannon_divergence,
    solve_irwls_batch_multimodal,
)
from rctd._multimodal_api import Modality, MultiModalResult, run_rctd_multimodal
from rctd._normalize import fit_bulk
from rctd._protein import (
    bootstrap_protein_profiles,
    build_signed_profile,
    normalize_protein,
    scgate_signatures,
)
from rctd._rctd import RCTD, run_rctd
from rctd._reference import Reference
from rctd._sigma import choose_sigma
from rctd._types import (
    SPOT_CLASS_NAMES,
    DoubletResult,
    FullResult,
    MultiResult,
    RCTDConfig,
)

try:
    from rctd._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "RCTD",
    "Reference",
    "run_rctd",
    "RCTDConfig",
    "FullResult",
    "DoubletResult",
    "MultiResult",
    "fit_bulk",
    "choose_sigma",
    "run_full_mode",
    "run_doublet_mode",
    "run_multi_mode",
    "SPOT_CLASS_NAMES",
    "normalize_protein",
    "bootstrap_protein_profiles",
    "build_signed_profile",
    "scgate_signatures",
    "run_rctd_multimodal",
    "Modality",
    "MultiModalResult",
    "solve_irwls_batch_multimodal",
    "adaptive_modality_weights",
    "jensen_shannon_divergence",
    "deconvolution_metrics",
]
