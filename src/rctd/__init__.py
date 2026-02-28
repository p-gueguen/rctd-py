from rctd._doublet import run_doublet_mode
from rctd._full import run_full_mode
from rctd._multi import run_multi_mode
from rctd._normalize import fit_bulk
from rctd._rctd import RCTD, run_rctd
from rctd._reference import Reference
from rctd._sigma import choose_sigma
from rctd._types import DoubletResult, FullResult, MultiResult, RCTDConfig

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
]
