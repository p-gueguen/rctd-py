"""CUDA C++ extension loader with automatic fallback.

CUDA-Agent-inspired approach: JIT-compiles a custom CUDA kernel that fuses
the entire prediction → spline interpolation → gradient → hessian pipeline
into a single kernel launch. Falls back to the torch.compile fused function
when CUDA toolkit is unavailable or compilation fails.

The custom CUDA kernel uses native CUDA sqrt/floor/ceil which produce
bit-identical results to PyTorch's intrinsics (unlike Triton's tl.math.sqrt
which diverges at float64 boundaries).
"""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_cuda_ext = None
_load_attempted = False


def _try_load_cuda_ext():
    """Attempt to JIT-compile and load the CUDA extension."""
    global _cuda_ext, _load_attempted
    if _load_attempted:
        return _cuda_ext
    _load_attempted = True

    if not torch.cuda.is_available():
        logger.debug("CUDA not available, skipping CUDA extension")
        return None

    try:
        from torch.utils.cpp_extension import load

        csrc_dir = Path(__file__).parent / "csrc"
        cuda_src = csrc_dir / "fused_irwls_kernel.cu"

        if not cuda_src.exists():
            logger.debug("CUDA source not found at %s", cuda_src)
            return None

        _cuda_ext = load(
            name="fused_irwls_cuda",
            sources=[str(cuda_src)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        logger.info("Loaded CUDA extension for fused IRWLS kernel")
        return _cuda_ext
    except Exception as e:
        logger.debug("Failed to load CUDA extension: %s", e)
        return None


def fused_predict_and_derivatives_cuda(
    w_act: torch.Tensor,
    P: torch.Tensor,
    P_T: torch.Tensor,
    Y_act: torch.Tensor,
    nUMI_act: torch.Tensor,
    thresh_act: torch.Tensor,
    Q_mat: torch.Tensor,
    SQ_mat: torch.Tensor,
    x_vals: torch.Tensor,
    K_val: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Call the CUDA fused kernel if available.

    Returns (grad, hess) tuple or None if CUDA extension is unavailable.
    """
    ext = _try_load_cuda_ext()
    if ext is None:
        return None

    solution = torch.clamp(w_act, min=0.0)
    try:
        grad, hess = ext.fused_predict_and_derivatives(
            solution,
            P,
            P_T,
            Y_act,
            nUMI_act,
            thresh_act,
            Q_mat,
            SQ_mat,
            x_vals,
            K_val,
        )
        return grad, hess
    except Exception as e:
        logger.warning("CUDA fused kernel failed, falling back: %s", e)
        return None
