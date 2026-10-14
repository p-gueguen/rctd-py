# Installation

```bash
uv pip install rctd-py   # or: pip install rctd-py
```

rctd-py runs on CPU out of the box and uses a GPU automatically when PyTorch can see one.

## Q-matrices (offline and HPC setups)

The first run needs `q_matrices.npz`, the precomputed Poisson-Lognormal likelihood tables for
every sigma between 0.10 and 2.00. It is 404 MB and does not compress, so it ships as a
[release asset](https://github.com/p-gueguen/rctd-py/releases/download/v0.1.1/q_matrices.npz)
rather than inside the wheel, and is fetched once into `~/.cache/rctd/` on first use.

That download fails on a compute node without internet access. Stage the file from a host that
has access and point rctd-py at it:

```bash
# once, somewhere every node can read
wget -P /shared/rctd https://github.com/p-gueguen/rctd-py/releases/download/v0.1.1/q_matrices.npz
export RCTD_Q_MATRICES=/shared/rctd          # a directory, or the .npz itself
```

`RCTD_Q_MATRICES` takes precedence over the cache, so one shared read-only copy serves every
user. If it is unset and the download fails, the error names the URL and the exact path to place
the file at. A file you point to deliberately is never re-downloaded over: if it is unreadable,
you are told so.

## GPU setup

On Linux x86_64, `pip install rctd-py` already pulls a CUDA build of PyTorch: the default PyPI
wheel has bundled CUDA 13.0 since torch 2.11, which needs an NVIDIA driver >= 580. On an older
driver, install a CUDA 12 build of PyTorch before rctd-py:

```bash
# Driver >= 580: default wheel, nothing extra to do
uv pip install rctd-py

# Driver 525-579: CUDA 12.6 build (not for Blackwell, which needs CUDA >= 12.8)
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
uv pip install rctd-py
```

macOS and Windows get the CPU (or MPS) wheel from PyPI; for other combinations use the selector
at [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

Check that the GPU is visible:

```python
import torch

print(torch.cuda.is_available())  # False means CPU-only torch or a driver issue
print(torch.cuda.get_device_name())  # e.g. 'NVIDIA L40S'
print(torch.version.cuda)  # e.g. '13.0'
```

No separate CUDA toolkit is needed: PyTorch ships its own CUDA runtime, so only the driver
version matters (`nvidia-smi`, top right).

| PyTorch wheel | Bundled CUDA | Minimum NVIDIA driver | Blackwell (sm_100 / sm_120) |
|---|---|---|---|
| PyPI default, or `cu130` / `cu132` index | CUDA 13.x | >= 580 | yes |
| `cu128` / `cu129` index | CUDA 12.8 / 12.9 | >= 525 | yes |
| `cu126` index | CUDA 12.6 | >= 525 | no |

### GPU memory

Peak VRAM scales with `batch_size * K^2`. From Python, `run_rctd` defaults to
`batch_size="auto"`, sized from the free GPU memory; the CLI defaults to `--batch-size 10000`.
Set it explicitly to fit a smaller card or a shared GPU:

| Available VRAM | Recommended `batch_size` | Peak VRAM (K=45) |
|---|---|---|
| 24+ GB | 10,000 | ~4 GB |
| 8-16 GB | 5,000 | ~2 GB |
| < 8 GB | 2,000 | ~1 GB |

Peak CPU RAM (RSS) is typically 2-3x peak VRAM.

## Known issue: NVRTC missing on Blackwell + CUDA 13

On a Blackwell GPU with a CUDA 13 driver (580.x), a CUDA 12 build of PyTorch can fail in the
`compile=False` path with:

```text
nvrtc: error: failed to open libnvrtc-builtins.so.13.0.
```

This is a runtime-library mismatch, not an rctd-py bug. Install the CUDA 13 NVRTC runtime and put
its directory on `LD_LIBRARY_PATH`:

```bash
pip install nvidia-cuda-nvrtc
find $CONDA_PREFIX -name "libnvrtc-builtins.so.13*"   # .../site-packages/nvidia/cu13/lib/
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.X/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```

Point `LD_LIBRARY_PATH` at `nvidia/cu13/lib/`, not `nvidia/cuda_nvrtc/lib/`. The CUDA 13 PyTorch
wheels (the PyPI default on Linux since torch 2.11) already pull this in, so most installs never
need it.
