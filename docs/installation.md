# Installation

```bash
uv pip install rctd-py   # or: pip install rctd-py
```

rctd-py runs on CPU out of the box and uses a GPU automatically when PyTorch can see one.

## Q-matrices (offline and HPC setups)

The first run needs `q_matrices.npz`, the precomputed Poisson-Lognormal likelihood tables for
126 sigma values from 0.10 to 2.00. It is 404 MB and already compressed, so it ships as a
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

macOS and Windows get a CPU build from PyPI, and rctd-py runs on the CPU there (Apple MPS is not
used); for other combinations use the selector
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

Peak VRAM grows with `batch_size` and with the number of fitted genes and cell types. By default
the whole spatial count matrix (pixels x fitted genes) is also copied to the GPU, so on large
datasets it grows with the number of pixels too: for 557,300 Visium HD bins and 4,638 fitted
genes, that matrix alone is about 19 GiB in float64. `dtype="float32"` about halves the count
matrix and the solver's working memory.

In full mode, `stream_gpu=True` keeps the count matrix in host memory and copies one batch at a
time to the GPU, so that part follows `batch_size` instead of the number of pixels. From Python,
`run_rctd(..., mode="full", stream_gpu=True)`; on the command line,
`rctd run --mode full --stream-gpu`. On Visium HD (103k bins, 4,041 fitted genes, 79 types,
float64, NVIDIA L40S) it lowered the full-mode peak from 6.9 GB to 4.0 GB at `batch_size=10000`
with no change in run time, and to about 1.1 GB at `batch_size=1024`. The output is bit-identical
to a run without it at the same `batch_size`. Doublet and multi mode always copy the whole count
matrix and reject `stream_gpu`.

From Python, `run_rctd` defaults to `batch_size="auto"`, sized from the free GPU memory; the CLI
defaults to `--batch-size 10000`. At `batch_size=10000` in float64, the
{doc}`benchmarks <benchmarks>` (0.4.0) peaked at 5.3-5.6 GiB on the small Xenium panels (242-376
genes), from multi mode, and at 7-16 GiB on the other datasets (up to 714,535 pixels). 0.4.1
brings the Xenium multi-mode peak to about 1 GiB (5,378 to 945 MiB on Xenium, float64). Set
`batch_size` explicitly to fit a smaller card or a shared GPU, or, in full mode, use `stream_gpu`
first:

| Available VRAM | Recommended `batch_size` |
|---|---|
| 24+ GB | 10,000 |
| 8-16 GB | 5,000 |
| < 8 GB | 2,000 |

Peak host RAM ranged from 4 to 136 GiB on the benchmark datasets and is highest on large
whole-transcriptome data: 136 GiB for 557,300 Visium HD bins, 113 GiB for 714,535 Atera cells.

PyTorch's CUDA caching allocator may reserve substantially more memory than is currently allocated by live tensors, especially on large workloads with changing allocation sizes. If reserved memory is much higher than allocated memory, setting

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

can reduce allocator fragmentation. On the tested Visium HD doublet workload in float32, this reduced peak reserved memory from 6.6 GB to 4.6 GB without changing the output. This affects allocator reservation and fragmentation; it does not reduce the memory required by live tensors themselves.

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
