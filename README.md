<p align="center">
  <h1 align="center">rctd-py</h1>
  <p align="center">
    <strong>GPU-accelerated spatial transcriptomics deconvolution — up to 8x faster than R</strong>
  </p>
  <p align="center">
    <a href="https://github.com/p-gueguen/rctd-py/actions/workflows/ci.yml"><img src="https://github.com/p-gueguen/rctd-py/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://pypi.org/project/rctd-py/"><img src="https://img.shields.io/pypi/v/rctd-py" alt="PyPI"></a>
    <a href="https://pypi.org/project/rctd-py/"><img src="https://img.shields.io/pypi/pyversions/rctd-py" alt="Python"></a>
    <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"></a>

  </p>
</p>

---

A Python reimplementation of the [spacexr](https://github.com/dmcable/spacexr) RCTD algorithm ([Cable et al., *Nature Biotechnology* 2022](https://doi.org/10.1038/s41587-021-00830-w)) with GPU acceleration via [PyTorch](https://pytorch.org/).

Deconvolve spatial transcriptomics spots (Visium, Xenium, MERFISH, Slide-seq, …) into cell type proportions using a scRNA-seq reference — in minutes instead of hours.

## ✨ Highlights

| | |
|---|---|
| 🚀 **6–15x end-to-end speedup** | Xenium 58k cells: **6.6 min** (GPU) vs 51 min (R, 8 CPU) |
| 🎯 **99.7% concordance** with R spacexr | **100%** with `sigma_override` — per-pixel solver is bit-identical |
| 🔧 **Drop-in replacement** | Same algorithm, same parameters, same results — just faster |
| 📦 **`pip install rctd-py`** | Pure Python, works on CPU out of the box |

## Quick Start

```python
from rctd import Reference, run_rctd
import anndata

# Load data
reference = Reference(anndata.read_h5ad("reference.h5ad"), cell_type_col="cell_type")
spatial = anndata.read_h5ad("spatial.h5ad")

# Run RCTD — handles normalization, sigma estimation, and deconvolution
result = run_rctd(spatial, reference, mode="doublet")
```

📓 **[Tutorial notebook](examples/tutorial.py)** (marimo) · 🌐 **[Rendered tutorial](https://p-gueguen.github.io/rctd-py/)**

## Installation

```bash
uv pip install rctd-py   # or: pip install rctd-py
```

<details>
<summary>GPU setup and CUDA compatibility</summary>

### Recommended setup

Install PyTorch with CUDA **before** installing rctd-py — `pip install rctd-py` alone pulls CPU-only PyTorch on most systems:

```bash
# CUDA 12.4 (recommended for drivers >= 550.54)
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install rctd-py

# CUDA 12.1 (for older drivers >= 530.30)
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (legacy, drivers >= 520.61)
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU detection

```python
import torch
print(torch.cuda.is_available())    # True  (False means CPU-only torch or driver issue)
print(torch.cuda.get_device_name()) # e.g. 'NVIDIA L40S'
print(torch.version.cuda)           # e.g. '12.4'
```

### CUDA compatibility table

**No separate CUDA toolkit installation needed.** PyTorch ships its own CUDA runtime — you only need a compatible NVIDIA driver.

| PyTorch version | Bundled CUDA | Minimum NVIDIA driver |
|-----------------|-------------|----------------------|
| 2.5+ | CUDA 12.4 | >= 550.54 |
| 2.3–2.4 | CUDA 12.1 | >= 530.30 |
| 2.0–2.2 | CUDA 11.8 | >= 520.61 |

> **Tip:** Check your driver version with `nvidia-smi` (top right of the output). This is the *driver* version, not the CUDA toolkit version — `nvcc --version` shows the toolkit version, which is irrelevant here since PyTorch bundles its own runtime.

### Tested GPUs

| GPU | VRAM | Speedup (58k cells, K=45) |
|-----|------|---------------------------|
| NVIDIA RTX PRO 6000 Blackwell | 96 GB | 7.7x (K=45) / 15.1x (K=22) |
| NVIDIA L40S | 48 GB | 4.2x (K=45) |

### Memory management

Peak VRAM scales with `batch_size * K^2`. Use the `batch_size` parameter to control GPU memory:

| Available VRAM | Recommended `batch_size` | Peak VRAM (K=45) |
|----------------|-------------------------|-------------------|
| 24+ GB | 10,000 (default) | ~4 GB |
| 8–16 GB | 5,000 | ~2 GB |
| < 8 GB | 2,000 | ~1 GB |

Peak CPU RAM (RSS) is typically 2–3x peak VRAM, dominated by intermediate arrays.

</details>

## Deconvolution Modes

| Mode | What it does | Best for |
|------|-------------|----------|
| **`full`** | Estimates weights for all K cell types per pixel (constrained IRWLS) | Visium, continuous mixtures |
| **`doublet`** | Classifies each pixel as singlet or doublet, estimates top 1–2 types | Slide-seq, sparse spatial |
| **`multi`** | Greedy forward selection of up to 4 cell types per pixel | Xenium, MERFISH, dense platforms |

## Benchmarks

Benchmarked on 3 datasets across all RCTD modes (full, doublet, multi) on an NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM) vs R spacexr with 8 CPU cores.

<p align="center">
  <img src="docs/benchmark.png" alt="Benchmark: runtime scalability and memory curves" width="800">
</p>

### Runtime (doublet mode)

| Dataset | Cells | K | R spacexr (8 CPU) | rctd-py (GPU) | Speedup |
|---------|-------|---|-------------------|---------------|---------|
| Xenium Region 1 | 13,940 | 45 | 14.1 min | 2.4 min | **6.0x** |
| Mouse Brain (CTX_HP) | 36,362 | 22 | 81.9 min | 5.4 min | **15.1x** |
| Xenium Region 3 | 58,191 | 45 | 51.1 min | 6.6 min | **7.7x** |

### Memory requirements

| Dataset | Cells | K | Peak VRAM | Peak RSS |
|---------|-------|---|-----------|----------|
| Xenium Region 1 | 13,940 | 45 | 2.6 GB | 34 GB |
| Mouse Brain | 36,362 | 22 | 2.6 GB | 5 GB |
| Xenium Region 3 | 58,191 | 45 | 2.6 GB | 34 GB |

Peak VRAM is ~2.6 GB across all tested datasets (doublet mode, default batch size). RSS is dominated by the reference matrix and scales with K. Use the `batch_size` parameter to control peak VRAM — smaller batches trade throughput for lower memory.

> **Note:** Speedup depends strongly on K (number of cell types). Smaller panels (K < 25) see 10–15x speedups because GPU-accelerated eigendecomposition handles all pairwise fits. Larger panels (K > 40) see 6–8x speedups, limited by CPU eigendecomposition for the K-dimensional full-mode fit.

## Validation

Validated against R spacexr on two Xenium datasets (45 cell types, 380 genes, doublet mode, `UMI_min=20`):

| Dataset | # cells | Dominant type agreement | With `sigma_override` |
|---------|--------|------------------------|-----------------------|
| Xenium (small) | 13,940 | **99.73%** | **100%** |
| Xenium (large) | 58,191 | **99.71%** | — |

The tiny default gap (0.27%) traces entirely to platform-effect estimation (`fit_bulk`), not the per-pixel solver — which is bit-identical to R. All disagreeing pixels are genuinely ambiguous (margin < 0.05 between top two types).

**`sigma_override` is not needed for normal use.** The default Python-estimated sigma is valid and produces near-identical results. It exists for specific scenarios:

- **Validation** — proving solver equivalence with R
- **Migration** — replicating exact R spacexr results when you already have R's sigma
- **Reproducibility** — locking sigma to a known value across runs

```python
# Only if you need exact R concordance and know R's sigma value:
result = run_rctd(spatial, reference, mode="doublet", sigma_override=62)
```

## API

<details>
<summary><strong>Click to expand full API reference</strong></summary>

### `run_rctd(spatial, reference, mode, config, batch_size, sigma_override)`

End-to-end pipeline. Takes an `AnnData` spatial object and a `Reference`, returns a typed result (`FullResult`, `DoubletResult`, or `MultiResult`). Pass `sigma_override` (int) to skip sigma estimation and use a known value (e.g. from R).

### `Reference(adata, cell_type_col, cell_min, n_max_cells, min_UMI)`

Constructs cell type profiles from a scRNA-seq `AnnData`. Filters cell types below `cell_min`, caps per-type cells at `n_max_cells`.

### `RCTD(spatial, reference, config)`

Stateful class for step-by-step control. Call `fit_platform_effects()`, then `run_full_mode`, `run_doublet_mode`, or `run_multi_mode`.

### `RCTDConfig` — key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `UMI_min` | 100 | Minimum UMI count per pixel |
| `UMI_min_sigma` | 300 | Minimum UMI for sigma estimation |
| `N_fit` | 100 | # pixels sampled for sigma fitting |
| `MAX_MULTI_TYPES` | 4 | Max cell types in multi mode |
| `CONFIDENCE_THRESHOLD` | 5.0 | Singlet confidence threshold |
| `DOUBLET_THRESHOLD` | 20.0 | Doublet certainty threshold |
| `device` | `"auto"` | `"auto"`, `"cpu"`, or `"cuda"` — force CPU/GPU |

### Result types

- **`FullResult`** — `weights` (N×K), `cell_type_names`, `converged`
- **`DoubletResult`** — `weights`, `weights_doublet` (N×2), `spot_class`, `first_type`, `second_type`
- **`MultiResult`** — `weights`, `cell_type_indices`, `n_types`, `conf_list`

</details>

## Citation

If you use rctd-py, please cite the original RCTD paper:

```bibtex
@article{cable2022robust,
  title={Robust decomposition of cell type mixtures in spatial transcriptomics},
  author={Cable, Dylan M and Murray, Evan and Zou, Luli S and Goeva, Aleksandrina and Macosko, Evan Z and Chen, Fei and Bhatt, Shreya and Denber, Hannah S and others},
  journal={Nature Biotechnology},
  volume={40},
  pages={517--526},
  year={2022},
  doi={10.1038/s41587-021-00830-w}
}
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, or open an [issue](https://github.com/p-gueguen/rctd-py/issues).

## License

[GNU General Public License v3.0](LICENSE)
