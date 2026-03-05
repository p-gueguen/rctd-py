<p align="center">
  <h1 align="center">rctd-py</h1>
  <p align="center">
    <strong>GPU-accelerated spatial transcriptomics deconvolution — 4× faster than R</strong>
  </p>
  <p align="center">
    <a href="https://github.com/p-gueguen/rctd-py/actions/workflows/test.yml"><img src="https://github.com/p-gueguen/rctd-py/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
    <a href="https://pypi.org/project/rctd-py/"><img src="https://img.shields.io/pypi/v/rctd-py" alt="PyPI"></a>
    <a href="https://pypi.org/project/rctd-py/"><img src="https://img.shields.io/pypi/pyversions/rctd-py" alt="Python"></a>
    <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"></a>
    <a href="https://codecov.io/gh/p-gueguen/rctd-py"><img src="https://codecov.io/gh/p-gueguen/rctd-py/branch/main/graph/badge.svg" alt="codecov"></a>
  </p>
</p>

---

A Python reimplementation of the [spacexr](https://github.com/dmcable/spacexr) RCTD algorithm ([Cable et al., *Nature Biotechnology* 2022](https://doi.org/10.1038/s41587-021-00830-w)) with GPU acceleration via [PyTorch](https://pytorch.org/).

Deconvolve spatial transcriptomics spots (Visium, Xenium, MERFISH, Slide-seq, …) into cell type proportions using a scRNA-seq reference — in minutes instead of hours.

## ✨ Highlights

| | |
|---|---|
| 🚀 **4× end-to-end speedup** | Xenium 58k pixels: **12 min** (L40S GPU) vs 51 min (R, 8 CPU) |
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
pip install rctd-py   # CPU (works everywhere; GPU auto-detected if CUDA available)
```

<details>
<summary>GPU setup and CUDA compatibility</summary>

### Recommended setup

Install PyTorch with CUDA **before** installing rctd-py — `pip install rctd-py` alone pulls CPU-only PyTorch on most systems:

```bash
# CUDA 12.4 (recommended for drivers >= 550.54)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install rctd-py

# CUDA 12.1 (for older drivers >= 530.30)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (legacy, drivers >= 520.61)
pip install torch --index-url https://download.pytorch.org/whl/cu118
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

| GPU | VRAM | Speedup (58k pixels, K=45) |
|-----|------|---------------------------|
| NVIDIA L40S | 48 GB | 4.3× |

### Memory management

Use the `batch_size` parameter in `run_rctd` to control GPU memory usage:

| VRAM | Recommended `batch_size` |
|------|-------------------------|
| 24+ GB | 10,000 (default) |
| 8–16 GB | 5,000 |
| < 8 GB | 2,000 |

</details>

## Deconvolution Modes

| Mode | What it does | Best for |
|------|-------------|----------|
| **`full`** | Estimates weights for all K cell types per pixel (constrained IRWLS) | Visium, continuous mixtures |
| **`doublet`** | Classifies each pixel as singlet or doublet, estimates top 1–2 types | Slide-seq, sparse spatial |
| **`multi`** | Greedy forward selection of up to 4 cell types per pixel | Xenium, MERFISH, dense platforms |

## Benchmarks

### End-to-end performance (Xenium, 45 cell types, doublet mode, L40S GPU)

<p align="center">
  <img src="docs/benchmark.png" alt="Benchmark barplot" width="700">
</p>

| Dataset | Pixels | R spacexr (8 CPU) | rctd-py (L40S GPU) | **Speedup** |
|---------|--------|-------------------|---------------------|-------------|
| Region 3 | 58k | 51.1 min | 11.8 min | **4.3×** |
| Region 1 | 14k | 14.1 min | 3.5 min | **4.0×** |

> **Note:** The IRWLS solver loop is memory-bandwidth bound for large cell type panels (K=45). Speedup scales with the number of cell types — smaller panels (K < 20) see larger speedups.

## Validation

Validated against R spacexr on two Xenium datasets (45 cell types, 380 genes, doublet mode, `UMI_min=20`):

| Dataset | Pixels | Dominant type agreement | With `sigma_override` |
|---------|--------|------------------------|-----------------------|
| Region 1 | 13,940 | **99.73%** | **100%** |
| Region 3 | 58,191 | **99.71%** | — |

The tiny default gap (0.27%) traces entirely to platform-effect estimation (`fit_bulk`), not the per-pixel solver — which is bit-identical to R. All 37 disagreeing pixels are genuinely ambiguous (margin < 0.05 between top two types).

Use `sigma_override` to inject R's sigma value and achieve exact concordance:

```python
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
| `N_fit` | 1000 | Pixels for sigma fitting |
| `MAX_MULTI_TYPES` | 4 | Max cell types in multi mode |
| `CONFIDENCE_THRESHOLD` | 5.0 | Singlet confidence threshold |
| `DOUBLET_THRESHOLD` | 20.0 | Doublet certainty threshold |

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
