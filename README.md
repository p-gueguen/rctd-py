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
| 🚀 **4× end-to-end speedup** | 58k-pixel Xenium dataset: **12 min** (GPU) vs 51 min (R) |
| 🎯 **99.7% concordance** with R spacexr | Median per-pixel weight correlation = **1.0000** |
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

📓 **[Tutorial notebook](examples/tutorial.ipynb)** · 🌐 **[Rendered tutorial](https://p-gueguen.github.io/rctd-py/)**

## Installation

```bash
pip install rctd-py   # CPU (works everywhere; GPU auto-detected if CUDA available)
```

<details>
<summary>Verify GPU detection</summary>

```python
import torch
print(torch.cuda.is_available())    # True
print(torch.cuda.get_device_name()) # e.g. 'NVIDIA L40S'
```

Use the `batch_size` parameter in `run_rctd` to control GPU memory. The default (10,000 pixels/batch) works well for 24+ GB VRAM.

</details>

## Deconvolution Modes

| Mode | What it does | Best for |
|------|-------------|----------|
| **`full`** | Estimates weights for all K cell types per pixel (constrained IRWLS) | Visium, continuous mixtures |
| **`doublet`** | Classifies each pixel as singlet or doublet, estimates top 1–2 types | Slide-seq, sparse spatial |
| **`multi`** | Greedy forward selection of up to 4 cell types per pixel | Xenium, MERFISH, dense platforms |

## Benchmarks

### End-to-end performance (Xenium, 58k pixels, 45 cell types, doublet mode)

| Backend | Sigma | Deconvolution | **Total** | **Speedup** |
|---------|-------|---------------|-----------|-------------|
| R spacexr (8 CPU cores) | ~49 min | ~2 min | ~51 min | 1× |
| **rctd-py — PyTorch GPU (L40S)** | **21s** | **~12 min** | **~12 min** | **4.2×** |
| **rctd-py — PyTorch GPU (RTX 6000 Blackwell)** | **21s** | **~11 min** | **~12 min** | **4.3×** |

> **Note:** The IRWLS solver loop is memory-bandwidth bound for large cell type panels (K=45). Speedup scales with the number of cell types — smaller panels (K < 20) see larger speedups.

## Validation

Validated against R spacexr on a Xenium dataset (45 cell types, 380 genes, ~58k pixels):

| Metric | Value |
|--------|-------|
| Dominant type agreement | **99.7%** |
| Median per-pixel weight correlation | **1.0000** |
| Mean per-pixel weight correlation | 0.9998 |
| Pixels with correlation > 0.8 | 99.98% |

Both implementations use identical parameters (`UMI_min=20`, doublet mode, `constrain=FALSE`).

## API

<details>
<summary><strong>Click to expand full API reference</strong></summary>

### `run_rctd(spatial, reference, mode, config, batch_size)`

End-to-end pipeline. Takes an `AnnData` spatial object and a `Reference`, returns a typed result (`FullResult`, `DoubletResult`, or `MultiResult`).

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
