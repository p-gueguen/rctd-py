# rctd-py

[![Tests](https://github.com/p-gueguen/rctd-py/actions/workflows/test.yml/badge.svg)](https://github.com/p-gueguen/rctd-py/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/rctd-py)](https://pypi.org/project/rctd-py/)
[![Python](https://img.shields.io/pypi/pyversions/rctd-py)](https://pypi.org/project/rctd-py/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![codecov](https://codecov.io/gh/p-gueguen/rctd-py/graph/badge.svg)](https://codecov.io/gh/p-gueguen/rctd-py)

JAX-accelerated Robust Cell Type Decomposition for spatial transcriptomics.

A Python reimplementation of the [spacexr](https://github.com/dmcable/spacexr) RCTD algorithm (Cable et al., 2022) with GPU acceleration via JAX. Deconvolves spatial transcriptomics data (Visium, Xenium, MERFISH, Slide-seq, etc.) into cell type proportions using a scRNA-seq reference.

## Installation

```bash
uv pip install rctd-py
```

Or with standard pip:

```bash
pip install rctd-py
```

With CUDA support (GPU acceleration):

```bash
uv pip install "rctd-py[cuda]"
```

For development:

```bash
git clone https://github.com/p-gueguen/rctd-py.git
cd rctd-py
uv pip install -e ".[dev]"
```

**Dependencies:** jax, jaxlib, numpy, scipy, anndata

## Quick Start

```python
from rctd import Reference, run_rctd
import anndata

# 1. Load reference scRNA-seq data
ref_adata = anndata.read_h5ad("reference.h5ad")
reference = Reference(ref_adata, cell_type_col="cell_type")

# 2. Load spatial data
spatial = anndata.read_h5ad("spatial.h5ad")

# 3. Run RCTD
result = run_rctd(spatial, reference, mode="doublet")
```

The `run_rctd` function handles the full pipeline: gene intersection, platform effect normalization, sigma estimation, and per-pixel deconvolution.

See the [tutorial notebook](examples/tutorial.ipynb) or the [rendered tutorial](https://p-gueguen.github.io/rctd-py/) for a complete walkthrough with synthetic data.

## Deconvolution Modes

RCTD supports three modes, selected via the `mode` parameter:

| Mode | Description | Use case |
|------|-------------|----------|
| `full` | Estimates weights for all K cell types per pixel using constrained IRWLS. | Continuous mixtures, Visium |
| `doublet` | Classifies each pixel as singlet or doublet, then estimates the top 1--2 cell type weights. Reports `spot_class` (singlet, doublet_certain, doublet_uncertain, reject). | Slide-seq, sparse spatial data |
| `multi` | Greedy forward selection of up to 4 cell types per pixel, adding types while the likelihood improves. | Dense spatial platforms (Xenium, MERFISH) |

## Benchmarks

### End-to-end (Xenium, 58k pixels, doublet mode)

Full pipeline on a 58k-pixel Xenium dataset (380 genes, 45 cell types):

| Backend | Sigma estimation | Deconvolution | Total |
|---------|-----------------|---------------|-------|
| R spacexr (8 CPU cores) | ~49 min | ~2 min | ~51 min |
| rctd-py — JAX GPU (Blackwell B200) | **~3 min** | ~36s | **~3.5 min** |
| rctd-py — JAX GPU (L40S) | ~3 min | ~55 min* | ~58 min |

*The L40S is a rendering/inference GPU (GDDR6, 864 GB/s) rather than an HPC card. Its memory bandwidth is ~9× lower than the B200 (HBM3e, ~8 TB/s), making the memory-bound doublet IRWLS loop much slower. On HPC-class GPUs (H100, A100, B200) the deconvolution step completes in under 1 minute.

**Sigma estimation** uses a Poisson-Lognormal model with cubic spline interpolation. After optimisation (cached matrix inverse, precomputed spline coefficients, vmapped JAX evaluation), sigma drops from ~66 min to ~3 min — a **~23× speedup** — and results are numerically identical.

### IRWLS solver only

Solver throughput measured on the spacexr vignette dataset (71 pixels, 313 genes, 19 cell types), scaled to larger pixel counts:

| Backend | Pixels/sec | Speedup vs R |
|---------|-----------|--------------|
| R spacexr (single-core) | ~62 | 1x |
| JAX CPU (16 threads) | ~374 | 6x |
| JAX GPU (L40S) | ~3,900 | 63x |
| JAX GPU (Blackwell B200) | ~4,450 | 72x |

GPU throughput saturates at ~3,900 pixels/sec on L40S at 7k+ pixels. JAX compilation overhead dominates at small pixel counts.

## Validation

Validated against R spacexr on a Xenium dataset (45 cell types, 380 genes, ~58k filtered pixels):

| Metric | Value |
|--------|-------|
| Dominant type agreement | **99.7%** |
| Median per-pixel weight correlation | **1.0000** |
| Mean per-pixel weight correlation | 0.9998 |
| Pixels with correlation > 0.8 | 99.98% |

Both implementations use identical parameters: `UMI_min=20`, doublet mode, `constrain=FALSE` for full-mode weight estimation. See the [validation report](https://p-gueguen.github.io/rctd-py/analysis_region3.html) for full details including spatial maps and per-type correlations.

## API Overview

### `run_rctd(spatial, reference, mode, config, batch_size)`

End-to-end pipeline. Takes an `AnnData` spatial object and a `Reference`, returns a typed result (`FullResult`, `DoubletResult`, or `MultiResult`).

### `Reference(adata, cell_type_col, cell_min, n_max_cells, min_UMI)`

Constructs cell type mean expression profiles from a scRNA-seq `AnnData`. Filters cell types below `cell_min` cells, caps per-type cells at `n_max_cells`, and removes cells below `min_UMI`.

### `RCTD(spatial, reference, config)`

Stateful class for step-by-step control. Call `fit_platform_effects()` to normalize, then use `run_full_mode`, `run_doublet_mode`, or `run_multi_mode` directly.

### `RCTDConfig`

Named tuple with all algorithm parameters. Key fields:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `UMI_min` | 100 | Minimum UMI count per spatial pixel |
| `UMI_min_sigma` | 300 | Minimum UMI for sigma estimation |
| `N_fit` | 1000 | Number of pixels for sigma fitting |
| `MAX_MULTI_TYPES` | 4 | Maximum cell types in multi mode |
| `CONFIDENCE_THRESHOLD` | 5.0 | Singlet confidence threshold (doublet mode) |
| `DOUBLET_THRESHOLD` | 20.0 | Doublet certainty threshold |
| `max_iter` | 50 | IRWLS maximum iterations |

### Result Types

- **`FullResult`**: `weights` (N x K), `cell_type_names`, `converged`
- **`DoubletResult`**: `weights` (N x K), `weights_doublet` (N x 2), `spot_class`, `first_type`, `second_type`, `first_class`, `second_class`, `min_score`, `singlet_score`, `cell_type_names`
- **`MultiResult`**: `weights` (N x K), `sub_weights`, `cell_type_indices`, `n_types`, `conf_list`, `min_score`, `cell_type_names`

## GPU Usage

JAX automatically detects available GPUs. To enable CUDA:

```bash
pip install "rctd-py[cuda]"
```

This installs `jax[cuda12]`. Verify GPU detection:

```python
import jax
print(jax.devices())  # [CudaDevice(id=0)]
```

Use the `batch_size` parameter in `run_rctd` to control GPU memory usage. The default (10,000 pixels per batch) works well for GPUs with 24+ GB VRAM.

## Project Structure

```
src/rctd/
  __init__.py        # Public API exports
  _types.py          # RCTDConfig, FullResult, DoubletResult, MultiResult
  _reference.py      # Reference class (profile computation, DE gene selection)
  _rctd.py           # RCTD class and run_rctd pipeline
  _normalize.py      # Platform effect estimation (fit_bulk)
  _sigma.py          # Sigma (overdispersion) estimation
  _likelihood.py     # Poisson-Lognormal model, Q-matrix interpolation
  _irwls.py          # Batched IRWLS solver (JAX jit + vmap)
  _simplex.py        # Simplex projection for constrained optimization
  _full.py           # Full mode deconvolution
  _doublet.py        # Doublet mode deconvolution
  _multi.py          # Multi mode deconvolution
```

## Contributing

Contributions are welcome! Please open an [issue](https://github.com/p-gueguen/rctd-py/issues) to discuss proposed changes or report bugs.

## Citation

If you use rctd-py, please cite the original spacexr RCTD paper:

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

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
