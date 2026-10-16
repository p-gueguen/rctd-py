<p align="center">
  <img src="https://raw.githubusercontent.com/p-gueguen/rctd-py/main/docs/logo_max.png" alt="rctd-py logo" width="500">
  <h1 align="center">rctd-py</h1>
  <p align="center">
    <strong>Spatial transcriptomics deconvolution, 10–57x faster than R spacexr on a GPU (doublet mode)</strong>
  </p>
  <p align="center">
    <a href="https://github.com/p-gueguen/rctd-py/actions/workflows/ci.yml"><img src="https://github.com/p-gueguen/rctd-py/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://pypi.org/project/rctd-py/"><img src="https://img.shields.io/pypi/v/rctd-py?cacheSeconds=3600" alt="PyPI"></a>
    <a href="https://pypi.org/project/rctd-py/"><img src="https://img.shields.io/pypi/pyversions/rctd-py?cacheSeconds=3600" alt="Python"></a>
    <a href="https://rctd-py.readthedocs.io"><img src="https://readthedocs.org/projects/rctd-py/badge/?version=latest" alt="Documentation"></a>
    <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"></a>

  </p>
</p>

> **Latest: v0.4.0**: rctd-py now makes the same calls as R spacexr wherever both pick the same sigma, after fixes to the bulk platform-effect fit, singlet labels, cell-type order, sigma sampling (`N_fit` 1000, as in R) and the QP solver. Calls can change from 0.3.9, most on sparse sequencing data (Visium, Visium HD, StrataMap); pin `rctd-py==0.3.9` to keep old results. See the [changelog](https://github.com/p-gueguen/rctd-py/blob/main/CHANGELOG.md).

---

A Python reimplementation of the [spacexr](https://github.com/dmcable/spacexr) RCTD algorithm ([Cable et al., *Nature Biotechnology* 2022](https://doi.org/10.1038/s41587-021-00830-w)) with GPU acceleration via [PyTorch](https://pytorch.org/).

Deconvolve spatial transcriptomics spots (Visium, Xenium, MERFISH, Slide-seq, …) into cell type proportions using a scRNA-seq reference — in minutes instead of hours.

**Documentation: [rctd-py.readthedocs.io](https://rctd-py.readthedocs.io)** (installation, tutorials, CLI, API reference, coming from R spacexr)

## ✨ Highlights

| | |
|---|---|
| ⚡ **10–57x faster than R spacexr** | Doublet mode on 19 datasets, one GPU vs 8 R cores. Xenium Prime 5K, 503k cells fitted: **34 min** vs 31 hours in R |
| 🎯 **Matches R spacexr** | Same calls as R wherever both pick the same sigma (11 of 19 datasets); 99.0–99.9% of spot classes on the rest, where sigma differs by 0.01–0.02, as it does between two R runs ([details](https://rctd-py.readthedocs.io/en/latest/r_spacexr.html)) |
| 🔧 **Familiar to spacexr users** | Same algorithm and, with a few exceptions, the same parameter names ([mapping](https://rctd-py.readthedocs.io/en/latest/r_spacexr.html)) |
| 📦 **`pip install rctd-py`** | Pure Python, works on CPU out of the box (3.7–14x faster than R in doublet mode), uses a GPU when there is one |
| 🧬 **Whole-transcriptome ready** | 10x **Atera** WTA (18,028-gene panel): 714k cells deconvolved in 9 min on GPU (doublet mode, 7 cell types) |

## Installation

```bash
uv pip install rctd-py   # or: pip install rctd-py
```

GPU builds of PyTorch, older NVIDIA drivers and offline / HPC installs (the 404 MB Q-matrices
file) are covered in the [installation guide](https://rctd-py.readthedocs.io/en/latest/installation.html).

## Quick Start

```python
from rctd import Reference, run_rctd
import anndata

reference = Reference(anndata.read_h5ad("reference.h5ad"), cell_type_col="cell_type")
spatial = anndata.read_h5ad("spatial.h5ad")

# normalisation, sigma estimation and deconvolution in one call
result = run_rctd(spatial, reference, mode="doublet")
```

| Mode | What it does | Best for |
|------|-------------|----------|
| **`full`** | Weights for all K cell types per pixel | Visium spots, continuous mixtures |
| **`doublet`** | Singlet or doublet call per pixel, top 1–2 types | Slide-seq, Visium HD bins, segmented cells (Xenium, MERFISH, CosMx) |
| **`multi`** | Greedy selection of up to 4 types per pixel | Spots expected to hold more than 2 types (an alternative to `full`) |

📓 [Tutorial](https://rctd-py.readthedocs.io/en/latest/tutorials/slideseq_cerebellum.html) ([marimo notebook](https://github.com/p-gueguen/rctd-py/blob/main/examples/tutorial.py)) · 🤖 [Claude Code skill](https://github.com/p-gueguen/rctd-py/blob/main/.claude/skills/rctd-py/SKILL.md) (copy the `.claude/skills/rctd-py/` folder into `~/.claude/skills/` for assisted coding with rctd-py)

## Command Line

```bash
rctd info                                                # versions, GPU availability
rctd validate spatial.h5ad reference.h5ad                # fast input checks
rctd run spatial.h5ad reference.h5ad --mode doublet -o results.h5ad
rctd run experiment.zarr reference.h5ad --table cell_table   # SpatialData stores work too
```

All options and the output layout: [command line reference](https://rctd-py.readthedocs.io/en/latest/cli.html).

## Benchmarks

<p align="center">
  <img src="https://raw.githubusercontent.com/p-gueguen/rctd-py/main/docs/benchmark.png" alt="Doublet mode wall time of R spacexr, rctd-py CPU and rctd-py GPU on 19 datasets" width="900">
</p>

rctd-py on one NVIDIA L40S or 8 CPU threads, R spacexr 2.2.1 on 8 cores; rctd-py points are the
median of 3 runs. Full and multi mode, per-dataset tables and memory: [benchmarks](https://rctd-py.readthedocs.io/en/latest/benchmarks.html).

## Citation

If you use rctd-py, please cite the original RCTD paper:

```bibtex
@article{cable2022robust,
  title={Robust decomposition of cell type mixtures in spatial transcriptomics},
  author={Cable, Dylan M and Murray, Evan and Zou, Luli S and Goeva, Aleksandrina and Macosko, Evan Z and Chen, Fei and Irizarry, Rafael A},
  journal={Nature Biotechnology},
  volume={40},
  number={4},
  pages={517--526},
  year={2022},
  doi={10.1038/s41587-021-00830-w}
}
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](https://github.com/p-gueguen/rctd-py/blob/main/CONTRIBUTING.md) for setup instructions, or open an [issue](https://github.com/p-gueguen/rctd-py/issues).

## License

[GNU General Public License v3.0](https://github.com/p-gueguen/rctd-py/blob/main/LICENSE)
