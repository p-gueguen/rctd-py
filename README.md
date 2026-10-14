<p align="center">
  <img src="docs/logo_max.png" alt="rctd-py logo" width="500">
  <h1 align="center">rctd-py</h1>
  <p align="center">
    <strong>Spatial transcriptomics deconvolution — 4–41x faster than R spacexr with GPU</strong>
  </p>
  <p align="center">
    <a href="https://github.com/p-gueguen/rctd-py/actions/workflows/ci.yml"><img src="https://github.com/p-gueguen/rctd-py/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://pypi.org/project/rctd-py/"><img src="https://img.shields.io/pypi/v/rctd-py?cacheSeconds=3600" alt="PyPI"></a>
    <a href="https://pypi.org/project/rctd-py/"><img src="https://img.shields.io/pypi/pyversions/rctd-py?cacheSeconds=3600" alt="Python"></a>
    <a href="https://rctd-py.readthedocs.io"><img src="https://readthedocs.org/projects/rctd-py/badge/?version=latest" alt="Documentation"></a>
    <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"></a>

  </p>
</p>

> **Latest: v0.3.9** — the CLI reads SpatialData / AnnData Zarr stores directly (`--table` picks one of several tables, #28), and `RCTD_Q_MATRICES` points offline / HPC installs at a staged copy of the Q-matrices (#29). See [CHANGELOG.md](CHANGELOG.md).

---

A Python reimplementation of the [spacexr](https://github.com/dmcable/spacexr) RCTD algorithm ([Cable et al., *Nature Biotechnology* 2022](https://doi.org/10.1038/s41587-021-00830-w)) with GPU acceleration via [PyTorch](https://pytorch.org/).

Deconvolve spatial transcriptomics spots (Visium, Xenium, MERFISH, Slide-seq, …) into cell type proportions using a scRNA-seq reference — in minutes instead of hours.

**Documentation: [rctd-py.readthedocs.io](https://rctd-py.readthedocs.io)** (installation, tutorials, CLI, API reference, coming from R spacexr)

## ✨ Highlights

| | |
|---|---|
| ⚡ **4–41x faster than R spacexr** | Xenium 58k cells, multi mode: **3.7 min** vs 2.6 hours in R |
| 🎯 **Agrees with R spacexr** | 99.2–99.7% dominant-type agreement; most of the gap is sigma, which R itself estimates differently run to run ([details](https://rctd-py.readthedocs.io/en/latest/r_spacexr.html)) |
| 🔧 **Drop-in replacement** | Same algorithm and parameter names as spacexr |
| 📦 **`pip install rctd-py`** | Pure Python, works on CPU out of the box, uses a GPU when there is one |
| 🧬 **Whole-transcriptome ready** | 10x **Atera** WTA (18,028 genes): 714k cells deconvolved in ~10 min on GPU |

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
| **`doublet`** | Singlet or doublet call per pixel, top 1–2 types | Slide-seq, Visium HD bins, segmented cells |
| **`multi`** | Greedy selection of up to 4 types per pixel | Xenium, MERFISH, dense platforms |

📓 [Tutorial](https://rctd-py.readthedocs.io/en/latest/tutorials/slideseq_cerebellum.html) ([marimo notebook](examples/tutorial.py)) · 🤖 [Claude Code skill](.claude/skills/rctd-py/SKILL.md) (drop into `~/.claude/skills/` for assisted coding with rctd-py)

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
  <img src="docs/benchmark.png" alt="Benchmark: CPU vs GPU scalability, runtime, and memory" width="900">
</p>

Runtime and memory tables per dataset and mode: [benchmarks](https://rctd-py.readthedocs.io/en/latest/benchmarks.html).

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

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, or open an [issue](https://github.com/p-gueguen/rctd-py/issues).

## License

[GNU General Public License v3.0](LICENSE)
