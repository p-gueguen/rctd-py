# rctd-py

GPU-accelerated Robust Cell Type Decomposition (RCTD) for spatial transcriptomics.

rctd-py is a Python reimplementation of the [spacexr](https://github.com/dmcable/spacexr) RCTD
algorithm {cite:p}`cable2022robust`, with GPU acceleration through [PyTorch](https://pytorch.org/).
It deconvolves spatial transcriptomics pixels (Visium, Visium HD, Xenium, CosMx, MERFISH,
Slide-seq, ...) into cell type proportions using a single-cell RNA-seq reference, in minutes
instead of hours.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Installation
:link: installation
:link-type: doc

`pip install rctd-py`, GPU setup and offline (HPC) installs.
:::

:::{grid-item-card} Quick start
:link: quickstart
:link-type: doc

Run RCTD from Python in three lines, and pick a deconvolution mode.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

End-to-end walkthroughs on public data.
:::

:::{grid-item-card} Coming from R spacexr
:link: r_spacexr
:link-type: doc

Parameter mapping, agreement with R and how to reproduce an R run.
:::
::::

```{toctree}
:hidden: true
:maxdepth: 2
:caption: Getting started

installation
quickstart
tutorials/index
```

```{toctree}
:hidden: true
:maxdepth: 2
:caption: Using rctd-py

cli
r_spacexr
benchmarks
```

```{toctree}
:hidden: true
:maxdepth: 2
:caption: Reference

api
changelog
contributing
references
```
