# Quick start

```python
import anndata
from rctd import Reference, run_rctd

reference = Reference(anndata.read_h5ad("reference.h5ad"), cell_type_col="cell_type")
spatial = anndata.read_h5ad("spatial.h5ad")

# normalisation, sigma estimation and deconvolution in one call
result = run_rctd(spatial, reference, mode="doublet")
```

Both inputs hold raw counts. The result is a typed tuple ({class}`~rctd.FullResult`,
{class}`~rctd.DoubletResult` or {class}`~rctd.MultiResult`) whose arrays cover the pixels that
passed the UMI filters; `spatial[result.pixel_mask]` lines up with them row for row.

```python
import numpy as np
from rctd import SPOT_CLASS_NAMES

names = np.asarray(result.cell_type_names)
fitted = spatial[result.pixel_mask].copy()
fitted.obs["spot_class"] = np.asarray(SPOT_CLASS_NAMES)[result.spot_class]
fitted.obs["first_type"] = names[result.first_type]
```

To get the results straight into the AnnData instead, with the same keys `rctd run` writes to its
output file, use {func}`~rctd.deconvolve` in place of `run_rctd`. It takes the same arguments,
writes into `spatial` in place (scanpy style; `copy=True` returns a new object) and marks filtered
pixels `"filtered"` / NaN:

```python
from rctd import deconvolve

deconvolve(spatial, reference, mode="doublet")
spatial.obs["rctd_first_type"]  # also rctd_spot_class, rctd_second_type, ...
spatial.obsm["rctd_weights"]  # pixels x cell types, columns in uns["rctd_cell_type_names"]
```

## Deconvolution modes

| Mode | What it does | Best for |
|---|---|---|
| `full` | Weights for all K cell types per pixel | Visium spots, continuous mixtures |
| `doublet` | Classifies each pixel as singlet or doublet and reports the top 1-2 types | Slide-seq, Visium HD bins, segmented cells (Xenium, MERFISH, CosMx) |
| `multi` | Greedy forward selection of up to 4 types per pixel | Spots expected to hold more than 2 types (an alternative to `full`) |

These follow R spacexr's own recommendations: doublet mode for high-resolution data, full or
multi mode for larger spots.

## Configuration

{class}`~rctd.RCTDConfig` holds every parameter; its defaults match R spacexr's (see
{doc}`r_spacexr`).

```python
from rctd import RCTDConfig

config = RCTDConfig(UMI_min=20, device="cuda", dtype="float32")
result = run_rctd(spatial, reference, mode="doublet", config=config, batch_size=5000)
```

On GPUs with limited fp64 throughput (L40S, RTX and Blackwell workstation cards),
`dtype="float32"` is usually faster and needs about half the VRAM on whole-transcriptome data. On
an L40S in doublet mode (0.4.0, CUDA 13) it was 1.3x faster on Xenium (K=45) and 1.9x on Visium HD (K=78), with 99.99% or more of the spot classes
unchanged. The default `float64` matches R's
arithmetic.

In full mode on a large dataset, `run_rctd(spatial, reference, mode="full", stream_gpu=True)`
copies the counts to the GPU one batch at a time instead of all at once, with the same output; see
[GPU memory](installation.md#gpu-memory).

## Hierarchical cell types (`class_df`)

When the reference has fine subtypes (for example several T cell states), doublet mode may not be
able to pick one subtype confidently. As in R spacexr, `class_df` maps each subtype to a parent
class; if only the class is certain, the call is flagged instead of forced:

```python
class_df = {
    "CD4_TH1": "T_cell",
    "CD4_TH2": "T_cell",
    "CD8": "T_cell",
    "B_naive": "B_cell",
    "B_memory": "B_cell",
    "Macrophage": "Macrophage",
}
result = run_rctd(spatial, reference, mode="doublet", config=RCTDConfig(class_df=class_df))

# result.first_class[n]      True: subtype ambiguous, trust the class label
# result.first_class_name[n] "T_cell"
# result.first_type[n]       still the best-scoring subtype
```

On the command line, pass a TSV with `cell_type` and `class` columns to `--class-df`.

## Step by step

{class}`~rctd.RCTD` exposes the stages `run_rctd` chains together, for example to inspect the
platform-effect fit before deconvolving:

```python
from rctd import RCTD

rctd = RCTD(spatial, reference, RCTDConfig())
rctd.fit_platform_effects()  # gene lists, bulk fit, sigma
print(rctd.sigma / 100)  # sigma chosen for this dataset
```
