---
name: rctd-py
description: GPU-accelerated Python RCTD cell type deconvolution for spatial transcriptomics (Visium, VisiumHD, Xenium, MERFISH, Slide-seq). Use when running rctd-py, when a Python alternative to R spacexr is needed, when deconvolving spatial data on GPU, when preparing h5ad references from Seurat/scanpy/CELLxGENE, or when integrating rctd-py results back into a Seurat/AnnData workflow. Covers the CLI (`rctd run`, `rctd validate`, `rctd info`), the Python API (`run_rctd`, `Reference`, `RCTDConfig`), all three modes (doublet / multi / full), GPU dtype selection, per-arch eigh tuning, and result interpretation. Triggers on "rctd-py", "python rctd", "GPU RCTD", "spacexr python", "h5ad deconvolution", "spatial cell type deconvolution python", "convert spacexr reference to h5ad".
---

# rctd-py: GPU-Accelerated Spatial Deconvolution

Python reimplementation of RCTD (Cable et al., *Nature Biotechnology* 2022) with PyTorch GPU acceleration. ~4-41× faster than R `spacexr` with >99% cell-type-call concordance on identical pixel sets. Pure GPU implementation -- no R runtime required.

## When to use

- Cell-type deconvolution on spatial transcriptomics (Visium, VisiumHD, Xenium, MERFISH, Slide-seq) where R `spacexr` is too slow or impractical to integrate into a Python pipeline.
- Large datasets (>50k spots) on GPU.
- Workflows that read/write h5ad and want to keep the deconvolution in-process with scanpy/anndata.
- Validating R `spacexr` results against a faster Python reference.

For the R `spacexr` workflow itself, use that package directly; this skill assumes you have already decided on the Python path.

## Install

```bash
# 1. PyTorch with CUDA matching your driver (one of):
uv pip install torch                                                # CPU only
uv pip install torch --index-url https://download.pytorch.org/whl/cu126  # CUDA 12.6
uv pip install torch --index-url https://download.pytorch.org/whl/cu128  # CUDA 12.8

# 2. rctd-py
uv pip install rctd-py

# 3. Sanity check
rctd info
```

`rctd info` prints torch version + CUDA availability + GPU name. If `cuda: False` is reported on a machine with a GPU, the wrong torch wheel was installed -- reinstall from the CUDA-matched index URL.

## Quick start: CLI

```bash
# Pre-flight (fast, no GPU)
rctd validate spatial.h5ad reference.h5ad --cell-type-col cell_type

# Run
rctd run spatial.h5ad reference.h5ad \
    --mode doublet \
    --device cuda \
    --umi-min 20 \
    --output results.h5ad \
    --json
```

`--json` prints a one-line summary on completion (mode, pixel count, runtime, output path). Default output is `<input_stem>_rctd.h5ad`.

## Quick start: Python API

```python
import anndata
from rctd import Reference, RCTDConfig, run_rctd

spatial = anndata.read_h5ad("spatial.h5ad")
ref = Reference(anndata.read_h5ad("reference.h5ad"), cell_type_col="cell_type")

config = RCTDConfig(device="cuda", UMI_min=20, dtype="float32")
result = run_rctd(spatial, ref, mode="doublet", config=config, batch_size=10000)
# DoubletResult: .weights, .weights_doublet, .spot_class, .first_type, .second_type,
#                .first_class, .second_class, .singlet_score, .min_score, .cell_type_names
```

For results stored back into the spatial AnnData, use the CLI -- it does the merge for you. From Python, see "Result interpretation" below.

## Mode selection

| Mode | Best for | What it returns |
|------|----------|-----------------|
| `doublet` (default) | Xenium, MERFISH, Slide-seq, any single-cell-resolution data | 1-2 cell types per spot with confidence class |
| `multi` | Dense spatial with several contributing types (Xenium with heavy contamination) | Up to `MAX_MULTI_TYPES` (default 4) types per spot via greedy forward selection |
| `full` | Visium, VisiumHD, anything with multi-cell spots | Continuous weight vector over all K types (unconstrained LS) |

For single-cell-resolution platforms, prefer `doublet`. Reach for `multi` only when a domain-specific reason makes >2 types per spot biologically expected.

## Reference preparation

Reference inputs are `AnnData` objects with:

- `.X` = **raw integer counts** (cells × genes). Not normalized, not log-transformed.
- `.obs[cell_type_col]` = string cell-type label.

Per-type subsampling to ~500 cells is recommended (more cells rarely improves the fit and slows reference loading).

### From CZ CELLxGENE via `gget`

```python
import anndata
import gget

gget.setup("cellxgene")  # one-time
gget.cellxgene(
    species="mus_musculus", tissue="cortex", disease="normal",
    out="reference.h5ad",
)
ref = anndata.read_h5ad("reference.h5ad")
# gget returns Ensembl IDs in var_names — remap to gene symbols
ref.var_names = ref.var["feature_name"].values
ref.var_names_make_unique()
ref.write_h5ad("reference.h5ad")
```

### From Seurat (R → h5ad)

```r
library(anndataR)

seurat_ref <- readRDS("reference.rds")          # or qs2::qs_read
Idents(seurat_ref) <- "cell_type"
seurat_ref <- subset(seurat_ref, downsample = 500)

adata <- as_anndata(seurat_ref, x_layer = "counts")  # MUST be raw counts
adata$obs$cell_type <- as.character(seurat_ref$cell_type)
write_h5ad(adata, "reference.h5ad")
```

### From a pre-built `spacexr::Reference` `.rds`

The `.rds` may be either a bare `spacexr::Reference` S4 object or a list containing one as `$reference`. Handle both:

```r
library(spacexr)
library(anndata)

obj <- readRDS("spacexr_reference.rds")
ref <- if (is.list(obj) && !is.null(obj$reference)) obj$reference else obj

# spacexr stores genes × cells; AnnData expects cells × genes
counts <- t(ref@counts)
obs <- data.frame(
    cell_type = as.character(ref@cell_types),
    nUMI = ref@nUMI,
    row.names = names(ref@cell_types)
)
write_h5ad(AnnData(X = counts, obs = obs), "reference.h5ad")
```

### From existing scanpy / h5ad

```python
import anndata
adata = anndata.read_h5ad("reference.h5ad")
assert "cell_type" in adata.obs.columns
# If raw counts live in a layer rather than .X:
# adata.X = adata.layers["counts"]
```

## CLI flag reference (`rctd run`)

| Flag | Default | Notes |
|------|---------|-------|
| `--mode` | `doublet` | `full`, `doublet`, `multi` |
| `--output / -o` | `<stem>_rctd.h5ad` | Output h5ad path |
| `--device` | `auto` | `auto`, `cpu`, `cuda` |
| `--batch-size` | `auto` | Pixels per IRWLS launch; `auto` sizes from free VRAM |
| `--dtype` | `float64` | `float32` is ~2× faster on consumer GPUs (see GPU note below) with identical cell-type calls |
| `--cell-type-col` | `cell_type` | Reference `obs` column with labels |
| `--umi-min` | `100` | **Use 20 for Xenium**; 100 is the R `spacexr` default |
| `--umi-max` | `20000000` | Per-pixel UMI ceiling |
| `--sigma-override` | _(auto)_ | Skip sigma estimation, force a fixed value (use for exact R concordance testing) |
| `--gene-cutoff` | `0.000125` | Bulk gene expression threshold |
| `--fc-cutoff` | `0.5` | Bulk fold-change threshold |
| `--gene-cutoff-reg` | `0.0002` | DE gene expression threshold |
| `--fc-cutoff-reg` | `0.75` | DE fold-change threshold |
| `--max-multi-types` | `4` | Max types per spot (`multi` mode) |
| `--confidence-threshold` | `5.0` | Singlet confidence margin |
| `--doublet-threshold` | `20.0` | Doublet certainty margin |
| `--cell-min` | `25` | Min cells per reference type |
| `--n-max-cells` | `10000` | Max cells per reference type (downsampling) |
| `--eigh-threshold` | _(arch default)_ | Force the K cutoff for the GPU-vs-CPU eigh dispatch (issue #22 diagnostic) |
| `--json` | off | Machine-readable summary |
| `--quiet / -q` | off | Suppress progress output |

`RCTDConfig` exposes the same fields under the Python API; consult `_types.py` for the canonical list.

## GPU notes

### `dtype="float32"` is usually the better default on GPU

The default `dtype="float64"` matches R `spacexr` arithmetic and is the safest choice when validating against R. On a data-center GPU with strong fp64 (A100, H100) the cost is small. On consumer / workstation GPUs (RTX, L40S, anything Blackwell sm_120, anything pre-Hopper) fp64 throughput is 1/32-1/64 of fp32, and `dtype="float32"` is roughly **2× faster on doublet mode** with the same cell-type calls. Empirically (L40S, sm_89, N=20000 spots, K=30 types): 76 s → 39 s; `spot_class` agreement with the fp64 run is **100%**, weights diff `<1e-6`. See `tests/test_fp32_concordance.py` for the locked-in assertion.

If you need bit-for-bit R `spacexr` concordance for validation, stay on fp64.

### Per-architecture K cutoff for the eigh dispatch

`_psd_batch` decides between batched GPU eigh (cuSOLVER) and CPU offload (LAPACK `syevd`) based on K and the GPU's compute capability:

- `sm < 9` (Volta, Turing, Ampere, Ada / L40S): `K ≤ 16` stays on GPU; `K > 16` offloads to CPU.
- `sm ≥ 9` (Hopper, Blackwell): `K ≤ 128` stays on GPU.

This was tuned empirically and is correct for the configurations we've tested. If you hit a perf cliff or a `CUSOLVER_STATUS_INVALID_VALUE` crash, override it:

- `--eigh-threshold 0` (or `RCTDConfig(eigh_threshold=0)`): force CPU eigh on every architecture. Cheap and robust if a particular cuSOLVER batch shape is misbehaving.
- `--eigh-threshold 64` etc.: force GPU eigh up to a higher K. Useful on V100/L20 if the CPU offload turns out to be slower than GPU eigh for your specific workload.

### CPU eigh thread oversubscription (fixed in v0.3.5)

On hosts with many cores running on Volta / Turing / Ampere / Ada (i.e. forced into the CPU eigh branch at moderate K), versions ≤ 0.3.4 could stall for thousands of seconds because OpenBLAS spawned one thread per core under batched `syevd`. v0.3.5 wraps the CPU eigh call with `torch.set_num_threads(1)` and restores the caller's count on exit. **No user action required from v0.3.5 onwards.** If you're on an older release and see Step 1 of doublet mode stuck, upgrade or set `OMP_NUM_THREADS=1`.

### VRAM guidance

| Available VRAM | Recommended `--batch-size` | Peak VRAM (K=45) |
|----------------|---------------------------:|-----------------:|
| ≥ 24 GB | `auto` (often 10000+)      | ~4 GB |
| 8-16 GB | 5000                       | ~2 GB |
| < 8 GB  | 2000                       | ~1 GB |

`--batch-size auto` reads *free* VRAM at startup. On a shared GPU where another job's allocation can grow during the run, pin an explicit `--batch-size` to avoid mid-run OOM, and consider `PYTORCH_ALLOC_CONF=expandable_segments:True`.

### Generic SLURM template

```bash
#!/bin/bash
#SBATCH --job-name=rctdpy
#SBATCH --output=rctdpy_%j.log
#SBATCH --error=rctdpy_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu        # cluster-specific
#SBATCH --gres=gpu:1

set -euo pipefail
echo "Started: $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

rctd run spatial.h5ad reference.h5ad \
    --cell-type-col cell_type \
    --mode doublet \
    --device cuda \
    --dtype float32 \
    --umi-min 20 \
    --output results_rctd.h5ad \
    --json
```

## Result interpretation

The CLI writes results into a copy of the spatial AnnData:

| Slot | Content | Modes |
|------|---------|-------|
| `obs["rctd_dominant_type"]` | Top cell type per spot | all |
| `obs["rctd_spot_class"]` | `singlet` / `doublet_certain` / `doublet_uncertain` / `reject` / `filtered` | doublet |
| `obs["rctd_first_type"]` | Primary type (string) | doublet |
| `obs["rctd_second_type"]` | Secondary type (string) | doublet |
| `obs["rctd_converged"]` | IRWLS convergence flag | full |
| `obs["rctd_n_types"]` | Types assigned per spot | multi |
| `obsm["rctd_weights"]` | (N × K) weight matrix | all |
| `obsm["rctd_weights_doublet"]` | (N × 2) top-2 weights | doublet |
| `obsm["rctd_sub_weights"]` | Selected-type weights | multi |
| `uns["rctd_cell_type_names"]` | Ordered K labels for the weight matrix | all |
| `uns["rctd_mode"]` | Mode used | all |
| `uns["rctd_config"]` | Configuration dict | all |
| `uns["rctd_version"]` | rctd-py version that produced the file | all |

**Spot class is 0-indexed in the Python API** (unlike R `spacexr`'s 1-indexed `spot_class`):

| Code | Label | Meaning |
|------|-------|---------|
| 0 | `reject` | Confidence too low |
| 1 | `singlet` | Single dominant type |
| 2 | `doublet_certain` | Two types, high confidence |
| 3 | `doublet_uncertain` | Two types, lower confidence |

Filtered spots (below `UMI_min`) carry `NaN` weights and `"filtered"` labels.

```python
import anndata
adata = anndata.read_h5ad("results_rctd.h5ad")

print(adata.obs["rctd_spot_class"].value_counts())
singlets = adata[adata.obs["rctd_spot_class"] == "singlet"]
print(singlets.obs["rctd_first_type"].value_counts().head(10))

print(adata.obsm["rctd_weights"].shape)     # (N_pixels, K_types)
print(adata.uns["rctd_cell_type_names"])    # column order
```

## Integration back to Seurat

```r
library(anndataR)
library(Seurat)

result <- read_h5ad("results_rctd.h5ad")
obs <- as.data.frame(result$obs)

spatial_obj <- readRDS("spatial.rds")  # or qs2::qs_read
common <- intersect(colnames(spatial_obj), rownames(obs))

spatial_obj$rctd_first_type     <- obs[common, "rctd_first_type"]
spatial_obj$rctd_spot_class     <- obs[common, "rctd_spot_class"]
spatial_obj$rctd_dominant_type  <- obs[common, "rctd_dominant_type"]

W <- as.matrix(result$obsm[["rctd_weights"]])
colnames(W) <- result$uns[["rctd_cell_type_names"]]
for (ct in colnames(W)) {
    safe <- paste0("rctd.weight.", make.names(ct))
    spatial_obj[[safe]] <- W[match(colnames(spatial_obj), rownames(obs)), ct]
}
saveRDS(spatial_obj, "spatial_rctd.rds")
```

## Best practices

1. **Validate first.** `rctd validate` checks gene overlap, count integrality, and label coverage in seconds without a GPU.
2. **`--umi-min 20` for Xenium.** The 100 default is inherited from R `spacexr` and is set for Visium/Slide-seq UMI distributions.
3. **Match reference to sample condition.** If your cohort mixes diagnoses (e.g. malignancy A vs B), do *not* run every sample against one combined reference containing disease-specific malignant prototypes -- RCTD will misassign healthy/reactive cells of the wrong-diagnosis samples to those prototypes. Branch on the sample's metadata column and pick the reference per sample.
4. **Avoid `/` in cell-type names.** zarr/h5 may convert it to `_`. Pick consistent ASCII labels.
5. **Downsample large references** to ~500 cells/type; more rarely improves the fit.
6. **Prefer `dtype="float32"` on consumer GPUs.** Concordance is verified; speedup is ~2×. Stay fp64 only for exact R `spacexr` validation runs.

## Gotchas

- **`spot_class` is 0-indexed in Python; R `spacexr` is 1-indexed.** When cross-comparing, add 1 to the Python values (or subtract 1 from the R values).
- **MT-gene case sensitivity vs R `spacexr`.** The DE filter is case-insensitive (`.lower().startswith("mt-")`), which drops human `MT-CO1`-style symbols from the candidate pool. R `spacexr`'s case-sensitive `grep("mt-", ...)` keeps them. Net effect on real data: rctd-py keeps fewer pixels because the differing `gene_list_bulk` shifts the per-pixel `counts_MIN` filter. Cell-type calls on the pixels both runs keep agree at ~100%.
- **Concordance metrics that count filtered pixels as mismatches understate agreement.** When the pixel-filtering decision differs between rctd-py and R `spacexr`, a `concordance = same_first_type / total_pixels` will appear low even when the kept-pixel calls agree exactly. Compare assignments only on the pixel intersection.
- **Doublet mode under-assigns fine subtypes on small panels.** When a reference fine subtype has markers that the spatial panel can't reliably detect (e.g. CD4 markers on a 300-500-gene Xenium panel), doublet mode falls back to a coarser label or `doublet_uncertain` rather than over-calling the subtype. Always `value_counts()` your subtype × group counts before slicing by subtype for downstream analysis -- sparsity at the call layer can look like a biological finding. To re-split a coarse pool into subtypes, use a reference classifier (SingleR, ProjecTILs) rather than a per-cell marker argmax (low capture biases argmax toward high-capture markers).
- **`batch_size="auto"` on a shared GPU can OOM mid-run.** It sizes from free VRAM at startup; if another job grows afterwards, the IRWLS launch can fail. Pin `--batch-size 5000`-`8000` on contested nodes.
- **Determinism.** Python runs are bit-reproducible (`np.random.default_rng(42)` throughout). R `spacexr` with `max_cores > 1` is not (`mclapply` advances the parent's global RNG as a side effect, perturbing the later `sample()` in `chooseSigma`). Use `--sigma-override` for run-to-run determinism matching a fixed R baseline.

## Multi-modal (RNA + protein)

Joint RNA + protein deconvolution (Gaussian/WLS protein block sharing the per-spot cell-type weights) is in development on the `feat/multimodal-protein` branch. The objective is `L(w) = L_RNA(w) + λ · L_protein(w)`; protein enters the IRWLS gradient/Hessian and the doublet pair/singlet scores. Per-type protein profiles are bootstrapped from RNA-confident singlets, optionally combined with curated signed gates (positive + negative markers per type) to break RNA-degenerate types apart. Check the PR list on https://github.com/p-gueguen/rctd-py/pulls for current status before assuming the API is on `main`.

## Resources

- **PyPI:** `pip install rctd-py` (latest: see https://pypi.org/project/rctd-py/)
- **Source + issues:** https://github.com/p-gueguen/rctd-py
- **CHANGELOG:** https://github.com/p-gueguen/rctd-py/blob/main/CHANGELOG.md
- **Original RCTD paper:** Cable et al., *Nature Biotechnology* 2022 (R `spacexr`)
- **Tutorial notebook (Slide-seq cerebellum, end-to-end):** https://p-gueguen.github.io/rctd-py/
