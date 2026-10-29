# Fixed-data regression benchmark

This directory contains an independent benchmark harness; it does not modify
`src/rctd/`. Run it from the repository virtual environment on WSL/Linux.
The included configurations expect the real colon cancer dataset under
`~/rctd_data/visiumhd_human_colon_cancer_03`. Store outputs outside the
repository (for example under `~/rctd_bench`) so data and source snapshots are
not added to Git.

## Daily use

From the repository root:

```bash
# Once per case: freeze the data, settings, source revision, and calibration result.
.venv/bin/python benchmarks/regression/bench.py prepare \
  --config benchmarks/regression/colon_pilot.json \
  --case ~/rctd_bench/colon-pilot --source HEAD

# Keep this baseline and compare later changes with it.
.venv/bin/python benchmarks/regression/bench.py run \
  --case ~/rctd_bench/colon-pilot --source baseline \
  --out ~/rctd_bench/baseline

# After a code change, choose a new output directory. The harness snapshots
# uncommitted source as well.
.venv/bin/python benchmarks/regression/bench.py run \
  --case ~/rctd_bench/colon-pilot --source working --stream-gpu \
  --out ~/rctd_bench/change-001

.venv/bin/python benchmarks/regression/bench.py compare \
  --baseline ~/rctd_bench/baseline --candidate ~/rctd_bench/change-001 \
  --out ~/rctd_bench/comparison-001
```

`report.md` is the human-readable report, `comparison.json` is the complete
comparison, and `run.json` contains every measurement, the environment,
per-cell-type metrics, and repeatability checks. Each stage saves raw results
as `output_*.npz` and logs to `worker.log`. Output directories are never
overwritten, and there is no automatic "accept new baseline" operation.

Exit codes are 0 for success/numerical pass, 2 for a failed stage or numerical
regression, and 1 for invalid inputs, environment, or configuration. Performance
changes do not affect numerical PASS/FAIL; the report labels them separately as
`faster`, `slower`, or `inconclusive`.

`--source` may also name any local Git revision. Use `--stages full` to run
only the Full stage. `--bulk-input-cpu` affects only the standalone Normalize
stage input device, not the complete pipeline; use it only with implementations
that support CPU input. Enabling `--stream-gpu` against an older implementation
without that option is an explicit error. CPU input uses `as_tensor` to share
the frozen NumPy input instead of making another full CPU copy.

`colon_pilot.json` deterministically samples 256 real spots and must not be
treated as a full-scale result. `colon_full.json` uses all 11,385 input spots.
For a capacity curve, prepare separate cases with different `max_spots` values
while keeping batch size, dtype, and all other settings fixed. Compare identical
inputs at each size before comparing memory across sizes. OOMs and timeouts are
never numerical passes.

## Measurement scopes

| Stage | Inputs and measured scope | Purpose |
|---|---|---|
| normalize | The raw inputs passed to baseline `fit_bulk`, including input tensor creation/transfer and returned CPU results | Aggregation and Normalize numerical/memory regression |
| full | Frozen baseline genes, profiles, sigma, Q/SQ, counts; includes input transfer and CPU output | Isolate batching and solver changes |
| pipeline | The same raw AnnData reconstructed through `Reference`, `RCTD`, and the public `run_rctd` entry point | Capture gene selection, Normalize, sigma, and Full interactions |

H5AD reading, validation, and sampling are outside timed regions and recorded
as `input_load_validate_s`. Reference construction is included in pipeline
timing. Each stage runs in a fresh process, with one complete warmup followed by
five measurements. Every pipeline repeat reconstructs its objects so
`is_normalized` cannot skip work.

CUDA is synchronized around timing. Stage observers only time and read results;
they do not replace solvers. `platform_total_s` includes both `fit_bulk_s`
and `choose_sigma_s`, so those values must not be added again. Random seeds are
reset for each repeat to keep sigma sampling comparable.

`first_call_s` includes the first call in a fresh worker process, not a fully
cold machine start. Every worker has a separate compilation cache. The default
is `compile=false`; use a separate case to benchmark `compile=true`.

Runtime is reported as the median with all samples and ranges retained. The
comparison labels performance `faster` or `slower` only when there are at
least three repeats, the ranges do not overlap, and the medians differ by more
than 5%. This is a conservative description rule, not a confidence interval or
statistical significance test. Do not run another GPU workload concurrently.

CUDA allocated is peak live PyTorch tensor allocation; CUDA reserved is peak
memory held by the caching allocator. Reserved memory includes cache retained
after warmup because the harness does not call `empty_cache` between repeats.
Neither metric is whole-device `nvidia-smi` usage. RSS is sampled every 10 ms
and can miss a very short peak. Linux `ru_maxrss` for the complete worker
lifetime, including input loading, is also saved. The harness does not describe
a before/after `ru_maxrss` difference as stage peak memory.

## Numerical invariants

- Data files, Q tables, and frozen intermediate inputs are recorded by SHA256
  and revalidated before and after every run.
- The harness saves source file contents, Git SHA, working-tree diff, evaluator
  code, and environment versions.
- Comparisons require identical case, configuration, evaluator, software, and
  hardware environments. Algorithm source and explicit optimization flags may
  differ.
- Spot ID order, genes and their order, cell-type order, `pixel_mask`, sigma,
  and convergence masks must match exactly.
- Comparisons use raw, unnormalized weights, bulk weights, and normalized
  profiles. Non-finite values fail even if both sides contain them.
- Floating-point acceptance is
  `abs(candidate - baseline) <= atol + rtol * abs(baseline)`. The defaults are
  `atol=rtol=0`; tolerances must not be relaxed after looking at a failure.
- Every timed output is compared with the first baseline output, and both sides
  are checked for internal repeatability. Reports include maximum absolute
  error, RMSE, failing element count, and worst location and values.

If a future change permits numerical error, decide the tolerance before running
the comparison and prepare a new case. Keep separate baselines for FP32 and
FP64. Agreement with an older Python implementation is not R agreement; R
parity requires matching inputs and saved R outputs. Existing repository
concordance/parity tests remain separate.

## RCTD evaluation metrics and extensions

Metric definitions were cross-checked against the
[RCTD paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8606190/) and the
[official Figure 3 analysis](https://github.com/dmcable/spacexr/blob/master/AnalysisPaper/MainFigures/figure3.Rmd)
at blob `ab1f9d883e1d420dd2e2da4335f3522d1e4b2be9`. A score from one paper
experiment is not a universal acceptance threshold or an expected score for
this dataset.

| Category | Metric | Harness treatment |
|---|---|---|
| Official proportion recovery | RMSE between known and predicted mixtures | Overall and per-type RMSE; a full-mode real-data run is not represented as reproduction of the paper's simulated two-type experiment |
| Official classification | Singlet/doublet classification rate | Computed for `mode=doublet`, with denominators both excluding and including rejects |
| Official confident calls | Confident cell-class identification | In doublet mode, first type excludes rejects and second type uses only `doublet_certain`; true two-type spots with minority fraction >=0.25 are summarized by type pair when `config.class_df` is supplied |
| Official platform effect | Recovery R-squared against known effects | N/A without platform-effect ground truth |
| Official biological validation | Marker and spatial-distribution plausibility | N/A without a separate validation design |
| Additional proportions | MAE, per-type Pearson, macro Pearson, JSD | Explicit formulas and denominators below |
| Additional classification | Dominant-type argmax accuracy | Not official doublet classification; true tied maxima are reported and resolved by fixed column order |
| Engineering | Runtime, throughput, CUDA allocated/reserved, RSS, OOM, timeout, repeatability | Reported separately from biological metrics |

`mode` defaults to `full`; prepare a separate `mode=doublet` or `mode=multi`
case to evaluate those complete pipelines. The standalone Full stage always
tests the Full solver. `stream_gpu` currently supports only Full mode.
Full mode has no `spot_class`, so confidence classification metrics are N/A;
the harness does not infer confidence from argmax. Doublet proportion metrics
use selected `weights_doublet` types and treat rejects as invalid predictions.
Multi mode uses `sub_weights`.

For proportion metrics, prediction `w` and annotation `t` are normalized per
row to `p=w/sum(w)` and `q=t/sum(t)`. Overall RMSE is
`sqrt(mean((p-q)^2))`; MAE is `mean(abs(p-q))`; per-type metrics operate
across spots. Pearson is null for a constant per-type vector and macro Pearson
uses only defined types while reporting the count. JSD is
`0.5 KL2(p||(p+q)/2) + 0.5 KL2(q||(p+q)/2)`, without a square root.
Zero-annotation rows and predictions with negative, non-finite, or zero-total
weights are counted and excluded rather than clipped or replaced. Labels must
align exactly; missing cell types are not silently dropped.

The included `annotation.csv` appears to contain cell counts, while RCTD
weights represent RNA-contribution proportions. Treat these results as
agreement with the supplied annotation, not as ground-truth accuracy or paper
reproduction. The annotation generation process and its independence from the
reference still need confirmation. Improved annotation metrics never override
a failed raw-weight regression.

## Data conventions

The harness reads raw counts from `X`. Set `spatial_layer` or
`reference_layer` if raw counts are stored in a layer. The two current H5AD
files have duplicate gene names, so the configurations explicitly align on
unique `gene_ids` without modifying the source files. RCTD's internal
`mt-` name filter is no longer equivalent when gene IDs are used; if
mitochondrial genes must be removed, define and freeze that preprocessing
protocol instead of changing it between versions.

The reference uses `Level2` labels with the existing default filters and a
fixed sample of at most 10,000 cells per type. Spot sampling is performed over
all original genes, and the algorithm still computes nUMI itself.

The harness does not download data, install dependencies, switch branches,
commit code, modify source files, or modify input H5AD files. Failed output
directories retain their logs; choose a new directory before retrying. Complete
artifacts can be large and may be archived manually.

## Validation

```bash
.venv/bin/python -m pytest -q tests/test_regression_metrics.py tests/test_regression_harness.py
.venv/bin/ruff check benchmarks/regression tests/test_regression_metrics.py tests/test_regression_harness.py
```
