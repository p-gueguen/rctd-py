# Tutorial notebook (marimo) notes

Moved out of `CLAUDE.md`, which links here.

### Marimo gotchas

**Figures in static HTML export**: `plt.show()` does NOT produce capturable output in marimo's static export. The figure must be the **last expression** in the cell (like a return value), and it must use a **non-underscore name** (underscore-prefixed variables are cell-private in marimo and won't be rendered):

```python
# WRONG — no output in static HTML
_fig, _ax = plt.subplots()
_ax.plot(x, y)
plt.show()
return

# CORRECT — figure rendered in static HTML
fig_plot, ax_plot = plt.subplots()
ax_plot.plot(x, y)
fig_plot  # last expression, non-underscore name
```

**Underscore-prefixed names are cell-private**: Functions/variables starting with `_` (e.g., `_fig`, `_detect_ct_col`) are NOT exported from a cell to other cells or to static HTML output.

**Deprecated matplotlib API**: Use `plt.colormaps.get_cmap('tab20').resampled(n)` instead of `plt.cm.get_cmap('tab20', n)` (deprecated in matplotlib 3.7+).

### Doc-vs-code drift audits

Tutorial doc strings have drifted from code multiple times (e.g. v0.3.x
shipped with three errors in `examples/tutorial.py`: a "weights sum to 1"
claim that was never true under `constrain=FALSE`, a `N_fit` default of
1000 when the code then used 100, and a `doublet_mode_alpha` parameter that has never
existed in any version — `git log --all -S doublet_mode_alpha` shows it
only in tutorial commits).

After non-trivial changes to `RCTDConfig` defaults, mode behavior, or
weight normalization, run a verification script that executes the
tutorial workflow end-to-end and prints the empirical answer next to the
documented claim. Reference fix: commit `a05067f` (source) + `428da09`
(re-rendered HTML). Cross-check parameter names against `RCTDConfig` in
`_types.py` to catch hallucinated names early.

**Spot class encoding** is 0-indexed (unlike R spacexr which is 1-indexed):
```
0 = reject, 1 = singlet, 2 = doublet_certain, 3 = doublet_uncertain
```
Use `SPOT_CLASS_NAMES` from `rctd._types` (exported in public API) — never hardcode the mapping.
