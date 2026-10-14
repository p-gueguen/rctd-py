# Compile / JIT dispatch dynamics

Moved out of `CLAUDE.md`, which links here.

### Compile / JIT dispatch dynamics

`_solve_box_qp_batch` (`_irwls.py:466-515`) has three paths:

| `_USE_COMPILE` state | Path | Has fallback? |
|---|---|---|
| `False` (or `_solve_box_qp_batch_compiled is None`) | `_solve_box_qp_batch_adaptive_jit` (TorchScript JIT) | n/a |
| `None` (first call) | try compiled, except RuntimeError → fall back to eager | yes |
| `True` (after first success) | `_solve_box_qp_batch_compiled` directly | **no** |

**Footgun:** the lazy try/except only protects the *first* call. If a later
recompile fails (different shape, transient toolchain breakage), there is no
automatic recovery. For environments with unreliable torch.compile, set
`compile=False` explicitly via `RCTDConfig`.

**v0.3.2 behavior change:** the `compile=False` path used to return
`_solve_box_qp_batch_impl` (eager Python loop). It now returns
`_solve_box_qp_batch_adaptive_jit`. The two paths converge to ~2e-4
numerical agreement (different op ordering). When writing tests that
assert equivalence, compare against `_solve_box_qp_batch_adaptive_jit`,
not `_solve_box_qp_batch_impl`. See commit `c068109`.

**`calc_q_all`** (`_likelihood.py:320-353`) has the same lazy fallback
pattern but its `compile=False` path still routes to the eager
`_calc_q_all_impl` — not changed in v0.3.2.
