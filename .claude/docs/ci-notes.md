# CI notes

Moved out of `CLAUDE.md`, which links here.

### PyTorch API compatibility

PyTorch renamed `cuda.get_device_properties().total_mem` → `total_memory` in 2.10+. Use `getattr` fallback pattern (see `cli.py:59`). CI runs an older PyTorch than the GPU nodes — test both paths.

### GitHub Actions runner toolchain ICEs

The ubuntu-24.04 runner image periodically rolls forward to a g++ that
ICEs on Inductor's `vec512.h` C++ codegen:

```
internal compiler error: in gimple_duplicate_bb, at tree-cfg.cc:6545
```

This surfaces in `Run slow integration + R concordance tests` and is a
GCC bug we don't control. **Workaround:** pin slow-test fixtures that
exercise the box-QP solver to `compile=False` so they take the
TorchScript JIT path (no g++ involved). Reference: `tests/test_concordance.py`
fixture comment + commit `b2a206f`. Don't try to "fix" this by upgrading
actions — it's intermittent and tied to the runner image version.
