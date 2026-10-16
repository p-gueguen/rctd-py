"""Pixel-level parity with R spacexr: R's own per-pixel inputs in, identical calls out.

The fixture (tests/fixtures/hippocampus_pixels/README.md) holds 150 beads on which rctd-py <= 0.3.9
reported a different first_type than R, so this test fails on that code.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rctd import RCTDConfig, _irwls, _likelihood
from rctd._doublet import run_doublet_mode
from rctd._likelihood import compute_spline_coefficients, load_cached_q_matrices

FIX = Path(__file__).parent / "fixtures" / "hippocampus_pixels"
CLS = np.array(["reject", "singlet", "doublet_certain", "doublet_uncertain"])


@pytest.mark.slow
def test_doublet_calls_identical_to_r_given_r_inputs(monkeypatch):
    monkeypatch.setattr(_irwls, "_USE_COMPILE", False)  # JIT path: no compile wait on CPU runners
    monkeypatch.setattr(_likelihood, "_CALC_Q_USE_COMPILE", False)
    f = np.load(FIX / "inputs.npz")
    r = pd.read_csv(FIX / "r_results.csv", index_col=0)
    cache = load_cached_q_matrices()
    x = cache.pop("X_vals")
    q = cache[f"Q_{round(float(f['sigma']) * 100)}"]
    types = list(f["cell_types"])
    res = run_doublet_mode(
        f["counts"].astype(np.float64),
        f["nUMI"],
        f["profiles"],
        types,
        q,
        compute_spline_coefficients(q, x),
        x,
        RCTDConfig(device="cpu"),
        batch_size=300,
        device="cpu",
    )
    names = np.asarray(types)
    np.testing.assert_array_equal(CLS[res.spot_class], r["spot_class"].to_numpy())
    np.testing.assert_array_equal(names[res.first_type], r["first_type"].to_numpy())
    np.testing.assert_array_equal(names[res.second_type], r["second_type"].to_numpy())
    np.testing.assert_allclose(res.min_score, r["min_score"].to_numpy(), atol=1e-3)
    np.testing.assert_allclose(res.singlet_score, r["singlet_score"].to_numpy(), atol=1e-3)
