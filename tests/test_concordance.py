"""Concordance tests: rctd-py vs R spacexr on the official vignette dataset.

These tests download the spacexr vignette data (~500 KB) from GitHub,
run rctd-py, and compare results against pre-computed R spacexr outputs
stored in tests/fixtures/vignette/.

Run with: pytest tests/test_concordance.py -v -m slow
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rctd import RCTDConfig, Reference
from rctd._doublet import run_doublet_mode
from rctd._rctd import RCTD

FIXTURES = Path(__file__).parent / "fixtures" / "vignette"


def _load_r_fixtures():
    """Load pre-computed R spacexr results from fixtures directory."""
    r_weights_df = pd.read_csv(FIXTURES / "r_weights.csv")
    r_barcodes = r_weights_df["barcode"].values
    r_type_names = [c for c in r_weights_df.columns if c != "barcode"]
    r_weights = r_weights_df[r_type_names].values.astype(np.float64)

    r_results = pd.read_csv(FIXTURES / "r_results_df.csv")

    return {
        "barcodes": r_barcodes,
        "type_names": r_type_names,
        "weights": r_weights,
        "results_df": r_results,
    }


def _align_and_compare(py_weights, py_barcodes, r_fix):
    """Align Python and R weight matrices, return concordance metrics."""
    common_barcodes = sorted(set(py_barcodes) & set(r_fix["barcodes"]))

    py_idx = {b: i for i, b in enumerate(py_barcodes)}
    r_idx = {b: i for i, b in enumerate(r_fix["barcodes"])}

    py_rows = [py_idx[b] for b in common_barcodes]
    r_rows = [r_idx[b] for b in common_barcodes]

    # Align by column position (both have same number of types, sorted)
    K = min(py_weights.shape[1], r_fix["weights"].shape[1])
    py_w = py_weights[py_rows, :K].astype(np.float64)
    r_w = r_fix["weights"][r_rows, :K]

    # Normalize rows to sum=1
    py_w = py_w / np.maximum(py_w.sum(axis=1, keepdims=True), 1e-10)
    r_w = r_w / np.maximum(r_w.sum(axis=1, keepdims=True), 1e-10)

    # Dominant type agreement
    py_dominant = py_w.argmax(axis=1)
    r_dominant = r_w.argmax(axis=1)
    agreement = float((py_dominant == r_dominant).mean())

    # Per-bead weight correlation
    corrs = []
    for i in range(len(common_barcodes)):
        if py_w[i].std() > 1e-10 and r_w[i].std() > 1e-10:
            c = np.corrcoef(py_w[i], r_w[i])[0, 1]
            if not np.isnan(c):
                corrs.append(c)
    median_corr = float(np.median(corrs)) if corrs else 0.0

    return {
        "n_common": len(common_barcodes),
        "agreement": agreement,
        "median_corr": median_corr,
    }


@pytest.fixture(scope="module")
def rctd_doublet_result(vignette_data):
    """Run rctd-py doublet mode on vignette data (cached per module)."""
    reference = Reference(vignette_data["reference"], cell_type_col="cell_type", cell_min=25)
    config = RCTDConfig(UMI_min=100)
    rctd = RCTD(vignette_data["spatial"], reference, config)
    rctd.fit_platform_effects()

    result = run_doublet_mode(
        spatial_counts=rctd.counts,
        spatial_numi=rctd.nUMI,
        norm_profiles=rctd.norm_profiles,
        cell_type_names=reference.cell_type_names,
        q_mat=rctd.q_mat,
        sq_mat=rctd.sq_mat,
        x_vals=rctd.x_vals,
        config=config,
        batch_size=200,
    )

    all_barcodes = list(vignette_data["spatial"].obs_names)
    filtered_barcodes = [b for b, m in zip(all_barcodes, rctd._pixel_mask) if m]

    return {
        "result": result,
        "barcodes": filtered_barcodes,
    }


@pytest.mark.slow
class TestRConcordance:
    """Compare rctd-py doublet mode against R spacexr on vignette data."""

    def test_fixtures_exist(self):
        """R fixture files should be present in the repository."""
        assert (FIXTURES / "r_weights.csv").exists()
        assert (FIXTURES / "r_results_df.csv").exists()

    def test_doublet_dominant_type_agreement(self, rctd_doublet_result):
        """Dominant cell type should agree with R on >=80% of beads."""
        r_fix = _load_r_fixtures()

        metrics = _align_and_compare(
            rctd_doublet_result["result"].weights,
            rctd_doublet_result["barcodes"],
            r_fix,
        )

        assert metrics["n_common"] > 0, "No common barcodes between Python and R"
        assert metrics["agreement"] >= 0.80, (
            f"Dominant type agreement {metrics['agreement']:.2%} < 80% "
            f"on {metrics['n_common']} common beads"
        )

    def test_doublet_weight_correlation(self, rctd_doublet_result):
        """Median per-bead weight correlation with R should be >= 0.90."""
        r_fix = _load_r_fixtures()

        metrics = _align_and_compare(
            rctd_doublet_result["result"].weights,
            rctd_doublet_result["barcodes"],
            r_fix,
        )

        assert metrics["median_corr"] >= 0.90, (
            f"Median per-bead correlation {metrics['median_corr']:.4f} < 0.90"
        )
