"""Failure-oriented tests for the independent evaluation harness."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_path = Path(__file__).parents[1] / "benchmarks/regression/metrics.py"
_spec = importlib.util.spec_from_file_location("regression_metrics", _path)
metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metrics)


def test_official_doublet_denominators_and_confident_calls():
    result = metrics.doublet_call_metrics(
        [1, 0, 2, 3, 1],
        [0, -1, 0, 0, 0],
        [-1, -1, 1, 1, -1],
        [[1, 0], [1, 0], [1, 1], [1, 1], [1, 1]],
        ["a", "b"],
    )
    assert result["singlet_classification_excluding_reject"]["accuracy"] == 1
    assert result["singlet_classification_including_reject"]["accuracy"] == 0.5
    assert result["doublet_classification_excluding_reject"]["accuracy"] == pytest.approx(2 / 3)
    assert result["confident_call_accuracy_per_pair"]["a | b"]["denominator"] == 4
    assert result["confident_call_accuracy_macro"] == 1
    with pytest.raises(ValueError):
        metrics.doublet_call_metrics([2], [0], [5], [[1, 1]], ["a", "b"])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_identical_nonfinite_values_fail(bad):
    assert not metrics.numerical([bad], [bad])["pass"]


def test_raw_difference_not_hidden_by_normalization():
    assert not metrics.numerical([[1.0, 2.0]], [[2.0, 4.0]])["pass"]
    assert metrics.numerical([1.0], [1.0001], atol=0.001)["pass"]
    assert not metrics.numerical(np.ones((2, 1)), np.ones((1, 2)))["pass"]


def test_masks_sigma_names_exact_even_under_loose_numeric_tolerance():
    a = {
        "weights": np.array([1.0]),
        "sigma": np.array(100),
        "spot_ids": np.array(["a", "b"]),
        "converged": np.array([True]),
    }
    b = dict(a, sigma=np.array(101), spot_ids=np.array(["b", "a"]), converged=np.array([False]))
    comparison = metrics.compare_outputs(a, b, atol=100, rtol=100)
    assert not comparison["pass"]
    assert comparison["arrays"]["weights"]["pass"]
    assert not comparison["arrays"]["sigma"]["pass"]
    assert not comparison["arrays"]["spot_ids"]["pass"]


def test_known_proportions_and_js_divergence():
    same = metrics.annotation_metrics(
        [[2.0, 0.0], [0.0, 3.0]], [[7.0, 0.0], [0.0, 5.0]], ["a", "b"]
    )
    assert same["rmse"] == 0
    assert same["jsd_base2_mean"] == 0
    assert same["dominant_type_accuracy"] == 1
    opposite = metrics.annotation_metrics([[1.0, 0.0]], [[0.0, 1.0]], ["a", "b"])
    assert opposite["rmse"] == 1
    assert opposite["jsd_base2_mean"] == 1
    assert opposite["pearson_macro"] is None


def test_invalid_predictions_and_empty_truth_are_counted():
    value = metrics.annotation_metrics(
        [[1.0, 0.0], [-1.0, 2.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 2.0], [1.0, 1.0]], ["a", "b"]
    )
    assert value["n_evaluated"] == 1
    assert value["zero_truth_rows"] == 1
    assert value["invalid_prediction_rows"] == 1
    with pytest.raises(ValueError):
        metrics.annotation_metrics([[1.0, 0.0]], [[1.0, -1.0]], ["a", "b"])


def test_no_speed_claim_from_single_run_or_overlapping_ranges():
    assert metrics.timing_comparison([3.0], [1.0])["verdict"] == "inconclusive"
    assert metrics.timing_comparison([1.0, 2.0, 3.0], [1.0, 1.1, 2.0])["verdict"] == "inconclusive"
    assert metrics.timing_comparison([3.0, 3.1, 3.2], [1.0, 1.1, 1.2])["verdict"] == "faster"
