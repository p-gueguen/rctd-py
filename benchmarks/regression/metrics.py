"""Metrics with explicit denominators; no solver imports or implicit realignment."""

import numpy as np


def numerical(reference, candidate, atol=0.0, rtol=0.0):
    """Compare raw arrays. Identical NaN/Inf values must never pass."""
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape:
        return {
            "pass": False,
            "reason": "shape",
            "reference": list(a.shape),
            "candidate": list(b.shape),
        }
    if a.dtype.kind in "OUSb" or b.dtype.kind in "OUSb":
        same = a == b
        return {
            "pass": bool(same.all()),
            "exact": bool(same.all()),
            "mismatches": int((~same).sum()),
        }
    a, b = a.astype(np.float64), b.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {
            "pass": False,
            "reason": "nonfinite",
            "reference_nonfinite": int((~np.isfinite(a)).sum()),
            "candidate_nonfinite": int((~np.isfinite(b)).sum()),
        }
    delta = np.abs(b - a)
    failed = delta > atol + rtol * np.abs(a)
    worst = list(np.unravel_index(delta.argmax(), delta.shape)) if delta.size else []
    return {
        "pass": bool(not failed.any()),
        "exact": bool(np.array_equal(a, b)),
        "max_abs": float(delta.max(initial=0)),
        "rmse": float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0,
        "outside_tolerance": int(failed.sum()),
        "elements": int(delta.size),
        "worst_index": [int(i) for i in worst],
        "worst_reference": float(a[tuple(worst)]) if delta.size else None,
        "worst_candidate": float(b[tuple(worst)]) if delta.size else None,
    }


def compare_outputs(reference, candidate, atol, rtol):
    keys = set(reference) | set(candidate)
    result = {}
    for key in sorted(keys):
        if key not in reference or key not in candidate:
            result[key] = {"pass": False, "reason": "missing output"}
        else:
            # IDs, masks, sigma and convergence must match exactly, regardless of tolerance.
            floating = reference[key].dtype.kind == "f"
            result[key] = numerical(
                reference[key], candidate[key], atol if floating else 0, rtol if floating else 0
            )
    return {"pass": all(x["pass"] for x in result.values()), "arrays": result}


def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else None


def annotation_metrics(weights, truth, names):
    """Compare row-normalized weights with provided cell-count annotations.

    This is annotation agreement, not proof of biological accuracy: RCTD estimates
    RNA contributions, whereas the supplied annotations may count cells.
    Negative predictions are reported and excluded, never silently clipped.
    """
    w, t = np.asarray(weights, dtype=float), np.asarray(truth, dtype=float)
    if w.shape != t.shape or w.ndim != 2 or w.shape[1] != len(names):
        raise ValueError("Annotation and prediction shapes/types must be aligned explicitly")
    if not np.isfinite(t).all() or (t < 0).any():
        raise ValueError("Annotations must be finite, nonnegative counts/proportions")
    prediction_valid = np.isfinite(w).all(1) & (w >= 0).all(1) & (w.sum(1) > 0)
    truth_valid = t.sum(1) > 0
    valid = prediction_valid & truth_valid
    result = {
        "status": "ok" if valid.any() else "unavailable",
        "n_total": len(w),
        "n_evaluated": int(valid.sum()),
        "zero_truth_rows": int((~truth_valid).sum()),
        "invalid_prediction_rows": int((~prediction_valid).sum()),
        "interpretation": "RNA-weight versus provided cell-count proportion agreement",
    }
    if not valid.any():
        return result
    p, q = w[valid] / w[valid].sum(1, keepdims=True), t[valid] / t[valid].sum(1, keepdims=True)
    delta = p - q
    m = (p + q) / 2

    def kl(x):
        # KL base 2, with 0 log 0 = 0. JSD is divergence, not its square root.
        return np.sum(x * np.log2(np.divide(x, m, out=np.ones_like(x), where=x > 0)), 1)

    jsd = (kl(p) + kl(q)) / 2
    result.update(
        rmse=float(np.sqrt(np.mean(delta**2))),
        mae=float(np.abs(delta).mean()),
        jsd_base2_mean=float(jsd.mean()),
        dominant_type_accuracy=float((p.argmax(1) == q.argmax(1)).mean()),
        true_dominant_ties=int(((q == q.max(1, keepdims=True)).sum(1) > 1).sum()),
    )
    result["per_cell_type"] = {
        str(name): {
            "rmse": float(np.sqrt(np.mean(delta[:, k] ** 2))),
            "mae": float(np.abs(delta[:, k]).mean()),
            "pearson": _pearson(p[:, k], q[:, k]),
        }
        for k, name in enumerate(names)
    }
    correlations = [
        x["pearson"] for x in result["per_cell_type"].values() if x["pearson"] is not None
    ]
    result["pearson_macro"] = float(np.mean(correlations)) if correlations else None
    result["pearson_defined_types"] = len(correlations)
    # These require doublet-mode calls / independent validation data, not argmax(full weights).
    result["official_not_applicable"] = {
        "singlet_doublet_classification": "full mode has no spot_class/confident calls",
        "confident_call_cell_type_accuracy": "requires doublet-mode calls and class truth",
        "platform_effect_R2": "no known true platform-effect vector",
        "marker_spatial_validation": "requires independent marker/spatial validation protocol",
    }
    return result


def timing_comparison(reference, candidate, practical_change=0.05):
    """Conservative descriptive verdict, not a statistical significance test."""
    a, b = np.asarray(reference), np.asarray(candidate)
    am, bm = float(np.median(a)), float(np.median(b))
    verdict = "inconclusive"
    if len(a) >= 3 and len(b) >= 3:
        if b.max() < a.min() and bm < am * (1 - practical_change):
            verdict = "faster"
        elif b.min() > a.max() and bm > am * (1 + practical_change):
            verdict = "slower"
    return {
        "reference_median_s": am,
        "candidate_median_s": bm,
        "speedup": am / bm,
        "verdict": verdict,
        "method": "at least 3 repeats, disjoint ranges and >5% median change",
    }


def doublet_call_metrics(spot_class, first_type, second_type, truth, names, class_map=None):
    """Official-style singlet/doublet and confident-call metrics, with denominators.

    Fig. 3a excludes rejected calls from its classification fractions. Fig. 3b
    scores first calls except rejects and second calls only for doublet_certain,
    on true doublets with minority fraction >= 0.25, aggregated by type pair.
    With cell-count truth these are proxy metrics, not the paper's UMI simulation.
    """
    labels, first, second = map(np.asarray, (spot_class, first_type, second_type))
    t = np.asarray(truth, dtype=float)
    n, k = t.shape
    if k != len(names) or any(x.shape != (n,) for x in [labels, first, second]):
        raise ValueError("Doublet labels, indices and truth must align")
    if not np.isfinite(t).all() or (t < 0).any() or not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError("Invalid doublet classes or truth")
    accepted, confident_second = labels != 0, labels == 2
    if ((first[accepted] < 0) | (first[accepted] >= k)).any() or (
        (second[confident_second] < 0) | (second[confident_second] >= k)
    ).any():
        raise ValueError("Confident cell type index out of range")
    truth_types = t > 0
    n_types = truth_types.sum(1)
    valid = n_types > 0
    singleton, doublet = n_types == 1, n_types == 2

    def score(correct, eligible):
        return {
            "correct": int((correct & eligible).sum()),
            "denominator": int(eligible.sum()),
            "accuracy": float(correct[eligible].mean()) if eligible.any() else None,
        }

    result = {
        "interpretation": "official-style metrics; supplied truth units and classes must be stated",
        "rejected": int(((labels == 0) & valid).sum()),
        "valid_truth_spots": int(valid.sum()),
        "confidence_rate": float(accepted[valid].mean()) if valid.any() else None,
        "singlet_classification_excluding_reject": score(labels == 1, singleton & accepted),
        "doublet_classification_excluding_reject": score(
            np.isin(labels, [2, 3]), doublet & accepted
        ),
        "singlet_classification_including_reject": score(labels == 1, singleton),
        "doublet_classification_including_reject": score(np.isin(labels, [2, 3]), doublet),
        "true_more_than_two_types": int((n_types > 2).sum()),
    }
    mapping = class_map or {}
    classes = np.asarray([mapping.get(str(name), str(name)) for name in names])
    totals = t.sum(1)
    fractions = np.divide(t, totals[:, None], out=np.zeros_like(t), where=totals[:, None] > 0)
    balanced = doublet & (fractions.max(1) <= 0.75)
    pairs = {}
    for i in np.flatnonzero(balanced):
        true_indices = np.flatnonzero(truth_types[i])
        pair = " | ".join(str(names[j]) for j in true_indices)
        row = pairs.setdefault(pair, {"correct": 0, "denominator": 0})
        true_classes = classes[true_indices]
        for called, index in [(accepted[i], first[i]), (confident_second[i], second[i])]:
            if called:
                row["correct"] += int(classes[index] in true_classes)
                row["denominator"] += 1
    rates = []
    for row in pairs.values():
        row["accuracy"] = row["correct"] / row["denominator"] if row["denominator"] else None
        if row["accuracy"] is not None:
            rates.append(row["accuracy"])
    result["confident_call_accuracy_per_pair"] = pairs
    result["confident_call_accuracy_macro"] = float(np.mean(rates)) if rates else None
    result["confident_call_accuracy_sd"] = float(np.std(rates, ddof=1)) if len(rates) > 1 else None
    return result
