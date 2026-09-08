"""Measure the protein modality instead of assuming it.

Two things in the multi-modal path are hand-set: lambda (how much the protein
term counts) and the +/- magnitude written into a curated profile. Neither has
ever been chosen against a label the model did not itself produce - the
available checks are proxies (lineage-conflict rate, dot-plot purity), and a
proxy cannot tell "protein fixed this call" from "protein overrode a correct
one".

CellTune's answer (Bussi et al., Nat Methods 2026) is landmark cells: strict
two-sided gates that hand you a subset of cells whose identity is not in doubt,
which then serve as truth. ``_protein.gate_landmarks`` ports the gating; this
module does the scoring and the lambda sweep on top of it.

The trap, and the reason for the marker folds: landmarks gated on protein would
trivially favour a fit that leans on protein. So the markers are split into
folds - truth is gated with fold *f*, and the fit for that fold runs with fold
*f* REMOVED from both the observation matrix and the profile. A lambda that only
looks good because protein is scoring itself cannot survive that.

WHY A NULL, NOT A THRESHOLD
--------------------------
"Protein helped if F1 went up" is not good enough. Measured on the synthetic
fixture (180 cells, ~36 landmarks per fold): the gain a SHUFFLED panel achieves
is centred at zero, as it should be, but its spread is sd ~0.05-0.08 - the same
order as a real effect. A single shuffled sweep therefore proves nothing, and a
"beats lambda=0" rule picked lambda=4 on pure noise. So the sweep is run against
``n_null`` independently shuffled panels and the observed gain must beat the
LARGEST of them (a one-sided permutation test at p <= 1/(n_null+1)). On that
fixture: informative panel gain +0.122 vs null max +0.064 (z = 2.5, selected);
noise panel gain +0.039 vs null max +0.077 (z = 0.5, rejected -> lambda = 0).

WHAT THIS STILL ASSUMES, measured rather than guessed
-----------------------------------------------------
The marker folds stop one marker from scoring itself. They do NOT make the truth
independent of the protein modality as a whole. If a panel is internally
consistent but attached to the wrong cells - mis-registration, a segmentation
offset, protein assigned to the neighbour - then the gated truth and the fit read
the SAME wrong assignment and agree with each other, and this sweep can still
land on lambda > 0. Verified on synthetic data
(``tests/test_protein_landmarks.py``): permuting each marker independently (a
panel with no information at all) correctly selects lambda = 0, while permuting
whole rows (a mis-assigned panel) does not.

What separates the two cases is the ABSOLUTE score, not the direction: a
mis-assigned panel cannot reach the F1 a real one does, because its gated truth
then contradicts the RNA the fit also sees (0.38 vs 0.85 on that fixture). So
read the absolute macro F1, not only the argmax - a selected lambda sitting on a
low F1 means the landmarks themselves are not trustworthy. The only real fix is
truth that does not come from protein at all, which is what CellTune buys with
human labelling.
"""

import numpy as np

from rctd._protein import gate_landmarks, marker_folds
from rctd._types import SPOT_CLASS_REJECT

NO_CALL = -1


def landmark_macro_f1(
    labels_true: np.ndarray,
    first_type: np.ndarray,
    spot_class: np.ndarray | None = None,
    count_reject_as_miss: bool = True,
) -> tuple[float, dict]:
    """Macro F1 of ``first_type`` against protein-gated landmark labels.

    Scored on ``first_type``, NOT ``argmax(weights)``: the argmax silently
    bypasses the reject/doublet decision that doublet mode exists to make.

    A rejected landmark cell is counted as a MISS (it hurts recall) but never as
    a false positive for another type. Dropping rejects from the denominator
    instead would let a configuration win by refusing to call anything.

    Args:
        labels_true: (N,) landmark labels, -1 = not a landmark (ignored).
        first_type: (N,) predicted type index.
        spot_class: (N,) optional; ``SPOT_CLASS_REJECT`` becomes a no-call.
        count_reject_as_miss: set False to score rejects as ordinary calls.

    Returns:
        ``(macro_f1, per_type)``; ``per_type`` maps type index -> dict with
        precision / recall / f1 / support.
    """
    labels_true = np.asarray(labels_true)
    pred = np.asarray(first_type).copy()
    if spot_class is not None and count_reject_as_miss:
        pred = np.where(np.asarray(spot_class) == SPOT_CLASS_REJECT, NO_CALL, pred)

    ev = labels_true >= 0
    per_type: dict[int, dict] = {}
    if not ev.any():
        return float("nan"), per_type

    for k in np.unique(labels_true[ev]):
        k = int(k)
        tp = int(((labels_true == k) & (pred == k) & ev).sum())
        # false positives only among evaluated (landmark) cells - a call on a
        # non-landmark cell is unknown, not wrong
        fp = int(((labels_true != k) & (pred == k) & ev).sum())
        fn = int(((labels_true == k) & (pred != k) & ev).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_type[k] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": int(((labels_true == k) & ev).sum()),
        }
    macro = float(np.mean([v["f1"] for v in per_type.values()])) if per_type else float("nan")
    return macro, per_type


def _subsample(labels: np.ndarray, max_pixels: int, seed: int = 0) -> np.ndarray:
    """All landmark cells plus a random background, capped at ``max_pixels``.

    The landmarks are what the score is computed on; the background is there so
    the fit still sees a realistic mixture of cell types.
    """
    n = labels.shape[0]
    if max_pixels is None or n <= max_pixels:
        return np.arange(n)
    lm = np.where(labels >= 0)[0]
    if lm.shape[0] >= max_pixels:
        return np.sort(np.random.default_rng(seed).choice(lm, max_pixels, replace=False))
    rest = np.where(labels < 0)[0]
    take = min(max_pixels - lm.shape[0], rest.shape[0])
    bg = np.random.default_rng(seed).choice(rest, take, replace=False)
    return np.sort(np.concatenate([lm, bg]))


def _column_shuffle(protein_std: np.ndarray, seed: int = 0) -> np.ndarray:
    """Permute each marker independently: keeps every marker's marginal
    distribution, destroys which cell carries which value. The null panel."""
    rng = np.random.default_rng(seed)
    out = np.asarray(protein_std, dtype=np.float64).copy()
    for m in range(out.shape[1]):
        out[:, m] = out[rng.permutation(out.shape[0]), m]
    return out


def _sweep_folds(
    *,
    run_doublet,
    fit_kwargs,
    protein_std,
    protein_profiles,
    inv_tau2,
    protein_mask,
    feature_names,
    signatures,
    cell_type_names,
    reliability,
    grid,
    n_folds,
    min_cells,
    max_pixels,
    verbose,
    tag="",
    type_to_class=None,
    class_names=None,
):
    """Gate truth from one marker fold, fit without that fold, score every lambda.

    With ``type_to_class`` (K,) and ``class_names``, landmarks are gated per CLASS
    (``signatures`` keyed by class) and ``first_type`` is mapped through
    ``type_to_class`` before scoring, so confusion inside a class is not an error.

    Returns ``(mean_f1_per_lambda, fold_records)``.
    """
    folds = marker_folds(feature_names, n_folds=n_folds)
    scores: dict[float, list] = {lam: [] for lam in grid}
    fold_records = []
    gate_names = list(cell_type_names) if type_to_class is None else list(class_names or [])

    def _pred(res):
        return res.first_type if type_to_class is None else type_to_class[res.first_type]

    for f, fold in enumerate(folds):
        keep = np.setdiff1d(np.arange(len(feature_names)), fold)
        if keep.size == 0:
            continue
        labels, info = gate_landmarks(
            protein_std,
            feature_names,
            signatures,
            gate_names,
            reliability=reliability,
            min_cells=min_cells,
            restrict_markers=fold,
        )
        gated = [t for t, d in info["per_type"].items() if d.get("n", 0) >= min_cells]
        n_lm = int((labels >= 0).sum())
        rec = {
            "fold": f,
            "gating_markers": [feature_names[i] for i in fold],
            "fit_markers": [feature_names[i] for i in keep],
            "n_landmarks": n_lm,
            "types_gated": gated,
            "scores": {},
        }
        if n_lm < min_cells or not gated:
            rec["skipped"] = "no usable landmarks from this fold's markers"
            fold_records.append(rec)
            if verbose:
                print(f"  [lambda]{tag} fold {f}: skipped ({n_lm} landmarks, {len(gated)} types)")
            continue

        idx = _subsample(labels, max_pixels, seed=f)
        labels_sub = labels[idx]
        kw = dict(fit_kwargs)
        kw["spatial_counts"] = fit_kwargs["spatial_counts"][idx]
        kw["spatial_numi"] = fit_kwargs["spatial_numi"][idx]
        kw["pixel_mask"] = None
        Yp = protein_std[np.ix_(idx, keep)]
        Pp = protein_profiles[keep, :]
        it2 = inv_tau2[keep]
        pm = None if protein_mask is None else np.asarray(protein_mask)[idx]

        def _fit(lam):
            return run_doublet(
                **kw,
                protein_profiles=Pp,
                protein_intensity=Yp,
                inv_tau2=it2,
                protein_lambda=lam,
                protein_mask=pm,
            )

        # Baseline first, and FIX the evaluation set to the landmark cells the
        # RNA-only fit was willing to call. Without that, any strong extra
        # likelihood term wins by widening the singlet/reject gap: cells flip from
        # reject (scored as a miss) to a call that is right ~1/K of the time by
        # luck. Measured on the column-shuffled fixture in
        # tests/test_protein_landmarks.py.
        base = _fit(0.0)
        called = base.spot_class != SPOT_CLASS_REJECT
        labels_eval = np.where(called, labels_sub, -1)
        n_eval = int((labels_eval >= 0).sum())
        rec["n_evaluated"] = n_eval
        rec["baseline_reject_frac"] = float((~called).mean())
        if n_eval < min_cells:
            rec["skipped"] = "too few landmark cells survive the RNA-only reject filter"
            fold_records.append(rec)
            if verbose:
                print(f"  [lambda]{tag} fold {f}: skipped ({n_eval} evaluable landmarks)")
            continue

        for lam in grid:
            res = base if lam == 0.0 else _fit(lam)
            macro, per_type = landmark_macro_f1(labels_eval, _pred(res), res.spot_class)
            scores[lam].append(macro)
            rec["scores"][lam] = {
                "macro_f1": macro,
                "per_type": per_type,
                "reject_frac": float((res.spot_class == SPOT_CLASS_REJECT).mean()),
            }
            if verbose:
                print(f"  [lambda]{tag} fold {f} lambda={lam:g}: F1 = {macro:.4f} (n={n_eval})")
        fold_records.append(rec)

    mean_f1 = {
        lam: (float(np.mean(v)) if v and np.isfinite(v).all() else float("nan"))
        for lam, v in scores.items()
    }
    return mean_f1, fold_records


def select_protein_weight(
    *,
    run_doublet,
    fit_kwargs: dict,
    protein_std: np.ndarray,
    protein_profiles: np.ndarray,
    inv_tau2: np.ndarray,
    protein_mask: np.ndarray,
    feature_names: list[str],
    signatures: dict,
    cell_type_names: list[str],
    reliability: np.ndarray | None = None,
    lambda_grid=(0.0, 0.5, 1.0, 2.0, 4.0, 8.0),
    n_folds: int = 2,
    min_cells: int = 20,
    max_pixels: int | None = 10000,
    n_null: int = 6,
    verbose: bool = True,
    type_to_class: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> tuple[float, dict]:
    """Pick lambda by measured accuracy on held-out landmark cells, against a
    permutation null.

    For each marker fold: gate truth from that fold's markers only, drop those
    markers from the fit, sweep ``lambda_grid``, and score macro F1 on the
    landmark cells the RNA-only fit was willing to CALL (fixed across the sweep,
    so a lambda cannot win by rejecting less).

    Then the part that is easy to skip and shouldn't be: **the same sweep is run
    on ``n_null`` column-shuffled copies of the panel**. Shuffling each marker
    independently keeps its marginal distribution and destroys which cell carries
    which value, so those sweeps measure what a NON-informative panel scores on
    this exact dataset. Their gain is centred at zero but spread widely when
    landmarks are few (sd ~0.05-0.08 at 36 landmarks per fold), which is why a
    single null is useless and a "beats lambda = 0" rule accepted pure noise. The
    observed gain must beat the largest null gain.

    Args:
        run_doublet: ``_doublet.run_doublet_mode``, injected to avoid a circular
            import.
        fit_kwargs: the RNA side (``spatial_counts``, ``spatial_numi``,
            ``norm_profiles``, likelihood tables, ``config``, ``batch_size``,
            ``device``); subsampled per fold.
        n_null: permutation-null realisations. The observed gain must beat the
            LARGEST gain any of them achieves, so this is a one-sided permutation
            test at p <= 1/(n_null+1). Costs (n_null+1)x the sweep, which is the
            price of a lambda you can defend; lower it to 3 for a weaker test, or
            0 to skip the null entirely and accept any gain over lambda=0 (as
            optimistic as the gradient heuristic it replaces).
        type_to_class, class_names: score at a coarser level - gate landmarks per
            class (``signatures`` keyed by class) and map ``first_type`` through
            ``type_to_class`` (K,) first. Use when the panel can only adjudicate
            lineages; a type mapped to -1 never matches a landmark.

    Returns:
        ``(best_lambda, curve)``. ``curve`` carries ``mean_f1``, ``gain``,
        ``null_mean_f1``, ``null_gain``, and the per-fold records (landmark counts,
        which types each fold could gate, reject fractions), so a choice can be
        audited instead of trusted. ``best_lambda`` is ``0.0`` when nothing beats
        the null - a real answer, not a failure.
    """
    grid = [float(x) for x in lambda_grid]
    if 0.0 not in grid:
        grid = [0.0] + grid  # the baseline is not optional
    common = dict(
        run_doublet=run_doublet,
        fit_kwargs=fit_kwargs,
        protein_profiles=protein_profiles,
        inv_tau2=inv_tau2,
        protein_mask=protein_mask,
        feature_names=feature_names,
        signatures=signatures,
        cell_type_names=cell_type_names,
        reliability=reliability,
        grid=grid,
        n_folds=n_folds,
        min_cells=min_cells,
        max_pixels=max_pixels,
        verbose=verbose,
        type_to_class=None if type_to_class is None else np.asarray(type_to_class),
        class_names=class_names,
    )
    mean_f1, folds = _sweep_folds(protein_std=protein_std, **common)
    curve: dict = {"grid": grid, "mean_f1": mean_f1, "folds": folds}

    usable = {lam: v for lam, v in mean_f1.items() if np.isfinite(v)}
    if not usable or not np.isfinite(mean_f1.get(0.0, np.nan)):
        curve["best"] = 0.0
        curve["reason"] = "no fold produced usable landmarks; protein not shown to help"
        if verbose:
            print("  [lambda] no usable landmarks — falling back to lambda=0 (RNA-only)")
        return 0.0, curve

    base_f1 = mean_f1[0.0]
    gains = {lam: v - base_f1 for lam, v in usable.items() if lam > 0.0}
    curve["gain"] = max(gains.values()) if gains else 0.0

    null_gain = 0.0
    if n_null > 0:
        if verbose:
            print(f"  [lambda] permutation null: {n_null} shuffled-panel sweep(s)")
        per_null = []
        curve["null_mean_f1"] = []
        for r in range(n_null):
            null_f1, _ = _sweep_folds(
                protein_std=_column_shuffle(protein_std, seed=r),
                tag=f" [null {r}]",
                **common,
            )
            curve["null_mean_f1"].append(null_f1)
            if not np.isfinite(null_f1.get(0.0, np.nan)):
                continue
            g = [v - null_f1[0.0] for lam, v in null_f1.items() if lam > 0.0 and np.isfinite(v)]
            if g:
                per_null.append(max(g))
        curve["null_gains"] = per_null
        if per_null:
            arr = np.asarray(per_null, dtype=float)
            curve["null_mean"] = float(arr.mean())
            curve["null_sd"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            null_gain = float(arr.max())
    curve["null_gain"] = null_gain
    if "null_sd" in curve and curve["null_sd"] > 0:
        curve["z"] = (curve["gain"] - curve["null_mean"]) / curve["null_sd"]

    winners = {lam: v for lam, v in usable.items() if lam > 0.0 and (v - base_f1) > null_gain}
    best = max(winners, key=lambda lam: winners[lam]) if winners else 0.0
    curve["best"] = best
    if verbose:
        if best == 0.0:
            print(
                f"  [lambda] selected lambda=0 (RNA-only): best gain "
                f"{curve['gain']:+.4f} does not beat the permutation null "
                f"{null_gain:+.4f}"
            )
        else:
            z = curve.get("z")
            print(
                f"  [lambda] selected lambda={best:g}: F1 {usable[best]:.4f} "
                f"(gain {usable[best] - base_f1:+.4f} vs null max {null_gain:+.4f}"
                + (f", z={z:.2f})" if z is not None else ")")
            )
    return best, curve
