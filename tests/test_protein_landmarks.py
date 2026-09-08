"""Landmark cells, per-marker calibration, spillover reliability, and the
modality-disagreement flag - the CellTune-derived additions to the protein path.

Every claim here is checked against something the model did not produce. The two
controls matter more than the accuracy numbers:

* POSITIVE control - the gate must recover planted labels. A gate that finds
  nothing looks exactly like clean data.
* NEGATIVE control - with the protein rows shuffled, the gate's precision must
  collapse to chance and lambda selection must pick 0. If a shuffled modality
  still wins lambda > 0, the selection is rewarding protein for merely existing
  and the whole harness is worthless.
"""

import anndata
import numpy as np
import pytest
import torch

from rctd import (
    RCTD,
    RCTDConfig,
    Reference,
    calibrate_signed_levels,
    gate_landmarks,
    landmark_macro_f1,
    marker_folds,
    neighbour_reliability,
    run_rctd,
)
from rctd._irwls import solve_irwls_batch_shared
from rctd._likelihood import build_x_vals, compute_q_matrix, compute_spline_coefficients
from rctd._types import SPOT_CLASS_REJECT

TYPES = ["A", "B", "C"]
SIGS = {
    "A": {"positive": ["m0"], "negative": ["m1"]},
    "B": {"positive": ["m1"], "negative": ["m0"]},
    "C": {"positive": ["m2", "m3"], "negative": []},
}
NAMES = [f"m{i}" for i in range(6)]


def _planted_protein(n=600, seed=0, separation=6.0):
    """(Z, truth): standardized protein with three cleanly separable populations."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 3, n)
    Z = rng.normal(0, 1, (n, len(NAMES)))
    Z[truth == 0, 0] += separation
    Z[truth == 0, 1] -= 3.0
    Z[truth == 1, 1] += separation
    Z[truth == 1, 0] -= 3.0
    Z[truth == 2, 2] += separation
    Z[truth == 2, 3] += separation
    return Z, truth


@pytest.fixture(scope="module")
def likelihood_tables():
    x_vals = build_x_vals()
    Q_mat = compute_q_matrix(sigma=1.0, x_vals=x_vals, K_val=100)
    SQ_mat = compute_spline_coefficients(Q_mat, x_vals)
    return torch.tensor(Q_mat), torch.tensor(SQ_mat), torch.tensor(x_vals)


# ── Landmark gating ──────────────────────────────────────────────────────────


@pytest.mark.protein
def test_gate_landmarks_recovers_planted_labels():
    """POSITIVE CONTROL: the gate must find the planted populations, and be right."""
    Z, truth = _planted_protein()
    labels, info = gate_landmarks(Z, NAMES, SIGS, TYPES, min_cells=20)

    ev = labels >= 0
    assert ev.sum() >= 60, f"gate found only {ev.sum()} landmarks — it cannot find anything"
    assert (labels[ev] == truth[ev]).mean() >= 0.95
    for t in TYPES:
        assert info["per_type"][t]["n"] >= 20, f"type {t} under-populated: {info['per_type'][t]}"
    assert info["underpopulated"] == []


@pytest.mark.protein
def test_gate_landmarks_shuffled_protein_is_chance():
    """NEGATIVE CONTROL: break the cell-to-protein correspondence and precision
    must fall to chance (~1/3 here). Proves the gate is reading the data, not the
    signature table."""
    Z, truth = _planted_protein()
    shuffled = Z[np.random.default_rng(1).permutation(Z.shape[0])]
    labels, _ = gate_landmarks(shuffled, NAMES, SIGS, TYPES, min_cells=20)
    ev = labels >= 0
    assert ev.sum() > 0  # the gate still fires — it just should not be right
    assert (labels[ev] == truth[ev]).mean() < 0.6


@pytest.mark.protein
def test_gate_landmarks_drops_ambiguous_cells():
    """A cell claimed by two types is not truth. It must be dropped, not assigned."""
    Z, _ = _planted_protein()
    Z[:20, 0] = 12.0  # simultaneously A-positive ...
    Z[:20, 1] = 12.0  # ... and B-positive, and neither's negative gate holds
    labels, info = gate_landmarks(Z, NAMES, SIGS, TYPES, min_cells=5)
    assert (labels[:20] == -1).all()


@pytest.mark.protein
def test_gate_landmarks_reports_ungateable_type():
    """A type with no positive marker in the panel is REPORTED, never silently
    dropped — a missing type in the truth set changes what a macro F1 means."""
    sigs = dict(SIGS)
    sigs["D"] = {"positive": ["not_in_panel"], "negative": []}
    Z, _ = _planted_protein()
    labels, info = gate_landmarks(Z, NAMES, sigs, TYPES + ["D"], min_cells=20)
    assert info["per_type"]["D"]["reason"] == "no positive marker"
    assert 3 not in np.unique(labels[labels >= 0])


@pytest.mark.protein
def test_marker_folds_partition_without_overlap():
    folds = marker_folds(NAMES, n_folds=2)
    assert len(folds) == 2
    assert sorted(np.concatenate(folds).tolist()) == list(range(len(NAMES)))
    assert set(folds[0].tolist()).isdisjoint(folds[1].tolist())


@pytest.mark.protein
def test_landmark_macro_f1_counts_reject_as_miss():
    """A rejected landmark cell must cost recall. Dropping rejects from the
    denominator would let a configuration win by refusing to call anything."""
    truth = np.array([0, 0, 1, 1, -1, -1])
    pred = np.array([0, 0, 1, 1, 2, 2])
    all_singlet = np.ones(6, dtype=int)
    f1_clean, _ = landmark_macro_f1(truth, pred, all_singlet)
    assert f1_clean == pytest.approx(1.0)

    sc = all_singlet.copy()
    sc[2] = SPOT_CLASS_REJECT  # a true type-1 landmark gets rejected
    f1_reject, per_type = landmark_macro_f1(truth, pred, sc)
    assert f1_reject < f1_clean
    assert per_type[1]["recall"] == pytest.approx(0.5)


# ── Per-marker calibration ───────────────────────────────────────────────────


@pytest.mark.protein
def test_calibrated_levels_recover_the_true_separation():
    """The fixed +/-1.5 magnitude is a guess. With a planted separation of z=6 the
    calibrated level must find ~6, not 1.5 — that gap is what a large lambda was
    silently paying for."""
    Z, _ = _planted_protein(separation=6.0)
    labels, _ = gate_landmarks(Z, NAMES, SIGS, TYPES, min_cells=20)
    pos, neg = calibrate_signed_levels(Z, NAMES, SIGS, TYPES, labels)
    for m in (0, 1, 2, 3):
        assert 4.5 < pos[m] < 7.5, f"marker {m} positive level {pos[m]:.2f} not near the planted 6"
    assert neg[0] < -1.5 and neg[1] < -1.5


@pytest.mark.protein
def test_calibrated_profile_at_low_lambda_matches_fixed_at_high_lambda(
    multimodal_synthetic_data, likelihood_tables
):
    """The Change-2 prediction, tested rather than asserted: a profile scaled to
    the data's real separation does at lambda=1 what an understated +/-1.5 profile
    needs a much larger lambda to do. If this fails, 'lambda=4 was compensating
    for a mis-scaled magnitude' is wrong and should be dropped from the docs."""
    d = multimodal_synthetic_data
    Q_mat, SQ_mat, x_vals = likelihood_tables
    P = torch.tensor(d["profiles"])
    common = dict(
        P=P,
        Y_batch=torch.tensor(d["counts"], dtype=P.dtype),
        nUMI_batch=torch.tensor(d["nUMI"], dtype=P.dtype),
        Q_mat=Q_mat,
        SQ_mat=SQ_mat,
        x_vals=x_vals,
        constrain=True,
        inv_tau2=torch.tensor(d["inv_tau2"], dtype=P.dtype),
        Y_prot_batch=torch.tensor(d["protein"], dtype=P.dtype),
    )
    i, j = d["degenerate_pair"]
    true_w = d["true_weights"]

    def mae(P_prot, lam):
        w = solve_irwls_batch_shared(**common, P_prot=torch.tensor(P_prot, dtype=P.dtype), lam=lam)[
            0
        ].numpy()
        return np.abs(w[:, [i, j]] - true_w[:, [i, j]]).mean()

    P_true = d["P_prot"]  # +/-3.0 on marker 0 — the real separation
    P_understated = np.sign(P_true) * 1.5  # what a fixed magnitude would write

    mae_calibrated_low_lam = mae(P_true, 1.0)
    mae_understated_low_lam = mae(P_understated, 1.0)
    mae_understated_high_lam = mae(P_understated, 4.0)

    assert mae_calibrated_low_lam < mae_understated_low_lam
    assert mae_calibrated_low_lam <= mae_understated_high_lam * 1.10


# ── Spillover: per-cell reliability ──────────────────────────────────────────


@pytest.mark.protein
def test_neighbour_reliability_separates_source_from_bleeder():
    rng = np.random.default_rng(4)
    n = 300
    coords = rng.uniform(0, 200, (n, 2))
    coords[1] = coords[0] + 0.4  # cell 1 sits on top of cell 0
    X = np.zeros((n, 4))
    X[0, 0] = 8.0  # the source
    X[1, 0] = 2.0  # bathed in the source's signal
    X[2:, 0] = rng.uniform(0.0, 0.5, n - 2)
    r = neighbour_reliability(X, coords, k=3)
    assert r[0] == pytest.approx(1.0)
    assert r[1] < 0.5
    assert r.min() >= 0.05 and r.max() <= 1.0


@pytest.mark.protein
def test_reliability_weight_recovers_a_contaminated_fit(
    multimodal_synthetic_data, likelihood_tables
):
    """SPILLOVER CASE: copy a neighbour's marker value into a fraction of cells.
    Trusting that protein equally hurts; downweighting those cells by their
    neighbour ratio must beat it."""
    d = multimodal_synthetic_data
    Q_mat, SQ_mat, x_vals = likelihood_tables
    rng = np.random.default_rng(9)
    N = d["protein"].shape[0]
    i, j = d["degenerate_pair"]
    true_w = d["true_weights"]

    # Place cells on a grid, then contaminate 30% of them with their neighbour's
    # marker-0 value (the marker that separates the RNA-degenerate pair).
    side = int(np.ceil(np.sqrt(N)))
    coords = np.stack(np.unravel_index(np.arange(N), (side, side)), axis=1).astype(float)
    protein = d["protein"].astype(np.float64).copy()
    victims = rng.choice(N, size=int(0.3 * N), replace=False)
    donors = (victims + 1) % N
    protein[victims, 0] = np.maximum(protein[donors, 0], protein[victims, 0]) + 2.0

    P = torch.tensor(d["profiles"])
    common = dict(
        P=P,
        Y_batch=torch.tensor(d["counts"], dtype=P.dtype),
        nUMI_batch=torch.tensor(d["nUMI"], dtype=P.dtype),
        Q_mat=Q_mat,
        SQ_mat=SQ_mat,
        x_vals=x_vals,
        constrain=True,
        P_prot=torch.tensor(d["P_prot"], dtype=P.dtype),
        Y_prot_batch=torch.tensor(protein, dtype=P.dtype),
        inv_tau2=torch.tensor(d["inv_tau2"], dtype=P.dtype),
        lam=1.0,
    )

    r = neighbour_reliability(protein, coords, k=4)
    assert r[victims].mean() < r.mean(), "the ratio must flag the contaminated cells"

    w_trusting = solve_irwls_batch_shared(**common)[0].numpy()
    w_weighted = solve_irwls_batch_shared(**common, prot_mask=torch.tensor(r, dtype=P.dtype))[
        0
    ].numpy()

    mae_trusting = np.abs(w_trusting[:, [i, j]] - true_w[:, [i, j]]).mean()
    mae_weighted = np.abs(w_weighted[:, [i, j]] - true_w[:, [i, j]]).mean()
    assert mae_weighted < mae_trusting


@pytest.mark.protein
def test_float_prot_mask_of_ones_equals_bool_mask(multimodal_synthetic_data, likelihood_tables):
    """The bool -> float generalization must not move the existing answer: an
    all-True bool and an all-ones float have to give the same weights."""
    d = multimodal_synthetic_data
    Q_mat, SQ_mat, x_vals = likelihood_tables
    P = torch.tensor(d["profiles"])
    N = d["protein"].shape[0]
    common = dict(
        P=P,
        Y_batch=torch.tensor(d["counts"], dtype=P.dtype),
        nUMI_batch=torch.tensor(d["nUMI"], dtype=P.dtype),
        Q_mat=Q_mat,
        SQ_mat=SQ_mat,
        x_vals=x_vals,
        constrain=True,
        P_prot=torch.tensor(d["P_prot"], dtype=P.dtype),
        Y_prot_batch=torch.tensor(d["protein"], dtype=P.dtype),
        inv_tau2=torch.tensor(d["inv_tau2"], dtype=P.dtype),
        lam=1.0,
    )
    w_bool = solve_irwls_batch_shared(**common, prot_mask=torch.ones(N, dtype=torch.bool))[0]
    w_float = solve_irwls_batch_shared(**common, prot_mask=torch.ones(N, dtype=P.dtype))[0]
    np.testing.assert_array_equal(w_bool.numpy(), w_float.numpy())


@pytest.mark.protein
def test_zero_reliability_equals_rna_only(multimodal_synthetic_data, likelihood_tables):
    """r = 0 for every cell must reproduce the RNA-only solve exactly — the same
    guarantee lambda=0 gives, through the other knob."""
    d = multimodal_synthetic_data
    Q_mat, SQ_mat, x_vals = likelihood_tables
    P = torch.tensor(d["profiles"])
    N = d["protein"].shape[0]
    rna_only = dict(
        P=P,
        Y_batch=torch.tensor(d["counts"], dtype=P.dtype),
        nUMI_batch=torch.tensor(d["nUMI"], dtype=P.dtype),
        Q_mat=Q_mat,
        SQ_mat=SQ_mat,
        x_vals=x_vals,
        constrain=True,
    )
    w_rna = solve_irwls_batch_shared(**rna_only)[0]
    w_zero = solve_irwls_batch_shared(
        **rna_only,
        P_prot=torch.tensor(d["P_prot"], dtype=P.dtype),
        Y_prot_batch=torch.tensor(d["protein"], dtype=P.dtype),
        inv_tau2=torch.tensor(d["inv_tau2"], dtype=P.dtype),
        lam=1.0,
        prot_mask=torch.zeros(N, dtype=P.dtype),
    )[0]
    np.testing.assert_array_equal(w_rna.numpy(), w_zero.numpy())


# ── lambda by measurement, and the disagreement flag (end to end) ────────────


def _degenerate_dataset(n_per_type=60, n_genes=90, seed=5, contradict=0.0, shuffle=None):
    """RNA where types A and B are INDISTINGUISHABLE, protein where they are not.

    Returns ``(spatial, reference_adata, truth)``. ``contradict`` gives that
    fraction of A cells B-like protein (used for the disagreement flag).

    ``shuffle`` breaks the protein in one of two very different ways:

    * ``"columns"`` - permute each marker independently. Destroys ALL structure:
      the panel now carries no information about anything. This is the negative
      control lambda selection must pass.
    * ``"rows"`` - permute whole cells. Each row stays internally consistent (a
      B-looking row is still B-looking across every marker) but is attached to
      the wrong cell. This is a mis-registered / mis-segmented panel, and the
      landmark harness CANNOT detect it - see the test below.
    """
    rng = np.random.default_rng(seed)
    K, N = 3, n_per_type * 3
    profiles = rng.exponential(0.001, size=(n_genes, K))
    block = n_genes // K
    for k in range(K):
        profiles[k * block : (k + 1) * block, k] *= 12.0
    profiles[:, 1] = profiles[:, 0]  # B is RNA-identical to A
    profiles /= profiles.sum(axis=0, keepdims=True)

    truth = np.repeat(np.arange(K), n_per_type)
    nUMI = rng.integers(2500, 5000, size=N)
    counts = np.zeros((N, n_genes), dtype=np.float32)
    for i in range(N):
        w = np.full(K, 0.01)
        w[truth[i]] = 1.0 - 0.01 * (K - 1)
        counts[i] = rng.poisson((profiles @ w) * nUMI[i])

    # protein: m0 -> A, m1 -> B, m2 & m3 -> C (so each marker fold can gate 2 types)
    protein = rng.normal(0.0, 1.0, size=(N, 4))
    protein[truth == 0, 0] += 7.0
    protein[truth == 1, 1] += 7.0
    protein[truth == 2, 2] += 7.0
    protein[truth == 2, 3] += 7.0
    if contradict > 0:
        a_cells = np.where(truth == 0)[0]
        liars = a_cells[: int(contradict * a_cells.size)]
        protein[liars, 0] -= 7.0  # looks like anything but A ...
        protein[liars, 1] += 7.0  # ... specifically, like B
    if shuffle == "rows":
        protein = protein[rng.permutation(N)]
    elif shuffle == "columns":
        for m in range(protein.shape[1]):
            protein[:, m] = protein[rng.permutation(N), m]
    elif shuffle is not None:
        raise ValueError(f"shuffle must be None, 'rows' or 'columns', got {shuffle!r}")

    spatial = anndata.AnnData(X=counts, obs={"i": np.arange(N)})
    spatial.var_names = [f"Gene_{i}" for i in range(n_genes)]
    spatial.obs_names = [f"Pix_{i}" for i in range(N)]
    spatial.obsm["protein"] = protein.astype(np.float32)
    spatial.uns["protein_feature_names"] = [f"m{m}" for m in range(4)]

    # reference: pure cells per type from the same profiles
    n_ref = 40 * K
    ref_counts = np.zeros((n_ref, n_genes), dtype=np.float32)
    ref_types = np.repeat(np.arange(K), 40)
    for i in range(n_ref):
        ref_counts[i] = rng.poisson(profiles[:, ref_types[i]] * 3000)
    ref = anndata.AnnData(X=ref_counts, obs={"cell_type": [TYPES[t] for t in ref_types]})
    ref.var_names = spatial.var_names
    ref.obs_names = [f"Ref_{i}" for i in range(n_ref)]
    return spatial, ref, truth


_SELECT_CFG = dict(
    compile=False,
    protein_profile_source="curated",
    protein_signatures=SIGS,
    protein_landmark_min_cells=10,
    protein_lambda_grid=(0.0, 1.0, 4.0),
    protein_lambda_max_pixels=None,
)


def _select(spatial, ref_adata):
    reference = Reference(ref_adata, cell_min=5, min_UMI=10)
    obj = RCTD(spatial, reference, RCTDConfig(protein_weight="landmark", **_SELECT_CFG))
    obj.fit_platform_effects()
    obj.prepare_protein()
    return obj, reference


def _accuracy_vs_planted(spatial, reference, truth, lam):
    """Independent yardstick, available only because this data is synthetic: how
    often the call matches the PLANTED type. Rejects count as misses."""
    cfg = {k: v for k, v in _SELECT_CFG.items() if not k.startswith("protein_lambda")}
    res = run_rctd(
        spatial,
        reference,
        mode="doublet",
        config=RCTDConfig(protein_weight=lam, **cfg),
        batch_size=200,
    )
    pred = np.where(res.spot_class == SPOT_CLASS_REJECT, -1, res.first_type)
    return float((pred == truth).mean())


@pytest.mark.protein
def test_landmark_lambda_selection_picks_protein_and_planted_truth_agrees():
    """With A and B RNA-identical, protein is the only thing that can separate
    them, so a measured lambda must come out > 0 - and the choice must hold up
    against the planted truth, which the selection never saw."""
    spatial, ref_adata, truth = _degenerate_dataset()
    obj, reference = _select(spatial, ref_adata)
    curve = obj.protein_lambda_curve

    assert curve is not None and curve["folds"], "no fold produced landmarks"
    assert obj.protein_lambda > 0.0, f"selection curve: {curve['mean_f1']}"
    assert curve["mean_f1"][obj.protein_lambda] > curve["mean_f1"][0.0] + 0.05
    # the gain has to beat the LARGEST gain any shuffled panel got on this data
    assert curve["gain"] > curve["null_gain"]
    assert curve["z"] > 2.0, f"gain {curve['gain']:+.4f} is only z={curve['z']:.2f} from the null"

    acc_rna = _accuracy_vs_planted(spatial, reference, truth, 0.0)
    acc_sel = _accuracy_vs_planted(spatial, reference, truth, obj.protein_lambda)
    assert acc_sel > acc_rna + 0.1, (
        f"landmark F1 improved but planted accuracy did not: {acc_rna:.3f} -> {acc_sel:.3f}"
    )


@pytest.mark.protein
def test_landmark_lambda_selection_rejects_uninformative_protein():
    """NEGATIVE CONTROL: permute each marker independently, so the panel carries
    no information about any cell, and lambda must come out 0.

    What this test taught, and why the selector looks the way it does: a noise
    panel still scored gain +0.039 here, and a single shuffled sweep drew +0.038,
    so "beats lambda = 0" and "beats one null" BOTH selected lambda = 4 on pure
    noise. The null gain is centred at zero (mean -0.001) but spreads as wide as a
    real effect at 36 landmarks per fold (sd 0.08). Only comparing against the
    largest of several nulls separates them: +0.039 (z = 0.5) loses to a null max
    of +0.077, while a real panel's +0.122 (z = 2.5) wins."""
    spatial, ref_adata, _ = _degenerate_dataset(shuffle="columns")
    obj, _ = _select(spatial, ref_adata)
    curve = obj.protein_lambda_curve
    assert len(curve["null_gains"]) >= 3, "the null must be sampled, not assumed"
    assert obj.protein_lambda == 0.0, (
        f"noise protein won lambda={obj.protein_lambda}: gain {curve['gain']:+.4f} "
        f"vs null max {curve['null_gain']:+.4f} (z={curve.get('z', float('nan')):.2f}), "
        f"curve {curve['mean_f1']}"
    )


@pytest.mark.protein
def test_landmark_truth_cannot_see_a_consistently_misassigned_panel():
    """The LIMIT of protein-gated truth, asserted rather than hoped for.

    Permute whole rows: every cell now carries some other cell's protein, but
    each row is still internally consistent. The landmark labels and the fit read
    the SAME wrong assignment, so the marker folds - which only stop a marker
    from scoring itself - cannot catch it, and the sweep may still land on
    lambda > 0.

    What does survive is the SCALE: a mis-assigned panel cannot reach the F1 a
    real one does, because the gated truth is then inconsistent with the RNA the
    fit also sees. Treat a low absolute landmark F1 as the warning it is; the
    harness assumes the protein-to-cell assignment itself is correct."""
    good_spatial, good_ref, _ = _degenerate_dataset()
    bad_spatial, bad_ref, _ = _degenerate_dataset(shuffle="rows")
    good, _ = _select(good_spatial, good_ref)
    bad, _ = _select(bad_spatial, bad_ref)

    f1_good = good.protein_lambda_curve["mean_f1"][good.protein_lambda]
    f1_bad = bad.protein_lambda_curve["mean_f1"][bad.protein_lambda]
    assert f1_good > 0.75, f"informative panel should score high, got {f1_good:.3f}"
    assert f1_bad < f1_good - 0.2, (
        f"a mis-assigned panel scored {f1_bad:.3f} against {f1_good:.3f} — the "
        "absolute F1 is the only signal that separates them, so it must separate them"
    )


@pytest.mark.protein
def test_modality_conflict_flags_contradicting_cells():
    """Cells given B-like protein while their RNA says A must show up as a
    modality conflict, and the confusion matrix must count them."""
    spatial, ref_adata, truth = _degenerate_dataset(contradict=0.5, seed=6)
    reference = Reference(ref_adata, cell_min=5, min_UMI=10)
    res = run_rctd(
        spatial,
        reference,
        mode="doublet",
        config=RCTDConfig(
            compile=False,
            protein_weight=1.0,
            protein_profile_source="curated",
            protein_signatures=SIGS,
        ),
        batch_size=200,
    )
    assert res.modality_conflict is not None
    assert res.rna_first_type is not None and res.prot_first_type is not None
    assert res.modality_confusion.shape == (len(TYPES), len(TYPES))

    scored = (res.rna_first_type >= 0) & (res.prot_first_type >= 0)
    assert res.modality_confusion.sum() == int(scored.sum())
    assert 0 < res.modality_conflict.sum() < scored.sum(), "flag fires for all or for none"

    # off-diagonal mass must exist: some cells' modalities pick different types
    off_diag = res.modality_confusion.sum() - np.trace(res.modality_confusion)
    assert off_diag > 0


@pytest.mark.protein
def test_landmark_classes_forgive_within_class_confusion():
    """protein_landmark_classes: gate and score landmarks per CLASS.

    A and B are RNA-identical, so at lambda=0 the fine-type F1 is poor. Mapped to
    one class "AB" (gated on m0, the A marker), that confusion is inside the class
    and the class-level F1 at lambda=0 must be clearly higher. The gated landmark
    report must be keyed by class, not by reference type."""
    spatial, ref_adata, _ = _degenerate_dataset()
    fine, _ = _select(spatial, ref_adata)

    reference = Reference(ref_adata, cell_min=5, min_UMI=10)
    cfg = RCTDConfig(
        protein_weight="landmark",
        protein_landmark_classes={"A": "AB", "B": "AB", "C": "C"},
        protein_landmark_signatures={
            "AB": {"positive": ["m0"], "negative": ["m2"]},
            "C": {"positive": ["m2", "m3"], "negative": []},
        },
        **_SELECT_CFG,
    )
    coarse = RCTD(spatial, reference, cfg)
    coarse.fit_platform_effects()
    coarse.prepare_protein()

    assert set(coarse.protein_landmark_info["per_type"]) == {"AB", "C"}
    f_fine = fine.protein_lambda_curve["mean_f1"][0.0]
    f_coarse = coarse.protein_lambda_curve["mean_f1"][0.0]
    assert f_coarse > f_fine + 0.15, f"fine {f_fine:.3f} vs class-level {f_coarse:.3f}"
    # the curated PROFILE still uses the per-type signatures (3 curated columns)
    assert coarse.reference.protein_profiles.shape[1] == 3


@pytest.mark.protein
def test_calibrated_refuses_class_level_landmarks():
    spatial, ref_adata, _ = _degenerate_dataset()
    reference = Reference(ref_adata, cell_min=5, min_UMI=10)
    cfg = RCTDConfig(
        protein_weight=1.0,
        protein_landmark_classes={"A": "AB", "B": "AB", "C": "C"},
        protein_landmark_signatures={"AB": {"positive": ["m0"], "negative": []}},
        **{**_SELECT_CFG, "protein_signature_magnitude": "calibrated"},
    )
    obj = RCTD(spatial, reference, cfg)
    obj.fit_platform_effects()
    with pytest.raises(ValueError, match="cannot be combined"):
        obj.prepare_protein()
