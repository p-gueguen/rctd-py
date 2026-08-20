"""Protein (immunofluorescence) modality helpers for multi-modal RCTD.

Continuous IF intensity is fused into RCTD as a Gaussian / weighted-least-squares
likelihood block that shares the per-pixel cell-type weights (see the protein
blocks in ``_irwls.py``). This module provides the two pieces that live outside
the solver:

* :func:`normalize_protein` - per-marker standardization. The default
  ``arcsinh_robust`` (per-marker arcsinh + median/MAD z-score) is preferred over
  per-cell CLR for IF: with a handful of markers the per-cell geometric mean
  couples channels, and CLR applies no per-marker exposure/autofluorescence
  correction. CLR is kept as an alternative for A/B comparison.
* :func:`bootstrap_protein_profiles` - build per-type protein profiles ``P_prot``
  from the confident singlets of an RNA-only first pass (no external CITE-seq
  reference required).
"""

import numpy as np


def normalize_protein(
    P_raw: np.ndarray,
    method: str = "arcsinh_robust",
    background: np.ndarray | None = None,
    cofactor: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize raw IF intensity, per marker.

    Args:
        P_raw: (N, M) raw intensity matrix; rows may be all-NaN for cells with no
            protein measurement.
        method: ``"arcsinh_robust"`` (default) or ``"clr"``.
        background: optional (M,) per-marker background subtracted before the
            arcsinh transform (autofluorescence / isotype floor). ``None`` = no-op.
        cofactor: arcsinh cofactor (controls the linear-to-log transition).

    Returns:
        ``(P_std, tau, valid_mask)``:
            * ``P_std`` (N, M): standardized values. Non-finite / invalid entries
              are set to 0 so they contribute nothing to the WLS block.
            * ``tau`` (M,): per-marker scale (1.0 after robust-z; refined by the
              bootstrap to the pooled within-type residual std).
            * ``valid_mask`` (N,): pixels with at least one finite marker.
    """
    P_raw = np.asarray(P_raw, dtype=np.float64)
    if P_raw.ndim != 2:
        raise ValueError(f"P_raw must be 2D (N, M), got shape {P_raw.shape}")
    N, M = P_raw.shape
    valid_mask = np.isfinite(P_raw).any(axis=1)

    if method == "arcsinh_robust":
        x = P_raw.copy()
        if background is not None:
            x = np.maximum(x - np.asarray(background, dtype=np.float64)[None, :], 0.0)
        a = np.arcsinh(x / cofactor)
        P_std = np.zeros_like(a)
        tau = np.ones(M)
        for m in range(M):
            col = a[:, m]
            finite = np.isfinite(col)
            if finite.sum() < 2:
                continue
            med = np.median(col[finite])
            mad = 1.4826 * np.median(np.abs(col[finite] - med))
            if mad <= 1e-9:
                std = np.std(col[finite])
                mad = std if std > 1e-9 else 1.0
            P_std[:, m] = (col - med) / mad
    elif method == "clr":
        # Per-cell centered log-ratio (the user's current project default). Kept
        # for A/B testing; questionable for continuous IF (see module docstring).
        x = np.clip(P_raw, 0.0, None) + 1.0
        logx = np.log(x)
        P_std = logx - logx.mean(axis=1, keepdims=True)
        tau = np.ones(M)
    else:
        raise ValueError(f"unknown protein normalization method: {method!r}")

    P_std = np.where(np.isfinite(P_std), P_std, 0.0)
    P_std[~valid_mask] = 0.0
    return P_std.astype(np.float64), tau, valid_mask


def bootstrap_protein_profiles(
    protein_std: np.ndarray,
    singlet_type_idx: np.ndarray,
    n_types: int,
    min_cells: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-type protein profiles from confident RNA-only singlets.

    For each cell type k, the profile is the mean standardized protein over the
    confident singlets assigned to k. Types with fewer than ``min_cells`` confident
    singlets get a zero profile (they then contribute no protein signal, so the
    solve falls back to RNA for those types).

    Args:
        protein_std: (N, M) standardized protein (output of :func:`normalize_protein`).
        singlet_type_idx: (N,) int type index of the confident singlet, ``-1`` for
            cells that are not confident singlets (excluded from the estimate).
        n_types: number of cell types K.
        min_cells: minimum confident singlets required to estimate a type's profile.

    Returns:
        ``(P_prot, tau, n_used)``:
            * ``P_prot`` (M, K): mean standardized protein per type, columns aligned
              to the reference cell-type order.
            * ``tau`` (M,): pooled within-type residual std (the WLS weight is
              1/tau_m^2), floored to stay finite.
            * ``n_used`` (K,): number of confident singlets used per type.
    """
    protein_std = np.asarray(protein_std, dtype=np.float64)
    N, M = protein_std.shape
    singlet_type_idx = np.asarray(singlet_type_idx)

    P_prot = np.zeros((M, n_types), dtype=np.float64)
    n_used = np.zeros(n_types, dtype=int)
    resid_blocks = []
    for k in range(n_types):
        sel = singlet_type_idx == k
        c = int(sel.sum())
        n_used[k] = c
        if c >= min_cells:
            mu = protein_std[sel].mean(axis=0)
            P_prot[:, k] = mu
            resid_blocks.append(protein_std[sel] - mu[None, :])

    if resid_blocks:
        pooled = np.concatenate(resid_blocks, axis=0)
        tau = np.sqrt(np.maximum(pooled.var(axis=0), 1e-12))
    else:
        tau = np.ones(M)
    return P_prot, tau, n_used


def build_signed_profile(
    cell_type_names: list[str],
    feature_names: list[str],
    signatures: dict,
    magnitude: float = 1.5,
    levels: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a curated signed protein profile (M x K, in standardized z-units) from
    positive/negative marker sets.

    Unlike the bootstrap (which inherits the RNA reference's errors), a curated profile
    can encode NEGATIVE markers - e.g. NK = CD16 positive but CD3E/CD8A negative - which is
    what lets protein actively REJECT a contaminant rather than just sharpen.

    Args:
        cell_type_names: K reference cell types (column order of the profile).
        feature_names: M protein markers (row order; must match the spatial protein columns).
        signatures: ``{cell_type: {"positive": [markers], "negative": [markers]}}``. Markers
            absent from ``feature_names`` are ignored; cell types absent from ``signatures``
            get an all-zero (neutral) column.
        magnitude: the +/- z value written for positive / negative markers. Used
            only when ``levels`` is None.
        levels: optional ``(pos_level (M,), neg_level (M,))`` from
            :func:`calibrate_signed_levels` - PER-MARKER levels measured from
            landmark cells, which override the single global ``magnitude``.

    Returns:
        ``(P_prot (M, K), curated_mask (K,) bool)`` - ``curated_mask[k]`` is True iff type k had
        a signature (so a caller can fill the rest from a bootstrap pass: hybrid mode).
    """
    M, K = len(feature_names), len(cell_type_names)
    fidx = {f: i for i, f in enumerate(feature_names)}
    if levels is None:
        pos_level = np.full(M, float(magnitude))
        neg_level = np.full(M, -float(magnitude))
    else:
        pos_level, neg_level = (np.asarray(a, dtype=np.float64) for a in levels)
        if pos_level.shape != (M,) or neg_level.shape != (M,):
            raise ValueError(
                f"levels must both be ({M},), got {pos_level.shape} / {neg_level.shape}"
            )
    P = np.zeros((M, K), dtype=np.float64)
    mask = np.zeros(K, dtype=bool)
    for k, t in enumerate(cell_type_names):
        spec = signatures.get(t)
        if not spec:
            continue
        mask[k] = True
        for mk in spec.get("positive", []):
            if mk in fidx:
                P[fidx[mk], k] = pos_level[fidx[mk]]
        for mk in spec.get("negative", []):
            if mk in fidx:
                P[fidx[mk], k] = neg_level[fidx[mk]]
    return P, mask


def scgate_signatures(
    master_table_path: str,
    type_to_model: dict,
    gene2protein: dict,
    panel: list[str] | None = None,
) -> dict:
    """Derive curated signed PROTEIN signatures from carmonalab/scGate_models gating models.

    scGate models are modular: a per-lineage ``*_scGate_Model.tsv`` references named signatures
    (positive/negative across hierarchical levels), and ``master_table.tsv`` maps each signature
    name to genes (a trailing ``-`` marks a NEGATIVE gene). This parses both, nets the signed
    gene votes per cell type, and maps genes to a protein panel.

    Args:
        master_table_path: path to the collection's ``master_table.tsv``.
        type_to_model: ``{cell_type_name: path to that type's *_scGate_Model.tsv}``.
        gene2protein: ``{gene: protein_marker}`` mapping (drop pan-lineage genes with no specific
            protein, e.g. LCK/SPI1, so they don't cancel real gates).
        panel: optional list of protein markers to restrict to.

    Returns:
        ``{cell_type: {"positive": [markers], "negative": [markers]}}`` for use with
        :func:`build_signed_profile` / ``RCTDConfig(protein_signatures=...)``.
    """
    sig2genes: dict[str, list] = {}
    with open(master_table_path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2 or p[0] == "name":
                continue
            genes = []
            for g in p[1].split(";"):
                g = g.strip()
                if g:
                    genes.append((g[:-1], -1) if g.endswith("-") else (g, 1))
            sig2genes[p[0]] = genes

    out = {}
    for t, mpath in type_to_model.items():
        votes: dict[str, int] = {}
        with open(mpath) as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) < 3 or p[0] == "levels":
                    continue
                use_as, name = p[1], p[2]
                for gene, gsign in sig2genes.get(name, []):
                    if use_as == "positive":
                        votes[gene] = votes.get(gene, 0) + gsign
                    elif gsign > 0:  # negative gate -> that lineage's markers should be low
                        votes[gene] = votes.get(gene, 0) - 1
        prot: dict[str, int] = {}
        for gene, v in votes.items():
            pm = gene2protein.get(gene)
            if pm is None or (panel is not None and pm not in panel):
                continue
            prot[pm] = prot.get(pm, 0) + int(np.sign(v))
        out[t] = {
            "positive": [m for m, s in prot.items() if s > 0],
            "negative": [m for m, s in prot.items() if s < 0],
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Spatial spillover: per-cell protein reliability
# ─────────────────────────────────────────────────────────────────────────────


def _knn_indices(coords: np.ndarray, k: int = 6) -> np.ndarray:
    """(N, k_eff) indices of each cell's k nearest OTHER cells.

    Uses ``scipy.spatial.cKDTree`` (scipy is already a hard dependency). Do NOT
    swap this for the dense pairwise-distance kNN in ``_multimodal._morans_i``:
    that materializes an N x N matrix and dies on a 60k-cell Xenium section.
    """
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coords must be (N, >=2), got shape {coords.shape}")
    N = coords.shape[0]
    if N < 2:
        raise ValueError(f"need >= 2 cells for a kNN graph, got {N}")
    k_eff = min(k, N - 1)

    # query k_eff+1 because the point itself is always among its own neighbours
    _, idx = cKDTree(coords).query(coords, k=k_eff + 1)
    idx = np.atleast_2d(idx)
    # Drop the self column. With duplicated coordinates self is not necessarily
    # column 0, so locate it per row rather than slicing [:, 1:].
    self_col = idx == np.arange(N)[:, None]
    first_self = np.argmax(self_col, axis=1)
    keep = np.ones(idx.shape, dtype=bool)
    keep[np.arange(N), first_self] = False
    return idx[keep].reshape(N, -1)[:, :k_eff]


def neighbour_reliability(
    protein_std: np.ndarray,
    coords: np.ndarray,
    k: int = 6,
    high_pct: float = 75.0,
    floor: float = 0.05,
) -> np.ndarray:
    """Per-cell protein reliability in ``[floor, 1]`` from the neighbour-max ratio.

    CellTune's anti-spillover test (Bussi et al., Nat Methods 2026): a marker
    value is only credible as cell-intrinsic if it stands above the SAME marker
    in the cell's immediate neighbours - otherwise it is bleed-through from a
    bright neighbour across an imperfect segmentation boundary.

    Per cell, over the markers it reads high on (above the per-marker
    ``high_pct`` percentile of the positive part), we take
    ``own / max(neighbour)`` and reduce with the median. Values are computed on
    ``max(z, 0)``: a robust-z of 0 is the marker's median, so the positive part
    is the natural "how far above background" scale, and a ratio in those units
    separates a source (z=6 beside z=2 -> 0.33) from two genuine neighbouring
    positives (z=6 beside z=5.5 -> 0.92).

    Cells with no high marker get 1.0 - there is nothing to doubt, and their
    protein is uninformative anyway.

    Args:
        protein_std: (N, M) standardized protein, output of :func:`normalize_protein`.
        coords: (N, 2) spatial coordinates, same row order.
        k: neighbours per cell.
        high_pct: percentile (over the positive part of each marker) above which a
            cell counts as reading high on that marker.
        floor: lower clamp, so a cell is never fully stripped of its protein term.

    Returns:
        ``r`` (N,) float64 in ``[floor, 1.0]``. Feed it to
        ``RCTDConfig(protein_reliability="neighbour_ratio")``, or multiply it into
        the ``protein_mask`` handed to the solver.
    """
    P = np.asarray(protein_std, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"protein_std must be 2D (N, M), got {P.shape}")
    N, M = P.shape
    if np.asarray(coords).shape[0] != N:
        raise ValueError(f"coords has {np.asarray(coords).shape[0]} rows, protein_std has {N}")

    x = np.clip(np.where(np.isfinite(P), P, 0.0), 0.0, None)  # (N, M) above-median part
    nn = _knn_indices(coords, k=k)  # (N, k_eff)
    nbr_max = x[nn].max(axis=1)  # (N, M) brightest neighbour per marker

    # Per-marker "reads high" cut, taken over the positive part only so a marker
    # that is off in most cells does not set an absurdly low bar.
    hi = np.zeros((N, M), dtype=bool)
    for m in range(M):
        pos = x[:, m] > 0
        if pos.sum() < 2:
            continue
        hi[:, m] = x[:, m] > np.percentile(x[pos, m], high_pct)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(nbr_max > 1e-12, x / np.maximum(nbr_max, 1e-12), 1.0)
    ratio = np.clip(ratio, 0.0, 1.0)

    r = np.ones(N, dtype=np.float64)
    any_hi = hi.any(axis=1)
    if any_hi.any():
        # only rows with at least one high marker — an all-NaN nanmedian slice is
        # both a warning and meaningless
        med = np.nanmedian(np.where(hi[any_hi], ratio[any_hi], np.nan), axis=1)
        r[any_hi] = np.where(np.isfinite(med), med, 1.0)
    return np.clip(r, floor, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Landmark cells: an internal gold standard gated on protein alone
# ─────────────────────────────────────────────────────────────────────────────


def _percentile_ranks(P: np.ndarray) -> np.ndarray:
    """(N, M) per-marker percentile rank in [0, 1] (ties broken by sort order)."""
    N = P.shape[0]
    if N < 2:
        return np.zeros_like(P)
    order = np.argsort(P, axis=0, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.arange(N)[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(rows, P.shape).astype(np.float64), axis=0)
    return ranks / (N - 1)


def marker_folds(feature_names: list[str], n_folds: int = 2) -> list[np.ndarray]:
    """Deterministic round-robin partition of marker indices into ``n_folds`` folds.

    Used for leave-markers-out landmark evaluation: gate truth with fold f, fit
    WITHOUT fold f. Deterministic (no RNG) so a reported score is reproducible.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    idx = np.arange(len(feature_names))
    return [idx[f::n_folds] for f in range(n_folds)]


def gate_landmarks(
    protein_std: np.ndarray,
    feature_names: list[str],
    signatures: dict,
    cell_type_names: list[str],
    reliability: np.ndarray | None = None,
    min_cells: int = 20,
    t_start: float = 0.95,
    t_min: float = 0.70,
    t_step: float = 0.05,
    min_reliability: float = 0.5,
    restrict_markers: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Gate high-confidence landmark cells from protein alone (CellTune's
    automated landmarking, ported).

    A cell is a landmark for type k when EVERY positive marker of k sits above
    the threshold and EVERY negative marker sits below its mirror, in per-marker
    percentiles. Thresholds start strict (0.95) and relax by ``t_step`` until the
    type reaches ``min_cells`` - CellTune's ladder, which is what keeps rare
    types representable without loosening the common ones.

    Two things make these labels usable as truth rather than as another
    prediction: a cell claimed by more than one type is DROPPED (truth must be
    unambiguous), and when ``reliability`` is supplied a cell whose signal is
    bleed from a neighbour cannot become truth.

    ``restrict_markers`` limits gating to a subset of marker columns - pass one
    fold from :func:`marker_folds` and fit without that fold, or the landmarks
    trivially agree with whatever the protein term already says.

    Args:
        protein_std: (N, M) standardized protein.
        feature_names: M marker names (column order).
        signatures: ``{cell_type: {"positive": [...], "negative": [...]}}``.
        cell_type_names: K reference type names; labels index into this list.
        reliability: optional (N,) from :func:`neighbour_reliability`.
        min_cells: per-type target before the ladder stops relaxing.
        t_start, t_min, t_step: the relaxation ladder.
        min_reliability: cells below this are never landmarks.
        restrict_markers: optional marker-column indices to gate on.

    Returns:
        ``(labels, info)``. ``labels`` (N,) int: type index, or -1 for "not a
        landmark". ``info`` carries per-type threshold / count / markers used, the
        conflict count, and the list of types that never reached ``min_cells`` -
        report those, never silently drop them.
    """
    P = np.asarray(protein_std, dtype=np.float64)
    N, M = P.shape
    if len(feature_names) != M:
        raise ValueError(f"{len(feature_names)} feature names for {M} protein columns")

    u = _percentile_ranks(np.where(np.isfinite(P), P, 0.0))
    fidx = {f: i for i, f in enumerate(feature_names)}
    allowed = None if restrict_markers is None else set(np.asarray(restrict_markers).tolist())

    usable = np.ones(N, dtype=bool)
    if reliability is not None:
        usable &= np.asarray(reliability, dtype=np.float64) >= min_reliability

    ladder = []
    t = t_start
    while t >= t_min - 1e-9:
        ladder.append(round(t, 4))
        t -= t_step

    claims = np.zeros((N, len(cell_type_names)), dtype=bool)
    info: dict = {"per_type": {}, "underpopulated": [], "n_conflicts": 0, "ladder": ladder}

    for k, tname in enumerate(cell_type_names):
        spec = signatures.get(tname) or {}
        pos = [fidx[m] for m in spec.get("positive", []) if m in fidx]
        neg = [fidx[m] for m in spec.get("negative", []) if m in fidx]
        if allowed is not None:
            pos = [i for i in pos if i in allowed]
            neg = [i for i in neg if i in allowed]
        if not pos:
            info["per_type"][tname] = {"n": 0, "threshold": None, "reason": "no positive marker"}
            continue

        chosen = None
        for t in ladder:
            ok = usable.copy()
            for i in pos:
                ok &= u[:, i] >= t
            for i in neg:
                ok &= u[:, i] <= 1.0 - t
            chosen = (t, ok)
            if int(ok.sum()) >= min_cells:
                break

        t_used, ok = chosen
        n = int(ok.sum())
        claims[:, k] = ok
        info["per_type"][tname] = {
            "n": n,
            "threshold": t_used,
            "positive": [feature_names[i] for i in pos],
            "negative": [feature_names[i] for i in neg],
        }
        if n < min_cells:
            info["underpopulated"].append(tname)

    n_claims = claims.sum(axis=1)
    labels = np.full(N, -1, dtype=np.int64)
    single = n_claims == 1
    labels[single] = np.argmax(claims[single], axis=1)
    info["n_conflicts"] = int((n_claims > 1).sum())
    info["n_landmarks"] = int(single.sum())
    return labels, info


def calibrate_signed_levels(
    protein_std: np.ndarray,
    feature_names: list[str],
    signatures: dict,
    cell_type_names: list[str],
    labels: np.ndarray,
    min_cells: int = 10,
    fallback_pct: tuple[float, float] = (90.0, 10.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Per-marker positive / negative levels in standardized units, measured from
    landmark cells instead of assumed.

    ``protein_signature_magnitude`` writes one global +/- z (1.5 by default) for
    every marker of every type. Real positive populations sit wherever they sit,
    and a magnitude that understates the true separation has to be paid for with
    a larger lambda - which over-weights the whole panel, not just that marker.
    This is CellTune's per-marker positivity calibration, minus the CNN: the
    level is the median standardized value of the landmark cells that should be
    positive (resp. negative) for that marker.

    Markers with too few landmarks on a side fall back to the ``fallback_pct``
    percentiles of the marker itself.

    Returns:
        ``(pos_level (M,), neg_level (M,))`` for :func:`build_signed_profile`.
    """
    P = np.asarray(protein_std, dtype=np.float64)
    M = P.shape[1]
    labels = np.asarray(labels)
    fidx = {f: i for i, f in enumerate(feature_names)}

    pos_types: dict[int, list[int]] = {}
    neg_types: dict[int, list[int]] = {}
    for k, tname in enumerate(cell_type_names):
        spec = signatures.get(tname) or {}
        for mk in spec.get("positive", []):
            if mk in fidx:
                pos_types.setdefault(fidx[mk], []).append(k)
        for mk in spec.get("negative", []):
            if mk in fidx:
                neg_types.setdefault(fidx[mk], []).append(k)

    hi_fb, lo_fb = fallback_pct
    pos_level = np.zeros(M)
    neg_level = np.zeros(M)
    for m in range(M):
        col = P[:, m]
        finite = np.isfinite(col)
        sel_p = np.isin(labels, pos_types.get(m, [])) & finite
        sel_n = np.isin(labels, neg_types.get(m, [])) & finite
        pos_level[m] = (
            np.median(col[sel_p])
            if sel_p.sum() >= min_cells
            else np.percentile(col[finite], hi_fb)
            if finite.any()
            else 0.0
        )
        neg_level[m] = (
            np.median(col[sel_n])
            if sel_n.sum() >= min_cells
            else np.percentile(col[finite], lo_fb)
            if finite.any()
            else 0.0
        )
    return pos_level, neg_level
