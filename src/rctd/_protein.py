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
        magnitude: the +/- z value written for positive / negative markers.

    Returns:
        ``(P_prot (M, K), curated_mask (K,) bool)`` - ``curated_mask[k]`` is True iff type k had
        a signature (so a caller can fill the rest from a bootstrap pass: hybrid mode).
    """
    M, K = len(feature_names), len(cell_type_names)
    fidx = {f: i for i, f in enumerate(feature_names)}
    P = np.zeros((M, K), dtype=np.float64)
    mask = np.zeros(K, dtype=bool)
    for k, t in enumerate(cell_type_names):
        spec = signatures.get(t)
        if not spec:
            continue
        mask[k] = True
        for mk in spec.get("positive", []):
            if mk in fidx:
                P[fidx[mk], k] = magnitude
        for mk in spec.get("negative", []):
            if mk in fidx:
                P[fidx[mk], k] = -magnitude
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
