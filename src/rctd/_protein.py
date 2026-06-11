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
