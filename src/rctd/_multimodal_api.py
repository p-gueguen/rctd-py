"""High-level multimodal RCTD API.

``run_rctd_multimodal`` mirrors ``run_rctd`` but accepts several
(spatial, reference) modality pairs that share the same K cell types, fits
each modality's platform effects and noise model independently, aligns the
common pixels across modalities, and runs a single joint IRWLS that shares
one cell-type weight vector per pixel across all modalities.
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata
import numpy as np
import torch

from rctd._irwls import solve_irwls_batch_shared
from rctd._multimodal import adaptive_modality_weights, solve_irwls_batch_multimodal
from rctd._rctd import RCTD
from rctd._types import RCTDConfig, resolve_device


@dataclass
class Modality:
    """One measurement modality for joint deconvolution.

    Attributes:
        name:      label (e.g. "RNA", "protein").
        spatial:   AnnData of spatial counts for this modality (cells/spots x features).
        reference: AnnData scRNA/protein reference with a cell-type column.
        beta:      relative weight of this modality in the joint objective (default 1.0).
    """

    name: str
    spatial: anndata.AnnData
    reference: anndata.AnnData
    beta: float = 1.0


@dataclass
class MultiModalResult:
    weights: np.ndarray  # (N, K) normalized cell-type proportions
    cell_type_names: list[str]
    converged: np.ndarray  # (N,) bool
    barcodes: np.ndarray  # (N,) shared pixel/cell ids used in the joint fit
    per_modality_sigma: dict  # name -> sigma/100
    modality_names: list[str]
    weights_se: np.ndarray | None = None  # (N, K) per-weight standard error, if requested


def run_rctd_multimodal(
    modalities: list[Modality],
    cell_type_col: str = "cell_type",
    config: RCTDConfig | None = None,
    batch_size: int = 10000,
    constrain: bool = True,
    return_uncertainty: bool = False,
    adaptive_weighting: str | None = None,
    fusion: str = "protein_wls",
    protein_lam: float | str = 1.0,
    protein_norm: str = "arcsinh_robust",
) -> MultiModalResult:
    """Joint RCTD deconvolution across several modalities.

    Each modality is fit through the stock RCTD platform-effect + sigma
    pipeline independently (its own DE genes, normalized profiles, and noise
    table). Pixels present in *all* modalities are then deconvolved jointly:
    the per-modality IRWLS gradients and Hessians are summed (weighted by
    ``beta``) before a single shared PSD -> box-QP -> simplex update.

    All modalities must share an identical set of cell-type names in their
    references. Pixels are matched by ``obs_names``; only the intersection
    (that also survived each modality's UMI filtering) is deconvolved.

    Args (beyond the modalities/reference basics):
        return_uncertainty: if True, populate ``result.weights_se`` with a
            per-pixel, per-cell-type standard error from the inverse-Hessian
            (Laplace) approximation at the solution.
        fusion: ``"protein_wls"`` (default) enters the SECOND modality as a
            Gaussian/WLS block on standardized values - correct for continuous
            protein intensity and for protein counts alike. ``"count"`` models every
            modality as Poisson counts and is only valid for genuine integer count
            modalities; it raises on continuous input.
        adaptive_weighting: if set to ``"depth_share"`` or ``"sqrt_depth"``,
            replace the fixed per-modality ``beta`` with data-driven per-pixel
            weights derived from each modality's relative sequencing depth
            (see ``adaptive_modality_weights``). ``None`` (default) keeps the
            fixed ``Modality.beta`` values.

    Returns a MultiModalResult with one weight vector per shared pixel (and, if
    requested, a matching ``weights_se`` array).
    """
    if len(modalities) < 1:
        raise ValueError("need at least one modality")
    if fusion not in ("protein_wls", "count"):
        raise ValueError(f"unknown fusion={fusion!r}; expected 'protein_wls' or 'count'")
    if fusion == "count":
        # Count fusion models EVERY modality as Poisson counts. Fed continuous
        # intensity it collapses to one constant weight vector for every pixel
        # (tests/test_fusion_switch.py asserts exactly that degeneracy), so refuse
        # non-integer input rather than returning a plausible-looking answer.
        for mod in modalities:
            X = mod.spatial.X
            sample = np.asarray(
                (X[: min(1000, X.shape[0])]).todense()
                if hasattr(X, "todense")
                else X[: min(1000, X.shape[0])]
            )
            if sample.size and not np.allclose(sample, np.round(sample)):
                raise ValueError(
                    f"fusion='count' requires integer counts, but modality "
                    f"'{mod.name}' holds continuous values. Use fusion='protein_wls', "
                    "which enters the second modality as a Gaussian/WLS block."
                )
    config = config or RCTDConfig()
    device = resolve_device(config.device)

    fits = []
    ct_ref = None
    for mod in modalities:
        ref_obj = _make_reference(mod.reference, cell_type_col, config)
        r = RCTD(mod.spatial, ref_obj, config)
        r.fit_platform_effects()
        ct = list(r.reference.cell_type_names)
        if ct_ref is None:
            ct_ref = ct
        elif ct != ct_ref:
            raise ValueError(
                f"modality '{mod.name}' has cell types {ct} but expected {ct_ref}; "
                "all references must define the same K cell types in the same order."
            )
        fits.append((mod, r))

    # ── Align pixels across modalities by surviving barcode ──
    def surviving_barcodes(mod, r):
        return np.asarray(mod.spatial.obs_names)[r._pixel_mask]

    bc_sets = [surviving_barcodes(mod, r) for mod, r in fits]
    shared_set = set(bc_sets[0])
    for bc in bc_sets[1:]:
        shared_set &= set(bc)
    # Preserve the first modality's (input) pixel order rather than sorting,
    # so result rows align with the caller's original spatial ordering.
    shared = [b for b in bc_sets[0] if b in shared_set]
    if len(shared) == 0:
        raise ValueError("no pixels survive filtering in all modalities simultaneously")

    # Per-modality row index for the shared barcodes (into that modality's kept rows)
    P_list, Y_list, n_list, Q_list, SQ_list, beta_list = [], [], [], [], [], []
    x_vals = None
    sigmas = {}
    for (mod, r), bc in zip(fits, bc_sets):
        pos = {b: i for i, b in enumerate(bc)}
        sel = np.array([pos[b] for b in shared], dtype=np.int64)
        P_list.append(torch.tensor(r.norm_profiles, device=device))
        Y_list.append(torch.tensor(r.counts[sel], device=device))
        n_list.append(torch.tensor(r.nUMI[sel], device=device))
        Q_list.append(torch.tensor(r.q_mat, device=device))
        SQ_list.append(torch.tensor(r.sq_mat, device=device))
        beta_list.append(float(mod.beta))
        sigmas[mod.name] = r.sigma / 100.0
        x_vals = torch.tensor(r.x_vals, device=device)

    N = len(shared)
    K = P_list[0].shape[1]
    all_w = np.zeros((N, K), dtype=np.float64)
    all_c = np.zeros(N, dtype=bool)
    all_se = np.zeros((N, K), dtype=np.float64) if return_uncertainty else None

    if fusion == "protein_wls":
        # ── PR#25 fusion: RNA count block + protein Gaussian/WLS block ──
        # Exactly two modalities: [0]=RNA (Poisson counts), [1]=protein entered as a
        # standardized weighted-least-squares block, NOT as a second count modality.
        # Same shared-weight IRWLS as count fusion, but the second modality contributes
        # 0.5*lam*sum_m (Y_prot - P_prot w)^2 / tau_m^2 instead of a Poisson term -- the
        # design that keeps protein on a per-marker z-scale so ~1e7 intensity magnitudes
        # cannot swamp the RNA gradient (the failure mode of count fusion on IF data).
        if len(modalities) != 2:
            raise ValueError("fusion='protein_wls' expects exactly [RNA, protein] modalities")
        P_prot_np, Y_prot_np, inv_tau2_np = _build_protein_wls_block(
            modalities[1], shared, ct_ref, cell_type_col, protein_norm, config
        )
        P_prot_t = torch.tensor(P_prot_np, device=device, dtype=P_list[0].dtype)
        Y_prot_t = torch.tensor(Y_prot_np, device=device, dtype=P_list[0].dtype)
        inv_tau2_t = torch.tensor(inv_tau2_np, device=device, dtype=P_list[0].dtype)
        if isinstance(protein_lam, str) and protein_lam == "auto":
            lam = _auto_protein_lambda(
                P_list[0],
                Y_list[0],
                n_list[0],
                Q_list[0],
                SQ_list[0],
                x_vals,
                P_prot_t,
                Y_prot_t,
                inv_tau2_t,
            )
        else:
            lam = float(protein_lam)
        sigmas["_protein_lambda"] = float(lam)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            w, c = solve_irwls_batch_shared(
                P=P_list[0],
                Y_batch=Y_list[0][start:end],
                nUMI_batch=n_list[0][start:end],
                Q_mat=Q_list[0],
                SQ_mat=SQ_list[0],
                x_vals=x_vals,
                max_iter=config.max_iter,
                min_change=config.MIN_CHANGE_REG,
                step_size=config.step_size,
                constrain=constrain,
                P_prot=P_prot_t,
                Y_prot_batch=Y_prot_t[start:end],
                inv_tau2=inv_tau2_t,
                lam=lam,
            )
            all_w[start:end] = w.cpu().numpy()
            all_c[start:end] = c.cpu().numpy()
        # solve_irwls_batch_shared has no uncertainty return; all_se stays None.
    else:
        # ── count fusion (opt-in): every modality as a Poisson count block ──
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            n_batch = [n[start:end] for n in n_list]
            # Per-pixel adaptive modality weights override the fixed betas when
            # requested (depth_share / sqrt_depth); otherwise use the fixed betas.
            if adaptive_weighting:
                batch_beta = adaptive_modality_weights(n_batch, mode=adaptive_weighting)
            else:
                batch_beta = beta_list
            res = solve_irwls_batch_multimodal(
                P_list=P_list,
                Y_list=[Y[start:end] for Y in Y_list],
                nUMI_list=n_batch,
                Q_list=Q_list,
                SQ_list=SQ_list,
                x_vals=x_vals,
                beta_list=batch_beta,
                max_iter=config.max_iter,
                min_change=config.MIN_CHANGE_REG,
                step_size=config.step_size,
                constrain=constrain,
                bulk_mode=False,
                return_uncertainty=return_uncertainty,
            )
            if return_uncertainty:
                w, c, se = res
                all_se[start:end] = se.cpu().numpy()
            else:
                w, c = res
            all_w[start:end] = w.cpu().numpy()
            all_c[start:end] = c.cpu().numpy()

    if constrain:
        row = all_w.sum(1, keepdims=True)
        all_w = all_w / np.clip(row, 1e-12, None)

    return MultiModalResult(
        weights=all_w,
        cell_type_names=ct_ref,
        converged=all_c,
        barcodes=np.array(shared),
        per_modality_sigma=sigmas,
        modality_names=[m.name for m, _ in fits],
        weights_se=all_se,
    )


def _make_reference(adata, cell_type_col, config):
    from rctd._reference import Reference

    return Reference(adata, cell_type_col=cell_type_col)


def _dense(X):
    """Densify a possibly-sparse AnnData X block to a float64 ndarray."""
    return np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float64)


def _build_protein_wls_block(prot_mod, shared, ct_ref, cell_type_col, protein_norm, config):
    """Assemble the (P_prot, Y_prot, inv_tau2) tensors for the protein WLS block.

    Both the spatial protein (aligned to the ``shared`` pixels) and the protein
    reference are standardized per-marker (arcsinh + robust-z) on their own scale -
    scale-free and direction-preserving, so summed pseudospots and single reference
    cells land on a comparable z-scale without needing a per-spot depth term. The
    per-type profile ``P_prot`` (M x K, columns in ``ct_ref`` order) and pooled
    within-type ``tau`` reuse :func:`bootstrap_protein_profiles` on the reference
    labels (the benchmark has a trusted reference, so no confident-singlet bootstrap
    from the spatial data is needed).
    """
    from rctd._protein import bootstrap_protein_profiles, normalize_protein

    markers = list(prot_mod.reference.var_names)  # M markers; defines column order
    cof = config.protein_arcsinh_cofactor

    # Spatial protein: align rows to shared barcodes, columns to the marker order.
    sp = prot_mod.spatial
    sp_pos = {b: i for i, b in enumerate(np.asarray(sp.obs_names))}
    srow = np.array([sp_pos[b] for b in shared], dtype=np.int64)
    Xsp = _dense(sp[:, markers].X)[srow]  # (N, M)
    Y_prot_std, _, _ = normalize_protein(Xsp, method=protein_norm, cofactor=cof)

    # Reference protein: per-type mean standardized profile + pooled within-type tau.
    ref = prot_mod.reference
    Xref = _dense(ref[:, markers].X)  # (Ncells, M)
    ref_std, _, _ = normalize_protein(Xref, method=protein_norm, cofactor=cof)
    lab2idx = {str(t): i for i, t in enumerate(ct_ref)}
    ref_type_idx = np.array(
        [lab2idx.get(str(l), -1) for l in np.asarray(ref.obs[cell_type_col].values)]
    )
    P_prot, tau, _ = bootstrap_protein_profiles(ref_std, ref_type_idx, len(ct_ref), min_cells=1)
    inv_tau2 = 1.0 / np.maximum(tau**2, config.protein_tau_floor**2)
    return P_prot, Y_prot_std, inv_tau2


def _auto_protein_lambda(P, Y, nUMI, Q, SQ, x_vals, P_prot, Y_prot, inv_tau2, max_pixels=2000):
    """Per-feature gradient-norm balance of protein vs RNA at the uniform init w=1/K.

    Mirrors ``RCTD._estimate_protein_lambda``: divide each modality's median gradient
    norm by its feature count so one protein marker carries ~the influence of one gene,
    landing lambda in the O(1) regime the synthetic recovery sweep found optimal.
    """
    from rctd._likelihood import calc_q_all

    G, K = P.shape
    N = Y.shape[0]
    device = P.device
    idx = torch.arange(N, device=device)
    if N > max_pixels:  # deterministic thin (no RNG, so the value is reproducible)
        step = max(1, N // max_pixels)
        idx = idx[::step][:max_pixels]
    Ys, ns = Y[idx], nUMI[idx]
    w0 = torch.full((K,), 1.0 / K, device=device, dtype=P.dtype)
    pred = torch.clamp(torch.abs(ns[:, None] * (w0 @ P.T)), min=1e-4)
    kval = Q.shape[0] - 3
    _, d1, _ = calc_q_all(torch.clamp(Ys, max=kval).reshape(-1), pred.reshape(-1), Q, SQ, x_vals)
    grad_rna = -((d1.reshape(len(idx), G) * ns[:, None]) @ P)
    rna_norm = torch.linalg.norm(grad_rna, dim=1)

    Yp = Y_prot[idx]
    mu = w0 @ P_prot.T
    grad_prot = -(((Yp - mu[None, :]) * inv_tau2[None, :]) @ P_prot)
    prot_norm = torch.linalg.norm(grad_prot, dim=1)

    M = P_prot.shape[0]
    med_prot = float(torch.median(prot_norm).item())
    if med_prot <= 1e-12:
        return 0.0
    return (float(torch.median(rna_norm).item()) / G) / (med_prot / M)
