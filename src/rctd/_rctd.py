import time
from typing import Literal, Union

import anndata
import numpy as np
import torch

from rctd._doublet import run_doublet_mode
from rctd._full import run_full_mode
from rctd._likelihood import compute_spline_coefficients, load_cached_q_matrices
from rctd._multi import run_multi_mode
from rctd._normalize import fit_bulk
from rctd._reference import Reference
from rctd._sigma import choose_sigma
from rctd._types import (
    DoubletResult,
    FullResult,
    MultiResult,
    RCTDConfig,
    auto_batch_size,
    resolve_device,
)


class RCTD:
    """Robust Cell Type Decomposition (RCTD)."""

    def __init__(
        self,
        spatial: anndata.AnnData,
        reference: Reference,
        config: RCTDConfig | None = None,
    ):
        """Initialize RCTD object.

        Args:
            spatial: AnnData object containing spatial transcriptomics counts.
                     X should be the raw count matrix (N, G).
            reference: Reference object with cell type profiles.
            config: RCTD configuration parameters.
        """
        self.spatial = spatial
        self.reference = reference
        self.config = config if config is not None else RCTDConfig()

        # Internal state
        self.is_normalized = False
        self.norm_profiles = None
        self.sigma = None
        self.q_mat = None
        self.sq_mat = None
        self.x_vals = None

        # Extract common data once
        self._extract_spatial_data()

    def _extract_spatial_data(self):
        """Extract spatial data and find intersecting genes with reference.

        Mirrors R spacexr ``create.RCTD`` flow:
        1. Compute nUMI from ALL spatial genes (R: ``SpatialRNA`` default
           ``nUMI = colSums(counts)`` over the full count matrix).
        2. Filter pixels by UMI range using total nUMI.
        3. Extract counts for common genes only (for DE gene selection).

        The nUMI is **never recomputed** after gene restriction — this matches
        R's ``restrict_counts()`` which preserves the original ``puck@nUMI``.
        """
        import scipy.sparse as sp

        # ── Total nUMI from ALL spatial genes (R: colSums(counts)) ──
        full_X = self.spatial.X
        if sp.issparse(full_X):
            nUMI_all = np.array(full_X.sum(axis=1)).flatten().astype(np.float32)
        else:
            nUMI_all = np.array(full_X.sum(axis=1), dtype=np.float32).flatten()

        # ── UMI filter based on total nUMI (R: restrict_counts with UMI_thresh) ──
        umi_mask = (nUMI_all >= self.config.UMI_min) & (nUMI_all <= self.config.UMI_max)
        n_before = len(nUMI_all)
        self.nUMI = nUMI_all[umi_mask]
        self._pixel_mask = umi_mask
        n_after = int(umi_mask.sum())
        if n_after < n_before:
            print(
                f"UMI filter: kept {n_after}/{n_before} pixels "
                f"(UMI range [{self.config.UMI_min}, {self.config.UMI_max}])"
            )

        # ── Common genes: intersection of spatial and reference ──
        spatial_genes = set(self.spatial.var_names)
        ref_genes = set(self.reference.gene_names)
        self.common_genes = sorted(list(spatial_genes & ref_genes))

        spatial_gene_idx = [self.spatial.var_names.get_loc(g) for g in self.common_genes]
        counts_sub = self.spatial.X[np.where(umi_mask)[0]][:, spatial_gene_idx]
        if sp.issparse(counts_sub):
            counts_sub = counts_sub.toarray()
        self.counts = np.array(counts_sub, dtype=np.float32)

        # Restrict reference profiles to common genes
        self.base_profiles = self.reference.get_profiles_for_genes(self.common_genes)

    def fit_platform_effects(self, sigma_override: int | None = None):
        """Estimate platform effects and compute normalized reference profiles.

        Mirrors R spacexr ``create.RCTD`` two-gene-list strategy:
        * ``gene_list_bulk`` — broad set (gene_cutoff / fc_cutoff) for fitBulk
        * ``gene_list_reg``  — DE set  (gene_cutoff_reg / fc_cutoff_reg) for
          choose_sigma and per-pixel IRWLS

        Args:
            sigma_override: If set, skip auto-calibration and use this sigma
                value directly (integer, same scale as Q_mat keys, e.g. 84 = sigma_c=0.84).
        """
        if self.is_normalized:
            return

        device = resolve_device(self.config.device)

        # ── 1. Gene-list split (R: get_de_genes) ──
        # spatial_bulk must be indexed by ALL reference genes (not just common)
        # so that get_de_genes gene indices align with the reference profile matrix
        common_gene_to_idx = {g: i for i, g in enumerate(self.common_genes)}
        spatial_bulk_common = self.counts.sum(axis=0)  # (G_common,)
        spatial_bulk_full = np.zeros(self.reference.n_genes, dtype=np.float32)
        for g in self.common_genes:
            ref_idx = self.reference.gene_names.index(g)
            spatial_bulk_full[ref_idx] = spatial_bulk_common[common_gene_to_idx[g]]

        gene_list_bulk = self.reference.get_de_genes(
            spatial_bulk=spatial_bulk_full,
            fc_thresh=self.config.fc_cutoff,
            expr_thresh=self.config.gene_cutoff,
            min_obs=self.config.MIN_OBS,
        )
        gene_list_reg = self.reference.get_de_genes(
            spatial_bulk=spatial_bulk_full,
            fc_thresh=self.config.fc_cutoff_reg,
            expr_thresh=self.config.gene_cutoff_reg,
            min_obs=self.config.MIN_OBS,
        )
        # Fall back to all common genes if DE selection is too small
        if len(gene_list_bulk) < 10:
            print(
                f"Warning: only {len(gene_list_bulk)} bulk genes selected, "
                f"using all {len(self.common_genes)} common genes"
            )
            gene_list_bulk = self.common_genes
        if len(gene_list_reg) < 10:
            print(
                f"Warning: only {len(gene_list_reg)} reg genes selected, "
                f"using all bulk genes ({len(gene_list_bulk)})"
            )
            gene_list_reg = gene_list_bulk

        # Keep only genes that are in our common gene set
        bulk_set = set(gene_list_bulk) & set(self.common_genes)
        reg_set = set(gene_list_reg) & set(self.common_genes)
        gene_list_bulk = sorted(bulk_set)
        gene_list_reg = sorted(reg_set)
        print(
            f"Gene lists: bulk={len(gene_list_bulk)}, reg={len(gene_list_reg)} "
            f"(from {len(self.common_genes)} common)"
        )

        # ── 2. fitBulk on gene_list_bulk ──
        # R: fitBulk uses puck@counts (restricted to gene_list_bulk) and
        #    puck@nUMI (total across ALL original genes, never recomputed)
        print("Fitting bulk platform effects...")
        bulk_profiles = self.reference.get_profiles_for_genes(gene_list_bulk)
        bulk_gene_idx = [self.common_genes.index(g) for g in gene_list_bulk]
        bulk_counts = self.counts[:, bulk_gene_idx]

        bulk_weights, norm_prof_bulk = fit_bulk(
            cell_type_profiles=torch.tensor(bulk_profiles, device=device),
            spatial_counts=torch.tensor(bulk_counts, device=device),
            spatial_nUMI=torch.tensor(self.nUMI, device=device),
            min_change=self.config.MIN_CHANGE_BULK,
        )

        # ── 3. Restrict to gene_list_reg for pixel-level fitting ──
        # Subset the bulk-normalized profiles to reg genes (matching R's approach:
        # R normalizes on gene_list_bulk via get_norm_ref, then subsets to gene_list_reg)
        norm_prof_bulk_np = norm_prof_bulk.cpu().numpy()  # (G_bulk, K)
        bulk_gene_map = {g: i for i, g in enumerate(gene_list_bulk)}
        reg_in_bulk_idx = [bulk_gene_map[g] for g in gene_list_reg]
        self.norm_profiles = norm_prof_bulk_np[reg_in_bulk_idx]  # (G_reg, K)

        reg_gene_idx = [self.common_genes.index(g) for g in gene_list_reg]
        self.counts = self.counts[:, reg_gene_idx]
        # NOTE: self.nUMI stays as total nUMI from ALL spatial genes.
        # R's puck@nUMI is never recomputed after gene restriction.

        self._gene_list_reg = gene_list_reg
        print(f"Using {len(gene_list_reg)} DE genes for pixel-level fitting")

        # ── 4. choose_sigma on reg genes ──
        cache = load_cached_q_matrices()
        self.x_vals = cache.pop("X_vals")
        q_matrices = {k.replace("Q_", ""): v for k, v in cache.items()}

        # Precompute all spline coefficients once (MI inverse is cached internally)
        sq_matrices = {
            k: compute_spline_coefficients(v, self.x_vals) for k, v in q_matrices.items()
        }

        if sigma_override is not None:
            # Bypass auto-calibration — use externally provided sigma (e.g. from R)
            self.sigma = int(sigma_override)
            print(f"Using provided sigma override: {self.sigma / 100.0}")
        else:
            print("Estimating sigma...")
            t_sigma = time.time()
            self.sigma = choose_sigma(
                spatial_counts=self.counts,
                spatial_numi=self.nUMI,
                norm_profiles=self.norm_profiles,
                q_matrices=q_matrices,
                x_vals=self.x_vals,
                sigma_init=100,
                min_umi=self.config.UMI_min_sigma,
                n_fit=self.config.N_fit,
                n_epoch=self.config.N_epoch,
                k_val=self.config.K_val,
                sq_matrices=sq_matrices,
                device=self.config.device,
            )
            sigma_elapsed = time.time() - t_sigma
            print(f"Chosen sigma: {self.sigma / 100.0} ({sigma_elapsed:.1f}s)")
        best_q_key = str(self.sigma)
        if best_q_key not in q_matrices:
            import re

            available_sigmas = [
                int(re.sub(r"[^0-9]", "", k)) for k in q_matrices.keys() if re.sub(r"[^0-9]", "", k)
            ]
            nearest = min(available_sigmas, key=lambda x: abs(x - self.sigma))
            best_q_key = str(nearest)

        self.q_mat = q_matrices[best_q_key]
        self.sq_mat = sq_matrices[best_q_key]

        # Cast to target dtype for GPU memory efficiency
        target_dtype = np.float32 if self.config.dtype == "float32" else np.float64
        self.norm_profiles = self.norm_profiles.astype(target_dtype)
        self.q_mat = self.q_mat.astype(target_dtype)
        self.sq_mat = self.sq_mat.astype(target_dtype)
        self.x_vals = self.x_vals.astype(target_dtype)
        self.counts = self.counts.astype(target_dtype)

        self.is_normalized = True


def run_rctd(
    spatial: anndata.AnnData,
    reference: Reference,
    mode: Literal["full", "doublet", "multi"] = "doublet",
    config: RCTDConfig | None = None,
    batch_size: int | str = "auto",
    sigma_override: int | None = None,
) -> Union[FullResult, DoubletResult, MultiResult]:  # noqa: UP007
    """Run RCTD pipeline on spatial data.

    Args:
        spatial: AnnData object with spatial counts
        reference: Reference object with cell type profiles
        mode: Deconvolution mode ('full', 'doublet', or 'multi')
        config: Configuration parameters
        batch_size: GPU batch size for pixel processing. "auto" calculates
            optimal size from available VRAM.

    Returns:
        Result object containing weights and predictions.
    """
    if mode not in ["full", "doublet", "multi"]:
        raise ValueError(f"Unknown mode: {mode}")

    rctd = RCTD(spatial, reference, config)
    rctd.fit_platform_effects(sigma_override=sigma_override)

    G = rctd.norm_profiles.shape[0]
    K = rctd.norm_profiles.shape[1]
    dtype_bytes = 8 if rctd.config.dtype == "float64" else 4

    if batch_size == "auto":
        batch_size = auto_batch_size(G, K, dtype_bytes)
        print(f"Auto batch size: {batch_size}")

    print(f"Running in {mode} mode...")

    kwargs = {
        "spatial_counts": rctd.counts,
        "spatial_numi": rctd.nUMI,
        "norm_profiles": rctd.norm_profiles,
        "cell_type_names": rctd.reference.cell_type_names,
        "q_mat": rctd.q_mat,
        "sq_mat": rctd.sq_mat,
        "x_vals": rctd.x_vals,
        "batch_size": batch_size,
        "device": rctd.config.device,
    }

    if mode == "full":
        res = run_full_mode(**kwargs)
    elif mode == "doublet":
        res = run_doublet_mode(**kwargs, config=rctd.config)
    elif mode == "multi":
        res = run_multi_mode(**kwargs, config=rctd.config)

    return res
