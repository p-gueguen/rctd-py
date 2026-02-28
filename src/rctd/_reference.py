"""Reference processing for RCTD.

Ports get_cell_type_info, get_norm_ref, get_de_genes from R spacexr.
"""

from collections import Counter

import anndata
import numpy as np
from scipy import sparse


class Reference:
    """scRNA-seq reference for RCTD decomposition.

    Computes cell type mean expression profiles and selects DE genes.
    """

    def __init__(
        self,
        adata: anndata.AnnData,
        cell_type_col: str = "cell_type",
        cell_min: int = 25,
        n_max_cells: int = 10000,
        min_UMI: int = 100,
    ):
        if cell_type_col not in adata.obs.columns:
            raise ValueError(f"Column '{cell_type_col}' not found in adata.obs")

        cell_types = np.asarray(adata.obs[cell_type_col].values)
        unique_types = sorted(set(cell_types))

        # Check minimum cell counts
        type_counts = Counter(cell_types)
        low_types = [ct for ct, count in type_counts.items() if count < cell_min]
        if low_types:
            remaining = [ct for ct in unique_types if ct not in low_types]
            if len(remaining) < 2:
                raise ValueError(
                    f"After filtering cell types with fewer than {cell_min} cells "
                    f"({', '.join(low_types)}), fewer than 2 cell types remain. "
                    f"Lower cell_min or add more cells."
                )
            unique_types = remaining
            # Keep only cells from valid types
            keep_type_mask = np.isin(cell_types, unique_types)
        else:
            keep_type_mask = np.ones(len(cell_types), dtype=bool)

        # Filter cells by UMI
        X = adata.X
        if sparse.issparse(X):
            nUMI = np.asarray(X.sum(axis=1)).ravel()
        else:
            nUMI = np.asarray(X.sum(axis=1)).ravel()

        keep_umi_mask = nUMI >= min_UMI
        keep = keep_type_mask & keep_umi_mask

        if keep.sum() == 0:
            raise ValueError(
                f"No cells remain after filtering (min_UMI={min_UMI}). "
                "Lower min_UMI or ensure cells have sufficient UMI counts."
            )

        X = X[keep]
        cell_types = cell_types[keep]
        nUMI = nUMI[keep]

        # Re-check cell type counts after UMI filtering
        type_counts_post = Counter(cell_types)
        valid_types = [ct for ct in unique_types if type_counts_post.get(ct, 0) >= cell_min]
        if len(valid_types) < 2:
            raise ValueError(
                f"After UMI filtering, fewer than 2 cell types have minimum {cell_min} cells."
            )
        if len(valid_types) < len(unique_types):
            # Re-filter to only valid types
            keep_valid = np.isin(cell_types, valid_types)
            X = X[keep_valid]
            cell_types = cell_types[keep_valid]
            nUMI = nUMI[keep_valid]
            unique_types = valid_types

        if sparse.issparse(X):
            X = np.asarray(X.todense())
        else:
            X = np.asarray(X, dtype=np.float64)

        # Downsample if needed
        rng = np.random.default_rng(42)
        keep_idx = []
        for ct in unique_types:
            mask = cell_types == ct
            indices = np.where(mask)[0]
            if len(indices) > n_max_cells:
                indices = rng.choice(indices, n_max_cells, replace=False)
            keep_idx.extend(indices)
        keep_idx = sorted(keep_idx)
        X = X[keep_idx]
        cell_types = cell_types[keep_idx]
        nUMI = nUMI[keep_idx]

        self.cell_type_names = unique_types
        self.n_types = len(unique_types)
        self.gene_names = list(adata.var_names)
        self.n_genes = len(self.gene_names)

        # Compute profiles: mean normalized expression per type
        # Ports get_cell_type_info from R processRef.R
        # For each cell type: normalize each cell by its UMI count, then average
        profiles = np.zeros((self.n_genes, self.n_types), dtype=np.float64)
        for k, ct in enumerate(unique_types):
            mask = cell_types == ct
            ct_data = X[mask]  # (n_cells_k, G)
            ct_nUMI = nUMI[mask]  # (n_cells_k,)
            # sweep(cell_type_data, 2, cell_type_umi, `/`) then rowSums / ncol
            normed = ct_data / ct_nUMI[:, None]
            profiles[:, k] = normed.mean(axis=0)

        self.profiles = profiles  # (G, K), columns sum to ~1

    def get_de_genes(
        self,
        spatial_bulk: np.ndarray | None = None,
        fc_thresh: float = 0.75,
        expr_thresh: float = 0.0002,
        min_obs: int = 3,
    ) -> list[str]:
        """Select differentially expressed genes.

        Ports get_de_genes from R utils.R.

        Parameters
        ----------
        spatial_bulk : array of shape (G,), optional
            Total counts per gene in the spatial dataset. Used to filter
            genes with too few observations. If None, no spatial filtering.
        fc_thresh : float
            Minimum log_e fold change required for a gene.
        expr_thresh : float
            Minimum expression threshold (normalized, proportion out of 1).
        min_obs : int
            Minimum number of occurrences of each gene in spatial data.

        Returns
        -------
        list of str
            Differentially expressed gene names.
        """
        epsilon = 1e-9
        # Start with all gene indices
        gene_list = list(range(self.n_genes))

        # Filter mitochondrial genes (case-insensitive mt- prefix)
        gene_list = [g for g in gene_list if not self.gene_names[g].lower().startswith("mt-")]

        # Filter by spatial observation count if provided
        if spatial_bulk is not None:
            gene_list = [g for g in gene_list if spatial_bulk[g] >= min_obs]

        if len(gene_list) == 0:
            return []

        gene_arr = np.array(gene_list)
        selected = set()

        for k in range(self.n_types):
            if self.n_types > 2:
                other_cols = [j for j in range(self.n_types) if j != k]
                other_mean = self.profiles[np.ix_(gene_arr, other_cols)].mean(axis=1)
            else:
                other_col = 1 - k
                other_mean = self.profiles[gene_arr, other_col]

            type_expr = self.profiles[gene_arr, k]
            logFC = np.log(type_expr + epsilon) - np.log(other_mean + epsilon)

            de_mask = (logFC > fc_thresh) & (type_expr > expr_thresh)
            for idx in np.where(de_mask)[0]:
                selected.add(gene_arr[idx])

        return [self.gene_names[g] for g in sorted(selected)]

    def get_profiles_for_genes(self, gene_names: list[str]) -> np.ndarray:
        """Get profile matrix restricted to specified genes.

        Parameters
        ----------
        gene_names : list of str
            Gene names to include.

        Returns
        -------
        ndarray of shape (len(gene_names), K)
        """
        gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}
        gene_idx = [gene_to_idx[g] for g in gene_names if g in gene_to_idx]
        return self.profiles[gene_idx, :]

    def normalize_to_spatial(
        self,
        spatial_bulk: np.ndarray,
        spatial_nUMI_total: float,
        gene_names: list[str],
        proportions: np.ndarray,
    ) -> np.ndarray:
        """Renormalize profiles to match spatial platform.

        Ports get_norm_ref from R processRef.R.

        Parameters
        ----------
        spatial_bulk : array of shape (G_full,)
            Total counts per gene across all spatial pixels (rowSums of counts).
        spatial_nUMI_total : float
            Total UMI count across all spatial pixels (sum of nUMI).
        gene_names : list of str
            Genes to use for normalization.
        proportions : array of shape (K,)
            Estimated proportion of each cell type in the bulk.

        Returns
        -------
        ndarray of shape (len(gene_names), K)
            Renormalized profiles for the given genes.
        """
        gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}
        gene_idx = np.array([gene_to_idx[g] for g in gene_names])
        profiles_sub = self.profiles[gene_idx, :]

        # R: rowSums(sweep(cell_type_means, 2, proportions/sum(proportions), '*'))
        prop_norm = proportions / proportions.sum()
        weight_avg = (profiles_sub * prop_norm[None, :]).sum(axis=1)

        # target_means = bulk_vec[gene_list] / sum(puck@nUMI)
        target_means = spatial_bulk[gene_idx] / spatial_nUMI_total

        # R: sweep(cell_type_means, 1, weight_avg / target_means, '/')
        ratio = np.where(target_means > 1e-10, weight_avg / target_means, 1.0)
        renorm = profiles_sub / ratio[:, None]
        return renorm
