#!/usr/bin/env python3
"""Debug profile normalization to understand weight scale issues."""
import sys
import numpy as np
import scipy.io
import scipy.sparse as sp
import pandas as pd
import anndata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_reference(ref_dir):
    counts = scipy.io.mmread(ref_dir / "ref_counts.mtx").T.tocsc()
    meta = pd.read_csv(ref_dir / "ref_meta.csv")
    genes = pd.read_csv(ref_dir / "ref_genes.txt", header=None)[0].tolist()
    return anndata.AnnData(
        X=counts,
        obs=meta.set_index(meta.columns[0]),
        var=pd.DataFrame(index=genes),
    )


ref_dir = Path("/home/pgueguen/git/rctd-py/data")
h5_path = "/srv/gstore/projects/p36005/Xenium_Data/data_20240724__105930__o35844/output-XETG00404__0022381__Region_3__20240724__110035/cell_feature_matrix.h5"

from rctd import Reference, RCTDConfig
from rctd._rctd import RCTD
import jax.numpy as jnp

print("Loading reference...")
ref_adata = load_reference(ref_dir)
reference = Reference(ref_adata, cell_type_col="cell_type")
print(f"  Reference: {reference.n_types} types, {reference.n_genes} genes")
print(f"  Profile column sums (full): {reference.profiles.sum(axis=0)[:5]}")

import scanpy as sc
print("\nLoading spatial...")
spatial = sc.read_10x_h5(h5_path)
spatial.var_names_make_unique()

config = RCTDConfig(UMI_min=20)
rctd = RCTD(spatial, reference, config)

print(f"\nCommon genes: {len(rctd.common_genes)}")
print(f"Base profiles shape: {rctd.base_profiles.shape}")
print(f"Base profile column sums: {rctd.base_profiles.sum(axis=0)[:5]}")
print(f"Counts shape: {rctd.counts.shape}")
print(f"nUMI: min={rctd.nUMI.min():.0f}, median={np.median(rctd.nUMI):.0f}, max={rctd.nUMI.max():.0f}")

# Manually trace through fit_platform_effects
common_gene_to_idx = {g: i for i, g in enumerate(rctd.common_genes)}
spatial_bulk_common = rctd.counts.sum(axis=0)
spatial_bulk_full = np.zeros(reference.n_genes, dtype=np.float32)
for g in rctd.common_genes:
    ref_idx = reference.gene_names.index(g)
    spatial_bulk_full[ref_idx] = spatial_bulk_common[common_gene_to_idx[g]]

gene_list_bulk = reference.get_de_genes(
    spatial_bulk=spatial_bulk_full,
    fc_thresh=config.fc_cutoff,
    expr_thresh=config.gene_cutoff,
    min_obs=config.MIN_OBS,
)
gene_list_reg = reference.get_de_genes(
    spatial_bulk=spatial_bulk_full,
    fc_thresh=config.fc_cutoff_reg,
    expr_thresh=config.gene_cutoff_reg,
    min_obs=config.MIN_OBS,
)

bulk_set = set(gene_list_bulk) & set(rctd.common_genes)
reg_set = set(gene_list_reg) & set(rctd.common_genes)
gene_list_bulk = sorted(bulk_set)
gene_list_reg = sorted(reg_set)
print(f"\ngene_list_bulk: {len(gene_list_bulk)}")
print(f"gene_list_reg: {len(gene_list_reg)}")

# fitBulk on gene_list_bulk
from rctd._normalize import fit_bulk
bulk_profiles = reference.get_profiles_for_genes(gene_list_bulk)
bulk_gene_idx = [rctd.common_genes.index(g) for g in gene_list_bulk]
bulk_counts = rctd.counts[:, bulk_gene_idx]

print(f"\nbulk_profiles shape: {bulk_profiles.shape}")
print(f"bulk_profiles column sums: {bulk_profiles.sum(axis=0)[:5]}")

bulk_weights, norm_prof_bulk = fit_bulk(
    cell_type_profiles=jnp.array(bulk_profiles),
    spatial_counts=jnp.array(bulk_counts),
    spatial_nUMI=jnp.array(rctd.nUMI),
    min_change=config.MIN_CHANGE_BULK,
    max_iter=config.max_iter,
)

bulk_w_np = np.array(bulk_weights)
print(f"\nbulk_weights: {bulk_w_np[:5]}")
print(f"bulk_weights sum: {bulk_w_np.sum()}")
print(f"norm_prof_bulk column sums: {np.array(norm_prof_bulk).sum(axis=0)[:5]}")

# Now compute reg normalization
reg_gene_idx = [rctd.common_genes.index(g) for g in gene_list_reg]
counts_reg = rctd.counts[:, reg_gene_idx]
nUMI_reg = counts_reg.sum(axis=1)

reg_profiles = reference.get_profiles_for_genes(gene_list_reg)
print(f"\nreg_profiles shape: {reg_profiles.shape}")
print(f"reg_profiles column sums: {reg_profiles.sum(axis=0)[:5]}")

bulk_w_np = np.maximum(bulk_w_np, 0.0)
bulk_w_np = bulk_w_np / bulk_w_np.sum()

expected = reg_profiles @ bulk_w_np
obs_freq = counts_reg.sum(axis=0) / max(nUMI_reg.sum(), 1e-10)
platform_ratio = expected / np.maximum(obs_freq, 1e-10)

print(f"\nexpected: min={expected.min():.6f}, max={expected.max():.6f}, sum={expected.sum():.6f}")
print(f"obs_freq: min={obs_freq.min():.6f}, max={obs_freq.max():.6f}, sum={obs_freq.sum():.6f}")
print(f"platform_ratio: min={platform_ratio.min():.6f}, max={platform_ratio.max():.6f}, mean={platform_ratio.mean():.6f}")

norm_profiles_reg = reg_profiles / np.maximum(platform_ratio[:, None], 1e-10)
print(f"\nnorm_profiles_reg column sums: {norm_profiles_reg.sum(axis=0)[:5]}")
print(f"norm_profiles_reg column sums range: min={norm_profiles_reg.sum(axis=0).min():.4f}, max={norm_profiles_reg.sum(axis=0).max():.4f}")

# Now compare: what if we just subset norm_prof_bulk to reg genes?
# This is what R effectively does
norm_prof_bulk_np = np.array(norm_prof_bulk)  # (len(gene_list_bulk), 45)
# Map from bulk to reg genes via common_genes
# gene_list_reg should be a subset of gene_list_bulk (stricter thresholds)
reg_in_bulk = [g for g in gene_list_reg if g in set(gene_list_bulk)]
print(f"\nreg genes in bulk: {len(reg_in_bulk)}/{len(gene_list_reg)}")

if len(reg_in_bulk) > 0:
    bulk_gene_map = {g: i for i, g in enumerate(gene_list_bulk)}
    reg_from_bulk_idx = [bulk_gene_map[g] for g in reg_in_bulk]
    norm_prof_from_bulk = norm_prof_bulk_np[reg_from_bulk_idx]
    print(f"norm_prof_from_bulk column sums: {norm_prof_from_bulk.sum(axis=0)[:5]}")
    print(f"norm_prof_from_bulk column sums range: min={norm_prof_from_bulk.sum(axis=0).min():.4f}, max={norm_prof_from_bulk.sum(axis=0).max():.4f}")

    # Compare the two normalization approaches
    # Align: only use genes common to both
    reg_gene_map = {g: i for i, g in enumerate(gene_list_reg)}
    common_reg_bulk = sorted(set(gene_list_reg) & set(gene_list_bulk))
    idx_in_reg = [reg_gene_map[g] for g in common_reg_bulk]
    idx_in_bulk = [bulk_gene_map[g] for g in common_reg_bulk]

    my_norm = norm_profiles_reg[idx_in_reg]
    bulk_norm = norm_prof_bulk_np[idx_in_bulk]
    ratio = my_norm / np.maximum(bulk_norm, 1e-10)
    print(f"\nRatio my_norm / bulk_norm: min={ratio.min():.4f}, max={ratio.max():.4f}, mean={ratio.mean():.4f}")
    print(f"Per gene ratio (first 10): {ratio[:10, 0]}")

print("\nDone!")
