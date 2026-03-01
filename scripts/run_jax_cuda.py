
import os
os.environ['JAX_PLATFORMS'] = 'cuda'
import time
import pandas as pd
import numpy as np
import sys
import scanpy as sc
import anndata as ad
sys.path.insert(0, os.path.abspath("src"))
from rctd import run_rctd, Reference, RCTDConfig

refdir = "/home/pgueguen/git/spacexr/inst/extdata/Reference/Vignette"
counts = pd.read_csv(refdir+"/dge.csv", index_col=0)
meta = pd.read_csv(refdir+"/meta_data.csv")
adata_ref = ad.AnnData(X=counts.values.T, obs=meta.set_index("barcode"))
adata_ref.var_names = counts.index.values
adata_ref.obs_names = meta["barcode"].values

datadir = "/home/pgueguen/git/spacexr/inst/extdata/SpatialRNA/Vignette"
spatial_counts = pd.read_csv(datadir+"/MappedDGEForR.csv", index_col=0)
coords = pd.read_csv(datadir+"/BeadLocationsForR.csv")
spatial = ad.AnnData(X=spatial_counts.values.T, obs=coords.set_index("barcodes"))
spatial.var_names = spatial_counts.index.values
spatial.obs_names = coords["barcodes"].values

ref_jax = Reference(adata_ref, cell_type_col="cluster")
# Use R's calibrated sigma (0.84) for fair comparison across all implementations
config_jax = RCTDConfig()

start = time.time()
res_jax = run_rctd(spatial=spatial, reference=ref_jax, mode="doublet", config=config_jax, sigma_override=84)
elapsed = time.time() - start

nUMI_all = np.array(spatial.X.sum(axis=1)).flatten()
umi_mask = (nUMI_all >= config_jax.UMI_min) & (nUMI_all <= config_jax.UMI_max)
filtered_obs = spatial.obs_names[umi_mask]

# Map mappings
spot_class_map = {0: "reject", 1: "singlet", 2: "doublet_certain", 3: "doublet_uncertain"}
s_classes = [spot_class_map[i] for i in np.array(res_jax.spot_class)]
f_types = [ref_jax.cell_type_names[i] for i in np.array(res_jax.first_type)]
s_types = [ref_jax.cell_type_names[i] for i in np.array(res_jax.second_type)]

df_jax = pd.DataFrame(index=spatial.obs_names)
df_jax['spot_class'] = "unfiltered"
df_jax.loc[filtered_obs, 'spot_class'] = s_classes
df_jax.loc[filtered_obs, 'first_type'] = f_types
df_jax.loc[filtered_obs, 'second_type'] = s_types
df_jax.loc[filtered_obs, 'min_score'] = res_jax.min_score
df_jax.loc[filtered_obs, 'singlet_score'] = res_jax.singlet_score

suffix = 'cpu' if 'cuda' == 'cpu' else 'gpu'
df_jax.to_csv(f"data/res_jax_{suffix}.csv")

full_weights = np.full((spatial.n_obs, res_jax.weights.shape[1]), np.nan)
full_weights[umi_mask] = res_jax.weights
np.save(f"data/weights_jax_{suffix}.npy", full_weights)

with open(f"data/jax_{suffix}_time.txt", "w") as f:
    f.write(str(elapsed))
