import time
import os
import sys
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import torch
import gc

# 1. pyrctd (PyTorch)
from pyrctd import RCTD as RCTD_torch

# 2. rctd-py (JAX)
sys.path.insert(0, os.path.abspath("src"))
from rctd import run_rctd as run_rctd_jax
from rctd import Reference as Reference_jax
from rctd import RCTDConfig as RCTDConfig_jax

def load_reference():
    refdir = "/home/pgueguen/git/spacexr/inst/extdata/Reference/Vignette"
    counts = pd.read_csv(f"{refdir}/dge.csv", index_col=0)
    meta = pd.read_csv(f"{refdir}/meta_data.csv")
    
    cells = meta["barcode"].values
    genes = counts.index.values
    
    adata_ref = ad.AnnData(X=counts.values.T, obs=meta.set_index("barcode"))
    adata_ref.var_names = genes
    adata_ref.obs_names = cells
    
    if not np.issubdtype(adata_ref.X.dtype, np.integer):
        adata_ref.X = np.round(adata_ref.X).astype(np.int32)
        
    return adata_ref

import subprocess
import json

def run_jax_process(device_type):
    script = f"""
import os
os.environ['JAX_PLATFORMS'] = '{device_type}'
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
spot_class_map = {{0: "reject", 1: "singlet", 2: "doublet_certain", 3: "doublet_uncertain"}}
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

suffix = 'cpu' if '{device_type}' == 'cpu' else 'gpu'
df_jax.to_csv(f"data/res_jax_{{suffix}}.csv")

full_weights = np.full((spatial.n_obs, res_jax.weights.shape[1]), np.nan)
full_weights[umi_mask] = res_jax.weights
np.save(f"data/weights_jax_{{suffix}}.npy", full_weights)

with open(f"data/jax_{{suffix}}_time.txt", "w") as f:
    f.write(str(elapsed))
"""
    with open(f"scripts/run_jax_{device_type}.py", "w") as f:
        f.write(script)
    
    subprocess.run([sys.executable, f"scripts/run_jax_{device_type}.py"], check=True)
    
    suffix = 'cpu' if device_type == 'cpu' else 'gpu'
    with open(f"data/jax_{suffix}_time.txt", "r") as f:
        return float(f.read().strip())

def run_models_and_cache():
    datadir = "/home/pgueguen/git/spacexr/inst/extdata/SpatialRNA/Vignette"
    print(f"Loading Vignette spatial data from {datadir}...")
    spatial_counts = pd.read_csv(f"{datadir}/MappedDGEForR.csv", index_col=0)
    coords = pd.read_csv(f"{datadir}/BeadLocationsForR.csv")
    
    spatial = ad.AnnData(X=spatial_counts.values.T, obs=coords.set_index("barcodes"))
    spatial.var_names = spatial_counts.index.values
    spatial.obs_names = coords["barcodes"].values
    
    if not np.issubdtype(spatial.X.dtype, np.integer):
        spatial.X = np.round(spatial.X).astype(np.int32)
        
    ref_adata = load_reference()
    print(f"Spatial data shape: {spatial.n_obs} cells.")
    
    results = {}
    time_log = {}

    import pyrctd
    umi_mask = (np.array(spatial.X.sum(axis=1)).flatten() >= 100)
    filtered_obs = spatial.obs_names[umi_mask]

    # === 1. PyTorch GPU ===
    print("\n[1] Running PyTorch GPU...")
    start = time.time()
    try:
        model_gpu = RCTD_torch(spatial, ref_adata, device="cuda", cell_type_col="cluster", sigma=0.84)
        model_gpu.run(mode="doublet")
        time_log['pytorch_gpu'] = time.time() - start
        
        res_df = model_gpu.get_results_df()
        res_df.index = filtered_obs
        res_df.to_csv("data/res_pytorch_gpu.csv")
        
        w_gpu = model_gpu.get_weights()
        full_weights = np.full((spatial.n_obs, w_gpu.shape[1]), np.nan)
        full_weights[umi_mask] = w_gpu
        np.save("data/weights_pytorch_gpu.npy", full_weights)
        
        results['pytorch_gpu'] = True
    except Exception as e:
        print(f"Failed: {e}")
        time_log['pytorch_gpu'] = None
        
    torch.cuda.empty_cache()
    gc.collect()

    # === 2. PyTorch CPU ===
    print("\n[2] Running PyTorch CPU (16 cores)...")
    start = time.time()
    try:
        model_cpu = RCTD_torch(spatial, ref_adata, device="cpu", cell_type_col="cluster", max_cores=16, sigma=0.84)
        model_cpu.run(mode="doublet")
        time_log['pytorch_cpu'] = time.time() - start
        
        res_df = model_cpu.get_results_df()
        res_df.index = filtered_obs
        res_df.to_csv("data/res_pytorch_cpu.csv")
        
        w_cpu = model_cpu.get_weights()
        full_weights = np.full((spatial.n_obs, w_cpu.shape[1]), np.nan)
        full_weights[umi_mask] = w_cpu
        np.save("data/weights_pytorch_cpu.npy", full_weights)
        
        results['pytorch_cpu'] = True
    except Exception as e:
        print(f"Failed: {e}")
        time_log['pytorch_cpu'] = None
        
    gc.collect()

    # === 3. JAX CPU + GPU ===
    try:
        print("\n[3] Running JAX CPU...")
        time_log['jax_cpu'] = run_jax_process("cpu")
        print("\n[4] Running JAX GPU...")
        time_log['jax_gpu'] = run_jax_process("cuda")
    except Exception as e:
        import traceback
        traceback.print_exc()

    # Save time log
    pd.Series(time_log).to_csv("data/benchmark_times.csv")
    print("Done generating cache files.")

if __name__ == "__main__":
    run_models_and_cache()
