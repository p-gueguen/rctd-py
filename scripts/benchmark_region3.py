import time
import os
import sys
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, os.path.abspath("src"))

from rctd import run_rctd as run_rctd_jax
from rctd import Reference as Reference_jax
from rctd import RCTDConfig as RCTDConfig_jax

from pyrctd import RCTD as RCTD_torch

# Force JAX to use CPU for this test
os.environ["JAX_PLATFORM_NAME"] = "cpu"

def load_reference():
    print("Loading Extracted Reference...")
    counts = scipy.io.mmread("data/ref_counts.mtx").tocsc().T # R writeMM format usually needs transpose
    meta = pd.read_csv("data/ref_meta.csv", index_col=0)
    with open("data/ref_genes.txt") as f:
        genes = [line.strip() for line in f]
    with open("data/ref_cells.txt") as f:
        cells = [line.strip() for line in f]
        
    # We transpose so cells x genes
    adata_ref = ad.AnnData(X=counts, obs=meta)
    adata_ref.var_names = genes
    adata_ref.obs_names = cells
    
    # Needs to be integer
    if not np.issubdtype(adata_ref.X.dtype, np.integer):
        adata_ref.X = np.round(adata_ref.X).astype(np.int32)
        
    return adata_ref

def main():
    data_dir = "/srv/gstore/projects/p36005/Xenium_Data/data_20240724__105930__o35844/output-XETG00404__0022381__Region_3__20240724__110035"
    print(f"Loading Xenium Region 3 data from {data_dir}...")
    spatial = sc.read_10x_h5(f"{data_dir}/cell_feature_matrix.h5")
    spatial.var_names_make_unique()
    
    ref_adata = load_reference()
    
    # Subsample spatial data to make testing faster for this run
    # we want to see if it works, maybe use 200 cells for immediate CPU benchmarking
    sc.pp.subsample(spatial, n_obs=200)
    print(f"Subsampled spatial data to {spatial.n_obs} cells for fast benchmarking")
    
    print("\n" + "="*50)
    print("1. Running pyrctd (PyTorch - CPU Mode)")
    print("="*50)
    
    start_time = time.time()
    try:
        model_torch = RCTD_torch(
            spatial, 
            ref_adata, 
            device="cpu", # Forcing CPU
            cell_type_col="cell_type",
            max_cores=16 # Let it use high concurrency
        )
        model_torch.run(mode="doublet")
        res_torch_weights = model_torch.get_weights()
        torch_time = time.time() - start_time
        print(f"pyrctd (PyTorch CPU) completed in {torch_time:.2f}s")
    except Exception as e:
        import traceback
        traceback.print_exc()
        torch_time = None
        res_torch_weights = None
        
    print("\n" + "="*50)
    print("2. Running rctd-py (JAX - CPU Mode)")
    print("="*50)
    
    start_time = time.time()
    try:
        ref_jax = Reference_jax(ref_adata, cell_type_col="cell_type")
        config_jax = RCTDConfig_jax()
        res_jax = run_rctd_jax(
            spatial=spatial,
            reference=ref_jax,
            mode="doublet",
            config=config_jax,
            batch_size=10000
        )
        jax_time = time.time() - start_time
        res_jax_weights = res_jax.weights
        print(f"rctd-py (JAX CPU) completed in {jax_time:.2f}s")
    except Exception as e:
        import traceback
        traceback.print_exc()
        jax_time = None
        res_jax_weights = None
        
    print("\n" + "="*50)
    print("Comparison Result:")
    print("="*50)
    if torch_time and jax_time:
        print(f"PyTorch CPU Runtime: {torch_time:.2f}s")
        print(f"JAX CPU Runtime: {jax_time:.2f}s")
        
        # Check numerical parity
        if res_torch_weights.shape == res_jax_weights.shape:
            diff = np.max(np.abs(res_torch_weights - res_jax_weights))
            print(f"Maximum absolute difference in weights: {diff:.6f}")
            if diff < 1e-3:
                print("-> NUMERICAL MATCH CONFIRMED")
            else:
                print("-> PARITY FAILED")
        else:
            print("Weight shapes do not match!")
            
if __name__ == "__main__":
    main()
