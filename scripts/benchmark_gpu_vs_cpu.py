import time
import os
import sys
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import scipy.io

from pyrctd import RCTD

def load_reference():
    counts = scipy.io.mmread("data/ref_counts.mtx").tocsc().T 
    meta = pd.read_csv("data/ref_meta.csv", index_col=0)
    with open("data/ref_genes.txt") as f:
        genes = [line.strip() for line in f]
    with open("data/ref_cells.txt") as f:
        cells = [line.strip() for line in f]
        
    adata_ref = ad.AnnData(X=counts, obs=meta)
    adata_ref.var_names = genes
    adata_ref.obs_names = cells
    
    if not np.issubdtype(adata_ref.X.dtype, np.integer):
        adata_ref.X = np.round(adata_ref.X).astype(np.int32)
        
    return adata_ref

def main():
    data_dir = "/srv/gstore/projects/p36005/Xenium_Data/data_20240724__105930__o35844/output-XETG00404__0022381__Region_3__20240724__110035"
    print(f"Loading Xenium Region 3 data from {data_dir}...")
    spatial = sc.read_10x_h5(f"{data_dir}/cell_feature_matrix.h5")
    spatial.var_names_make_unique()
    
    ref_adata = load_reference()
    
    # Subsample to 5,000 cells for a realistic timed comparison
    # (CPU will take a few minutes, GPU will be fast)
    sc.pp.subsample(spatial, n_obs=5000, random_state=42)
    print(f"Subsampled spatial data to {spatial.n_obs} cells.")
    
    print("\n" + "="*50)
    print("1. Running pyrctd (PyTorch - CPU Mode)")
    print("="*50)
    start_time = time.time()
    
    try:
        model_cpu = RCTD(
            spatial, 
            ref_adata, 
            device="cpu", 
            cell_type_col="cell_type",
            max_cores=16
        )
        model_cpu.run(mode="doublet")
        cpu_time = time.time() - start_time
        print(f"pyrctd (PyTorch CPU) completed in {cpu_time:.2f}s")
    except Exception as e:
        import traceback
        traceback.print_exc()
        cpu_time = None
        
    print("\n" + "="*50)
    print("2. Running pyrctd (PyTorch - GPU Mode)")
    print("="*50)
    start_time = time.time()
    
    try:
        model_gpu = RCTD(
            spatial, 
            ref_adata, 
            device="cuda", 
            cell_type_col="cell_type"
        )
        model_gpu.run(mode="doublet")
        gpu_time = time.time() - start_time
        print(f"pyrctd (PyTorch GPU) completed in {gpu_time:.2f}s")
    except Exception as e:
        import traceback
        traceback.print_exc()
        gpu_time = None
        
    print("\n" + "="*50)
    print("Final Comparison")
    print("="*50)
    if cpu_time and gpu_time:
        print(f"CPU Runtime: {cpu_time:.2f} seconds")
        print(f"GPU Runtime: {gpu_time:.2f} seconds")
        speedup = cpu_time / gpu_time
        gain_pct = ((cpu_time - gpu_time) / cpu_time) * 100
        print(f"GPU is {speedup:.2f}x faster than CPU")
        print(f"Time savings (gain %): {gain_pct:.2f}%")

if __name__ == "__main__":
    main()
