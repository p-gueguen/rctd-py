import time
import scanpy as sc
import anndata as ad
import numpy as np
import sys

# Add rctd-py src to path
import os
sys.path.insert(0, os.path.abspath("src"))

# 1. rctd-py (JAX)
from rctd import run_rctd as run_rctd_jax
from rctd import Reference as Reference_jax
from rctd import RCTDConfig as RCTDConfig_jax

# 2. pyrctd (PyTorch/CPU)
from pyrctd import RCTD as RCTD_torch

def create_mock_reference(spatial_adata: ad.AnnData) -> ad.AnnData:
    print("Creating mock reference from spatial data...")
    np.random.seed(42)
    n_cells = min(spatial_adata.n_obs, 2000)
    # Subsample for reference if spatial is too big
    idx = np.random.choice(spatial_adata.n_obs, size=n_cells, replace=False)
    ref_adata = spatial_adata[idx].copy()
    
    cell_types = [f"Type_{i}" for i in range(5)]
    mock_types = np.random.choice(cell_types, size=n_cells)
    ref_adata.obs["cell_type"] = mock_types
    
    if not np.issubdtype(ref_adata.X.dtype, np.integer):
        ref_adata.X = np.round(ref_adata.X).astype(np.int32)
        
    return ref_adata


def main():
    data_dir = "/srv/gstore/projects/p36005/Xenium_Data/data_20240724__105930__o35844/output-XETG00404__0022378__Region_2__20240724__110035"
    print(f"Loading Xenium data from {data_dir}...")
    spatial = sc.read_10x_h5(f"{data_dir}/cell_feature_matrix.h5")
    spatial.var_names_make_unique()
    
    # Subsample spatial data to make testing faster
    sc.pp.subsample(spatial, n_obs=10000)
    
    ref_adata = create_mock_reference(spatial)
    
    print("\n" + "="*50)
    print("1. Running pyrctd (PyTorch)")
    print("="*50)
    
    start_time = time.time()
    # pyrctd needs string path to data or directly uses AnnData arrays. It builds the model internally.
    try:
        model_torch = RCTD_torch(
            spatial, 
            ref_adata, 
            device="cuda", 
            cell_type_col="cell_type",
            umi_min=10,
            umi_min_sigma=10,
            counts_min=10,
            cell_min_instance=10
        )
        model_torch.run(mode="doublet")
        res_torch_weights = model_torch.get_weights()
        torch_time = time.time() - start_time
        print(f"pyrctd (PyTorch) completed in {torch_time:.2f}s")
    except Exception as e:
        import traceback
        traceback.print_exc()
        torch_time = None
        
    print("\n" + "="*50)
    print("2. Running rctd-py (JAX)")
    print("="*50)
    
    start_time = time.time()
    try:
        ref_jax = Reference_jax(ref_adata, cell_type_col="cell_type", min_UMI=10, cell_min=10)
        config_jax = RCTDConfig_jax(UMI_min=10, UMI_min_sigma=10)
        res_jax = run_rctd_jax(
            spatial=spatial,
            reference=ref_jax,
            mode="doublet",
            config=config_jax,
            batch_size=10000
        )
        jax_time = time.time() - start_time
        print(f"rctd-py (JAX) completed in {jax_time:.2f}s")
    except Exception as e:
        import traceback
        traceback.print_exc()
        jax_time = None
        
    print("\n" + "="*50)
    print("Performance Summary")
    print("="*50)
    if torch_time:
        print(f"pyrctd (PyTorch): {torch_time:.2f}s")
    if jax_time:
        print(f"rctd-py (JAX):   {jax_time:.2f}s")

if __name__ == "__main__":
    main()
