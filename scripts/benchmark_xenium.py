import time
import scanpy as sc
import anndata as ad
import numpy as np

from rctd import run_rctd, Reference, RCTDConfig

def create_mock_reference(spatial_adata: ad.AnnData) -> Reference:
    """Create a mock reference from the spatial data itself for testing purposes.
    In a real scenario, this would be a separate scRNA-seq dataset.
    """
    print("Creating mock reference from spatial data...")
    # Add mock cell types based on clustering or random assignment
    np.random.seed(42)
    n_cells = spatial_adata.n_obs
    cell_types = [f"Type_{i}" for i in range(5)]
    mock_types = np.random.choice(cell_types, size=n_cells)
    
    ref_adata = spatial_adata.copy()
    ref_adata.obs["cell_type"] = mock_types
    
    # Needs to be integer counts for Reference
    if not np.issubdtype(ref_adata.X.dtype, np.integer):
        ref_adata.X = np.round(ref_adata.X).astype(np.int32)
        
    reference = Reference(ref_adata, cell_type_col="cell_type", min_UMI=10, cell_min=10)
    return reference

def main():
    # Path to Xenium data
    data_dir = "/srv/gstore/projects/p36005/Xenium_Data/data_20240724__105930__o35844/output-XETG00404__0022378__Region_2__20240724__110035"
    
    print(f"Loading Xenium data from {data_dir}...")
    start_time = time.time()
    
    # 10x Xenium output wrapper in scanpy/squidpy usually reads cell_feature_matrix
    spatial = sc.read_10x_h5(f"{data_dir}/cell_feature_matrix.h5")
    
    # Make sure gene names are unique
    spatial.var_names_make_unique()
    print(f"Loaded spatial data containing {spatial.n_obs} cells and {spatial.n_vars} genes in {time.time() - start_time:.2f}s")
    
    # We don't have a matching snRNA-seq reference for this specific dataset right now.
    # To benchmark the *algorithm performance*, we'll create a mock reference using the spatial data itself.
    reference = create_mock_reference(spatial)
    
    config = RCTDConfig(
        UMI_min=10, # lower thresholds for testing
        UMI_min_sigma=10,
    )
    
    print("Starting RCTD Doublet Mode...")
    rctd_start = time.time()
    
    res = run_rctd(
        spatial=spatial,
        reference=reference,
        mode="doublet",
        config=config,
        batch_size=10000
    )
    
    rctd_end = time.time()
    print(f"RCTD Doublet Mode completed in {rctd_end - rctd_start:.2f}s")
    
    # Output some summaries
    print("\nResult Summary:")
    print(f"Total cells processed: {res.weights.shape[0]}")
    unique_classes, class_counts = np.unique(res.spot_class, return_counts=True)
    class_names = ["reject", "singlet", "doublet_certain", "doublet_uncertain"]
    print("Spot Class Distribution:")
    for cls_val, count in zip(unique_classes, class_counts):
        print(f"  {class_names[cls_val]}: {count}")

if __name__ == "__main__":
    main()
