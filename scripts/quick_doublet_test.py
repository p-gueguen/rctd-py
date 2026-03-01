#!/usr/bin/env python3
"""Quick test: run doublet mode with sigma_override=72 to isolate performance."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    import numpy as np
    import scipy.io
    import pandas as pd
    import anndata
    import scanpy as sc

    from rctd import Reference, run_rctd, RCTDConfig

    ref_dir = Path("data")
    spatial_path = sys.argv[1] if len(sys.argv) > 1 else "/scratch/cell_feature_matrix.h5"

    # Load reference
    counts = scipy.io.mmread(ref_dir / "ref_counts.mtx").T.tocsc()
    ref_meta = pd.read_csv(ref_dir / "ref_meta.csv")
    genes = pd.read_csv(ref_dir / "ref_genes.txt", header=None)[0].tolist()
    ref_adata = anndata.AnnData(X=counts, obs=ref_meta.set_index(ref_meta.columns[0]),
                                var=pd.DataFrame(index=genes))
    reference = Reference(ref_adata, cell_type_col="cell_type")

    # Load spatial
    spatial = sc.read_10x_h5(spatial_path)
    spatial.var_names_make_unique()
    print(f"Spatial: {spatial.n_obs} cells, {spatial.n_vars} genes")

    config = RCTDConfig(UMI_min=20)

    # Test with sigma_override=72 (skips choose_sigma entirely)
    print("\n=== Doublet mode with sigma_override=72 (no sigma estimation) ===")
    t0 = time.time()
    result = run_rctd(spatial, reference, mode="doublet", config=config,
                      batch_size=5000, sigma_override=72)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")
    print(f"Spot classes: reject={sum(result.spot_class==0)}, singlet={sum(result.spot_class==1)}, "
          f"doublet_certain={sum(result.spot_class==2)}, doublet_uncertain={sum(result.spot_class==3)}")

if __name__ == "__main__":
    main()
