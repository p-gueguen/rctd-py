import sys
sys.path.insert(0, '/home/pgueguen/git/rctd-py/scripts')
from compare_r_python import *
import anndata as ad
import pandas as pd
import pyrctd

sp_dir = '/home/pgueguen/git/spacexr/inst/extdata/SpatialRNA/Vignette'
sp = pd.read_csv(sp_dir+'/MappedDGEForR.csv', index_col=0)
coords = pd.read_csv(sp_dir+'/BeadLocationsForR.csv')
spatial = ad.AnnData(X=sp.values.T, obs=coords.set_index("barcodes"))
spatial.var_names = sp.index.values

refdir = "/home/pgueguen/git/spacexr/inst/extdata/Reference/Vignette"
counts = pd.read_csv(refdir+"/dge.csv", index_col=0)
meta = pd.read_csv(refdir+"/meta_data.csv")
adata_ref = ad.AnnData(X=counts.values.T, obs=meta.set_index("barcode"))
adata_ref.var_names = counts.index.values
adata_ref.obs_names = meta["barcode"].values

model_gpu = pyrctd.RCTD_torch(spatial, adata_ref, device="cuda", cell_type_col="cluster", sigma=0.84)
model_gpu.run(mode="doublet")
w_pt = model_gpu.get_weights()[:10]

print("PT sums:", w_pt.sum(axis=1))
