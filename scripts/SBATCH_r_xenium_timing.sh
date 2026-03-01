#!/bin/bash
#SBATCH --job-name=rctd_r_time
#SBATCH --output=/home/pgueguen/rctd_r_timing_%j.log
#SBATCH --error=/home/pgueguen/rctd_r_timing_%j.log
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=employee

echo "Timing R spacexr on Xenium Region 3 at $(date)"

module load Dev/R/4.5.0
cd /home/pgueguen/git/rctd-py

Rscript -e '
library(spacexr)
library(Seurat)

output_dir <- "data/xenium_validation"

cat("Loading Xenium spatial data...\n")
h5_path <- "/srv/gstore/projects/p36005/Xenium_Data/data_20240724__105930__o35844/output-XETG00404__0022381__Region_3__20240724__110035/cell_feature_matrix.h5"
spatial <- Read10X_h5(h5_path)
if (is.list(spatial)) spatial <- spatial[[1]]
cat("  Spatial:", ncol(spatial), "cells,", nrow(spatial), "genes\n")

coords <- data.frame(x = seq_len(ncol(spatial)), y = seq_len(ncol(spatial)),
                      row.names = colnames(spatial))
puck <- SpatialRNA(coords, spatial)

ref_rds <- "/srv/GT/databases/RCTD_References/p36005_UM_references/duodenum/duodenum_rctd.rds"
obj <- readRDS(ref_rds)
if (is.list(obj) && !is.null(obj$reference)) obj <- obj$reference
if (inherits(obj, "Seurat")) {
    counts <- obj@assays$RNA$counts
    if (is.null(counts)) counts <- obj@assays$RNA@counts
    cell_types <- setNames(as.factor(obj@meta.data$cell_type), colnames(counts))
    nUMI <- colSums(counts)
    reference <- Reference(counts, cell_types, nUMI)
} else {
    reference <- obj
}
cat("  Reference:", length(reference@cell_types), "cells,",
    nrow(reference@counts), "genes\n")

cat("\nRunning R spacexr RCTD (doublet mode, UMI_min=20, 8 cores)...\n")
t0 <- proc.time()
myRCTD <- create.RCTD(puck, reference, max_cores = 8, UMI_min = 20)
myRCTD <- run.RCTD(myRCTD, doublet_mode = "doublet")
elapsed <- (proc.time() - t0)["elapsed"]

n_px <- nrow(myRCTD@results$results_df)
cat("\n  R elapsed:", elapsed, "seconds\n")
cat("  R pixels:", n_px, "\n")
cat("  R pixels/sec:", n_px / elapsed, "\n")

write.csv(data.frame(
    implementation = "R_spacexr_8cores",
    elapsed_s = as.numeric(elapsed),
    n_pixels = n_px,
    n_types = length(unique(reference@cell_types)),
    pixels_per_sec = n_px / as.numeric(elapsed)
), file.path(output_dir, "r_xenium_timing.csv"), row.names = FALSE)
cat("Saved timing CSV\n")
' 2>&1

echo "Completed at $(date)"
