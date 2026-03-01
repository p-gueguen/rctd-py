#!/usr/bin/env Rscript

library(Seurat)
library(dplyr)
library(Matrix)
devtools::load_all("/home/pgueguen/git/spacexr")
source("/home/pgueguen/git/spacexr/R/fitPixels_gpu.R")

message("Loading Reference...")
refdir <- system.file("extdata",'Reference/Vignette',package = 'spacexr')
counts <- read.csv(file.path(refdir,"dge.csv"))
rownames(counts) <- counts[,1]; counts[,1] <- NULL
meta_data <- read.csv(file.path(refdir,"meta_data.csv"))
cell_types <- as.factor(setNames(meta_data$cluster, meta_data$barcode))
nUMI_ref <- setNames(meta_data$nUMI, meta_data$barcode)
reference <- Reference(counts, cell_types, nUMI_ref)

message("Loading target cell subset from Python run...")
datadir <- system.file("extdata",'SpatialRNA/Vignette',package = 'spacexr')
sp_counts <- read.csv(file.path(datadir,"MappedDGEForR.csv"))
coords <- read.csv(file.path(datadir,"BeadLocationsForR.csv"))
rownames(sp_counts) <- sp_counts[,1]; sp_counts[,1] <- NULL
rownames(coords) <- coords$barcodes; coords$barcodes <- NULL
sp_nUMI <- colSums(sp_counts)
puck <- SpatialRNA(coords, sp_counts, sp_nUMI)

myRCTD <- create.RCTD(puck, reference, max_cores = 16)

message("Running setup tasks (fitBulk, choose_sigma_c)...")
myRCTD <- fitBulk(myRCTD)
myRCTD <- choose_sigma_c(myRCTD)

message("1. Running spacexr CPU (Gold Standard)...")
time_cpu <- system.time({
    myRCTD_cpu <- fitPixels(myRCTD, doublet_mode = "doublet")
})

message("Extracting CPU results...")
res_cpu_df <- myRCTD_cpu@results$results_df
w_cpu <- myRCTD_cpu@results$weights
write.csv(res_cpu_df, "data/res_spacexr_cpu.csv")
writeMM(as(w_cpu, "sparseMatrix"), "data/weights_spacexr_cpu.mtx")
write.table(rownames(w_cpu), "data/weights_spacexr_cpu_cells.txt", row.names=F, col.names=F, quote=F)

message("2. Running spacexr GPU (Reticulate)...")
time_gpu <- system.time({
    myRCTD_gpu <- fitPixels_gpu(myRCTD, mode = "doublet")
})

message("Extracting GPU results...")
res_gpu_df <- myRCTD_gpu@results$results_df
w_gpu <- myRCTD_gpu@results$weights
write.csv(res_gpu_df, "data/res_spacexr_gpu.csv")
writeMM(as(w_gpu, "sparseMatrix"), "data/weights_spacexr_gpu.mtx")
write.table(rownames(w_gpu), "data/weights_spacexr_gpu_cells.txt", row.names=F, col.names=F, quote=F)

# Save times
times <- data.frame(
    model = c("spacexr_cpu", "spacexr_gpu"),
    elapsed = c(time_cpu["elapsed"], time_gpu["elapsed"])
)
write.csv(times, "data/spacexr_times.csv", row.names=FALSE)
message("Done.")
