#!/usr/bin/env Rscript
library(spacexr)

# Load vignette data
refdir <- system.file("extdata", "Reference", "Vignette", package = "spacexr")
counts <- as.matrix(read.csv(file.path(refdir, "dge.csv"), row.names = 1))
meta <- read.csv(file.path(refdir, "meta_data.csv"))
cell_types <- setNames(as.factor(meta$cluster), meta$barcode)
nUMI_ref <- setNames(meta$nUMI, meta$barcode)
reference <- Reference(counts, cell_types, nUMI_ref)

spdir <- system.file("extdata", "SpatialRNA", "Vignette", package = "spacexr")
sp_counts <- as.matrix(read.csv(file.path(spdir, "MappedDGEForR.csv"), row.names = 1))
coords <- read.csv(file.path(spdir, "BeadLocationsForR.csv"), row.names = 1)
sp_nUMI <- colSums(sp_counts)
puck <- SpatialRNA(coords, sp_counts, sp_nUMI)

# Time full RCTD pipeline (create + fitBulk + choose_sigma + fitPixels)
n_reps <- 3
times <- numeric(n_reps)
for (i in seq_len(n_reps)) {
    t0 <- proc.time()
    myRCTD <- create.RCTD(puck, reference, max_cores = 1)
    myRCTD <- run.RCTD(myRCTD, doublet_mode = "full")
    t1 <- proc.time()
    times[i] <- (t1 - t0)["elapsed"]
    cat(sprintf("  Run %d: %.3fs\n", i, times[i]))
}
cat(sprintf("R_TIMES:%s\n", paste(times, collapse = ",")))
