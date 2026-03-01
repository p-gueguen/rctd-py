#!/usr/bin/env Rscript
# Benchmark JUST the pixel decomposition (fitPixels) in R spacexr
# This is the apples-to-apples comparison with Python's solve_irwls_batch

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

# Do the setup once (this is not part of IRWLS timing)
myRCTD <- create.RCTD(puck, reference, max_cores = 1)

# Extract internals needed for fitPixels
# (This runs fitBulk + chooseSigma once)
myRCTD <- spacexr:::fitBulk(myRCTD)
myRCTD <- spacexr:::choose_sigma_c(myRCTD)

# Time just the pixel decomposition (fitPixels equivalent)
# This is the part that calls solveIRWLS.weights for each pixel
n_reps <- 5
times_full_pipe <- numeric(n_reps)
times_irwls_only <- numeric(n_reps)

for (i in seq_len(n_reps)) {
    # Time fitPixels only (single-core)
    t0 <- proc.time()
    results <- spacexr:::fitPixels(myRCTD, doublet_mode = "full")
    t1 <- proc.time()
    times_irwls_only[i] <- (t1 - t0)["elapsed"]
    cat(sprintf("  fitPixels run %d: %.4fs\n", i, times_irwls_only[i]))
}

cat(sprintf("IRWLS_TIMES:%s\n", paste(times_irwls_only, collapse = ",")))

# Also time with multi-core to show R's parallelization
for (cores in c(4, 8)) {
    myRCTD2 <- myRCTD
    myRCTD2@config$max_cores <- cores
    times_mc <- numeric(3)
    for (i in 1:3) {
        t0 <- proc.time()
        results <- spacexr:::fitPixels(myRCTD2, doublet_mode = "full")
        t1 <- proc.time()
        times_mc[i] <- (t1 - t0)["elapsed"]
    }
    cat(sprintf("R_%dcores_TIMES:%s\n", cores, paste(round(times_mc, 4), collapse = ",")))
}
