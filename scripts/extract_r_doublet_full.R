#!/usr/bin/env Rscript
# Extract FULL R doublet classifications (all pixels) for Python comparison
# Also time R's RCTD on the same Xenium data for speedup comparison
library(spacexr)

output_dir <- "/home/pgueguen/git/rctd-py/data/xenium_validation"

cat("=== Part 1: Extract full doublet results from saved R object ===\n")
myRCTD <- qs2::qs_read(file.path(output_dir, "r_rctd_object.qs2"), nthreads = 8)

results <- myRCTD@results
rdf <- results$results_df
cat("  results_df dim:", dim(rdf), "\n")
cat("  spot_class distribution:\n")
print(table(rdf$spot_class))

# Save FULL results_df (all pixels, not just 1000)
write.csv(rdf, file.path(output_dir, "r_results_df_full.csv"))
cat("  Saved r_results_df_full.csv:", nrow(rdf), "rows\n")

# Also extract weights_doublet (the 2-column doublet pair weights)
if (!is.null(results$weights_doublet)) {
    wd <- results$weights_doublet
    cat("  weights_doublet: list of", length(wd), "elements\n")

    # R stores weights_doublet as a list of per-pixel data frames
    # Each element has columns for the two types in the doublet pair
    # Extract into a matrix format
    n_pixels <- nrow(rdf)
    doublet_w1 <- numeric(n_pixels)
    doublet_w2 <- numeric(n_pixels)
    doublet_t1 <- character(n_pixels)
    doublet_t2 <- character(n_pixels)

    for (i in seq_len(n_pixels)) {
        wi <- wd[[i]]
        if (is.data.frame(wi) || is.matrix(wi)) {
            doublet_t1[i] <- colnames(wi)[1]
            doublet_t2[i] <- colnames(wi)[2]
            doublet_w1[i] <- wi[1, 1]
            doublet_w2[i] <- wi[1, 2]
        } else {
            doublet_t1[i] <- NA
            doublet_t2[i] <- NA
            doublet_w1[i] <- NA
            doublet_w2[i] <- NA
        }
    }

    doublet_df <- data.frame(
        barcode = rownames(rdf),
        first_type = rdf$first_type,
        second_type = rdf$second_type,
        spot_class = rdf$spot_class,
        first_class = rdf$first_class,
        second_class = rdf$second_class,
        min_score = rdf$min_score,
        singlet_score = rdf$singlet_score,
        doublet_weight_1 = doublet_w1,
        doublet_weight_2 = doublet_w2,
        doublet_type_1 = doublet_t1,
        doublet_type_2 = doublet_t2,
        stringsAsFactors = FALSE
    )
    write.csv(doublet_df, file.path(output_dir, "r_doublet_full.csv"), row.names = FALSE)
    cat("  Saved r_doublet_full.csv:", nrow(doublet_df), "rows\n")
    cat("  Sample:\n")
    print(head(doublet_df, 3))
}

cat("\n=== Part 2: Time R spacexr on same Xenium data ===\n")

# Load spatial data
library(Seurat)
cat("Loading Xenium spatial data...\n")
h5_path <- "/srv/gstore/projects/p36005/Xenium_Data/data_20240724__105930__o35844/output-XETG00404__0022381__Region_3__20240724__110035/cell_feature_matrix.h5"
spatial <- Read10X_h5(h5_path)
cat("  Spatial:", ncol(spatial), "cells,", nrow(spatial), "genes\n")

# Create SpatialRNA puck
coords <- data.frame(x = 1:ncol(spatial), y = 1:ncol(spatial), row.names = colnames(spatial))
puck <- SpatialRNA(coords, spatial)

# Load reference
ref_rds <- "/srv/GT/databases/RCTD_References/p36005_UM_references/duodenum/duodenum_rctd.rds"
reference <- readRDS(ref_rds)
cat("  Reference:", length(reference@cell_types), "cells,", nrow(reference@counts), "genes\n")

# Time RCTD
cat("\nRunning R spacexr RCTD (doublet mode, UMI_min=20)...\n")
t0 <- proc.time()

myRCTD_new <- create.RCTD(puck, reference, max_cores = 8, UMI_min = 20)
myRCTD_new <- run.RCTD(myRCTD_new, doublet_mode = "doublet")

elapsed <- (proc.time() - t0)["elapsed"]
cat("  R elapsed:", elapsed, "seconds\n")
cat("  R pixels/sec:", nrow(myRCTD_new@results$results_df) / elapsed, "\n")

# Save timing
write.csv(data.frame(
    implementation = "R_spacexr",
    elapsed_s = as.numeric(elapsed),
    n_pixels = nrow(myRCTD_new@results$results_df),
    n_types = length(reference@cell_types),
    pixels_per_sec = nrow(myRCTD_new@results$results_df) / as.numeric(elapsed)
), file.path(output_dir, "r_xenium_timing.csv"), row.names = FALSE)

cat("\nDone!\n")
