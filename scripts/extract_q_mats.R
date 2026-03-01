#!/usr/bin/env Rscript
library(spacexr)
library(reticulate)

# We use reticulate to save as .npz since it's easier than finding a native R way
# Make sure numpy is available in the python environment
np <- import("numpy")

out_dir <- "src/rctd/data"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("Extracting X_vals...\n")
X_vals <- readRDS(system.file("extdata", "Qmat/X_vals.rds", package = "spacexr"))

cat("Extracting Q matrices...\n")
Q1 <- readRDS(system.file("extdata", "Qmat/Q_mat_1.rds", package = "spacexr"))
Q2 <- readRDS(system.file("extdata", "Qmat/Q_mat_2.rds", package = "spacexr"))
Q3 <- readRDS(system.file("extdata", "Qmat/Q_mat_3.rds", package = "spacexr"))
Q4 <- readRDS(system.file("extdata", "Qmat/Q_mat_4.rds", package = "spacexr"))
Q5 <- readRDS(system.file("extdata", "Qmat/Q_mat_5.rds", package = "spacexr"))

Q_mat_all <- c(Q1, Q2, Q3, Q4, Q5)

# Convert to a dictionary for python
out_dict <- list()
out_dict[["X_vals"]] <- as.numeric(X_vals)

for (sigma in names(Q_mat_all)) {
  # Q_mat has K_val+3 rows and length(X_vals) cols
  out_dict[[paste0("Q_", sigma)]] <- as.matrix(Q_mat_all[[sigma]])
}

cat("Saving to src/rctd/data/q_matrices.npz...\n")
do.call(np$savez_compressed, c(list(file=file.path(out_dir, "q_matrices.npz")), out_dict))

cat("Done!\n")
