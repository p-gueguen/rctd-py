#!/usr/bin/env Rscript
# Cross-validation: Run spacexr RCTD on vignette data and export all intermediate results
# for comparison with rctd-py Python implementation.
library(spacexr)
library(reticulate)

np <- import("numpy")
out_dir <- "scripts/r_reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(42)

# ============================================================
# PART 1: Load vignette data (same as spacexr vignette)
# ============================================================
cat("=== Loading vignette data ===\n")

# --- Reference ---
refdir <- system.file("extdata", "Reference/Vignette", package = "spacexr")
ref_counts <- read.csv(file.path(refdir, "dge.csv"))
rownames(ref_counts) <- ref_counts[, 1]; ref_counts[, 1] <- NULL
meta_data <- read.csv(file.path(refdir, "meta_data.csv"))
cell_types <- meta_data$cluster; names(cell_types) <- meta_data$barcode
cell_types <- as.factor(cell_types)
nUMI_ref <- meta_data$nUMI; names(nUMI_ref) <- meta_data$barcode

reference <- Reference(ref_counts, cell_types, nUMI_ref)
cat("  Reference:", ncol(reference@counts), "cells,", nrow(reference@counts), "genes,",
    length(levels(reference@cell_types)), "types\n")
cat("  Cell types:", levels(reference@cell_types), "\n")

# --- Spatial ---
datadir <- system.file("extdata", "SpatialRNA/Vignette", package = "spacexr")
sp_counts <- read.csv(file.path(datadir, "MappedDGEForR.csv"))
rownames(sp_counts) <- sp_counts[, 1]; sp_counts[, 1] <- NULL
coords <- read.csv(file.path(datadir, "BeadLocationsForR.csv"))
rownames(coords) <- coords$barcodes; coords$barcodes <- NULL
nUMI_sp <- colSums(sp_counts)

puck <- SpatialRNA(coords, sp_counts, nUMI_sp)
cat("  Spatial:", ncol(puck@counts), "pixels,", nrow(puck@counts), "genes\n")

# ============================================================
# PART 2: Save raw input data
# ============================================================
cat("\n=== Saving input data ===\n")

# Reference counts (cells x genes in Python convention)
ref_mat <- as.matrix(reference@counts)  # genes x cells
np$save(file.path(out_dir, "ref_counts.npy"), t(ref_mat))  # cells x genes
np$save(file.path(out_dir, "ref_gene_names.npy"), rownames(ref_mat))
np$save(file.path(out_dir, "ref_cell_barcodes.npy"), colnames(ref_mat))
np$save(file.path(out_dir, "ref_cell_types.npy"), as.character(reference@cell_types))
np$save(file.path(out_dir, "ref_nUMI.npy"), as.double(reference@nUMI))

# Spatial counts (pixels x genes in Python convention)
sp_mat <- as.matrix(puck@counts)  # genes x pixels
np$save(file.path(out_dir, "sp_counts.npy"), t(sp_mat))  # pixels x genes
np$save(file.path(out_dir, "sp_gene_names.npy"), rownames(sp_mat))
np$save(file.path(out_dir, "sp_pixel_names.npy"), colnames(sp_mat))
np$save(file.path(out_dir, "sp_nUMI.npy"), as.double(puck@nUMI))

# ============================================================
# PART 3: Run RCTD step-by-step and save intermediates
# ============================================================
cat("\n=== Running RCTD pipeline ===\n")

# Step 3a: Create RCTD object (computes cell_type_info and DE genes)
myRCTD <- create.RCTD(puck, reference, max_cores = 1)

# Save cell type profiles (genes x cell_types)
cell_type_means <- myRCTD@cell_type_info$info[[1]]
cell_type_names <- myRCTD@cell_type_info$info[[2]]
cat("  Cell type profiles shape:", dim(cell_type_means), "\n")
cat("  Cell type names:", cell_type_names, "\n")
np$save(file.path(out_dir, "cell_type_means.npy"), as.matrix(cell_type_means))
np$save(file.path(out_dir, "cell_type_names.npy"), cell_type_names)

# Save gene lists
gene_list_reg <- myRCTD@internal_vars$gene_list_reg
gene_list_bulk <- myRCTD@internal_vars$gene_list_bulk
cat("  gene_list_reg:", length(gene_list_reg), "genes\n")
cat("  gene_list_bulk:", length(gene_list_bulk), "genes\n")
np$save(file.path(out_dir, "gene_list_reg.npy"), gene_list_reg)
np$save(file.path(out_dir, "gene_list_bulk.npy"), gene_list_bulk)

# Step 3b: fitBulk
cat("\n=== fitBulk ===\n")
myRCTD <- fitBulk(myRCTD)
proportions <- myRCTD@internal_vars$proportions
renorm_profiles <- myRCTD@cell_type_info$renorm[[1]]
cat("  Proportions:", proportions, "\n")
cat("  Renorm profiles shape:", dim(renorm_profiles), "\n")
np$save(file.path(out_dir, "bulk_proportions.npy"), as.double(proportions))
np$save(file.path(out_dir, "renorm_profiles.npy"), as.matrix(renorm_profiles))

# Step 3c: choose_sigma_c
cat("\n=== choose_sigma_c ===\n")
myRCTD <- choose_sigma_c(myRCTD)
sigma_val <- myRCTD@internal_vars$sigma
cat("  Sigma:", sigma_val, "\n")
np$save(file.path(out_dir, "sigma.npy"), sigma_val)

# Step 3d: Save Q_mat and X_vals used
Q_mat_final <- myRCTD@internal_vars$Q_mat
X_vals_final <- myRCTD@internal_vars$X_vals
np$save(file.path(out_dir, "Q_mat_final.npy"), Q_mat_final)
np$save(file.path(out_dir, "X_vals_final.npy"), X_vals_final)
SQ_mat_final <- spacexr:::solve_sq(Q_mat_final, X_vals_final)
np$save(file.path(out_dir, "SQ_mat_final.npy"), SQ_mat_final)

# ============================================================
# PART 4: Run decomposition modes
# ============================================================

# ---- 4a: Full mode ----
cat("\n=== Full mode ===\n")
myRCTD_full <- run.RCTD(myRCTD, doublet_mode = "full")
full_weights <- as.matrix(myRCTD_full@results$weights)
cat("  Full weights shape:", dim(full_weights), "\n")
np$save(file.path(out_dir, "full_weights.npy"), full_weights)
np$save(file.path(out_dir, "full_pixel_names.npy"), rownames(full_weights))

# ---- 4b: Doublet mode ----
cat("\n=== Doublet mode ===\n")
myRCTD_doublet <- run.RCTD(myRCTD, doublet_mode = "doublet")

# Extract doublet results
results_df <- myRCTD_doublet@results$results_df
doublet_weights <- as.matrix(myRCTD_doublet@results$weights)
cat("  Doublet results_df rows:", nrow(results_df), "\n")
cat("  Spot class counts:\n")
print(table(results_df$spot_class))

np$save(file.path(out_dir, "doublet_weights.npy"), doublet_weights)
np$save(file.path(out_dir, "doublet_spot_class.npy"), as.character(results_df$spot_class))
np$save(file.path(out_dir, "doublet_first_type.npy"), as.character(results_df$first_type))
np$save(file.path(out_dir, "doublet_second_type.npy"), as.character(results_df$second_type))
np$save(file.path(out_dir, "doublet_pixel_names.npy"), rownames(results_df))

# Also save the per-pixel doublet weights
if (!is.null(myRCTD_doublet@results$weights_doublet)) {
  dbl_w <- as.matrix(myRCTD_doublet@results$weights_doublet)
  np$save(file.path(out_dir, "doublet_pair_weights.npy"), dbl_w)
}

# ---- 4c: Multi mode ----
cat("\n=== Multi mode ===\n")
myRCTD_multi <- run.RCTD(myRCTD, doublet_mode = "multi")

# Multi mode results are stored per-pixel as a list
multi_results <- myRCTD_multi@results
# Extract per-pixel all_weights (full mode weights)
pixel_names_multi <- names(multi_results)
n_pix_multi <- length(pixel_names_multi)
n_types <- length(cell_type_names)
multi_all_weights <- matrix(0, n_pix_multi, n_types)
multi_n_types <- integer(n_pix_multi)
multi_min_score <- numeric(n_pix_multi)
for (i in seq_along(pixel_names_multi)) {
  pn <- pixel_names_multi[i]
  multi_all_weights[i, ] <- multi_results[[pn]]$all_weights
  multi_n_types[i] <- length(multi_results[[pn]]$cell_type_list)
  multi_min_score[i] <- multi_results[[pn]]$min_score
}
cat("  Multi all_weights shape:", dim(multi_all_weights), "\n")
cat("  Multi n_types distribution:", table(multi_n_types), "\n")
np$save(file.path(out_dir, "multi_all_weights.npy"), multi_all_weights)
np$save(file.path(out_dir, "multi_n_types.npy"), as.integer(multi_n_types))
np$save(file.path(out_dir, "multi_pixel_names.npy"), pixel_names_multi)

# ============================================================
# PART 5: Function-level comparison data
# ============================================================
cat("\n=== Function-level reference data ===\n")

# --- X_vals ---
X_vals <- readRDS(system.file("extdata", "Qmat/X_vals.rds", package = "spacexr"))
np$save(file.path(out_dir, "x_vals.npy"), X_vals)

# --- ht_pdf_norm ---
test_x <- seq(-5, 5, by = 0.1)
ht_vals <- spacexr:::ht_pdf_norm(test_x)
np$save(file.path(out_dir, "ht_pdf_norm_x.npy"), test_x)
np$save(file.path(out_dir, "ht_pdf_norm_y.npy"), ht_vals)

# --- calc_Q_all with final sigma ---
spacexr:::set_likelihood_vars(Q_mat_final, X_vals_final)
test_Y <- as.integer(c(0, 1, 2, 5, 10, 20, 50, 100))
test_lambda <- c(0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0)
result <- spacexr:::calc_Q_all(test_Y, test_lambda)
np$save(file.path(out_dir, "calc_q_all_Y.npy"), as.double(test_Y))
np$save(file.path(out_dir, "calc_q_all_lambda.npy"), test_lambda)
np$save(file.path(out_dir, "calc_q_all_d0.npy"), result$d0_vec)
np$save(file.path(out_dir, "calc_q_all_d1.npy"), result$d1_vec)
np$save(file.path(out_dir, "calc_q_all_d2.npy"), result$d2_vec)

# Batch test
set.seed(123)
batch_Y <- sample(0:80, 200, replace = TRUE)
batch_lambda <- runif(200, 0.01, 500.0)
batch_result <- spacexr:::calc_Q_all(batch_Y, batch_lambda)
np$save(file.path(out_dir, "calc_q_all_batch_Y.npy"), as.double(batch_Y))
np$save(file.path(out_dir, "calc_q_all_batch_lambda.npy"), batch_lambda)
np$save(file.path(out_dir, "calc_q_all_batch_d0.npy"), batch_result$d0_vec)
np$save(file.path(out_dir, "calc_q_all_batch_d1.npy"), batch_result$d1_vec)
np$save(file.path(out_dir, "calc_q_all_batch_d2.npy"), batch_result$d2_vec)

# --- log-likelihood ---
nll_val <- spacexr:::calc_log_l_vec(test_lambda, test_Y)
np$save(file.path(out_dir, "log_likelihood_val.npy"), nll_val)
batch_nll <- spacexr:::calc_log_l_vec(batch_lambda, batch_Y)
np$save(file.path(out_dir, "batch_log_likelihood_val.npy"), batch_nll)

# --- IRWLS on a single pixel from vignette data ---
cat("=== Single pixel IRWLS ===\n")
gene_list <- gene_list_reg
profiles_reg <- renorm_profiles[gene_list, ]
pixel_idx <- 1
pixel_name <- colnames(puck@counts)[pixel_idx]
pixel_counts <- as.double(puck@counts[gene_list, pixel_name])
pixel_nUMI <- as.double(puck@nUMI[pixel_name])

S_single <- as.matrix(profiles_reg * pixel_nUMI)

# NOTE: R's fitPixels always uses constrain=FALSE for all modes
# Test unconstrained (matching R's actual pipeline)
res_single_uc <- spacexr:::solveIRWLS.weights(S_single, pixel_counts, pixel_nUMI,
                                  constrain = FALSE, n.iter = 50, MIN_CHANGE = 0.001)
cat("  Single pixel weights (unconstrained):", round(res_single_uc$weights, 4), "\n")
cat("  Converged:", res_single_uc$converged, "\n")
np$save(file.path(out_dir, "single_pixel_S.npy"), S_single)
np$save(file.path(out_dir, "single_pixel_Y.npy"), pixel_counts)
np$save(file.path(out_dir, "single_pixel_nUMI.npy"), pixel_nUMI)
np$save(file.path(out_dir, "single_pixel_weights.npy"), as.double(res_single_uc$weights))

# Also test constrained for reference
res_single_c <- spacexr:::solveIRWLS.weights(S_single, pixel_counts, pixel_nUMI,
                                  constrain = TRUE, n.iter = 50, MIN_CHANGE = 0.001)
cat("  Single pixel weights (constrained):", round(res_single_c$weights, 4), "\n")
np$save(file.path(out_dir, "single_pixel_weights_constrained.npy"), as.double(res_single_c$weights))

# --- Multi-sigma spline check ---
cat("=== Multi-sigma calc_Q_all ===\n")
Q_mat_all <- spacexr:::get_Q_all()
for (sig in c(10, 50, 100, 150, 200)) {
  q <- Q_mat_all[[as.character(sig)]]
  spacexr:::set_likelihood_vars(q, X_vals)
  d <- spacexr:::calc_Q_all(test_Y, test_lambda)
  np$save(file.path(out_dir, paste0("sigma_", sig, "_d0.npy")), d$d0_vec)
  np$save(file.path(out_dir, paste0("sigma_", sig, "_d1.npy")), d$d1_vec)
  np$save(file.path(out_dir, paste0("sigma_", sig, "_d2.npy")), d$d2_vec)
}

cat("\n=== All reference data saved to", out_dir, "===\n")
cat("Files saved:\n")
cat(paste0("  ", list.files(out_dir), "\n"))
