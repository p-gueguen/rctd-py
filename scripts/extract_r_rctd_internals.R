#!/usr/bin/env Rscript
# Extract internal state from saved R RCTD object for Python comparison
library(spacexr)
library(Matrix)

output_dir <- "/home/pgueguen/git/rctd-py/data/xenium_validation"

cat("Loading saved RCTD object...\n")
myRCTD <- qs2::qs_read(file.path(output_dir, "r_rctd_object.qs2"), nthreads = 8)

cat("RCTD object loaded\n")
cat("  Slots:", slotNames(myRCTD), "\n")

# 1. Normalized profiles (cell_type_info$renorm)
renorm <- myRCTD@cell_type_info$renorm
if (!is.null(renorm)) {
    norm_profiles <- as.matrix(renorm[[1]])
    cat("\nNormalized profiles (renorm[[1]]):\n")
    cat("  dim:", dim(norm_profiles), "\n")
    cat("  colnames:", head(colnames(norm_profiles), 5), "\n")
    cat("  rownames sample:", head(rownames(norm_profiles), 5), "\n")
    cat("  colSums:", head(colSums(norm_profiles), 5), "\n")
    cat("  range:", range(norm_profiles), "\n")
    write.csv(as.data.frame(norm_profiles),
              file.path(output_dir, "r_norm_profiles.csv"))
    cat("  Saved r_norm_profiles.csv\n")
} else {
    cat("WARNING: cell_type_info$renorm is NULL!\n")
}

# 2. Gene lists
gene_list_reg <- myRCTD@internal_vars$gene_list_reg
gene_list_bulk <- myRCTD@internal_vars$gene_list_bulk
cat("\nGene lists:\n")
cat("  gene_list_reg:", length(gene_list_reg), "\n")
cat("  gene_list_bulk:", length(gene_list_bulk), "\n")
writeLines(gene_list_reg, file.path(output_dir, "r_gene_list_reg.txt"))
writeLines(gene_list_bulk, file.path(output_dir, "r_gene_list_bulk.txt"))

# 3. Sigma
sigma_de <- myRCTD@internal_vars_de$sigma
if (!is.null(sigma_de)) {
    cat("\nSigma (internal_vars_de):", sigma_de, "\n")
    writeLines(as.character(sigma_de), file.path(output_dir, "r_sigma.txt"))
}
sigma_iv <- myRCTD@internal_vars$sigma
if (!is.null(sigma_iv)) {
    cat("Sigma (internal_vars):", sigma_iv, "\n")
}

# 4. Puck nUMI info
puck <- myRCTD@spatialRNA
cat("\nPuck:\n")
cat("  n_pixels:", length(puck@nUMI), "\n")
cat("  n_genes:", nrow(puck@counts), "\n")
cat("  nUMI range:", range(puck@nUMI), "\n")
cat("  nUMI mean:", mean(puck@nUMI), "\n")
cat("  nUMI median:", median(puck@nUMI), "\n")
cat("  Puck gene count:", nrow(puck@counts), "(should match gene_list_bulk:", length(gene_list_bulk), ")\n")

# Check if puck nUMI is recomputed or original
# If original, nUMI != colSums(counts) for restricted genes
puck_numi_check <- colSums(puck@counts)
cat("  colSums(puck@counts) range:", range(puck_numi_check), "\n")
cat("  puck@nUMI range:", range(puck@nUMI), "\n")
cat("  nUMI == colSums?", all.equal(puck@nUMI, puck_numi_check), "\n")

# 5. Config
cat("\nConfig:\n")
for (nm in names(myRCTD@config)) {
    val <- myRCTD@config[[nm]]
    if (length(val) <= 5) {
        cat("  ", nm, ":", paste(val, collapse=", "), "\n")
    }
}

# 6. Proportions
proportions <- myRCTD@internal_vars$proportions
if (!is.null(proportions)) {
    cat("\nBulk proportions:\n")
    cat("  sum:", sum(proportions), "\n")
    cat("  top 5:\n")
    print(head(sort(proportions, decreasing=TRUE), 5))
    write.csv(data.frame(type=names(proportions), proportion=as.numeric(proportions)),
              file.path(output_dir, "r_bulk_proportions.csv"), row.names=FALSE)
} else {
    cat("\nProportions: NULL\n")
}

# 7. Per-pixel results (first 1000)
results <- myRCTD@results
if (!is.null(results)) {
    cat("\nResults:\n")
    cat("  names:", names(results), "\n")

    if (!is.null(results$weights)) {
        w <- as.matrix(results$weights)
        cat("  weights dim:", dim(w), "\n")
        cat("  weight sums (first 5):", head(rowSums(w), 5), "\n")
        cat("  weight range:", range(w, na.rm=TRUE), "\n")
        n_save <- min(1000, nrow(w))
        write.csv(as.data.frame(w[1:n_save, ]),
                  file.path(output_dir, "r_pixel_weights_sample.csv"))
        writeLines(rownames(w)[1:n_save],
                   file.path(output_dir, "r_pixel_barcodes_sample.txt"))
    }

    if (!is.null(results$results_df)) {
        rdf <- results$results_df
        cat("  results_df dim:", dim(rdf), "\n")
        cat("  spot_class distribution:\n")
        print(table(rdf$spot_class))
        write.csv(rdf[1:min(1000, nrow(rdf)), ],
                  file.path(output_dir, "r_results_df_sample.csv"))
    }
}

# 8. Normalized profiles for reg genes
if (!is.null(renorm)) {
    norm_prof_reg <- as.matrix(renorm[[1]][gene_list_reg, ])
    cat("\nNormalized profiles restricted to gene_list_reg:\n")
    cat("  dim:", dim(norm_prof_reg), "\n")
    cat("  colSums:", head(colSums(norm_prof_reg)), "\n")
    write.csv(as.data.frame(norm_prof_reg),
              file.path(output_dir, "r_norm_profiles_reg.csv"))
}

# 9. Raw profiles for bulk genes
info <- myRCTD@cell_type_info$info
if (!is.null(info)) {
    raw_profiles <- as.matrix(info[[1]])
    cat("\nRaw profiles (info[[1]]):\n")
    cat("  dim:", dim(raw_profiles), "\n")
    cat("  colSums (first 5):", head(colSums(raw_profiles)), "\n")
    raw_prof_bulk <- raw_profiles[gene_list_bulk, ]
    write.csv(as.data.frame(raw_prof_bulk),
              file.path(output_dir, "r_raw_profiles_bulk.csv"))
}

# 10. nUMI for all pixels
write.csv(data.frame(barcode=names(puck@nUMI), nUMI=puck@nUMI),
          file.path(output_dir, "r_puck_nUMI_all.csv"), row.names=FALSE)

cat("\nAll extractions complete!\n")
