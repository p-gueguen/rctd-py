#!/usr/bin/env Rscript
# Re-run R spacexr RCTD on the same Xenium data and save internal state
library(spacexr)
library(Matrix)

output_dir <- "/home/pgueguen/git/rctd-py/data/xenium_validation"

# Load reference (same as Python uses)
cat("Loading reference...\n")
ref_counts <- readMM("/home/pgueguen/git/rctd-py/data/ref_counts.mtx")
ref_meta <- read.csv("/home/pgueguen/git/rctd-py/data/ref_meta.csv")
ref_genes <- readLines("/home/pgueguen/git/rctd-py/data/ref_genes.txt")
ref_cells <- readLines("/home/pgueguen/git/rctd-py/data/ref_cells.txt")

# Check dims: ref_counts from readMM may be cells x genes or genes x cells
cat("  Raw matrix dims:", dim(ref_counts), "\n")
cat("  Expected:", length(ref_cells), "cells x", length(ref_genes), "genes\n")

# Determine orientation
if (nrow(ref_counts) == length(ref_genes) && ncol(ref_counts) == length(ref_cells)) {
    # genes x cells -> need to transpose for spacexr (which wants genes x cells dgCMatrix)
    ref_counts_gc <- ref_counts
    rownames(ref_counts_gc) <- ref_genes
    colnames(ref_counts_gc) <- ref_cells
} else if (nrow(ref_counts) == length(ref_cells) && ncol(ref_counts) == length(ref_genes)) {
    # cells x genes -> transpose to genes x cells
    ref_counts_gc <- t(ref_counts)
    rownames(ref_counts_gc) <- ref_genes
    colnames(ref_counts_gc) <- ref_cells
} else {
    stop("Unexpected matrix dimensions")
}
ref_counts_gc <- as(ref_counts_gc, "dgCMatrix")
cat("  Final ref counts:", nrow(ref_counts_gc), "genes x", ncol(ref_counts_gc), "cells\n")

cell_types <- setNames(factor(ref_meta$cell_type), ref_cells)
nUMI_ref <- setNames(Matrix::colSums(ref_counts_gc), ref_cells)

reference <- Reference(ref_counts_gc, cell_types, nUMI_ref)
cat("  Reference:", length(unique(cell_types)), "types,", ncol(ref_counts), "genes\n")

# Load spatial
cat("Loading spatial...\n")
library(Seurat)
spatial_obj <- Read10X_h5("/srv/gstore/projects/p36005/Xenium_Data/data_20240724__105930__o35844/output-XETG00404__0022381__Region_3__20240724__110035/cell_feature_matrix.h5")
# Xenium h5 returns a list of modalities; use Gene Expression
if (is.list(spatial_obj)) {
    cat("  Modalities:", names(spatial_obj), "\n")
    spatial_obj <- spatial_obj[["Gene Expression"]]
}
# Make fake coords (RCTD needs them but doesn't use them for deconvolution)
n_cells <- ncol(spatial_obj)
coords <- data.frame(x = seq_len(n_cells), y = seq_len(n_cells))
rownames(coords) <- colnames(spatial_obj)

puck <- SpatialRNA(coords, spatial_obj)
cat("  Spatial:", n_cells, "cells,", nrow(spatial_obj), "genes\n")

# Create RCTD with same params as ezRun
cat("Creating RCTD (UMI_min=20, doublet mode)...\n")
myRCTD <- create.RCTD(puck, reference, max_cores = 8, UMI_min = 20)

# Extract internal state BEFORE running
cat("\n== Internal state after create.RCTD ==\n")
gene_list <- myRCTD@config$gene_list
cat("gene_list (reg):", length(gene_list), "class:", class(gene_list), "\n")
if (length(gene_list) > 0) {
    writeLines(as.character(gene_list), file.path(output_dir, "r_gene_list_reg.txt"))
    cat("  First 5:", head(as.character(gene_list), 5), "\n")
} else {
    cat("  gene_list is empty, checking internal_vars...\n")
}

cat("Config:\n")
for (name in names(myRCTD@config)) {
    val <- myRCTD@config[[name]]
    if (length(val) <= 5) {
        cat("  ", name, ":", paste(val, collapse=", "), "\n")
    } else {
        cat("  ", name, ": [", length(val), "elements]\n")
    }
}

# cell_type_info
cti <- myRCTD@cell_type_info
cat("\ncell_type_info slots:", names(cti), "\n")
for (slot_name in names(cti)) {
    slot_val <- cti[[slot_name]]
    if (is.null(slot_val)) {
        cat("  ", slot_name, ": NULL\n")
        next
    }
    cat("  ", slot_name, ": class=", class(slot_val), "length=", length(slot_val), "\n")
    if (is.list(slot_val) && length(slot_val) >= 2) {
        cat("    [[1]] class:", class(slot_val[[1]]), "dim:", dim(slot_val[[1]]),
            "length:", length(slot_val[[1]]), "\n")
        cat("    [[2]] class:", class(slot_val[[2]]), "dim:", dim(slot_val[[2]]), "\n")
        if (length(slot_val) >= 3) {
            cat("    [[3]]:", slot_val[[3]], "\n")
        }

        # Save proportions
        props <- unlist(slot_val[[1]])
        cat("    Proportions:", head(props, 5), "\n")
        write.csv(data.frame(type=names(props), proportion=as.numeric(props)),
                  file.path(output_dir, paste0("r_", slot_name, "_proportions.csv")),
                  row.names=FALSE)

        # Save profiles
        profiles <- as.matrix(slot_val[[2]])
        cat("    Profile dim:", dim(profiles), "\n")
        cat("    Profile colSums:", head(colSums(profiles)), "\n")
        write.csv(as.data.frame(profiles),
                  file.path(output_dir, paste0("r_", slot_name, "_profiles.csv")))

        # Save nUMI
        if (length(slot_val) >= 3) {
            writeLines(as.character(slot_val[[3]]),
                       file.path(output_dir, paste0("r_", slot_name, "_nUMI.txt")))
        }
    }
}

# renorm already handled in loop above

# internal_vars
cat("\ninternal_vars:", names(myRCTD@internal_vars), "\n")
if (!is.null(myRCTD@internal_vars$sigma)) {
    cat("  sigma:", myRCTD@internal_vars$sigma, "\n")
    writeLines(as.character(myRCTD@internal_vars$sigma),
               file.path(output_dir, "r_sigma.txt"))
}
if (!is.null(myRCTD@internal_vars$gene_list_bulk)) {
    gene_list_bulk <- myRCTD@internal_vars$gene_list_bulk
    cat("  gene_list_bulk:", length(gene_list_bulk), "genes\n")
    writeLines(gene_list_bulk, file.path(output_dir, "r_gene_list_bulk.txt"))
}
if (!is.null(myRCTD@internal_vars$gene_list_reg)) {
    gene_list_reg_internal <- myRCTD@internal_vars$gene_list_reg
    cat("  gene_list_reg:", length(gene_list_reg_internal), "genes\n")
}

# Spatial puck info after restriction
puck_restricted <- myRCTD@spatialRNA
cat("\nRestricted puck:\n")
cat("  nUMI range:", range(puck_restricted@nUMI), "\n")
cat("  n_pixels:", length(puck_restricted@nUMI), "\n")
cat("  n_genes:", nrow(puck_restricted@counts), "\n")
cat("  Gene names sample:", head(rownames(puck_restricted@counts), 5), "\n")

# Save puck nUMI for comparison
write.csv(data.frame(barcode=names(puck_restricted@nUMI),
                     nUMI=puck_restricted@nUMI),
          file.path(output_dir, "r_puck_nUMI_restricted.csv"), row.names=FALSE)

# Save the config as well
config_df <- data.frame(
    param = names(myRCTD@config),
    value = sapply(myRCTD@config, function(x) {
        if (length(x) <= 1) as.character(x) else paste0("[", length(x), " elements]")
    })
)
write.csv(config_df, file.path(output_dir, "r_config.csv"), row.names=FALSE)

cat("\n== Running RCTD ==\n")
myRCTD <- run.RCTD(myRCTD, doublet_mode = "doublet")

# Save post-run internal state
cat("\n== Post-run state ==\n")
cat("internal_vars_de:", names(myRCTD@internal_vars_de), "\n")
if (!is.null(myRCTD@internal_vars_de$sigma)) {
    cat("  sigma_de:", myRCTD@internal_vars_de$sigma, "\n")
    writeLines(as.character(myRCTD@internal_vars_de$sigma),
               file.path(output_dir, "r_sigma_de.txt"))
}

# Save the full RCTD object for further debugging
cat("Saving full RCTD object...\n")
qs2::qs_save(myRCTD, file.path(output_dir, "r_rctd_object.qs2"), nthreads = 8)

cat("\nDone!\n")
