#!/usr/bin/env Rscript
# Minimal: create RCTD object and save before/after running
library(spacexr)
library(Matrix)
library(Seurat)

output_dir <- "/home/pgueguen/git/rctd-py/data/xenium_validation"

# Load reference
cat("Loading reference...\n")
ref_counts <- readMM("/home/pgueguen/git/rctd-py/data/ref_counts.mtx")
ref_meta <- read.csv("/home/pgueguen/git/rctd-py/data/ref_meta.csv")
ref_genes <- readLines("/home/pgueguen/git/rctd-py/data/ref_genes.txt")
ref_cells <- readLines("/home/pgueguen/git/rctd-py/data/ref_cells.txt")

rownames(ref_counts) <- ref_genes
colnames(ref_counts) <- ref_cells
ref_counts <- as(ref_counts, "CsparseMatrix")
cell_types <- setNames(factor(ref_meta$cell_type), ref_cells)
nUMI_ref <- setNames(Matrix::colSums(ref_counts), ref_cells)
reference <- Reference(ref_counts, cell_types, nUMI_ref)

# Load spatial
cat("Loading spatial...\n")
spatial_obj <- Read10X_h5("/srv/gstore/projects/p36005/Xenium_Data/data_20240724__105930__o35844/output-XETG00404__0022381__Region_3__20240724__110035/cell_feature_matrix.h5")
if (is.list(spatial_obj)) spatial_obj <- spatial_obj[["Gene Expression"]]
n_cells <- ncol(spatial_obj)
coords <- data.frame(x = seq_len(n_cells), y = seq_len(n_cells))
rownames(coords) <- colnames(spatial_obj)
puck <- SpatialRNA(coords, spatial_obj)

# Create RCTD
cat("Creating RCTD...\n")
myRCTD <- create.RCTD(puck, reference, max_cores = 8, UMI_min = 20)

# Explore the structure to understand it
cat("\n== Exploring cell_type_info structure ==\n")
cti <- myRCTD@cell_type_info

# Save the raw cell_type_info for Python to read
cat("Saving cell_type_info...\n")
saveRDS(cti, file.path(output_dir, "r_cell_type_info.rds"))

# Print what we can about the structure
cat("cell_type_info names:", names(cti), "\n")
for (nm in names(cti)) {
    x <- cti[[nm]]
    cat("\n  ", nm, ":\n")
    if (is.null(x)) { cat("    NULL\n"); next }
    cat("    class:", class(x), "\n")
    if (is.list(x)) {
        cat("    length:", length(x), "\n")
        for (i in seq_along(x)) {
            cat("      [[", i, "]] class:", class(x[[i]]),
                "dim:", if(!is.null(dim(x[[i]]))) paste(dim(x[[i]]), collapse="x") else "NULL",
                "length:", length(x[[i]]), "\n")
            if (is.data.frame(x[[i]]) || is.matrix(x[[i]])) {
                cat("        rownames sample:", head(rownames(x[[i]]), 3), "\n")
                cat("        colnames sample:", head(colnames(x[[i]]), 3), "\n")
                cat("        colSums:", head(colSums(as.matrix(x[[i]]))), "\n")
            } else if (is.numeric(x[[i]])) {
                cat("        values:", head(x[[i]], 5), "\n")
            } else if (is.character(x[[i]])) {
                cat("        values:", head(x[[i]], 5), "\n")
            }
        }
    }
}

# Print internal_vars
cat("\n== internal_vars ==\n")
cat("names:", names(myRCTD@internal_vars), "\n")
for (nm in names(myRCTD@internal_vars)) {
    x <- myRCTD@internal_vars[[nm]]
    if (is.null(x)) { cat(nm, ": NULL\n"); next }
    if (is.atomic(x) && length(x) <= 10) {
        cat(nm, ":", x, "\n")
    } else {
        cat(nm, ": class=", class(x), "length=", length(x), "\n")
    }
}

# Print puck info
cat("\n== Restricted puck ==\n")
rpuck <- myRCTD@spatialRNA
cat("n_pixels:", length(rpuck@nUMI), "\n")
cat("n_genes:", nrow(rpuck@counts), "\n")
cat("nUMI range:", range(rpuck@nUMI), "\n")
cat("Gene names sample:", head(rownames(rpuck@counts), 5), "\n")

# Now run RCTD
cat("\n== Running RCTD (doublet mode) ==\n")
myRCTD <- run.RCTD(myRCTD, doublet_mode = "doublet")

# Save the full RCTD object
cat("Saving RCTD object...\n")
qs2::qs_save(myRCTD, file.path(output_dir, "r_rctd_object.qs2"), nthreads = 8)

# Post-run info
cat("\n== Post-run ==\n")
cat("internal_vars_de names:", names(myRCTD@internal_vars_de), "\n")
for (nm in names(myRCTD@internal_vars_de)) {
    x <- myRCTD@internal_vars_de[[nm]]
    if (is.atomic(x) && length(x) <= 10) cat(nm, ":", x, "\n")
}

cat("\nDone!\n")
