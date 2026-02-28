#!/usr/bin/env Rscript
#' Convert an R spacexr Reference or Seurat object to text/MTX files
#' that rctd-py can read.
#'
#' Usage:
#'   Rscript scripts/convert_reference.R input.rds output_dir/
#'
#' Outputs (in output_dir/):
#'   ref_counts.mtx  - Sparse count matrix (genes x cells, MatrixMarket)
#'   ref_meta.csv    - Cell metadata with cell_type column
#'   ref_genes.txt   - Gene names (one per line)
#'   ref_cells.txt   - Cell barcodes (one per line)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
    stop("Usage: Rscript convert_reference.R <input.rds> <output_dir/>")
}

in_rds <- args[1]
out_dir <- args[2]

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
    library(Matrix)
})

message("Loading ", in_rds)
obj <- readRDS(in_rds)

if (is.list(obj) && !is.null(obj$reference)) {
    obj <- obj$reference
}

if (inherits(obj, "Seurat")) {
    counts <- obj@assays$RNA$counts
    if (is.null(counts)) counts <- obj@assays$RNA@counts
    meta <- obj@meta.data

    writeMM(counts, file = file.path(out_dir, "ref_counts.mtx"))
    write.csv(meta, file = file.path(out_dir, "ref_meta.csv"), row.names = TRUE)
    write.table(rownames(counts), file = file.path(out_dir, "ref_genes.txt"),
                row.names = FALSE, col.names = FALSE, quote = FALSE)
    write.table(colnames(counts), file = file.path(out_dir, "ref_cells.txt"),
                row.names = FALSE, col.names = FALSE, quote = FALSE)
    message("Saved Seurat reference to ", out_dir)

} else if (inherits(obj, "Reference")) {
    counts <- obj@counts
    cell_types <- obj@cell_types
    nUMI <- obj@nUMI

    writeMM(counts, file = file.path(out_dir, "ref_counts.mtx"))
    meta <- data.frame(cell_type = as.character(cell_types),
                       nUMI = as.numeric(nUMI),
                       row.names = colnames(counts))
    write.csv(meta, file = file.path(out_dir, "ref_meta.csv"), row.names = TRUE)
    write.table(rownames(counts), file = file.path(out_dir, "ref_genes.txt"),
                row.names = FALSE, col.names = FALSE, quote = FALSE)
    write.table(colnames(counts), file = file.path(out_dir, "ref_cells.txt"),
                row.names = FALSE, col.names = FALSE, quote = FALSE)
    message("Saved spacexr Reference to ", out_dir)
} else {
    stop("Unknown object type: ", class(obj))
}
