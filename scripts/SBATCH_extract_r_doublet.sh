#!/bin/bash
#SBATCH --job-name=rctd_r_extract
#SBATCH --output=/home/pgueguen/rctd_r_extract_%j.log
#SBATCH --error=/home/pgueguen/rctd_r_extract_%j.log
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=employee

echo "Extracting R doublet results + timing at $(date)"

module load Dev/R/4.5.0
cd /home/pgueguen/git/rctd-py

Rscript scripts/extract_r_doublet_full.R 2>&1

echo "Completed at $(date)"
