#!/bin/bash
#SBATCH --job-name=extract_rctd
#SBATCH --output=/home/pgueguen/git/rctd-py/extract_rctd_%j.log
#SBATCH --error=/home/pgueguen/git/rctd-py/extract_rctd_%j.log
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=employee

echo "Extracting R RCTD internals at $(date)"
module load Dev/R/4.5.0
cd /home/pgueguen/git/rctd-py
Rscript scripts/extract_r_rctd_internals.R
echo "Completed at $(date)"
