#!/bin/bash
#SBATCH --job-name=r_rctd_debug
#SBATCH --output=/home/pgueguen/git/rctd-py/r_rctd_debug_%j.log
#SBATCH --error=/home/pgueguen/git/rctd-py/r_rctd_debug_%j.log
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=employee

echo "Running R spacexr RCTD debug at $(date)"
module load Dev/R/4.5.0
cd /home/pgueguen/git/rctd-py
Rscript scripts/run_r_rctd_save.R
echo "Completed at $(date)"
