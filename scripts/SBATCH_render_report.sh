#!/bin/bash
#SBATCH --job-name=rctd_report
#SBATCH --output=/home/pgueguen/rctd_render_%j.log
#SBATCH --error=/home/pgueguen/rctd_render_%j.log
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=employee

echo "Rendering rctd-py analysis report at $(date)"

module load Dev/R/4.5.0
cd /home/pgueguen/git/rctd-py
export PATH="/home/pgueguen/git/rctd-py/.venv/bin:$PATH"
export RETICULATE_PYTHON="/home/pgueguen/git/rctd-py/.venv/bin/python"

Rscript -e 'rmarkdown::render("analysis_region3.qmd", output_file = "analysis_region3.html")' 2>&1

echo "Completed at $(date)"
