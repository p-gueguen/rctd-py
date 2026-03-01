#!/bin/bash
#SBATCH --job-name=vignette_rctd_benchmark
#SBATCH --output=/home/pgueguen/git/rctd-py/vignette_bench_%j.log
#SBATCH --error=/home/pgueguen/git/rctd-py/vignette_bench_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1

cd /home/pgueguen/git/rctd-py

echo "============================================="
echo "1. Initiating Python Architecture Benchmarks"
echo "============================================="
PYTHONPATH=src ./.venv/bin/python scripts/generate_report_data.py

echo "============================================="
echo "2. Initiating R Spacexr Architecture Benchmarks"
echo "============================================="
module load Dev/R/4.5.0
Rscript scripts/generate_r_report_data.R

echo "Benchmark data successfully generated!"

echo "Benchmark execution and rendering successfully completed!"
