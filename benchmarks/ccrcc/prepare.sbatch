#!/bin/bash
#SBATCH --job-name=ar_ccrcc_prepare
#SBATCH --output=/misc/GT/analysis/pgueguen/rctd-py/autoresearch_ccrcc/wt/benchmarks/ccrcc/out/logs/prepare_%j.log
#SBATCH --error=/misc/GT/analysis/pgueguen/rctd-py/autoresearch_ccrcc/wt/benchmarks/ccrcc/out/logs/prepare_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodelist=fgcz-r-023
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
cd /misc/GT/analysis/pgueguen/rctd-py/autoresearch_ccrcc/wt || exit 1
echo "node=$(hostname) date=$(date)"
.venv/bin/python benchmarks/ccrcc/prepare.py
echo "EXIT=$? date=$(date)"
