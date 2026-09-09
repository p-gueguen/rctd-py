#!/bin/bash
#SBATCH --job-name=ar_ccrcc_eval
#SBATCH --output=/misc/GT/analysis/pgueguen/rctd-py/autoresearch_ccrcc/wt/benchmarks/ccrcc/out/logs/eval_%j.log
#SBATCH --error=/misc/GT/analysis/pgueguen/rctd-py/autoresearch_ccrcc/wt/benchmarks/ccrcc/out/logs/eval_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodelist=fgcz-r-023
# READ-ONLY harness file. AR_SPLIT / AR_TAG / AR_EXTRA come from verify.py via --export.
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
cd /misc/GT/analysis/pgueguen/rctd-py/autoresearch_ccrcc/wt || exit 1
echo "node=$(hostname) date=$(date) sha=$(git rev-parse --short HEAD) split=${AR_SPLIT:-tune} tag=${AR_TAG}"
.venv/bin/python benchmarks/ccrcc/eval.py --split "${AR_SPLIT:-tune}" --tag "${AR_TAG}" ${AR_EXTRA}
echo "EXIT=$? date=$(date)"
