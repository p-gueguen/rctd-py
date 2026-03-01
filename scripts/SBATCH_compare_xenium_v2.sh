#!/bin/bash
#SBATCH --job-name=rctd_cmp_v2
#SBATCH --output=/home/pgueguen/git/rctd-py/xenium_compare_v2_%j.log
#SBATCH --error=/home/pgueguen/git/rctd-py/xenium_compare_v2_%j.log
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=employee

echo "Starting R vs Python Xenium comparison v2 at $(date)"
echo "Fixes: constrain=True for doublet pairs, subset norm_profiles from bulk"
cd /home/pgueguen/git/rctd-py

export PATH="/home/pgueguen/git/rctd-py/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1

python scripts/compare_r_vs_python_xenium.py 2>&1

echo "Completed at $(date)"
