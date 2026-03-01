#!/bin/bash
#SBATCH --job-name=rctd_compare
#SBATCH --output=/home/pgueguen/git/rctd-py/xenium_compare_%j.log
#SBATCH --error=/home/pgueguen/git/rctd-py/xenium_compare_%j.log
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=employee

echo "Starting R vs Python Xenium comparison at $(date)"
cd /home/pgueguen/git/rctd-py

# Use the project venv
export PATH="/home/pgueguen/git/rctd-py/.venv/bin:$PATH"
echo "Python: $(which python)"

# Run comparison
python scripts/compare_r_vs_python_xenium.py 2>&1

echo "Completed at $(date)"
