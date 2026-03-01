#!/bin/bash
#SBATCH --job-name=rctd_s72
#SBATCH --output=/home/pgueguen/xenium_compare_sigma72_%j.log
#SBATCH --error=/home/pgueguen/xenium_compare_sigma72_%j.log
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1

echo "Starting R vs Python Xenium comparison (GPU, sigma=72 override) at $(date)"
echo "Test: using R's exact sigma=72 to isolate sigma impact on agreement"

cd /home/pgueguen/git/rctd-py

export PATH="/home/pgueguen/git/rctd-py/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1

# Copy spatial data to scratch for I/O performance
echo "Staging data to /scratch..."
cp /srv/gstore/projects/p36005/Xenium_Data/data_20240724__105930__o35844/output-XETG00404__0022381__Region_3__20240724__110035/cell_feature_matrix.h5 /scratch/cell_feature_matrix.h5

python scripts/compare_r_vs_python_xenium.py --spatial /scratch/cell_feature_matrix.h5 --sigma-override 72 2>&1

# Cleanup
rm -f /scratch/cell_feature_matrix.h5
echo "Completed at $(date)"
