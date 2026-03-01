#!/bin/bash
#SBATCH --job-name=rctd_sweeps
#SBATCH --output=/home/pgueguen/rctd_sweeps_%j.log
#SBATCH --error=/home/pgueguen/rctd_sweeps_%j.log
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1

echo "Testing QP sweep counts at $(date)"

cd /home/pgueguen/git/rctd-py
export PATH="/home/pgueguen/git/rctd-py/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1

cp /srv/gstore/projects/p36005/Xenium_Data/data_20240724__105930__o35844/output-XETG00404__0022381__Region_3__20240724__110035/cell_feature_matrix.h5 /scratch/cell_feature_matrix.h5

python scripts/test_sweep_counts.py --spatial /scratch/cell_feature_matrix.h5 2>&1

rm -f /scratch/cell_feature_matrix.h5
echo "Completed at $(date)"
