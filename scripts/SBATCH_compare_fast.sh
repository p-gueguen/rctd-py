#!/bin/bash
#SBATCH --job-name=rctd_fast
#SBATCH --output=/home/pgueguen/xenium_fast_%j.log
#SBATCH --error=/home/pgueguen/xenium_fast_%j.log
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1

echo "Fast validation (1000 pixels) at $(date)"
echo "Fixes: nUMI all genes, norm_ref normalized props, full mode constrain=False, doublet weight comparison, QP warm start fix"

cd /home/pgueguen/git/rctd-py
export PATH="/home/pgueguen/git/rctd-py/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1

cp /srv/gstore/projects/p36005/Xenium_Data/data_20240724__105930__o35844/output-XETG00404__0022381__Region_3__20240724__110035/cell_feature_matrix.h5 /scratch/cell_feature_matrix.h5

echo "=== RUN 1: Auto sigma ==="
python scripts/compare_r_vs_python_xenium.py --spatial /scratch/cell_feature_matrix.h5 --n-pixels 1000 2>&1

echo ""
echo "=== RUN 2: Sigma=72 (R's value) ==="
python scripts/compare_r_vs_python_xenium.py --spatial /scratch/cell_feature_matrix.h5 --n-pixels 1000 --sigma-override 72 2>&1

rm -f /scratch/cell_feature_matrix.h5
echo "Completed at $(date)"
