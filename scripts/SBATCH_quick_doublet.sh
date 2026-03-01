#!/bin/bash
#SBATCH --job-name=rctd_quick_dbl
#SBATCH --output=/home/pgueguen/rctd_quick_dbl_%j.log
#SBATCH --error=/home/pgueguen/rctd_quick_dbl_%j.log
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodelist=fgcz-r-023

echo "Quick doublet test at $(date)"
echo "Node: $(hostname)"

cd /home/pgueguen/git/rctd-py
export PATH="/home/pgueguen/git/rctd-py/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1

python -c "import jax; print('JAX devices:', jax.devices())"

cp /srv/gstore/projects/p36005/Xenium_Data/data_20240724__105930__o35844/output-XETG00404__0022381__Region_3__20240724__110035/cell_feature_matrix.h5 /scratch/cell_feature_matrix.h5
python scripts/quick_doublet_test.py /scratch/cell_feature_matrix.h5 2>&1
rm -f /scratch/cell_feature_matrix.h5

echo "Completed at $(date)"
