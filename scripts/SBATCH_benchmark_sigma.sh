#!/bin/bash
#SBATCH --job-name=rctd_sigma_bench
#SBATCH --output=/home/pgueguen/rctd_sigma_bench_%j.log
#SBATCH --error=/home/pgueguen/rctd_sigma_bench_%j.log
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodelist=fgcz-r-023

echo "Sigma optimization benchmark at $(date)"
echo "Node: $(hostname)"

cd /home/pgueguen/git/rctd-py
export PATH="/home/pgueguen/git/rctd-py/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1

# Verify GPU is available
python -c "import jax; print('JAX devices:', jax.devices()); assert any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in jax.devices()), 'No GPU!'" 2>&1

echo ""
echo "=== Xenium Region 3 (58k pixels, sigma estimation + doublet) ==="
# Stage data to local scratch to avoid NFS bottleneck
cp /srv/gstore/projects/p36005/Xenium_Data/data_20240724__105930__o35844/output-XETG00404__0022381__Region_3__20240724__110035/cell_feature_matrix.h5 /scratch/cell_feature_matrix.h5
python scripts/generate_xenium_report_data.py --spatial /scratch/cell_feature_matrix.h5 2>&1
rm -f /scratch/cell_feature_matrix.h5

echo ""
echo "=== Rendering HTML ==="
module load Dev/R/4.5.0
Rscript -e "rmarkdown::render('analysis_region3.qmd')" 2>&1

echo "Completed at $(date)"
