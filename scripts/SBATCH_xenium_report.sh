#!/bin/bash
#SBATCH --job-name=rctd_report
#SBATCH --output=/home/pgueguen/rctd_report_%j.log
#SBATCH --error=/home/pgueguen/rctd_report_%j.log
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=16
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1

echo "Generating Xenium report data + re-rendering at $(date)"

cd /home/pgueguen/git/rctd-py
export PATH="/home/pgueguen/git/rctd-py/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1

# Regenerate vignette JAX data (constrain=False fix affects doublet mode)
echo "=== Vignette JAX GPU ==="
python scripts/run_jax_cuda.py 2>&1
echo ""
echo "=== Vignette JAX CPU ==="
python scripts/run_jax_cpu.py 2>&1

python -c "
import pandas as pd
times = pd.read_csv('data/benchmark_times.csv', index_col=0, header=None).squeeze('columns')
with open('data/jax_gpu_time.txt') as f: times['jax_gpu'] = float(f.read().strip())
with open('data/jax_cpu_time.txt') as f: times['jax_cpu'] = float(f.read().strip())
times.to_csv('data/benchmark_times.csv')
print('Updated benchmark_times.csv:')
print(times)
"

echo ""
echo "=== Xenium Region 3 data ==="
cp /srv/gstore/projects/p36005/Xenium_Data/data_20240724__105930__o35844/output-XETG00404__0022381__Region_3__20240724__110035/cell_feature_matrix.h5 /scratch/cell_feature_matrix.h5
python scripts/generate_xenium_report_data.py --spatial /scratch/cell_feature_matrix.h5 2>&1
rm -f /scratch/cell_feature_matrix.h5

# Step 3: Render report
echo ""
echo "=== Step 3: Rendering HTML ==="
module load Dev/R/4.5.0
Rscript -e "rmarkdown::render('analysis_region3.qmd')"

echo "Completed at $(date)"
