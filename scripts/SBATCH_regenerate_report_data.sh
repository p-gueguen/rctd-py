#!/bin/bash
#SBATCH --job-name=rctd_regen
#SBATCH --output=/home/pgueguen/rctd_regen_%j.log
#SBATCH --error=/home/pgueguen/rctd_regen_%j.log
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1

echo "Regenerating JAX report data (post QP warm-start fix) at $(date)"

cd /home/pgueguen/git/rctd-py
export PATH="/home/pgueguen/git/rctd-py/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "=== JAX GPU ==="
python scripts/run_jax_cuda.py 2>&1

echo ""
echo "=== JAX CPU ==="
python scripts/run_jax_cpu.py 2>&1

# Update benchmark_times.csv with new JAX times
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
echo "=== Re-rendering report ==="
module load Dev/R/4.5.0
Rscript -e "rmarkdown::render('analysis_region3.qmd')"

echo "Completed at $(date)"
