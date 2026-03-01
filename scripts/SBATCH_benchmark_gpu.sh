#!/bin/bash
#SBATCH --job-name=rctd_benchmark
#SBATCH --output=/home/pgueguen/git/rctd-py/scripts/benchmark_gpu_%j.log
#SBATCH --error=/home/pgueguen/git/rctd-py/scripts/benchmark_gpu_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodelist=fgcz-c-056

cd /home/pgueguen/git/rctd-py

# Use the venv Python with JAX
export JAX_ENABLE_X64=1
.venv/bin/python scripts/benchmark_gpu.py

echo "Completed at $(date)"
