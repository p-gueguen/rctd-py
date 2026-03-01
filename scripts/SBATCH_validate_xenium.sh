#!/bin/bash
#SBATCH --job-name=rctd_xenium_validate
#SBATCH --output=/home/pgueguen/git/rctd-py/xenium_validate_%j.log
#SBATCH --error=/home/pgueguen/git/rctd-py/xenium_validate_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=employee

cd /home/pgueguen/git/rctd-py

# Activate virtual environment
source .venv/bin/activate

echo "Starting Xenium validation at $(date)"
echo "Python: $(which python)"
echo "JAX version: $(python -c 'import jax; print(jax.__version__)')"

# Run full mode validation
echo ""
echo "=== Full Mode ==="
python scripts/validate_xenium.py --mode full --batch-size 5000

# Run doublet mode validation
echo ""
echo "=== Doublet Mode ==="
python scripts/validate_xenium.py --mode doublet --batch-size 5000

echo ""
echo "Completed at $(date)"
