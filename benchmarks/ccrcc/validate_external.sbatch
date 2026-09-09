#!/bin/bash
#SBATCH --job-name=ar_ccrcc_validate_ext
#SBATCH --output=/misc/GT/analysis/pgueguen/rctd-py/autoresearch_ccrcc/wt/benchmarks/ccrcc/out/logs/validate_ext_%j.log
#SBATCH --error=/misc/GT/analysis/pgueguen/rctd-py/autoresearch_ccrcc/wt/benchmarks/ccrcc/out/logs/validate_ext_%j.err
#SBATCH --time=02:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodelist=fgcz-r-023
# External transfer section (different patient block, 405-gene panel): baseline, final, final protein off,
# two shuffled-marker nulls; then the region-annotation checks.
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
cd /misc/GT/analysis/pgueguen/rctd-py/autoresearch_ccrcc/wt || exit 1
O=benchmarks/ccrcc/out
echo "node=$(hostname) date=$(date) sha=$(git rev-parse --short HEAD)"
for P in baseline final final_rna; do
  echo "=== external: $P"; .venv/bin/python benchmarks/ccrcc/eval.py --split external --pipeline $O/pipeline_$P.py --tag external_$P || echo "FAILED $P"
done
for s in 0 1; do
  echo "=== external: final, shuffled fit markers seed $s"
  .venv/bin/python benchmarks/ccrcc/eval.py --split external --pipeline $O/pipeline_final.py --tag external_final_shuffle$s --shuffle-protein-seed $s || echo "FAILED shuffle$s"
done
.venv/bin/python benchmarks/ccrcc/external_metrics.py
echo "EXIT=$? date=$(date)"
