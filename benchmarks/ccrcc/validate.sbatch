#!/bin/bash
#SBATCH --job-name=ar_ccrcc_validate
#SBATCH --output=/misc/GT/analysis/pgueguen/rctd-py/autoresearch_ccrcc/wt/benchmarks/ccrcc/out/logs/validate_%j.log
#SBATCH --error=/misc/GT/analysis/pgueguen/rctd-py/autoresearch_ccrcc/wt/benchmarks/ccrcc/out/logs/validate_%j.err
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodelist=fgcz-r-023
# Final validation on the held-out TEST split (east half, 100k cells), never touched by the loop:
# baseline pipeline, final pipeline, final pipeline with protein off, and 3 permutation nulls
# (each fit marker shuffled independently before the final pipeline sees it).
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
cd /misc/GT/analysis/pgueguen/rctd-py/autoresearch_ccrcc/wt || exit 1
O=benchmarks/ccrcc/out
echo "node=$(hostname) date=$(date) sha=$(git rev-parse --short HEAD)"
for P in baseline final final_rna; do
  echo "=== test split: $P"; .venv/bin/python benchmarks/ccrcc/eval.py --split test --pipeline $O/pipeline_$P.py --tag test_$P || echo "FAILED $P"
done
for s in 0 1 2; do
  echo "=== test split: final, shuffled fit markers seed $s"
  .venv/bin/python benchmarks/ccrcc/eval.py --split test --pipeline $O/pipeline_final.py --tag test_final_shuffle$s --shuffle-protein-seed $s || echo "FAILED shuffle$s"
done
echo "=== tune split: baseline RNA-only and final RNA-only (protein contribution on the tune split)"
.venv/bin/python benchmarks/ccrcc/eval.py --split tune --pipeline $O/pipeline_final_rna.py --tag tune_final_rna || echo "FAILED tune_final_rna"
echo "EXIT=$? date=$(date)"
