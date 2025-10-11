#!/bin/bash
#SBATCH --job-name=on-batch
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=on_batch.out
#SBATCH --error=on_batch.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

python3 run_all_geometries.py --start 1 --end 180 \
    --runs-root lbm_runs_b_on --output lbm_runs_b_on/batch_results.csv \
    --type1-bouzidi on