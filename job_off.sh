#!/bin/bash
#SBATCH --job-name=off-batch
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=off_batch.out
#SBATCH --error=off_batch.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

python3 run_all_geometries.py --start 1 --end 180 \
    --runs-root lbm_runs_b_off --output lbm_runs_b_off/batch_results.csv \
    --type1-bouzidi off