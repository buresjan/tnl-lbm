#!/bin/bash
#SBATCH --job-name=lbm-batch
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=lbm_batch.out
#SBATCH --error=lbm_batch.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

python3 run_all_geometries.py --start 0 --end 179 --type1-bouzidi auto
