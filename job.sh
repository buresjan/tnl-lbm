#!/bin/bash
#SBATCH --job-name=lbm-batch
#SBATCH --time=3-23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=lbm_batch.out
#SBATCH --error=lbm_batch.err

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python3 run_all_geometries.py --start 0 --end 180 --type1-bouzidi on
