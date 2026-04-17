#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --account=def-lplevass
#SBATCH --job-name=psld_aggregate
#SBATCH --output=./logs/slurm_%j_aggregate.out
#SBATCH --error=./logs/slurm_%j_aggregate.err
#SBATCH --mail-user="sammy.sharief@umontreal.ca"
#SBATCH --mail-type=ALL

set -euo pipefail

SD_DIR="$SLURM_SUBMIT_DIR"
PSLD_ROOT="$(dirname "$SD_DIR")"
cd "$SD_DIR"

module purge
module load python/3.11
module load gcc
module load arrow/23.0.1

source "$PSLD_ROOT/psld_env/bin/activate"

export PYTHONUNBUFFERED=1

echo "Aggregating: $SAVE_DIR  (N=$N_POSTERIOR)"

python aggregate_psld.py \
    --save_dir    "$SAVE_DIR" \
    --n_posterior "$N_POSTERIOR" \
    --image_tag   "$IMAGE_TAG"
