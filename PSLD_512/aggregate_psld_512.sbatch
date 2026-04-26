#!/bin/bash
#SBATCH --job-name=psld512_aggregate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=0:50:00
#SBATCH --gpus-per-node=1
#SBATCH --account=ynx@v100
#SBATCH -C v100-32g
#SBATCH --output=./logs/psld512_aggregate_slurm_%j.out
#SBATCH --error=./logs/psld512_aggregate_slurm_%j.err
#SBATCH --mail-user="sammy.sharief@umontreal.ca"
#SBATCH --mail-type=ALL

# Aggregation step: loads all sample_*.pt tensors, computes posterior
# mean/std, saves diagnostic PNGs.  No GPU required.
# Submitted automatically by master_psld_512.sh after the array completes.

# set -euo pipefail

SD_DIR='/lustre/fsn1/projects/rech/ynx/uxl64xr/PSLD/stable-diffusion'

cd "$SD_DIR"

module purge
module load gcc
module load opencv
module load ffmpeg

export PATH="/lustre/fsn1/projects/rech/ynx/uxl64xr/conda_envs/mamba/bin:$PATH"
eval "$(mamba shell hook --shell bash)"
mamba activate ldm

export PYTHONUNBUFFERED=1

echo "Aggregating: $SAVE_DIR  (N=$N_POSTERIOR)"
echo "Start: $(date)"

python aggregate_psld.py \
    --save_dir    "$SAVE_DIR" \
    --n_posterior "$N_POSTERIOR" \
    --image_tag   "$IMAGE_TAG"

echo "Done: $(date)"
echo "Results in: $SAVE_DIR/"
