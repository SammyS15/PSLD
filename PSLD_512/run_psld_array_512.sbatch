#!/bin/bash
#SBATCH --job-name=psld512_sample_array
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=0:50:00
#SBATCH --gpus-per-node=1
#SBATCH --account=ynx@v100
#SBATCH -C v100-32g
#SBATCH --output=./logs/psld512_sample_array_slurm_%j.out
#SBATCH --error=./logs/psld512_sample_array_slurm_%j.err
#SBATCH --mail-user="sammy.sharief@umontreal.ca"
#SBATCH --mail-type=ALL

# Each task draws one independent PSLD posterior sample at 512×512 with SD 1.5.
# Submitted as an array job by master_psld_512.sh — do not run directly.

# set -euo pipefail

SD_DIR='/lustre/fsn1/projects/rech/ynx/uxl64xr/PSLD/stable-diffusion'
DPS_DIR='/lustre/fsn1/projects/rech/ynx/uxl64xr/PSLD/diffusion-posterior-sampling'

cd "$SD_DIR"

module purge
module load gcc
module load opencv
module load ffmpeg

export PATH="/lustre/fsn1/projects/rech/ynx/uxl64xr/conda_envs/mamba/bin:$PATH"
eval "$(mamba shell hook --shell bash)"
mamba activate ldm

export PYTHONUNBUFFERED=1
export PYTHONPATH="$SD_DIR:$DPS_DIR/util:${PYTHONPATH:-}"
export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_HOME="$SCRATCH/.cache/torch"

echo "Array task : $SLURM_ARRAY_TASK_ID"
echo "Save dir   : $SAVE_DIR"
echo "Image size : ${IMAGE_SIZE}×${IMAGE_SIZE}"
echo "ddim_steps=$DDIM_STEPS  scale=$SCALE  gamma=$GAMMA  omega=$OMEGA  eta=$DDIM_ETA"
echo "Node: $SLURMD_NODENAME  |  Start: $(date)"
echo "Python Path: $PYTHONPATH"

TASK_SEED=$((SEED + SLURM_ARRAY_TASK_ID))

echo "SEED: $TASK_SEED"


python scripts/sample_one.py \
    --sample_idx  "$SLURM_ARRAY_TASK_ID" \
    --save_dir    "$SAVE_DIR" \
    --file_path   "$IMAGE_PATH" \
    --task_config "$TASK_CONFIG" \
    --dps_path    "$DPS_DIR/" \
    --config      "$MODEL_CONFIG" \
    --ckpt        "$CKPT" \
    --ddim_steps  "$DDIM_STEPS" \
    --ddim_eta    "$DDIM_ETA" \
    --scale       "$SCALE" \
    --gamma       "$GAMMA" \
    --omega       "$OMEGA" \
    --seed        "$TASK_SEED" \
    --H           "$IMAGE_SIZE"

echo "Task $SLURM_ARRAY_TASK_ID done: $(date)"
