#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --account=rrg-lplevass
#SBATCH --gres=gpu:1
#SBATCH --job-name=psld_sample
#SBATCH --output=./logs/slurm_%A_%a.out
#SBATCH --error=./logs/slurm_%A_%a.err
#SBATCH --mail-user="sammy.sharief@umontreal.ca"
#SBATCH --mail-type=FAIL,ARRAY_TASKS

set -euo pipefail

SD_DIR="$SLURM_SUBMIT_DIR"
PSLD_ROOT="$(dirname "$SD_DIR")"
cd "$SD_DIR"

module purge
module load python/3.11
module load gcc
module load arrow/23.0.1
module load gcc opencv
module load ffmpeg

source "$PSLD_ROOT/psld_env/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$SD_DIR:$PSLD_ROOT/diffusion-posterior-sampling/util:$HOME/links/scratch/resample/taming-transformers:${PYTHONPATH:-}"
export HF_HOME="/lustre10/scratch/sammys15/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Save dir:   $SAVE_DIR"
echo "ddim_steps=$DDIM_STEPS  scale=$SCALE  gamma=$GAMMA  omega=$OMEGA"

python scripts/sample_one.py \
    --sample_idx  "$SLURM_ARRAY_TASK_ID" \
    --save_dir    "$SAVE_DIR" \
    --file_path   "$IMAGE_PATH" \
    --task_config "$TASK_CONFIG" \
    --config      "$MODEL_CONFIG" \
    --ckpt        "$CKPT" \
    --ddim_steps  "$DDIM_STEPS" \
    --ddim_eta    "$DDIM_ETA" \
    --scale       "$SCALE" \
    --gamma       "$GAMMA" \
    --omega       "$OMEGA" \
    --seed        "$SEED"

echo "Task $SLURM_ARRAY_TASK_ID done: $(date)"
