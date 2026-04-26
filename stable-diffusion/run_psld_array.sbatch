#!/bin/bash
#SBATCH --job-name=psld_array
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=0:40:00
#SBATCH --gpus-per-node=1
#SBATCH --account=ynx@v100
#SBATCH -C v100-32g
#SBATCH --output=./logs/psld_array_slurm_%j.out
#SBATCH --error=./logs/psld_array_slurm_%j.err
#SBATCH --mail-user="sammy.sharief@umontreal.ca"
#SBATCH --mail-type=ALL

cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc
module load opencv
module load ffmpeg

export PATH="/lustre/fsn1/projects/rech/ynx/uxl64xr/conda_envs/mamba/bin:$PATH"
eval "$(mamba shell hook --shell bash)"
mamba activate ldm

SD_DIR="$SLURM_SUBMIT_DIR"           # .../PSLD/stable-diffusion/
PSLD_ROOT="$(dirname "$SD_DIR")"     # .../PSLD/
cd "$SD_DIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PSLD_ROOT/diffusion-posterior-sampling/util:$HOME/links/scratch/resample/taming-transformers:${PYTHONPATH:-}"
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