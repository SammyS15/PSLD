#!/bin/bash
#SBATCH --job-name=psld_100samples
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=0:40:00
#SBATCH --gpus-per-node=1
#SBATCH --account=ynx@v100
#SBATCH -C v100-32g
#SBATCH --output=./logs/psld_100samples_slurm_%j.out
#SBATCH --error=./logs/psld_100samples_slurm_%j.err
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
export TORCH_HOME="$SCRATCH/.cache/torch"

# --- Best params: scale=7.5, eta=0.0, gamma=0.01, omega=0.5 ---
# ddim_steps=200 (vs 1000) so 100 samples fits in 24h (~20h estimate)
echo "Job started:  $(date)"
echo "Node:         $SLURMD_NODENAME"

python scripts/inverse.py \
    --file_path=/lustre/fsn1/projects/rech/ynx/uxl64xr/latent_model_test_images/apt.png \
    --task_config=configs/center_inpainting_config_psld.yaml \
    --inpainting=1 \
    --general_inverse=0 \
    --ddim_steps=200 \
    --ckpt=/lustre/fswork/projects/rech/ynx/uxl64xr/models/sd15/v1-5-pruned-emaonly.ckpt \
    --config=configs/stable-diffusion/v1-inference.yaml \
    --outdir=outputs/psld-100samples-apt \
    --gamma=0.01 \
    --omega=0.5 \
    --seed=42 \
    --scale=7.5 \
    --ddim_eta=0.0 \
    --n_posterior_samples=100 \
    --skip_low_res

echo "Job finished: $(date)"
