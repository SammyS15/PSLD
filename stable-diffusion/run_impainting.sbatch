#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=rrg-lplevass
#SBATCH --gres=gpu:1
#SBATCH --job-name=psld_100samples
#SBATCH --output=./logs/slurm_%j.out
#SBATCH --error=./logs/slurm_%j.err
#SBATCH --mail-user="sammy.sharief@umontreal.ca"
#SBATCH --mail-type=ALL

set -euo pipefail

# Go to /home/sammys15/links/scratch/PSLD
cd "$HOME/links/scratch/PSLD"
PSLD_ROOT="$HOME/links/scratch/PSLD"
module purge
module load python/3.11
module load gcc
module load arrow/23.0.1
module load gcc opencv
module load ffmpeg
source "$PSLD_ROOT/psld_env/bin/activate"

# Go to /home/sammys15/links/scratch/Latent_Posterior_Sampling_Method_Comparsion/PSLD
UPDATED_PSLD_ROOT="$HOME/links/scratch/Latent_Posterior_Sampling_Method_Comparsion/PSLD"
cd "$UPDATED_PSLD_ROOT"
SD_DIR="$UPDATED_PSLD_ROOT/stable-diffusion"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$SD_DIR:$UPDATED_PSLD_ROOT/diffusion-posterior-sampling/util:$HOME/links/scratch/resample/taming-transformers:${PYTHONPATH:-}"
export HF_HOME="/lustre10/scratch/sammys15/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# inverse.py must be run from inside stable-diffusion/
cd "$SD_DIR"

# --- Best params: scale=7.5, eta=0.0, gamma=0.01, omega=0.5 ---
# ddim_steps=200 (vs 1000) so 100 samples fits in 24h (~20h estimate)
echo "Job started:  $(date)"
echo "Node:         $SLURMD_NODENAME"

python scripts/inverse.py \
    --file_path=/home/sammys15/links/scratch/latent_model_test_images/apt.png \
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
