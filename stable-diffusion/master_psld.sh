#!/bin/bash
# Master script: submit 100 PSLD posterior-sample jobs then a dependent
# aggregation job.
#
# Run from the stable-diffusion/ directory:
#   bash master_psld.sh
#
# Override any parameter via env var:
#   N_POSTERIOR=10 DDIM_STEPS=500 bash master_psld.sh

#!/bin/bash
set -euo pipefail

# --- Parameters (override via env vars) ---
N_POSTERIOR="${N_POSTERIOR:-100}"
DDIM_STEPS="${DDIM_STEPS:-1000}"
DDIM_ETA="${DDIM_ETA:-0.0}"
SCALE="${SCALE:-7.5}"
GAMMA="${GAMMA:-0.01}"
OMEGA="${OMEGA:-0.5}"
SEED="${SEED:-42}"
IMAGE_DIR="${IMAGE_DIR:-/lustre/fsn1/projects/rech/ynx/uxl64xr/Images_Posterior_Method_Test_512}"
TASK_CONFIG="${TASK_CONFIG:-configs/center_inpainting_config_psld.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/stable-diffusion/v1-inference.yaml}"
CKPT="${CKPT:-/lustre/fswork/projects/rech/ynx/uxl64xr/models/sd15/v1-5-pruned-emaonly.ckpt}"

mkdir -p logs

echo "=============================================="
echo " PSLD Posterior Sampling — Multi-Image Run"
echo "  image_dir:   $IMAGE_DIR"
echo "  n_posterior: $N_POSTERIOR  (max 20 concurrent)"
echo "  ddim_steps:  $DDIM_STEPS"
echo "  scale=$SCALE  gamma=$GAMMA  omega=$OMEGA  eta=$DDIM_ETA"
echo "=============================================="

mapfile -t IMAGES < <(find "$IMAGE_DIR" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | sort)

if [[ ${#IMAGES[@]} -eq 0 ]]; then
    echo "ERROR: No images found in $IMAGE_DIR"
    exit 1
fi

echo "Found ${#IMAGES[@]} images — submitting one array job each."
echo ""

ARRAY_LAST=$((N_POSTERIOR - 1))

for IMAGE_PATH in "${IMAGES[@]}"; do
    IMAGE_TAG="$(basename "${IMAGE_PATH%.*}")"
    SAVE_DIR="results/psld_g${GAMMA}_w${OMEGA}_eta${DDIM_ETA}_sc${SCALE}_N${N_POSTERIOR}_${IMAGE_TAG}"

    ARRAY_JOB_ID=$(sbatch --parsable \
        --array="0-${ARRAY_LAST}%20" \
        --job-name="psld_${IMAGE_TAG}" \
        --output="logs/${IMAGE_TAG}_%A_%a.out" \
        --error="logs/${IMAGE_TAG}_%A_%a.err" \
        --export=ALL,SAVE_DIR="$SAVE_DIR",IMAGE_PATH="$IMAGE_PATH",TASK_CONFIG="$TASK_CONFIG",MODEL_CONFIG="$MODEL_CONFIG",CKPT="$CKPT",DDIM_STEPS="$DDIM_STEPS",DDIM_ETA="$DDIM_ETA",SCALE="$SCALE",GAMMA="$GAMMA",OMEGA="$OMEGA",SEED="$SEED" \
        run_psld_array.sbatch)

    echo "  Submitted $IMAGE_TAG  →  array job $ARRAY_JOB_ID"
done

echo ""
echo "Monitor all:  squeue -u \$USER"
echo "Results:      results/psld_*/"