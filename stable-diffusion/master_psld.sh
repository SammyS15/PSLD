#!/bin/bash
# Master script: submit 100 PSLD posterior-sample jobs then a dependent
# aggregation job.
#
# Run from the stable-diffusion/ directory:
#   bash master_psld.sh
#
# Override any parameter via env var:
#   N_POSTERIOR=10 DDIM_STEPS=500 bash master_psld.sh

set -euo pipefail

# --- Parameters (override via env vars) ---
N_POSTERIOR="${N_POSTERIOR:-100}"
DDIM_STEPS="${DDIM_STEPS:-1000}"
DDIM_ETA="${DDIM_ETA:-0.0}"
SCALE="${SCALE:-7.5}"
GAMMA="${GAMMA:-0.01}"
OMEGA="${OMEGA:-0.5}"
SEED="${SEED:-42}"
IMAGE_PATH="${IMAGE_PATH:-/home/sammys15/links/scratch/latent_model_test_images/apt.png}"
TASK_CONFIG="${TASK_CONFIG:-configs/center_inpainting_config_psld.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/stable-diffusion/v1-inference.yaml}"
CKPT="${CKPT:-/home/sammys15/links/scratch/Latent_Posterior_Sampling_Method_Comparsion/stable_diffusion_1_5_model/v1-5-pruned-emaonly.ckpt}"

IMAGE_TAG="$(basename "${IMAGE_PATH%.*}")"
SAVE_DIR="results/psld_g${GAMMA}_w${OMEGA}_eta${DDIM_ETA}_sc${SCALE}_N${N_POSTERIOR}_${IMAGE_TAG}"

mkdir -p logs

echo "=============================================="
echo " PSLD Posterior Sampling"
echo "  save_dir:    $SAVE_DIR"
echo "  n_posterior: $N_POSTERIOR  (max 20 concurrent)"
echo "  ddim_steps:  $DDIM_STEPS"
echo "  scale=$SCALE  gamma=$GAMMA  omega=$OMEGA  eta=$DDIM_ETA"
echo "  image:       $IMAGE_PATH"
echo "=============================================="

ARRAY_LAST=$((N_POSTERIOR - 1))

ARRAY_JOB_ID=$(sbatch --parsable \
    --array="0-${ARRAY_LAST}%20" \
    --export=ALL,SAVE_DIR="$SAVE_DIR",IMAGE_PATH="$IMAGE_PATH",TASK_CONFIG="$TASK_CONFIG",MODEL_CONFIG="$MODEL_CONFIG",CKPT="$CKPT",DDIM_STEPS="$DDIM_STEPS",DDIM_ETA="$DDIM_ETA",SCALE="$SCALE",GAMMA="$GAMMA",OMEGA="$OMEGA",SEED="$SEED" \
    run_psld_array.sbatch)

echo "Submitted array job:  $ARRAY_JOB_ID  (tasks 0-${ARRAY_LAST}, max 20 concurrent)"

AGG_JOB_ID=$(sbatch --parsable \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --export=ALL,SAVE_DIR="$SAVE_DIR",N_POSTERIOR="$N_POSTERIOR",IMAGE_TAG="$IMAGE_TAG" \
    aggregate_psld.sbatch)

echo "Submitted agg job:    $AGG_JOB_ID  (depends on $ARRAY_JOB_ID)"
echo ""
echo "Monitor with:"
echo "  squeue -j ${ARRAY_JOB_ID},${AGG_JOB_ID}"
echo ""
echo "Results will appear in: $SAVE_DIR/"
