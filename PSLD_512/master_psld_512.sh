#!/bin/bash
# Master orchestrator for PSLD at 512×512 with SD 1.5.
#
# Submits N_POSTERIOR independent array tasks (each draws one posterior sample),
# then a dependent aggregation job that computes mean/std/residuals and saves
# all diagnostic PNGs.
#
# Run from LATENT_METHODS_COMP/PSLD_512/:
#   bash master_psld_512.sh
#
# Override any parameter via environment variable:
#   IMAGE_PATH=/path/to/image.png TASK=inpainting bash master_psld_512.sh
#
# Supported TASK values:
#   inpainting           → center square mask
#   box_inpainting       → random box mask
#   super_resolution     → SR ×4
#   gaussian_deblur      → Gaussian blur
#   motion_deblur        → motion blur

set -euo pipefail

# ── Parameters (override via env vars) ──────────────────────────────────────
N_POSTERIOR="${N_POSTERIOR:-100}"
DDIM_STEPS="${DDIM_STEPS:-1000}"
DDIM_ETA="${DDIM_ETA:-0.0}"
SCALE="${SCALE:-7.5}"
GAMMA="${GAMMA:-0.01}"
OMEGA="${OMEGA:-0.5}"
SEED="${SEED:-42}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"     # explicit — do not change unless you know what you're doing

IMAGE_PATH="${IMAGE_PATH:-/lustre/fsn1/projects/rech/ynx/uxl64xr/latent_model_test_images/apt.png}"
TASK="${TASK:-inpainting}"          # maps to configs below

# Model paths (Jean Zay defaults)
MODEL_CONFIG="${MODEL_CONFIG:-configs/stable-diffusion/v1-inference.yaml}"
CKPT="${CKPT:-/lustre/fswork/projects/rech/ynx/uxl64xr/models/sd15/v1-5-pruned-emaonly.ckpt}"

# ── Resolve task config ──────────────────────────────────────────────────────
PSLD_ROOT="$(cd "$(dirname "$0")/" && pwd)"
DPS_CONFIGS="/lustre/fsn1/projects/rech/ynx/uxl64xr/PSLD/diffusion-posterior-sampling/configs"

case "$TASK" in
    inpainting)         TASK_CONFIG="configs/center_inpainting_config_psld.yaml" ;;
    box_inpainting)     TASK_CONFIG="configs/box_inpainting_config_psld.yaml" ;;
    super_resolution)   TASK_CONFIG="configs/super_resolution_config_psld.yaml" ;;
    gaussian_deblur)    TASK_CONFIG="configs/gaussian_deblur_config_psld.yaml" ;;
    motion_deblur)      TASK_CONFIG="configs/motion_deblur_config_psld.yaml" ;;
    *)
        echo "ERROR: Unknown TASK='$TASK'. Valid: inpainting, box_inpainting, super_resolution, gaussian_deblur, motion_deblur"
        exit 1
        ;;
esac
TASK_CONFIG="${TASK_CONFIG_OVERRIDE:-$TASK_CONFIG}"   # allow full override too

# ── Output directory ─────────────────────────────────────────────────────────
IMAGE_TAG="$(basename "${IMAGE_PATH%.*}")"
SAVE_DIR="results/${TASK}/psld512_g${GAMMA}_w${OMEGA}_eta${DDIM_ETA}_sc${SCALE}_N${N_POSTERIOR}_${IMAGE_TAG}"

mkdir -p logs

echo "============================================================"
echo " PSLD-512  Posterior Sampling  (SD 1.5 @ ${IMAGE_SIZE}×${IMAGE_SIZE})"
echo "  task:        $TASK"
echo "  task_config: $TASK_CONFIG"
echo "  save_dir:    $SAVE_DIR"
echo "  n_posterior: $N_POSTERIOR  (max 20 concurrent)"
echo "  ddim_steps:  $DDIM_STEPS   eta=$DDIM_ETA"
echo "  scale=$SCALE   gamma=$GAMMA   omega=$OMEGA"
echo "  image:       $IMAGE_PATH"
echo "  image_size:  ${IMAGE_SIZE}×${IMAGE_SIZE}"
echo "============================================================"

ARRAY_LAST=$((N_POSTERIOR - 1))
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ARRAY_JOB_ID=$(sbatch --parsable \
    --array="0-${ARRAY_LAST}%20" \
    --export=ALL,\
SAVE_DIR="$SAVE_DIR",\
IMAGE_PATH="$IMAGE_PATH",\
TASK_CONFIG="$TASK_CONFIG",\
MODEL_CONFIG="$MODEL_CONFIG",\
CKPT="$CKPT",\
DDIM_STEPS="$DDIM_STEPS",\
DDIM_ETA="$DDIM_ETA",\
SCALE="$SCALE",\
GAMMA="$GAMMA",\
OMEGA="$OMEGA",\
SEED="$SEED",\
IMAGE_SIZE="$IMAGE_SIZE",\
PSLD_ROOT="$PSLD_ROOT" \
    "$SCRIPT_DIR/run_psld_array_512.sbatch")

echo "Submitted array job : $ARRAY_JOB_ID  (tasks 0–${ARRAY_LAST}, ≤20 concurrent)"

# AGG_JOB_ID=$(sbatch --parsable \
#     --dependency="afterany:${ARRAY_JOB_ID}" \
#     --export=ALL,\
# SAVE_DIR="$SAVE_DIR",\
# N_POSTERIOR="$N_POSTERIOR",\
# IMAGE_TAG="$IMAGE_TAG",\
# PSLD_ROOT="$PSLD_ROOT" \
#     "$SCRIPT_DIR/aggregate_psld_512.sbatch")

# echo "Submitted agg job   : $AGG_JOB_ID  (depends on $ARRAY_JOB_ID)"
echo ""
echo "Monitor:  squeue -j ${ARRAY_JOB_ID}"
echo "Results:  $SAVE_DIR/"
