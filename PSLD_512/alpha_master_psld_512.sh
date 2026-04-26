#!/bin/bash
set -euo pipefail

IMAGE_DIR="/lustre/fsn1/projects/rech/ynx/uxl64xr/Images_Posterior_Method_Test_512"

TASK="${TASK:-inpainting}"
N_POSTERIOR="${N_POSTERIOR:-100}"

echo "Running PSLD batch over all images in: $IMAGE_DIR"
echo "Task: $TASK"
echo "N_POSTERIOR: $N_POSTERIOR"
echo "===================================================="

shopt -s nullglob

for img in "$IMAGE_DIR"/*.png; do
    echo ""
    echo "===================================================="
    echo "Submitting job for image: $(basename "$img")"
    echo "===================================================="

    IMAGE_PATH="$img" \
    TASK="$TASK" \
    N_POSTERIOR="$N_POSTERIOR" \
    bash master_psld_512.sh

    # optional: small delay to avoid SLURM bursts
    sleep 1
done

echo "All jobs submitted."