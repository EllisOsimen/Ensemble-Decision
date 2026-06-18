#!/bin/bash

set -euo pipefail

# Submit all three complete WORD evaluations in sequence. Each backbone gets a
# separate SLURM allocation and starts only if the preceding job succeeds.
#
# Usage:
#   bash submit_all_backbones.sh

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
SBATCH_SCRIPT="${PROJECT_DIR}/evaluate_word.sbatch"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs

UNET_JOB_ID="$(
    sbatch \
        --parsable \
        --job-name=suprem-word-unet \
        "${SBATCH_SCRIPT}" unet
)"
UNET_JOB_ID="${UNET_JOB_ID%%;*}"

SWIN_JOB_ID="$(
    sbatch \
        --parsable \
        --dependency="afterok:${UNET_JOB_ID}" \
        --job-name=suprem-word-swin \
        "${SBATCH_SCRIPT}" swinunetr
)"
SWIN_JOB_ID="${SWIN_JOB_ID%%;*}"

SEGRESNET_JOB_ID="$(
    sbatch \
        --parsable \
        --dependency="afterok:${SWIN_JOB_ID}" \
        --job-name=suprem-word-segresnet \
        "${SBATCH_SCRIPT}" segresnet
)"
SEGRESNET_JOB_ID="${SEGRESNET_JOB_ID%%;*}"

echo "Submitted sequential WORD evaluations:"
echo "  U-Net:      ${UNET_JOB_ID}"
echo "  Swin UNETR: ${SWIN_JOB_ID} (after U-Net succeeds)"
echo "  SegResNet:  ${SEGRESNET_JOB_ID} (after Swin UNETR succeeds)"
echo
echo "Monitor with:"
echo "  squeue -j ${UNET_JOB_ID},${SWIN_JOB_ID},${SEGRESNET_JOB_ID}"
