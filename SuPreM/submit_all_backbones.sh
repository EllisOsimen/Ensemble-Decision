#!/bin/bash

set -euo pipefail

# Submit the Swin UNETR and SegResNet WORD evaluations in sequence. Each
# backbone gets a separate SLURM allocation, and SegResNet starts only if
# Swin UNETR succeeds.
#
# Usage:
#   bash submit_all_backbones.sh

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
SBATCH_SCRIPT="${PROJECT_DIR}/evaluate_word.sbatch"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs

SWIN_JOB_ID="$(
    sbatch \
        --parsable \
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
echo "  Swin UNETR: ${SWIN_JOB_ID}"
echo "  SegResNet:  ${SEGRESNET_JOB_ID} (after Swin UNETR succeeds)"
echo
echo "Monitor with:"
echo "  squeue -j ${SWIN_JOB_ID},${SEGRESNET_JOB_ID}"
