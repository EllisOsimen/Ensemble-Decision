#!/bin/bash
#SBATCH --job-name=consensus-mask
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.err

set -euo pipefail

# Create consensus segmentation masks from the three CURVAS prediction folders.
#
# Outputs:
#   1. Label masks:      OUTPUT_DIR
#   2. Confidence masks: CONFIDENCE_DIR
#
# Usage:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/consensus_agreement_mask.sbatch
#
# Optional arguments:
#   sbatch sbatch/consensus_agreement_mask.sbatch [PREDICTION_ROOT] [OUTPUT_DIR] [CONFIDENCE_DIR] [EXTRA_ARGS...]
#
# Defaults:
#   PREDICTION_ROOT -> results/CURVAS_INFERENCE
#   OUTPUT_DIR      -> results/CURVAS_INFERENCE/agreement_masks
#   CONFIDENCE_DIR  -> results/CURVAS_INFERENCE/agreement_confidence_masks
#
# EXTRA_ARGS are forwarded to consensus_agreement_mask.py, e.g.:
#   --case-name UKCHLL003
#   --overwrite
#   --no-clip-fallback

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
PREDICTION_ROOT="${1:-results/CURVAS_INFERENCE}"
OUTPUT_DIR="${2:-results/CURVAS_INFERENCE/agreement_masks}"
CONFIDENCE_DIR="${3:-results/CURVAS_INFERENCE/agreement_confidence_masks}"

if (( $# > 0 )); then
    shift
fi
if (( $# > 0 )); then
    shift
fi
if (( $# > 0 )); then
    shift
fi

EXTRA_ARGS=("$@")

if [[ "${PREDICTION_ROOT}" != /* ]]; then
    PREDICTION_ROOT="${PROJECT_DIR}/${PREDICTION_ROOT}"
fi
if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
fi
if [[ "${CONFIDENCE_DIR}" != /* ]]; then
    CONFIDENCE_DIR="${PROJECT_DIR}/${CONFIDENCE_DIR}"
fi

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_DIR}" "${CONFIDENCE_DIR}"

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

python -c "import nibabel, numpy, scipy, tqdm" || {
    echo "The suprem-h200 environment is missing required dependencies." >&2
    exit 1
}

CMD=(
    python -u ensemble_agreement/consensus_agreement_mask.py
    "${PREDICTION_ROOT}"
    --model-dir swinunetr_5050
    --model-dir clip_universal_unet
    --model-dir suprem_segresnet
    --clip-fallback-dir clip_universal_unet
    --output-dir "${OUTPUT_DIR}"
    --confidence-dir "${CONFIDENCE_DIR}"
)

CMD+=("${EXTRA_ARGS[@]}")

printf 'Prediction root: %s\n' "${PREDICTION_ROOT}"
printf 'Output label mask dir: %s\n' "${OUTPUT_DIR}"
printf 'Confidence mask dir: %s\n' "${CONFIDENCE_DIR}"
printf 'Extra args:'
printf ' %q' "${EXTRA_ARGS[@]}"
printf '\n'

"${CMD[@]}"

echo "Completed consensus agreement mask generation."
echo "Label masks: ${OUTPUT_DIR}"
echo "Confidence masks: ${CONFIDENCE_DIR}"
