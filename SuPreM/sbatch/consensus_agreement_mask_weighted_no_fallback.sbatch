#!/bin/bash
#SBATCH --job-name=weighted-consensus
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.err

set -euo pipefail

# Create weighted foreground-threshold consensus masks from the three CURVAS
# prediction folders. CLIP fallback is disabled, so unresolved weighted voxels
# remain background/default with confidence 5.
#
# Outputs:
#   1. Label masks:      OUTPUT_DIR
#   2. Confidence masks: CONFIDENCE_DIR
#
# Usage:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/consensus_agreement_mask_weighted_no_fallback.sbatch
#
# Optional arguments:
#   sbatch sbatch/consensus_agreement_mask_weighted_no_fallback.sbatch [PREDICTION_ROOT] [OUTPUT_DIR] [CONFIDENCE_DIR] [EXTRA_ARGS...]
#
# Defaults:
#   PREDICTION_ROOT -> results/CURVAS_INFERENCE
#   OUTPUT_DIR      -> results/CURVAS_INFERENCE/agreement_masks_weighted
#   CONFIDENCE_DIR  -> results/CURVAS_INFERENCE/agreement_confidence_masks_weighted
#
# EXTRA_ARGS are forwarded to consensus_agreement_mask.py, e.g.:
#   --case-name UKCHLL003
#   --overwrite
#   --weak-thresholds 0.50 0.70 0.85
#   --strong-thresholds 0.65 0.85 1.20

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
PREDICTION_ROOT="${1:-results/CURVAS_INFERENCE}"
OUTPUT_DIR="${2:-results/CURVAS_INFERENCE/agreement_masks_weighted}"
CONFIDENCE_DIR="${3:-results/CURVAS_INFERENCE/agreement_confidence_masks_weighted}"

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
    --consensus-mode weighted
    --no-clip-fallback
    --model-dir swinunetr_5050
    --model-id swin5050
    --label-space btcv
    --model-dir clip_universal_unet
    --model-id clip_unet
    --label-space suprem
    --model-dir suprem_segresnet
    --model-id segresnet
    --label-space suprem
    --output-dir "${OUTPUT_DIR}"
    --confidence-dir "${CONFIDENCE_DIR}"
)

CMD+=("${EXTRA_ARGS[@]}")

printf 'Prediction root: %s\n' "${PREDICTION_ROOT}"
printf 'Output label mask dir: %s\n' "${OUTPUT_DIR}"
printf 'Confidence mask dir: %s\n' "${CONFIDENCE_DIR}"
printf 'Consensus mode: weighted\n'
printf 'CLIP fallback: disabled\n'
printf 'Extra args:'
printf ' %q' "${EXTRA_ARGS[@]}"
printf '\n'

"${CMD[@]}"

echo "Completed weighted consensus agreement mask generation."
echo "Label masks: ${OUTPUT_DIR}"
echo "Confidence masks: ${CONFIDENCE_DIR}"
