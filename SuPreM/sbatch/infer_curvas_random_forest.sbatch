#!/bin/bash
#SBATCH --job-name=curvas-rf
#SBATCH --partition=Teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-64%6
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

# Apply the fitted config_002 random forest to all 65 CURVAS test cases.
# Run after train_final_random_forest_config_002.sbatch completes.
#
# Optional arguments:
#   sbatch sbatch/infer_curvas_random_forest.sbatch \
#     [TEST_PREDICTION_ROOT] [TEST_CASES_ROOT] [MODEL_INPUT] [OUTPUT_ROOT]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${PROJECT_DIR}/.." && pwd)"
CURVAS_ROOT="${PROJECT_DIR}/results/all_datasets_curvas_inference_with_confidence"
PREDICTION_ROOT="${1:-${CURVAS_ROOT}/testing}"
CASES_ROOT="${2:-${REPO_DIR}/testing_set}"
MODEL_INPUT="${3:-${CURVAS_ROOT}/random_forest_config_002_final/rf_stacking_config_002.joblib}"
OUTPUT_ROOT="${4:-${CURVAS_ROOT}/random_forest_config_002_final/testing_predictions}"
OVERWRITE="${OVERWRITE:-0}"
EXPECTED_CASES=65

if [[ ! -f "${MODEL_INPUT}" || ! -f "${MODEL_INPUT%.joblib}.metadata.json" ]]; then
    echo "Random-forest model or metadata is missing: ${MODEL_INPUT}" >&2
    exit 1
fi

mapfile -d '' CASE_DIRS < <(
    find "${CASES_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)
if (( ${#CASE_DIRS[@]} != EXPECTED_CASES )); then
    echo "Expected ${EXPECTED_CASES} CURVAS test cases; found ${#CASE_DIRS[@]}." >&2
    exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array job.}"
if (( TASK_ID < 0 || TASK_ID >= EXPECTED_CASES )); then
    echo "Array task ${TASK_ID} is outside the expected range 0-64." >&2
    exit 1
fi

CASE_DIR="${CASE_DIRS[TASK_ID]}"
CASE_NAME="$(basename "${CASE_DIR}")"
AGREEMENT_OUTPUT="${OUTPUT_ROOT}/${CASE_NAME}/agreement_mask.nii.gz"
CONFIDENCE_OUTPUT="${OUTPUT_ROOT}/${CASE_NAME}/confidence_mask.nii.gz"

if [[ "${OVERWRITE}" == "0" && -f "${AGREEMENT_OUTPUT}" && -f "${CONFIDENCE_OUTPUT}" ]]; then
    echo "Random-forest outputs already exist for ${CASE_NAME}; skipping"
    exit 0
fi
if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
    echo "OVERWRITE must be 0 or 1; got ${OVERWRITE}" >&2
    exit 1
fi

INPUT_DIRS=(
    clip_universal_unet suprem_segresnet swinunetr_5050
    clip_universal_unet_confidence suprem_segresnet_confidence swinunetr_5050_confidence
    clip_universal_unet_validity suprem_segresnet_validity swinunetr_5050_validity
)
for INPUT_DIR in "${INPUT_DIRS[@]}"; do
    if [[ ! -f "${PREDICTION_ROOT}/${INPUT_DIR}/${CASE_NAME}.nii.gz" ]]; then
        echo "Missing input: ${PREDICTION_ROOT}/${INPUT_DIR}/${CASE_NAME}.nii.gz" >&2
        exit 1
    fi
done

source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-suprem-h200}"
cd "${PROJECT_DIR}"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" || -e "${AGREEMENT_OUTPUT}" || -e "${CONFIDENCE_OUTPUT}" ]]; then
    OVERWRITE_ARGS+=(--overwrite)
fi

python -u ensemble_agreement/random_forest_stacking_consensus.py \
    --inference-only \
    --model-input "${MODEL_INPUT}" \
    --val-prediction-root "${PREDICTION_ROOT}" \
    --val-cases-root "${CASE_DIR}" \
    --output-dir "${OUTPUT_ROOT}" \
    --model-dir clip_universal_unet \
    --model-id clip_unet \
    --label-space suprem \
    --confidence-dir clip_universal_unet_confidence \
    --validity-dir clip_universal_unet_validity \
    --model-dir suprem_segresnet \
    --model-id segresnet \
    --label-space suprem \
    --confidence-dir suprem_segresnet_confidence \
    --validity-dir suprem_segresnet_validity \
    --model-dir swinunetr_5050 \
    --model-id swin5050 \
    --label-space btcv \
    --confidence-dir swinunetr_5050_confidence \
    --validity-dir swinunetr_5050_validity \
    --target-annotation image.nii.gz \
    --predict-chunk-size 1000000 \
    "${OVERWRITE_ARGS[@]}"

echo "Completed random-forest inference for ${CASE_NAME}."
