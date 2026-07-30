#!/bin/bash
#SBATCH --job-name=external-rf
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-49%6
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

# Apply the final config_002 random-forest stacker to the 50 external cases.
# This is CPU-only. Each array task loads the three hard-label predictions,
# assigned-label confidence maps, and common inference-validity maps for one
# case, then writes a CURVAS-label consensus mask and forest confidence mask.
#
# Submit with defaults:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/infer_external_dataset_random_forest.sbatch
#
# Optional positional arguments:
#   sbatch sbatch/infer_external_dataset_random_forest.sbatch \
#     [PREDICTION_ROOT] [CASES_ROOT] [MODEL_INPUT] [OUTPUT_ROOT]
#
# Rerun and replace both outputs for every selected case:
#   sbatch --export=ALL,OVERWRITE=1 \
#     sbatch/infer_external_dataset_random_forest.sbatch

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${PROJECT_DIR}/.." && pwd)"
PREDICTION_ROOT="${1:-${PROJECT_DIR}/results/external_dataset_inference_with_confidence}"
CASES_ROOT="${2:-${REPO_DIR}/external_dataset}"
MODEL_INPUT="${3:-${PROJECT_DIR}/results/all_datasets_curvas_inference_with_confidence/random_forest_config_002_final/rf_stacking_config_002.joblib}"
OUTPUT_ROOT="${4:-${PROJECT_DIR}/results/external_dataset_random_forest_config_002}"
OVERWRITE="${OVERWRITE:-0}"
EXPECTED_CASES=50

if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
    echo "OVERWRITE must be 0 or 1; got ${OVERWRITE}" >&2
    exit 1
fi
if [[ ! -d "${PREDICTION_ROOT}" ]]; then
    echo "External prediction root not found: ${PREDICTION_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${CASES_ROOT}" ]]; then
    echo "External cases root not found: ${CASES_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_INPUT}" ]]; then
    echo "Trained random-forest model not found: ${MODEL_INPUT}" >&2
    exit 1
fi

MODEL_METADATA="${MODEL_INPUT%.*}.metadata.json"
if [[ ! -f "${MODEL_METADATA}" ]]; then
    echo "Random-forest metadata not found: ${MODEL_METADATA}" >&2
    exit 1
fi

mapfile -d '' CASE_DIRS < <(
    find "${CASES_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)
if (( ${#CASE_DIRS[@]} != EXPECTED_CASES )); then
    echo "Expected ${EXPECTED_CASES} external cases; found ${#CASE_DIRS[@]}." >&2
    exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array job.}"
if (( TASK_ID < 0 || TASK_ID >= EXPECTED_CASES )); then
    echo "Array task ${TASK_ID} is outside the expected range 0-49." >&2
    exit 1
fi

MODEL_DIRS=(
    clip_universal_unet
    suprem_segresnet
    swinunetr_5050
)
CONFIDENCE_DIRS=(
    clip_universal_unet_confidence
    suprem_segresnet_confidence
    swinunetr_5050_confidence
)
VALIDITY_DIRS=(
    clip_universal_unet_validity
    suprem_segresnet_validity
    swinunetr_5050_validity
)
INPUT_DIRS=(
    "${MODEL_DIRS[@]}"
    "${CONFIDENCE_DIRS[@]}"
    "${VALIDITY_DIRS[@]}"
)

# Strict dataset-wide preflight prevents the array from processing an
# incomplete or case-misaligned collection of base-model outputs.
for INPUT_DIR in "${INPUT_DIRS[@]}"; do
    FULL_INPUT_DIR="${PREDICTION_ROOT}/${INPUT_DIR}"
    if [[ ! -d "${FULL_INPUT_DIR}" ]]; then
        echo "Required input directory not found: ${FULL_INPUT_DIR}" >&2
        exit 1
    fi
    INPUT_COUNT=$(find "${FULL_INPUT_DIR}" -maxdepth 1 -type f \
        -name '*.nii.gz' | wc -l)
    if (( INPUT_COUNT != EXPECTED_CASES )); then
        echo "Expected ${EXPECTED_CASES} files in ${FULL_INPUT_DIR}; found ${INPUT_COUNT}." >&2
        exit 1
    fi
done

for CASE_DIR in "${CASE_DIRS[@]}"; do
    CASE_NAME="$(basename "${CASE_DIR}")"
    if [[ ! -f "${CASE_DIR}/ct.nii.gz" ]]; then
        echo "Missing external CT: ${CASE_DIR}/ct.nii.gz" >&2
        exit 1
    fi
    for INPUT_DIR in "${INPUT_DIRS[@]}"; do
        INPUT="${PREDICTION_ROOT}/${INPUT_DIR}/${CASE_NAME}.nii.gz"
        if [[ ! -f "${INPUT}" ]]; then
            echo "Missing input for ${CASE_NAME}: ${INPUT}" >&2
            exit 1
        fi
    done
done

CASE_DIR="${CASE_DIRS[TASK_ID]}"
CASE_NAME="$(basename "${CASE_DIR}")"
AGREEMENT_OUTPUT="${OUTPUT_ROOT}/${CASE_NAME}/agreement_mask.nii.gz"
CONFIDENCE_OUTPUT="${OUTPUT_ROOT}/${CASE_NAME}/confidence_mask.nii.gz"

if [[ "${OVERWRITE}" == "0" ]]; then
    if [[ -f "${AGREEMENT_OUTPUT}" && -f "${CONFIDENCE_OUTPUT}" ]]; then
        echo "Both random-forest outputs already exist for ${CASE_NAME}; skipping"
        exit 0
    fi
    if [[ -e "${AGREEMENT_OUTPUT}" || -e "${CONFIDENCE_OUTPUT}" ]]; then
        echo "Only one output exists for ${CASE_NAME}; refusing to leave a partial pair." >&2
        echo "Resubmit with OVERWRITE=1 to regenerate both outputs." >&2
        exit 1
    fi
fi

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_ROOT}"

source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-suprem-h200}"
cd "${PROJECT_DIR}"

python -c "import joblib, nibabel, numpy, scipy, sklearn, tqdm" || {
    echo "The suprem-h200 environment is missing required random-forest dependencies." >&2
    exit 1
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-4}"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
    OVERWRITE_ARGS+=(--overwrite)
fi

echo "Array job/task: ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Case: ${CASE_NAME}"
echo "External predictions: ${PREDICTION_ROOT}"
echo "Random-forest model: ${MODEL_INPUT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Overwrite: ${OVERWRITE}"

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
    --target-annotation ct.nii.gz \
    --predict-chunk-size 1000000 \
    "${OVERWRITE_ARGS[@]}"

echo "Completed random-forest ensemble inference for ${CASE_NAME}."
echo "Consensus: ${AGREEMENT_OUTPUT}"
echo "Confidence: ${CONFIDENCE_OUTPUT}"
