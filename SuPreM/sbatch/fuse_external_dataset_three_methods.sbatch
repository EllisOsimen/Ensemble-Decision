#!/bin/bash
#SBATCH --job-name=external-fuse
#SBATCH --partition=Teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-49%6
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

# Fuse the three external-dataset base predictions using the same unweighted,
# training-derived weighted, and STAPLE methods used for the CURVAS analysis.
# This is CPU-only: one array task processes one external case through all
# three fusion methods.
#
# Submit with defaults:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/fuse_external_dataset_three_methods.sbatch
#
# Optional positional arguments:
#   sbatch sbatch/fuse_external_dataset_three_methods.sbatch \
#     [PREDICTION_ROOT] [OUTPUT_ROOT] [EXTERNAL_DATASET_ROOT]
#
# Rerun and replace all three methods for every selected case:
#   sbatch --export=ALL,OVERWRITE=1 \
#     sbatch/fuse_external_dataset_three_methods.sbatch

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${PROJECT_DIR}/.." && pwd)"
PREDICTION_ROOT="${1:-${PROJECT_DIR}/results/external_dataset_inference_with_confidence}"
OUTPUT_ROOT="${2:-${PROJECT_DIR}/results/external_dataset_consensus}"
EXTERNAL_DATASET_ROOT="${3:-${REPO_DIR}/external_dataset}"
OVERWRITE="${OVERWRITE:-0}"

if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
    echo "OVERWRITE must be 0 or 1; got ${OVERWRITE}" >&2
    exit 1
fi

if [[ ! -d "${PREDICTION_ROOT}" ]]; then
    echo "Prediction root not found: ${PREDICTION_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${EXTERNAL_DATASET_ROOT}" ]]; then
    echo "External dataset root not found: ${EXTERNAL_DATASET_ROOT}" >&2
    exit 1
fi

mapfile -d '' CASE_DIRS < <(
    find "${EXTERNAL_DATASET_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)

CASE_COUNT="${#CASE_DIRS[@]}"
if (( CASE_COUNT != 50 )); then
    echo "Expected 50 external cases but discovered ${CASE_COUNT}." >&2
    echo "Update the Slurm array range if the dataset layout changed." >&2
    exit 1
fi
if (( SLURM_ARRAY_TASK_ID >= CASE_COUNT )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} exceeds ${CASE_COUNT} cases." >&2
    exit 0
fi

CASE_NAME="$(basename "${CASE_DIRS[SLURM_ARRAY_TASK_ID]}")"
CASE_FILE="${CASE_NAME}.nii.gz"

CLIP_DIR="${PREDICTION_ROOT}/clip_universal_unet"
SEGRESNET_DIR="${PREDICTION_ROOT}/suprem_segresnet"
SWIN_DIR="${PREDICTION_ROOT}/swinunetr_5050"
CLIP_VALIDITY_DIR="${PREDICTION_ROOT}/clip_universal_unet_validity"
SEGRESNET_VALIDITY_DIR="${PREDICTION_ROOT}/suprem_segresnet_validity"
SWIN_VALIDITY_DIR="${PREDICTION_ROOT}/swinunetr_5050_validity"

REQUIRED_INPUTS=(
    "${CLIP_DIR}/${CASE_FILE}"
    "${SEGRESNET_DIR}/${CASE_FILE}"
    "${SWIN_DIR}/${CASE_FILE}"
    "${CLIP_VALIDITY_DIR}/${CASE_FILE}"
    "${SEGRESNET_VALIDITY_DIR}/${CASE_FILE}"
    "${SWIN_VALIDITY_DIR}/${CASE_FILE}"
)
for INPUT in "${REQUIRED_INPUTS[@]}"; do
    if [[ ! -f "${INPUT}" ]]; then
        echo "Missing required input for ${CASE_NAME}: ${INPUT}" >&2
        exit 1
    fi
done

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_ROOT}"

source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-suprem-h200}"
cd "${PROJECT_DIR}"

python -c "import nibabel, numpy, scipy, tqdm" || {
    echo "The suprem-h200 environment is missing required fusion dependencies." >&2
    exit 1
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Array job/task: ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Case: ${CASE_NAME}"
echo "Prediction root: ${PREDICTION_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Overwrite: ${OVERWRITE}"

COMMON_ARGS=(
    "${PREDICTION_ROOT}"
    --case-name "${CASE_NAME}"
    --strict
    --model-dir "${CLIP_DIR}"
    --label-space suprem
    --model-dir "${SEGRESNET_DIR}"
    --label-space suprem
    --model-dir "${SWIN_DIR}"
    --label-space btcv
)

run_method() {
    local METHOD="$1"
    local CONSENSUS_DIR="${OUTPUT_ROOT}/${METHOD}/consensus_masks"
    local CONFIDENCE_DIR="${OUTPUT_ROOT}/${METHOD}/confidence_masks"
    shift

    local CONSENSUS_FILE="${CONSENSUS_DIR}/${CASE_FILE}"
    local CONFIDENCE_FILE="${CONFIDENCE_DIR}/${CASE_FILE}"
    local -a OVERWRITE_ARGS=()

    if [[ "${OVERWRITE}" == "0" \
        && -f "${CONSENSUS_FILE}" \
        && -f "${CONFIDENCE_FILE}" ]]; then
        echo "${METHOD}: both outputs already exist for ${CASE_NAME}; skipping"
        return
    fi

    # If a prior attempt produced only one of the two files, force this method
    # to regenerate the pair so the outputs cannot remain partially complete.
    if [[ "${OVERWRITE}" == "1" \
        || -f "${CONSENSUS_FILE}" \
        || -f "${CONFIDENCE_FILE}" ]]; then
        OVERWRITE_ARGS+=(--overwrite)
    fi

    echo "Running ${METHOD} fusion for ${CASE_NAME}"
    python -u ensemble_agreement/consensus_agreement_mask.py \
        "${COMMON_ARGS[@]}" \
        --output-dir "${CONSENSUS_DIR}" \
        --confidence-dir "${CONFIDENCE_DIR}" \
        "${OVERWRITE_ARGS[@]}" \
        "$@"
}

# Reported unweighted method: equal voting/local agreement with CLIP fallback.
run_method unweighted \
    --consensus-mode legacy \
    --clip-fallback-dir "${CLIP_DIR}"

# Reported weighted method: fixed organ/model weights and thresholds learned
# from the 20 CURVAS training cases. No CLIP fallback is used.
run_method weighted \
    --consensus-mode weighted \
    --model-id clip_unet \
    --model-id segresnet \
    --model-id swin5050 \
    --weak-thresholds \
        0.6510967049572083 \
        0.742517090301494 \
        0.751405718196963 \
    --strong-thresholds \
        0.8316425584923725 \
        1.2033290040589268 \
        1.2177773831809173 \
    --no-clip-fallback

# Reported per-organ STAPLE method. The model validity masks exclude restored
# crop padding from the EM estimation and force that exterior to background.
run_method staple \
    --consensus-mode staple \
    --validity-dir "${CLIP_VALIDITY_DIR}" \
    --validity-dir "${SEGRESNET_VALIDITY_DIR}" \
    --validity-dir "${SWIN_VALIDITY_DIR}" \
    --staple-prob-threshold 0.5 \
    --staple-margin-threshold 0.1 \
    --staple-max-iter 50 \
    --staple-tol 1e-5 \
    --staple-eps 1e-6 \
    --no-clip-fallback

echo "Completed unweighted, weighted, and STAPLE fusion for ${CASE_NAME}"
