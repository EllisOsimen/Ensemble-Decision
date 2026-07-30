#!/bin/bash
#SBATCH --job-name=curvas-fuse
#SBATCH --partition=Teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-64%6
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

# Generate unweighted, weighted, and STAPLE masks for the 65 CURVAS test cases.
# Run after infer_all_curvas_three_models_with_confidence.sbatch completes.
#
# Run from SuPreM/:
#   sbatch sbatch/curvas/fuse_curvas_three_methods.sbatch
#
# Optional arguments:
#   sbatch sbatch/curvas/fuse_curvas_three_methods.sbatch \
#     [TEST_PREDICTION_ROOT] [OUTPUT_ROOT] [TEST_CASES_ROOT]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_DIR="$(cd -- "${PROJECT_DIR}/.." && pwd)"
PREDICTION_ROOT="${1:-${PROJECT_DIR}/results/all_datasets_curvas_inference_with_confidence/testing}"
OUTPUT_ROOT="${2:-${PROJECT_DIR}/results/all_datasets_curvas_inference_with_confidence/consensus_testing}"
TEST_CASES_ROOT="${3:-${REPO_DIR}/testing_set}"
OVERWRITE="${OVERWRITE:-0}"
EXPECTED_CASES=65

if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
    echo "OVERWRITE must be 0 or 1; got ${OVERWRITE}" >&2
    exit 1
fi

mapfile -d '' CASE_DIRS < <(
    find "${TEST_CASES_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
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

CASE_NAME="$(basename "${CASE_DIRS[TASK_ID]}")"
CASE_FILE="${CASE_NAME}.nii.gz"
CLIP_DIR="${PREDICTION_ROOT}/clip_universal_unet"
SEGRESNET_DIR="${PREDICTION_ROOT}/suprem_segresnet"
SWIN_DIR="${PREDICTION_ROOT}/swinunetr_5050"
CLIP_VALIDITY_DIR="${PREDICTION_ROOT}/clip_universal_unet_validity"
SEGRESNET_VALIDITY_DIR="${PREDICTION_ROOT}/suprem_segresnet_validity"
SWIN_VALIDITY_DIR="${PREDICTION_ROOT}/swinunetr_5050_validity"

for INPUT_DIR in \
    "${CLIP_DIR}" "${SEGRESNET_DIR}" "${SWIN_DIR}" \
    "${CLIP_VALIDITY_DIR}" "${SEGRESNET_VALIDITY_DIR}" "${SWIN_VALIDITY_DIR}"
do
    if [[ ! -f "${INPUT_DIR}/${CASE_FILE}" ]]; then
        echo "Missing input for ${CASE_NAME}: ${INPUT_DIR}/${CASE_FILE}" >&2
        exit 1
    fi
done

source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-suprem-h200}"
cd "${PROJECT_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

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
    shift
    local CONSENSUS_DIR="${OUTPUT_ROOT}/${METHOD}/consensus_masks"
    local CONFIDENCE_DIR="${OUTPUT_ROOT}/${METHOD}/confidence_masks"
    local CONSENSUS_FILE="${CONSENSUS_DIR}/${CASE_FILE}"
    local CONFIDENCE_FILE="${CONFIDENCE_DIR}/${CASE_FILE}"
    local -a OVERWRITE_ARGS=()

    if [[ "${OVERWRITE}" == "0" && -f "${CONSENSUS_FILE}" && -f "${CONFIDENCE_FILE}" ]]; then
        echo "${METHOD}: outputs already exist for ${CASE_NAME}; skipping"
        return
    fi
    if [[ "${OVERWRITE}" == "1" || -e "${CONSENSUS_FILE}" || -e "${CONFIDENCE_FILE}" ]]; then
        OVERWRITE_ARGS+=(--overwrite)
    fi

    python -u ensemble_agreement/consensus_agreement_mask.py \
        "${COMMON_ARGS[@]}" \
        --output-dir "${CONSENSUS_DIR}" \
        --confidence-dir "${CONFIDENCE_DIR}" \
        "${OVERWRITE_ARGS[@]}" \
        "$@"
}

run_method unweighted \
    --consensus-mode legacy \
    --clip-fallback-dir "${CLIP_DIR}"

run_method weighted \
    --consensus-mode weighted \
    --model-id clip_unet \
    --model-id segresnet \
    --model-id swin5050 \
    --weak-thresholds 0.6510967049572083 0.742517090301494 0.751405718196963 \
    --strong-thresholds 0.8316425584923725 1.2033290040589268 1.2177773831809173 \
    --no-clip-fallback

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

echo "Completed unweighted, weighted, and STAPLE fusion for ${CASE_NAME}."
