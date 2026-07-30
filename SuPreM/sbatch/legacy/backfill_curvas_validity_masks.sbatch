#!/bin/bash
#SBATCH --job-name=curvas-validity
#SBATCH --partition=Teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-89%4
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%A_%a.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%A_%a.err

set -euo pipefail

# Add validity maps to an existing complete set of CURVAS hard-label and
# confidence outputs. This uses the inference scripts' preprocessing/inversion
# path but exits before model loading, so it needs CPU and memory but no GPU.
#
# Usage:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/legacy/backfill_curvas_validity_masks.sbatch
#
# Optional positional arguments:
#   sbatch sbatch/legacy/backfill_curvas_validity_masks.sbatch \
#     [OUTPUT_DIR] [TRAIN_ROOT] [VAL_ROOT] [TEST_ROOT]

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
OUTPUT_DIR="${1:-results/all_curvas_inference_with_confidence}"
TRAIN_ROOT="${2:-/home/s2347484/Seg/training_set/training_set}"
VAL_ROOT="${3:-/home/s2347484/Seg/validation_set}"
TEST_ROOT="${4:-/home/s2347484/Seg/testing_set}"
CHECKPOINT_DIR="${PROJECT_DIR}/pretrained_weights"

if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
fi

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_DIR}"

for INPUT_ROOT in "${TRAIN_ROOT}" "${VAL_ROOT}" "${TEST_ROOT}"; do
    if [[ ! -d "${INPUT_ROOT}" ]]; then
        echo "Input root not found: ${INPUT_ROOT}" >&2
        exit 1
    fi
done

declare -a SPLIT_NAMES
declare -a CASE_NAMES
declare -a INPUT_IMAGES

add_cases() {
    local SPLIT_NAME="$1"
    local INPUT_ROOT="$2"
    local CASE_DIR CASE_NAME INPUT_IMAGE
    local -a CASE_DIRS

    mapfile -d '' CASE_DIRS < <(
        find "${INPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
    )

    for CASE_DIR in "${CASE_DIRS[@]}"; do
        CASE_NAME="$(basename "${CASE_DIR}")"
        if [[ -f "${CASE_DIR}/ct.nii.gz" ]]; then
            INPUT_IMAGE="${CASE_DIR}/ct.nii.gz"
        elif [[ -f "${CASE_DIR}/image.nii.gz" ]]; then
            INPUT_IMAGE="${CASE_DIR}/image.nii.gz"
        else
            echo "Skipping ${SPLIT_NAME}/${CASE_NAME}: no CT image found" >&2
            continue
        fi
        SPLIT_NAMES+=("${SPLIT_NAME}")
        CASE_NAMES+=("${CASE_NAME}")
        INPUT_IMAGES+=("${INPUT_IMAGE}")
    done
}

add_cases "training" "${TRAIN_ROOT}"
add_cases "validation" "${VAL_ROOT}"
add_cases "testing" "${TEST_ROOT}"

CASE_COUNT="${#CASE_NAMES[@]}"
if (( CASE_COUNT != 90 )); then
    echo "Expected 90 CURVAS cases but discovered ${CASE_COUNT}." >&2
    exit 1
fi
if (( SLURM_ARRAY_TASK_ID >= CASE_COUNT )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} exceeds ${CASE_COUNT} cases." >&2
    exit 0
fi

SPLIT_NAME="${SPLIT_NAMES[SLURM_ARRAY_TASK_ID]}"
CASE_NAME="${CASE_NAMES[SLURM_ARRAY_TASK_ID]}"
INPUT_IMAGE="${INPUT_IMAGES[SLURM_ARRAY_TASK_ID]}"
SPLIT_OUTPUT_DIR="${OUTPUT_DIR}/${SPLIT_NAME}"

CLIP_OUTPUT="${SPLIT_OUTPUT_DIR}/clip_universal_unet/${CASE_NAME}.nii.gz"
CLIP_CONFIDENCE="${SPLIT_OUTPUT_DIR}/clip_universal_unet_confidence/${CASE_NAME}.nii.gz"
CLIP_VALIDITY="${SPLIT_OUTPUT_DIR}/clip_universal_unet_validity/${CASE_NAME}.nii.gz"
SEGRESNET_OUTPUT="${SPLIT_OUTPUT_DIR}/suprem_segresnet/${CASE_NAME}.nii.gz"
SEGRESNET_CONFIDENCE="${SPLIT_OUTPUT_DIR}/suprem_segresnet_confidence/${CASE_NAME}.nii.gz"
SEGRESNET_VALIDITY="${SPLIT_OUTPUT_DIR}/suprem_segresnet_validity/${CASE_NAME}.nii.gz"
SWIN5050_OUTPUT="${SPLIT_OUTPUT_DIR}/swinunetr_5050/${CASE_NAME}.nii.gz"
SWIN5050_CONFIDENCE="${SPLIT_OUTPUT_DIR}/swinunetr_5050_confidence/${CASE_NAME}.nii.gz"
SWIN5050_VALIDITY="${SPLIT_OUTPUT_DIR}/swinunetr_5050_validity/${CASE_NAME}.nii.gz"

for REQUIRED in \
    "${CLIP_OUTPUT}" "${CLIP_CONFIDENCE}" \
    "${SEGRESNET_OUTPUT}" "${SEGRESNET_CONFIDENCE}" \
    "${SWIN5050_OUTPUT}" "${SWIN5050_CONFIDENCE}"
do
    if [[ ! -f "${REQUIRED}" ]]; then
        echo "Cannot backfill ${SPLIT_NAME}/${CASE_NAME}; missing ${REQUIRED}" >&2
        echo "Run the full GPU inference array for incomplete cases." >&2
        exit 1
    fi
done

if [[ -f "${CLIP_VALIDITY}" && -f "${SEGRESNET_VALIDITY}" && -f "${SWIN5050_VALIDITY}" ]]; then
    echo "All validity maps already exist for ${SPLIT_NAME}/${CASE_NAME}; skipping"
    exit 0
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

echo "Array job/task: ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Split/case: ${SPLIT_NAME}/${CASE_NAME}"
echo "Input image: ${INPUT_IMAGE}"
echo "Mode: CPU-only validity backfill; neural networks will not run"

python -u inference/infer_clip_universal_unet.py \
    --image "${INPUT_IMAGE}" \
    --output "${CLIP_OUTPUT}" \
    --confidence-output "${CLIP_CONFIDENCE}" \
    --validity-output "${CLIP_VALIDITY}" \
    --checkpoint "${CHECKPOINT_DIR}/supervised_clip_driven_universal_unet_2100.pth" \
    --device cpu

python -u inference/infer_suprem_segresnet.py \
    --image "${INPUT_IMAGE}" \
    --output "${SEGRESNET_OUTPUT}" \
    --confidence-output "${SEGRESNET_CONFIDENCE}" \
    --validity-output "${SEGRESNET_VALIDITY}" \
    --checkpoint "${CHECKPOINT_DIR}/supervised_suprem_segresnet_2100.pth" \
    --device cpu

python -u inference/infer_swinunetr_5050.py \
    --image "${INPUT_IMAGE}" \
    --output "${SWIN5050_OUTPUT}" \
    --confidence-output "${SWIN5050_CONFIDENCE}" \
    --validity-output "${SWIN5050_VALIDITY}" \
    --checkpoint "${CHECKPOINT_DIR}/self_supervised_nv_swin_unetr_5050.pt" \
    --device cpu

echo "Completed validity backfill for ${SPLIT_NAME}/${CASE_NAME}"
