#!/bin/bash
#SBATCH --job-name=curvas-infer-conf
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-89%4
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%A_%a.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%A_%a.err

set -euo pipefail

# Regenerate all three base-model predictions, assigned-label confidence maps,
# and binary inference-validity masks for the 20 training, 5 validation, and 65
# testing CURVAS cases.
#
# Usage:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/infer_all_curvas_three_models_with_confidence.sbatch
#
# Run selected array tasks only (command-line --array overrides the directive):
#   sbatch --array=7 sbatch/infer_all_curvas_three_models_with_confidence.sbatch
#
# Rerun CLIP and SegResNet with the current maximum-score overlap rule while
# reusing the unchanged Swin outputs:
#   sbatch --export=ALL,RERUN_SIGMOID=1 \
#     sbatch/infer_all_curvas_three_models_with_confidence.sbatch
#
# Optional positional arguments:
#   sbatch sbatch/infer_all_curvas_three_models_with_confidence.sbatch \
#     [OUTPUT_DIR] [TRAIN_ROOT] [VAL_ROOT] [TEST_ROOT]

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
OUTPUT_DIR="${1:-results/all_curvas_inference_with_confidence}"
TRAIN_ROOT="${2:-/home/s2347484/Seg/training_set/training_set}"
VAL_ROOT="${3:-/home/s2347484/Seg/validation_set}"
TEST_ROOT="${4:-/home/s2347484/Seg/testing_set}"
CHECKPOINT_DIR="${PROJECT_DIR}/pretrained_weights"
RERUN_SIGMOID="${RERUN_SIGMOID:-0}"

if [[ "${RERUN_SIGMOID}" != "0" && "${RERUN_SIGMOID}" != "1" ]]; then
    echo "RERUN_SIGMOID must be 0 or 1; got ${RERUN_SIGMOID}" >&2
    exit 1
fi

CLIP_CHECKPOINT="${CHECKPOINT_DIR}/supervised_clip_driven_universal_unet_2100.pth"
SEGRESNET_CHECKPOINT="${CHECKPOINT_DIR}/supervised_suprem_segresnet_2100.pth"
SWIN5050_CHECKPOINT="${CHECKPOINT_DIR}/self_supervised_nv_swin_unetr_5050.pt"

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

for CHECKPOINT in \
    "${CLIP_CHECKPOINT}" \
    "${SEGRESNET_CHECKPOINT}" \
    "${SWIN5050_CHECKPOINT}"
do
    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "Missing checkpoint: ${CHECKPOINT}" >&2
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
    echo "Update the Slurm array range if the dataset layout changed." >&2
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

OUTPUTS=(
    "${CLIP_OUTPUT}"
    "${CLIP_CONFIDENCE}"
    "${CLIP_VALIDITY}"
    "${SEGRESNET_OUTPUT}"
    "${SEGRESNET_CONFIDENCE}"
    "${SEGRESNET_VALIDITY}"
    "${SWIN5050_OUTPUT}"
    "${SWIN5050_CONFIDENCE}"
    "${SWIN5050_VALIDITY}"
)
ALL_COMPLETE=true
for OUTPUT in "${OUTPUTS[@]}"; do
    if [[ ! -f "${OUTPUT}" ]]; then
        ALL_COMPLETE=false
        break
    fi
done
if [[ "${ALL_COMPLETE}" == true && "${RERUN_SIGMOID}" != "1" ]]; then
    echo "All label, confidence, and validity outputs exist for ${SPLIT_NAME}/${CASE_NAME}; skipping"
    exit 0
fi

SIGMOID_OVERWRITE_ARGS=()
if [[ "${RERUN_SIGMOID}" == "1" ]]; then
    SIGMOID_OVERWRITE_ARGS+=(--overwrite)
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

python -c "import nibabel, monai, numpy, torch" || {
    echo "The suprem-h200 environment is missing required dependencies." >&2
    exit 1
}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Array job/task: ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Split/case: ${SPLIT_NAME}/${CASE_NAME}"
echo "Input image: ${INPUT_IMAGE}"
echo "Output root: ${SPLIT_OUTPUT_DIR}"
echo "Confidence storage: scaled uint8 (decoded range [0,1])"
echo "Validity storage: uint8 (1=evaluated, 0=crop exterior)"
echo "Rerun sigmoid models: ${RERUN_SIGMOID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi

python -u inference/infer_clip_universal_unet.py \
    --image "${INPUT_IMAGE}" \
    --output "${CLIP_OUTPUT}" \
    --confidence-output "${CLIP_CONFIDENCE}" \
    --validity-output "${CLIP_VALIDITY}" \
    --confidence-storage uint8 \
    --checkpoint "${CLIP_CHECKPOINT}" \
    --device cuda \
    --sw-batch-size 1 \
    "${SIGMOID_OVERWRITE_ARGS[@]}"

python -u inference/infer_suprem_segresnet.py \
    --image "${INPUT_IMAGE}" \
    --output "${SEGRESNET_OUTPUT}" \
    --confidence-output "${SEGRESNET_CONFIDENCE}" \
    --validity-output "${SEGRESNET_VALIDITY}" \
    --confidence-storage uint8 \
    --checkpoint "${SEGRESNET_CHECKPOINT}" \
    --device cuda \
    --sw-batch-size 1 \
    "${SIGMOID_OVERWRITE_ARGS[@]}"

python -u inference/infer_swinunetr_5050.py \
    --image "${INPUT_IMAGE}" \
    --output "${SWIN5050_OUTPUT}" \
    --confidence-output "${SWIN5050_CONFIDENCE}" \
    --validity-output "${SWIN5050_VALIDITY}" \
    --confidence-storage uint8 \
    --checkpoint "${SWIN5050_CHECKPOINT}" \
    --device cuda \
    --sw-batch-size 1

echo "Completed labels, assigned-label confidence, and validity for ${SPLIT_NAME}/${CASE_NAME}"
