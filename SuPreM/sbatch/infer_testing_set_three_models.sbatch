#!/bin/bash
#SBATCH --job-name=eval-infer3
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-127%4
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%A_%a.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%A_%a.err

set -euo pipefail

# Run 3-model inference for one case/image per SLURM array task.
#
# Usage:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/infer_testing_set_three_models.sbatch
#
# Optional arguments:
#   sbatch sbatch/infer_testing_set_three_models.sbatch [OUTPUT_DIR] [INPUT_ROOT]
#
# Defaults:
#   OUTPUT_DIR -> results/eval_inference
#   INPUT_ROOT -> /home/s2347484/Seg/testing_set
#
# INPUT_ROOT can be either:
#   1. a testing_set-style directory with one folder per case, each containing
#      ct.nii.gz or image.nii.gz
#   2. a flat directory of .nii.gz files, such as /home/s2347484/Seg/Dataset/ImagesTr

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
OUTPUT_DIR="${1:-results/eval_inference}"
INPUT_ROOT="${2:-/home/s2347484/Seg/testing_set}"
CHECKPOINT_DIR="${PROJECT_DIR}/pretrained_weights"

CLIP_CHECKPOINT="${CHECKPOINT_DIR}/supervised_clip_driven_universal_unet_2100.pth"
SEGRESNET_CHECKPOINT="${CHECKPOINT_DIR}/supervised_suprem_segresnet_2100.pth"
SWIN5050_CHECKPOINT="${CHECKPOINT_DIR}/self_supervised_nv_swin_unetr_5050.pt"

if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
fi

mkdir -p "${PROJECT_DIR}/slurm_logs"

if [[ ! -d "${INPUT_ROOT}" ]]; then
    echo "Input root not found: ${INPUT_ROOT}" >&2
    exit 1
fi

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

declare -a CASE_NAMES
declare -a INPUT_IMAGES

mapfile -d '' CASE_DIRS < <(find "${INPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

if (( ${#CASE_DIRS[@]} > 0 )); then
    for CASE_DIR in "${CASE_DIRS[@]}"; do
        CASE_NAME="$(basename "${CASE_DIR}")"

        if [[ -f "${CASE_DIR}/ct.nii.gz" ]]; then
            INPUT_IMAGE="${CASE_DIR}/ct.nii.gz"
        elif [[ -f "${CASE_DIR}/image.nii.gz" ]]; then
            INPUT_IMAGE="${CASE_DIR}/image.nii.gz"
        else
            echo "Skipping ${CASE_NAME}: no ct.nii.gz or image.nii.gz found" >&2
            continue
        fi

        CASE_NAMES+=("${CASE_NAME}")
        INPUT_IMAGES+=("${INPUT_IMAGE}")
    done
else
    mapfile -d '' IMAGE_FILES < <(find "${INPUT_ROOT}" -mindepth 1 -maxdepth 1 -type f -name '*.nii.gz' -print0 | sort -z)

    for INPUT_IMAGE in "${IMAGE_FILES[@]}"; do
        CASE_NAME="$(basename "${INPUT_IMAGE}")"
        CASE_NAME="${CASE_NAME%.nii.gz}"

        CASE_NAMES+=("${CASE_NAME}")
        INPUT_IMAGES+=("${INPUT_IMAGE}")
    done
fi

CASE_COUNT="${#CASE_NAMES[@]}"
if (( CASE_COUNT == 0 )); then
    echo "No valid input images found in ${INPUT_ROOT}" >&2
    exit 1
fi

if (( SLURM_ARRAY_TASK_ID >= CASE_COUNT )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} exceeds ${CASE_COUNT} available cases." >&2
    echo "Adjust #SBATCH --array to 0-$((CASE_COUNT - 1))." >&2
    exit 0
fi

CASE_NAME="${CASE_NAMES[SLURM_ARRAY_TASK_ID]}"
INPUT_IMAGE="${INPUT_IMAGES[SLURM_ARRAY_TASK_ID]}"

CLIP_OUTPUT="${OUTPUT_DIR}/clip_universal_unet/${CASE_NAME}.nii.gz"
SEGRESNET_OUTPUT="${OUTPUT_DIR}/suprem_segresnet/${CASE_NAME}.nii.gz"
SWIN5050_OUTPUT="${OUTPUT_DIR}/swinunetr_5050/${CASE_NAME}.nii.gz"

if [[ -f "${CLIP_OUTPUT}" && -f "${SEGRESNET_OUTPUT}" && -f "${SWIN5050_OUTPUT}" ]]; then
    echo "All outputs already exist for ${CASE_NAME}; skipping"
    exit 0
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

echo "Array job/task: ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Case: ${CASE_NAME}"
echo "Input image: ${INPUT_IMAGE}"
echo "Input root: ${INPUT_ROOT}"
echo "Output root: ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi

python -c "import nibabel, monai, torch" || {
    echo "The suprem-h200 environment is missing required dependencies." >&2
    exit 1
}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u inference/infer_clip_universal_unet.py \
    --image "${INPUT_IMAGE}" \
    --output "${CLIP_OUTPUT}" \
    --checkpoint "${CLIP_CHECKPOINT}" \
    --device cuda \
    --sw-batch-size 1

python -u inference/infer_suprem_segresnet.py \
    --image "${INPUT_IMAGE}" \
    --output "${SEGRESNET_OUTPUT}" \
    --checkpoint "${SEGRESNET_CHECKPOINT}" \
    --device cuda \
    --sw-batch-size 1

python -u inference/infer_swinunetr_5050.py \
    --image "${INPUT_IMAGE}" \
    --output "${SWIN5050_OUTPUT}" \
    --checkpoint "${SWIN5050_CHECKPOINT}" \
    --device cuda \
    --sw-batch-size 1

echo "Completed inference for ${CASE_NAME}"
