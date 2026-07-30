#!/bin/bash
#SBATCH --job-name=external-infer-conf
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-49%4
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

# Run the three established base models on the prepared 50-case
# TotalSegmentator external dataset.
# Each array task processes one CT and runs the models sequentially on one GPU:
#   1. CLIP Universal U-Net
#   2. SuPreM SegResNet
#   3. Swin UNETR 5050
#
# In addition to hard-label predictions, the job saves assigned-label
# confidence and inference-validity maps for downstream fusion/stacking.
#
# Submit with defaults:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/totalsegmentator/infer_external_dataset_three_models_with_confidence.sbatch
#
# Optional positional arguments:
#   sbatch sbatch/totalsegmentator/infer_external_dataset_three_models_with_confidence.sbatch \
#     [INPUT_ROOT] [OUTPUT_DIR]
#
# Rerun and replace every output:
#   sbatch --export=ALL,OVERWRITE=1 \
#     sbatch/totalsegmentator/infer_external_dataset_three_models_with_confidence.sbatch

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_DIR="$(cd -- "${PROJECT_DIR}/.." && pwd)"
INPUT_ROOT="${1:-${REPO_DIR}/external_dataset}"
OUTPUT_DIR="${2:-${PROJECT_DIR}/results/external_dataset_inference_with_confidence}"
CHECKPOINT_DIR="${PROJECT_DIR}/pretrained_weights"
OVERWRITE="${OVERWRITE:-0}"

if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
    echo "OVERWRITE must be 0 or 1; got ${OVERWRITE}" >&2
    exit 1
fi
if [[ ! -d "${INPUT_ROOT}" ]]; then
    echo "Input root not found: ${INPUT_ROOT}" >&2
    exit 1
fi
if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
fi

CLIP_CHECKPOINT="${CHECKPOINT_DIR}/supervised_clip_driven_universal_unet_2100.pth"
SEGRESNET_CHECKPOINT="${CHECKPOINT_DIR}/supervised_suprem_segresnet_2100.pth"
SWIN5050_CHECKPOINT="${CHECKPOINT_DIR}/self_supervised_nv_swin_unetr_5050.pt"

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

mapfile -d '' CASE_DIRS < <(
    find "${INPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)

CASE_COUNT="${#CASE_DIRS[@]}"
if (( CASE_COUNT != 50 )); then
    echo "Expected 50 external cases but discovered ${CASE_COUNT}." >&2
    echo "Update the Slurm array range if the dataset size changed." >&2
    exit 1
fi
if (( SLURM_ARRAY_TASK_ID >= CASE_COUNT )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} exceeds ${CASE_COUNT} cases." >&2
    exit 0
fi

CASE_DIR="${CASE_DIRS[SLURM_ARRAY_TASK_ID]}"
CASE_NAME="$(basename "${CASE_DIR}")"
if [[ -f "${CASE_DIR}/ct.nii.gz" ]]; then
    INPUT_IMAGE="${CASE_DIR}/ct.nii.gz"
elif [[ -f "${CASE_DIR}/image.nii.gz" ]]; then
    INPUT_IMAGE="${CASE_DIR}/image.nii.gz"
else
    echo "No CT image found for ${CASE_NAME}" >&2
    exit 1
fi
if [[ ! -f "${CASE_DIR}/combined_mask.nii.gz" ]]; then
    echo "Ground-truth combined mask missing for ${CASE_NAME}" >&2
    exit 1
fi

CLIP_OUTPUT="${OUTPUT_DIR}/clip_universal_unet/${CASE_NAME}.nii.gz"
CLIP_CONFIDENCE="${OUTPUT_DIR}/clip_universal_unet_confidence/${CASE_NAME}.nii.gz"
CLIP_VALIDITY="${OUTPUT_DIR}/clip_universal_unet_validity/${CASE_NAME}.nii.gz"
SEGRESNET_OUTPUT="${OUTPUT_DIR}/suprem_segresnet/${CASE_NAME}.nii.gz"
SEGRESNET_CONFIDENCE="${OUTPUT_DIR}/suprem_segresnet_confidence/${CASE_NAME}.nii.gz"
SEGRESNET_VALIDITY="${OUTPUT_DIR}/suprem_segresnet_validity/${CASE_NAME}.nii.gz"
SWIN5050_OUTPUT="${OUTPUT_DIR}/swinunetr_5050/${CASE_NAME}.nii.gz"
SWIN5050_CONFIDENCE="${OUTPUT_DIR}/swinunetr_5050_confidence/${CASE_NAME}.nii.gz"
SWIN5050_VALIDITY="${OUTPUT_DIR}/swinunetr_5050_validity/${CASE_NAME}.nii.gz"

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
if [[ "${ALL_COMPLETE}" == true && "${OVERWRITE}" != "1" ]]; then
    echo "All label, confidence, and validity outputs exist for ${CASE_NAME}; skipping"
    exit 0
fi

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_DIR}"

source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-suprem-h200}"
cd "${PROJECT_DIR}"

python -c "import nibabel, monai, numpy, torch" || {
    echo "The suprem-h200 environment is missing required dependencies." >&2
    exit 1
}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OVERWRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
    OVERWRITE_ARGS+=(--overwrite)
fi

echo "Array job/task: ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Case: ${CASE_NAME}"
echo "Input image: ${INPUT_IMAGE}"
echo "Ground truth: ${CASE_DIR}/combined_mask.nii.gz"
echo "Output root: ${OUTPUT_DIR}"
echo "Overwrite: ${OVERWRITE}"
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
    "${OVERWRITE_ARGS[@]}"

python -u inference/infer_suprem_segresnet.py \
    --image "${INPUT_IMAGE}" \
    --output "${SEGRESNET_OUTPUT}" \
    --confidence-output "${SEGRESNET_CONFIDENCE}" \
    --validity-output "${SEGRESNET_VALIDITY}" \
    --confidence-storage uint8 \
    --checkpoint "${SEGRESNET_CHECKPOINT}" \
    --device cuda \
    --sw-batch-size 1 \
    "${OVERWRITE_ARGS[@]}"

python -u inference/infer_swinunetr_5050.py \
    --image "${INPUT_IMAGE}" \
    --output "${SWIN5050_OUTPUT}" \
    --confidence-output "${SWIN5050_CONFIDENCE}" \
    --validity-output "${SWIN5050_VALIDITY}" \
    --confidence-storage uint8 \
    --checkpoint "${SWIN5050_CHECKPOINT}" \
    --device cuda \
    --sw-batch-size 1 \
    "${OVERWRITE_ARGS[@]}"

echo "Completed all three models for ${CASE_NAME}"
