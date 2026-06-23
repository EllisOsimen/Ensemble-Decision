#!/bin/bash
#SBATCH --job-name=suprem-word-all
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-99%4
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%A_%a.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%A_%a.err

set -euo pipefail

# Runs one WORD image per SLURM array task. Submit with:
#
#   cd /home/s2347484/Seg/SuPreM
#   sbatch infer_word_images_three_models.sbatch
#
# Optional output directory:
#
#   sbatch infer_word_images_three_models.sbatch results/word_three_models

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
IMAGE_DIR="/home/s2347484/Seg/WORD/WORD-V0.1.0/imagesTr"
OUTPUT_DIR="${1:-results/word_three_models}"
CHECKPOINT_DIR="${PROJECT_DIR}/pretrained_weights"

# Interpret relative output paths from the SuPreM project directory.
if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
fi

mkdir -p "${PROJECT_DIR}/slurm_logs"

mapfile -d '' IMAGES < <(find "${IMAGE_DIR}" -maxdepth 1 -type f -name '*.nii.gz' \
    -print0 | sort -z)

IMAGE_COUNT="${#IMAGES[@]}"
if (( IMAGE_COUNT == 0 )); then
    echo "No .nii.gz images found in ${IMAGE_DIR}" >&2
    exit 1
fi

if (( SLURM_ARRAY_TASK_ID >= IMAGE_COUNT )); then
    echo "Array index ${SLURM_ARRAY_TASK_ID} exceeds the ${IMAGE_COUNT} available images." >&2
    echo "Adjust #SBATCH --array to 0-$((IMAGE_COUNT - 1))." >&2
    exit 2
fi

INPUT_CT="${IMAGES[SLURM_ARRAY_TASK_ID]}"
CASE_NAME="$(basename "${INPUT_CT}")"

UNET_OUTPUT="${OUTPUT_DIR}/unet/${CASE_NAME}"
SWIN_OUTPUT="${OUTPUT_DIR}/swinunetr/${CASE_NAME}"
SEGRESNET_OUTPUT="${OUTPUT_DIR}/segresnet/${CASE_NAME}"
AGREEMENT_OUTPUT="${OUTPUT_DIR}/agreement/${CASE_NAME}"

# A fully completed case exits without activating Python or loading checkpoints.
if [[ -f "${UNET_OUTPUT}" \
    && -f "${SWIN_OUTPUT}" \
    && -f "${SEGRESNET_OUTPUT}" \
    && -f "${AGREEMENT_OUTPUT}" ]]
then
    echo "All outputs already exist; skipping ${CASE_NAME}"
    exit 0
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

echo "Array job: ${SLURM_ARRAY_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}/${IMAGE_COUNT}"
echo "Node: $(hostname)"
echo "Input CT: ${INPUT_CT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi

python -c "import nibabel, monai, torch" || {
    echo "The suprem-h200 environment is missing required dependencies." >&2
    exit 1
}

# Verify the shared checkpoint files before starting inference. Array tasks
# read these files directly instead of copying three large checkpoints into
# /tmp for every concurrently running task.
for CHECKPOINT in \
    supervised_suprem_unet_2100.pth \
    supervised_suprem_swinunetr_2100.pth \
    supervised_suprem_segresnet_2100.pth
do
    SOURCE="${CHECKPOINT_DIR}/${CHECKPOINT}"
    if [[ ! -f "${SOURCE}" ]]; then
        echo "Missing checkpoint: ${SOURCE}" >&2
        exit 1
    fi
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u infer_single_image_three_models.py \
    --image "${INPUT_CT}" \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --device cuda \
    --sw-batch-size 1

echo "Completed: ${INPUT_CT}"
