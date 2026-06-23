#!/bin/bash
#SBATCH --job-name=word-swin5050
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

# Infer all WORD imagesTr scans with Tang et al.'s 14-channel BTCV Swin UNETR.
#
# Usage:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch infer_word_swinunetr_5050.sbatch
#
# Optional output directory:
#   sbatch infer_word_swinunetr_5050.sbatch results/word_swinunetr_5050

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
IMAGE_DIR="/home/s2347484/Seg/WORD/WORD-V0.1.0/imagesTr"
OUTPUT_DIR="${1:-results/word_swinunetr_5050}"
CHECKPOINT="${PROJECT_DIR}/pretrained_weights/self_supervised_nv_swin_unetr_5050.pt"

# Interpret a relative output directory from the SuPreM project root.
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
    echo "Array task ${SLURM_ARRAY_TASK_ID} exceeds ${IMAGE_COUNT} images." >&2
    exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

INPUT_CT="${IMAGES[SLURM_ARRAY_TASK_ID]}"
OUTPUT_PATH="${OUTPUT_DIR}/$(basename "${INPUT_CT}")"

# Resume safely: a completed case is not inferred again.
if [[ -f "${OUTPUT_PATH}" ]]; then
    echo "Output already exists; skipping: ${OUTPUT_PATH}"
    exit 0
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

echo "Array job/task: ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Input: ${INPUT_CT}"
echo "Output: ${OUTPUT_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u infer_swinunetr_5050.py \
    --image "${INPUT_CT}" \
    --output "${OUTPUT_PATH}" \
    --checkpoint "${CHECKPOINT}" \
    --device cuda \
    --sw-batch-size 1

echo "Completed: ${INPUT_CT}"
