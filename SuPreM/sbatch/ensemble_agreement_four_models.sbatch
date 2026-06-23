#!/bin/bash
#SBATCH --job-name=agreement4
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-99%8
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%A_%a.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%A_%a.err

set -euo pipefail

# Create one four-model agreement map per SLURM array task.
#
# Usage:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/ensemble_agreement_four_models.sbatch
#
# Optional output directory:
#   sbatch sbatch/ensemble_agreement_four_models.sbatch results/my_agreement_maps

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
UNET_DIR="${PROJECT_DIR}/results/word_three_models/unet"
SWINUNETR_DIR="${PROJECT_DIR}/results/word_three_models/swinunetr"
SEGRESNET_DIR="${PROJECT_DIR}/results/word_three_models/segresnet"
SWIN5050_DIR="${PROJECT_DIR}/results/word_swinunetr_5050"
OUTPUT_DIR="${1:-${PROJECT_DIR}/results/word_four_model_agreement}"

if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
fi

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_DIR}"

mapfile -d '' CASES < <(find "${UNET_DIR}" -maxdepth 1 -type f -name '*.nii.gz' \
    -print0 | sort -z)

CASE_COUNT="${#CASES[@]}"
if (( CASE_COUNT == 0 )); then
    echo "No predictions found in ${UNET_DIR}" >&2
    exit 1
fi
if (( SLURM_ARRAY_TASK_ID >= CASE_COUNT )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} exceeds ${CASE_COUNT} cases." >&2
    exit 2
fi

CASE_NAME="$(basename "${CASES[SLURM_ARRAY_TASK_ID]}")"
OUTPUT_PATH="${OUTPUT_DIR}/${CASE_NAME}"

# Safe resume: never replace a completed agreement map.
if [[ -f "${OUTPUT_PATH}" ]]; then
    echo "Output already exists; skipping: ${OUTPUT_PATH}"
    exit 0
fi

for DIRECTORY in \
    "${UNET_DIR}" \
    "${SWINUNETR_DIR}" \
    "${SEGRESNET_DIR}" \
    "${SWIN5050_DIR}"
do
    if [[ ! -f "${DIRECTORY}/${CASE_NAME}" ]]; then
        echo "Missing input: ${DIRECTORY}/${CASE_NAME}" >&2
        exit 1
    fi
done

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

echo "Array job/task: ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Case: ${CASE_NAME}"
echo "Output: ${OUTPUT_PATH}"

python -u ensemble_agreement/ensemble_agreement_four_models.py \
    --unet-dir "${UNET_DIR}" \
    --swinunetr-dir "${SWINUNETR_DIR}" \
    --segresnet-dir "${SEGRESNET_DIR}" \
    --swin5050-dir "${SWIN5050_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --case-name "${CASE_NAME}"

echo "Completed: ${CASE_NAME}"
