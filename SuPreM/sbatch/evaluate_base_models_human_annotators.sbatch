#!/bin/bash
#SBATCH --job-name=base-human-eval
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%A_%a.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%A_%a.err

set -euo pipefail

# Evaluate one base segmentation model per array task against all three expert
# annotations. Native WORD/SuPreM or BTCV labels are remapped to CURVAS labels
# during evaluation. Each task therefore computes a separate four-rater
# Fleiss' kappa (three experts plus one base model).

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
CASES_ROOT="/home/s2347484/Seg/testing_set"
PREDICTION_ROOT="${PROJECT_DIR}/results/all_curvas_inference_with_confidence/testing"
OUTPUT_ROOT="${PROJECT_DIR}/results/base_model_human_annotator_evaluation"

MODEL_NAMES=(clip_unet segresnet swin5050)
MODEL_DIRS=(clip_universal_unet suprem_segresnet swinunetr_5050)
LABEL_SPACES=(suprem suprem btcv)

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array job.}"
if (( TASK_ID < 0 || TASK_ID >= ${#MODEL_NAMES[@]} )); then
    echo "Array task ${TASK_ID} is outside the expected range 0-2." >&2
    exit 1
fi

MODEL_NAME="${MODEL_NAMES[TASK_ID]}"
MODEL_DIR="${PREDICTION_ROOT}/${MODEL_DIRS[TASK_ID]}"
LABEL_SPACE="${LABEL_SPACES[TASK_ID]}"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}"

if [[ ! -d "${CASES_ROOT}" ]]; then
    echo "Testing cases root not found: ${CASES_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "Prediction directory not found: ${MODEL_DIR}" >&2
    exit 1
fi

CASE_COUNT=$(find "${CASES_ROOT}" -mindepth 1 -maxdepth 1 -type d | wc -l)
PREDICTION_COUNT=$(find "${MODEL_DIR}" -maxdepth 1 -type f \
    \( -name '*.nii' -o -name '*.nii.gz' \) | wc -l)
if (( CASE_COUNT != 65 )); then
    echo "Expected exactly 65 testing case directories; found ${CASE_COUNT}." >&2
    exit 1
fi
if (( PREDICTION_COUNT != 65 )); then
    echo "Expected 65 ${MODEL_NAME} predictions; found ${PREDICTION_COUNT}." >&2
    exit 1
fi

for ANNOTATION_NAME in annotation_1.nii.gz annotation_2.nii.gz annotation_3.nii.gz; do
    ANNOTATION_COUNT=$(find "${CASES_ROOT}" -mindepth 2 -maxdepth 2 \
        -type f -name "${ANNOTATION_NAME}" | wc -l)
    if (( ANNOTATION_COUNT != 65 )); then
        echo "Expected 65 ${ANNOTATION_NAME} files; found ${ANNOTATION_COUNT}." >&2
        exit 1
    fi
done

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_DIR}"

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Node: $(hostname)"
echo "Model: ${MODEL_NAME}"
echo "Native label space: ${LABEL_SPACE}"
echo "Predictions: ${MODEL_DIR}"
echo "Cases root: ${CASES_ROOT}"
echo "Output directory: ${OUTPUT_DIR}"

python -u evaluation/evaluate_testing_set_human_agreement.py \
    --cases-root "${CASES_ROOT}" \
    --annotations \
        annotation_1.nii.gz \
        annotation_2.nii.gz \
        annotation_3.nii.gz \
    --prediction-dir "${MODEL_NAME}=${MODEL_DIR}:${LABEL_SPACE}" \
    --output-dir "${OUTPUT_DIR}" \
    --nsd-tolerance-mm 1.0

echo "Completed ${MODEL_NAME} evaluation against all three experts."
echo "Per-organ pairwise results: ${OUTPUT_DIR}/per_pair_per_class_summary.csv"
echo "Multiclass pairwise results: ${OUTPUT_DIR}/per_pair_multiclass_summary.csv"
echo "Four-rater Fleiss results: ${OUTPUT_DIR}/overall_summary.json"
