#!/bin/bash
#SBATCH --job-name=external-kappa
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%A_%a.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%A_%a.err

set -euo pipefail

# Run three independent pairwise evaluations:
#   task 0: combined_mask.nii.gz versus unweighted consensus
#   task 1: combined_mask.nii.gz versus weighted consensus
#   task 2: combined_mask.nii.gz versus STAPLE consensus
#
# Each task writes binary Cohen kappa per organ and whole-map multiclass Cohen
# kappa. Fleiss kappa is explicitly disabled because there is only one external
# ground-truth annotation. NSD is skipped because it is produced by the separate
# external consensus Dice/NSD evaluation job.
#
# Submit with defaults:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/legacy/evaluate_external_dataset_consensus_kappa.sbatch
#
# Optional positional arguments:
#   sbatch sbatch/legacy/evaluate_external_dataset_consensus_kappa.sbatch \
#     [CONSENSUS_ROOT] [CASES_ROOT] [OUTPUT_ROOT]

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
CONSENSUS_ROOT="${1:-${PROJECT_DIR}/results/external_dataset_consensus}"
CASES_ROOT="${2:-/home/s2347484/Seg/external_dataset}"
OUTPUT_ROOT="${3:-${PROJECT_DIR}/results/external_dataset_consensus_kappa_evaluation}"
ANNOTATION_NAME="combined_mask.nii.gz"
EXPECTED_CASES=50

METHOD_NAMES=(unweighted weighted staple)
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array job.}"
if (( TASK_ID < 0 || TASK_ID >= ${#METHOD_NAMES[@]} )); then
    echo "Array task ${TASK_ID} is outside the expected range 0-2." >&2
    exit 1
fi

METHOD="${METHOD_NAMES[TASK_ID]}"
PREDICTION_DIR="${CONSENSUS_ROOT}/${METHOD}/consensus_masks"
OUTPUT_DIR="${OUTPUT_ROOT}/${METHOD}"

if [[ ! -d "${CASES_ROOT}" ]]; then
    echo "External cases root not found: ${CASES_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${PREDICTION_DIR}" ]]; then
    echo "${METHOD} prediction directory not found: ${PREDICTION_DIR}" >&2
    exit 1
fi

mapfile -d '' CASE_DIRS < <(
    find "${CASES_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)
if (( ${#CASE_DIRS[@]} != EXPECTED_CASES )); then
    echo "Expected ${EXPECTED_CASES} external cases; found ${#CASE_DIRS[@]}." >&2
    exit 1
fi

for CASE_DIR in "${CASE_DIRS[@]}"; do
    CASE_NAME="$(basename "${CASE_DIR}")"
    ANNOTATION="${CASE_DIR}/${ANNOTATION_NAME}"
    PREDICTION="${PREDICTION_DIR}/${CASE_NAME}.nii.gz"

    if [[ ! -f "${ANNOTATION}" ]]; then
        echo "Missing ground truth: ${ANNOTATION}" >&2
        exit 1
    fi
    if [[ ! -f "${PREDICTION}" ]]; then
        echo "Missing ${METHOD} prediction: ${PREDICTION}" >&2
        exit 1
    fi
done

PREDICTION_COUNT=$(find "${PREDICTION_DIR}" -maxdepth 1 -type f \
    -name '*.nii.gz' | wc -l)
if (( PREDICTION_COUNT != EXPECTED_CASES )); then
    echo "Expected ${EXPECTED_CASES} ${METHOD} predictions; found ${PREDICTION_COUNT}." >&2
    exit 1
fi

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_DIR}"

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

python -c "import nibabel, numpy, tqdm" || {
    echo "The suprem-h200 environment is missing required evaluation dependencies." >&2
    exit 1
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Array job/task: ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Method: ${METHOD}"
echo "Ground truth: ${CASES_ROOT}/<case>/${ANNOTATION_NAME}"
echo "Predictions: ${PREDICTION_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Fleiss kappa: disabled"
echo "NSD: skipped (already evaluated separately)"

python -u evaluation/evaluate_testing_set_human_agreement.py \
    --cases-root "${CASES_ROOT}" \
    --annotations "${ANNOTATION_NAME}" \
    --prediction-dir "${METHOD}=${PREDICTION_DIR}:target" \
    --output-dir "${OUTPUT_DIR}" \
    --skip-fleiss

echo "Completed pairwise ${METHOD} evaluation against ${ANNOTATION_NAME}."
echo "Binary kappa: ${OUTPUT_DIR}/per_pair_per_class_summary.csv"
echo "Multiclass kappa: ${OUTPUT_DIR}/per_pair_multiclass_summary.csv"
