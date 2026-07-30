#!/bin/bash
#SBATCH --job-name=external-eval
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.err

set -euo pipefail

# Evaluate the unweighted, training-derived weighted, and STAPLE external
# consensus masks against each case's combined_mask.nii.gz ground truth.
# Predictions and ground truth already use the common target label space:
#   0=background, 1=pancreas, 2=kidney, 3=liver.
#
# Submit with defaults:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/legacy/evaluate_external_dataset_consensus_methods.sbatch
#
# Optional positional arguments:
#   sbatch sbatch/legacy/evaluate_external_dataset_consensus_methods.sbatch \
#     [CONSENSUS_ROOT] [CASES_ROOT] [OUTPUT_DIR]

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
CONSENSUS_ROOT="${1:-${PROJECT_DIR}/results/external_dataset_consensus}"
CASES_ROOT="${2:-/home/s2347484/Seg/external_dataset}"
OUTPUT_DIR="${3:-${PROJECT_DIR}/results/external_dataset_consensus_evaluation}"
ANNOTATION_NAME="combined_mask.nii.gz"
EXPECTED_CASES=50

METHOD_NAMES=(unweighted weighted staple)

if [[ ! -d "${CASES_ROOT}" ]]; then
    echo "External cases root not found: ${CASES_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${CONSENSUS_ROOT}" ]]; then
    echo "Consensus root not found: ${CONSENSUS_ROOT}" >&2
    exit 1
fi

mapfile -d '' CASE_DIRS < <(
    find "${CASES_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)
if (( ${#CASE_DIRS[@]} != EXPECTED_CASES )); then
    echo "Expected ${EXPECTED_CASES} external cases; found ${#CASE_DIRS[@]}." >&2
    exit 1
fi

# Require a ground truth and all three consensus predictions for every case.
# This prevents the evaluator from silently using only an intersection subset.
for CASE_DIR in "${CASE_DIRS[@]}"; do
    CASE_NAME="$(basename "${CASE_DIR}")"
    ANNOTATION="${CASE_DIR}/${ANNOTATION_NAME}"
    if [[ ! -f "${ANNOTATION}" ]]; then
        echo "Missing ground truth: ${ANNOTATION}" >&2
        exit 1
    fi

    for METHOD in "${METHOD_NAMES[@]}"; do
        PREDICTION="${CONSENSUS_ROOT}/${METHOD}/consensus_masks/${CASE_NAME}.nii.gz"
        if [[ ! -f "${PREDICTION}" ]]; then
            echo "Missing ${METHOD} prediction: ${PREDICTION}" >&2
            exit 1
        fi
    done
done

for METHOD in "${METHOD_NAMES[@]}"; do
    PREDICTION_DIR="${CONSENSUS_ROOT}/${METHOD}/consensus_masks"
    PREDICTION_COUNT=$(find "${PREDICTION_DIR}" -maxdepth 1 -type f \
        -name '*.nii.gz' | wc -l)
    if (( PREDICTION_COUNT != EXPECTED_CASES )); then
        echo "Expected ${EXPECTED_CASES} ${METHOD} predictions; found ${PREDICTION_COUNT}." >&2
        exit 1
    fi
done

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_DIR}"

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

python -c "import nibabel, numpy, surface_distance, tqdm" || {
    echo "The suprem-h200 environment is missing required evaluation dependencies." >&2
    exit 1
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Node: $(hostname)"
echo "Cases root: ${CASES_ROOT}"
echo "Annotation: ${ANNOTATION_NAME}"
echo "Consensus root: ${CONSENSUS_ROOT}"
echo "Evaluation output: ${OUTPUT_DIR}"
echo "Methods: ${METHOD_NAMES[*]}"
echo "NSD tolerance: 1.0 mm"

python -u evaluation/evaluate_testing_set_prediction_dirs.py \
    --cases-root "${CASES_ROOT}" \
    --annotation-name "${ANNOTATION_NAME}" \
    --model \
        "unweighted=${CONSENSUS_ROOT}/unweighted/consensus_masks:target" \
    --model \
        "weighted=${CONSENSUS_ROOT}/weighted/consensus_masks:target" \
    --model \
        "staple=${CONSENSUS_ROOT}/staple/consensus_masks:target" \
    --output-dir "${OUTPUT_DIR}" \
    --nsd-tolerance-mm 1.0

echo "Completed external consensus evaluation."
for METHOD in "${METHOD_NAMES[@]}"; do
    echo "${METHOD} summary: ${OUTPUT_DIR}/${METHOD}/per_class_summary.csv"
    echo "${METHOD} overall: ${OUTPUT_DIR}/${METHOD}/overall_summary.json"
done
