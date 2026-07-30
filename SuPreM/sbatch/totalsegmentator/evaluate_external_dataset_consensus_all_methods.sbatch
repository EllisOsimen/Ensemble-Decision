#!/bin/bash
#SBATCH --job-name=totalseg-eval
#SBATCH --partition=Teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%4
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

# Evaluate one TotalSegmentator external method per task against the combined
# reference mask: 0=unweighted, 1=weighted, 2=STAPLE, 3=random forest.
# Fleiss kappa is disabled because TotalSegmentator provides one reference.
#
# Optional arguments:
#   sbatch sbatch/totalsegmentator/evaluate_external_dataset_consensus_all_methods.sbatch \
#     [CONSENSUS_ROOT] [RF_ROOT] [CASES_ROOT] [OUTPUT_ROOT]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_DIR="$(cd -- "${PROJECT_DIR}/.." && pwd)"
CONSENSUS_ROOT="${1:-${PROJECT_DIR}/results/external_dataset_consensus}"
RF_ROOT="${2:-${PROJECT_DIR}/results/external_dataset_random_forest_config_002}"
CASES_ROOT="${3:-${REPO_DIR}/external_dataset}"
OUTPUT_ROOT="${4:-${PROJECT_DIR}/results/external_dataset_consensus_evaluation}"
EXPECTED_CASES=50
METHODS=(unweighted weighted staple random_forest)
PREDICTION_NAMES=(unweighted weighted staple random_forest_config_002)

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array job.}"
if (( TASK_ID < 0 || TASK_ID >= ${#METHODS[@]} )); then
    echo "Array task ${TASK_ID} is outside the expected range 0-3." >&2
    exit 1
fi
METHOD="${METHODS[TASK_ID]}"
PREDICTION_NAME="${PREDICTION_NAMES[TASK_ID]}"

if [[ "${METHOD}" == "random_forest" ]]; then
    PREDICTION_DIR="${RF_ROOT}"
else
    PREDICTION_DIR="${CONSENSUS_ROOT}/${METHOD}/consensus_masks"
fi
OUTPUT_DIR="${OUTPUT_ROOT}/${METHOD}"

mapfile -d '' CASE_DIRS < <(
    find "${CASES_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)
if (( ${#CASE_DIRS[@]} != EXPECTED_CASES )); then
    echo "Expected ${EXPECTED_CASES} TotalSegmentator cases; found ${#CASE_DIRS[@]}." >&2
    exit 1
fi

for CASE_DIR in "${CASE_DIRS[@]}"; do
    CASE_NAME="$(basename "${CASE_DIR}")"
    if [[ ! -f "${CASE_DIR}/combined_mask.nii.gz" ]]; then
        echo "Missing reference mask: ${CASE_DIR}/combined_mask.nii.gz" >&2
        exit 1
    fi
    if [[ "${METHOD}" == "random_forest" ]]; then
        PREDICTION="${PREDICTION_DIR}/${CASE_NAME}/agreement_mask.nii.gz"
    else
        PREDICTION="${PREDICTION_DIR}/${CASE_NAME}.nii.gz"
    fi
    if [[ ! -f "${PREDICTION}" ]]; then
        echo "Missing ${METHOD} prediction: ${PREDICTION}" >&2
        exit 1
    fi
done

source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-suprem-h200}"
cd "${PROJECT_DIR}"

python -u evaluation/evaluate_testing_set_human_agreement.py \
    --cases-root "${CASES_ROOT}" \
    --annotations combined_mask.nii.gz \
    --prediction-dir "${PREDICTION_NAME}=${PREDICTION_DIR}:target" \
    --output-dir "${OUTPUT_DIR}" \
    --nsd-tolerance-mm 1.0 \
    --skip-fleiss

echo "Completed ${METHOD} evaluation on ${EXPECTED_CASES} TotalSegmentator cases."
