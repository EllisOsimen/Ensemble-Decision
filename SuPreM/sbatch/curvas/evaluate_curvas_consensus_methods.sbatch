#!/bin/bash
#SBATCH --job-name=curvas-eval
#SBATCH --partition=Teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3%4
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

# Evaluate one CURVAS ensemble method per task against all three experts:
#   0=unweighted, 1=weighted, 2=STAPLE, 3=random forest.
#
# Optional arguments:
#   sbatch sbatch/curvas/evaluate_curvas_consensus_methods.sbatch \
#     [CURVAS_RESULT_ROOT] [TEST_CASES_ROOT] [OUTPUT_ROOT]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_DIR="$(cd -- "${PROJECT_DIR}/.." && pwd)"
CURVAS_ROOT="${1:-${PROJECT_DIR}/results/all_datasets_curvas_inference_with_confidence}"
CASES_ROOT="${2:-${REPO_DIR}/testing_set}"
OUTPUT_ROOT="${3:-${PROJECT_DIR}/results/results_65_testing_set}"
EXPECTED_CASES=65
METHODS=(unweighted weighted staple random_forest)
PREDICTION_NAMES=(agreement_mask weighted_training_consensus staple random_forest_config_002)

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array job.}"
if (( TASK_ID < 0 || TASK_ID >= ${#METHODS[@]} )); then
    echo "Array task ${TASK_ID} is outside the expected range 0-3." >&2
    exit 1
fi
METHOD="${METHODS[TASK_ID]}"
PREDICTION_NAME="${PREDICTION_NAMES[TASK_ID]}"

if [[ "${METHOD}" == "random_forest" ]]; then
    PREDICTION_DIR="${CURVAS_ROOT}/random_forest_config_002_final/testing_predictions"
    OUTPUT_DIR="${OUTPUT_ROOT}/random_forest_human_annotator_evaluation"
else
    PREDICTION_DIR="${CURVAS_ROOT}/consensus_testing/${METHOD}/consensus_masks"
    OUTPUT_DIR="${OUTPUT_ROOT}/${METHOD}_human_annotator_evaluation"
fi

mapfile -d '' CASE_DIRS < <(
    find "${CASES_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)
if (( ${#CASE_DIRS[@]} != EXPECTED_CASES )); then
    echo "Expected ${EXPECTED_CASES} CURVAS test cases; found ${#CASE_DIRS[@]}." >&2
    exit 1
fi

for CASE_DIR in "${CASE_DIRS[@]}"; do
    CASE_NAME="$(basename "${CASE_DIR}")"
    for ANNOTATION in annotation_1.nii.gz annotation_2.nii.gz annotation_3.nii.gz; do
        if [[ ! -f "${CASE_DIR}/${ANNOTATION}" ]]; then
            echo "Missing annotation: ${CASE_DIR}/${ANNOTATION}" >&2
            exit 1
        fi
    done

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
    --annotations annotation_1.nii.gz annotation_2.nii.gz annotation_3.nii.gz \
    --prediction-dir "${PREDICTION_NAME}=${PREDICTION_DIR}:target" \
    --output-dir "${OUTPUT_DIR}" \
    --nsd-tolerance-mm 1.0

echo "Completed ${METHOD} evaluation on ${EXPECTED_CASES} CURVAS cases."
