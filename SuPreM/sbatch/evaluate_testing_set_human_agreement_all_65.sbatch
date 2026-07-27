#!/bin/bash
#SBATCH --job-name=human-agreement-65
#SBATCH --partition=Teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.err

set -euo pipefail

# Measure agreement among the three expert annotations for every testing case.
# No model prediction is included and no case is excluded, so overall_summary.json
# contains human-only, three-rater Fleiss' kappa across all 65 cases.

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
CASES_ROOT="/home/s2347484/Seg/testing_set"
OUTPUT_DIR="${PROJECT_DIR}/results/results_65_testing_set/human_annotator_evaluation"

if [[ ! -d "${CASES_ROOT}" ]]; then
    echo "Testing cases root not found: ${CASES_ROOT}" >&2
    exit 1
fi

CASE_COUNT=$(find "${CASES_ROOT}" -mindepth 1 -maxdepth 1 -type d | wc -l)
if (( CASE_COUNT != 65 )); then
    echo "Expected exactly 65 testing case directories; found ${CASE_COUNT}." >&2
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
echo "Cases root: ${CASES_ROOT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Evaluating all 65 cases; no cases or annotations are excluded."

python -u evaluation/evaluate_testing_set_human_agreement.py \
    --cases-root "${CASES_ROOT}" \
    --annotations \
        annotation_1.nii.gz \
        annotation_2.nii.gz \
        annotation_3.nii.gz \
    --output-dir "${OUTPUT_DIR}" \
    --nsd-tolerance-mm 1.0

echo "Completed human-only agreement evaluation for all 65 testing cases."
echo "Pairwise Cohen kappa: ${OUTPUT_DIR}/per_pair_multiclass_summary.csv"
echo "Per-organ binary Cohen kappa: ${OUTPUT_DIR}/per_pair_per_class_summary.csv"
echo "Three-rater Fleiss kappa: ${OUTPUT_DIR}/overall_summary.json"
