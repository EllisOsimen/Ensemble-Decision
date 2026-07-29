#!/bin/bash
#SBATCH --job-name=external-rf-eval
#SBATCH --partition=Teaching
#SBATCH --exclude=damnii12
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.err

set -euo pipefail

# Evaluate the final config_002 random-forest consensus masks against the
# external dataset's combined_mask.nii.gz annotations. This uses the same
# pairwise evaluation as the external base models and reports:
#   - per-organ Dice
#   - per-organ 1 mm normalized surface Dice (NSD)
#   - per-organ binary Cohen kappa
#   - whole-map multiclass and foreground multiclass Cohen kappa
# Fleiss kappa is disabled because the external dataset has one reference mask.
#
# Submit with defaults:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/evaluate_external_dataset_random_forest.sbatch
#
# Optional positional arguments:
#   sbatch sbatch/evaluate_external_dataset_random_forest.sbatch \
#     [RF_OUTPUT_ROOT] [CASES_ROOT] [EVALUATION_OUTPUT_DIR]

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
RF_OUTPUT_ROOT="${1:-${PROJECT_DIR}/results/external_dataset_random_forest_config_002}"
CASES_ROOT="${2:-/home/s2347484/Seg/external_dataset}"
EVALUATION_OUTPUT_DIR="${3:-${PROJECT_DIR}/results/external_dataset_random_forest_config_002_evaluation}"
PREDICTION_DIR="${EVALUATION_OUTPUT_DIR}/prediction_masks"
ANNOTATION_NAME="combined_mask.nii.gz"
MODEL_NAME="random_forest_config_002"
EXPECTED_CASES=50

if [[ ! -d "${RF_OUTPUT_ROOT}" ]]; then
    echo "Random-forest output root not found: ${RF_OUTPUT_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${CASES_ROOT}" ]]; then
    echo "External cases root not found: ${CASES_ROOT}" >&2
    exit 1
fi

mapfile -d '' CASE_DIRS < <(
    find "${CASES_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)
if (( ${#CASE_DIRS[@]} != EXPECTED_CASES )); then
    echo "Expected ${EXPECTED_CASES} external cases; found ${#CASE_DIRS[@]}." >&2
    exit 1
fi

AGREEMENT_COUNT=$(find "${RF_OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 \
    -type f -name 'agreement_mask.nii.gz' | wc -l)
CONFIDENCE_COUNT=$(find "${RF_OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 \
    -type f -name 'confidence_mask.nii.gz' | wc -l)
if (( AGREEMENT_COUNT != EXPECTED_CASES )); then
    echo "Expected ${EXPECTED_CASES} random-forest agreement masks; found ${AGREEMENT_COUNT}." >&2
    exit 1
fi
if (( CONFIDENCE_COUNT != EXPECTED_CASES )); then
    echo "Expected ${EXPECTED_CASES} random-forest confidence masks; found ${CONFIDENCE_COUNT}." >&2
    exit 1
fi

# Strict case-wise preflight prevents evaluation on a silent intersection.
for CASE_DIR in "${CASE_DIRS[@]}"; do
    CASE_NAME="$(basename "${CASE_DIR}")"
    ANNOTATION="${CASE_DIR}/${ANNOTATION_NAME}"
    AGREEMENT="${RF_OUTPUT_ROOT}/${CASE_NAME}/agreement_mask.nii.gz"
    CONFIDENCE="${RF_OUTPUT_ROOT}/${CASE_NAME}/confidence_mask.nii.gz"

    if [[ ! -f "${ANNOTATION}" ]]; then
        echo "Missing external ground truth: ${ANNOTATION}" >&2
        exit 1
    fi
    if [[ ! -f "${AGREEMENT}" ]]; then
        echo "Missing random-forest agreement mask: ${AGREEMENT}" >&2
        exit 1
    fi
    if [[ ! -f "${CONFIDENCE}" ]]; then
        echo "Missing random-forest confidence mask: ${CONFIDENCE}" >&2
        exit 1
    fi
done

mkdir -p \
    "${PROJECT_DIR}/slurm_logs" \
    "${EVALUATION_OUTPUT_DIR}" \
    "${PREDICTION_DIR}"

# The common evaluator expects one flat prediction directory. Preserve the
# original case-wise outputs and expose only lightweight, case-named symlinks.
for CASE_DIR in "${CASE_DIRS[@]}"; do
    CASE_NAME="$(basename "${CASE_DIR}")"
    SOURCE="${RF_OUTPUT_ROOT}/${CASE_NAME}/agreement_mask.nii.gz"
    DESTINATION="${PREDICTION_DIR}/${CASE_NAME}.nii.gz"

    if [[ -L "${DESTINATION}" ]]; then
        if [[ "$(readlink "${DESTINATION}")" != "${SOURCE}" ]]; then
            echo "Existing symlink points to the wrong source: ${DESTINATION}" >&2
            exit 1
        fi
    elif [[ -e "${DESTINATION}" ]]; then
        echo "Refusing to replace existing non-symlink: ${DESTINATION}" >&2
        exit 1
    else
        ln -s "${SOURCE}" "${DESTINATION}"
    fi
done

PREDICTION_COUNT=$(find "${PREDICTION_DIR}" -maxdepth 1 -type l \
    -name '*.nii.gz' | wc -l)
if (( PREDICTION_COUNT != EXPECTED_CASES )); then
    echo "Expected ${EXPECTED_CASES} staged predictions; found ${PREDICTION_COUNT}." >&2
    exit 1
fi

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
echo "Cases: ${EXPECTED_CASES}"
echo "Ground truth: ${CASES_ROOT}/<case>/${ANNOTATION_NAME}"
echo "Random-forest outputs: ${RF_OUTPUT_ROOT}/<case>/agreement_mask.nii.gz"
echo "Staged prediction directory: ${PREDICTION_DIR}"
echo "Evaluation output: ${EVALUATION_OUTPUT_DIR}"
echo "Label space: target (0=background, 1=pancreas, 2=kidney, 3=liver)"
echo "NSD tolerance: 1.0 mm"
echo "Fleiss kappa: disabled"

python -u evaluation/evaluate_testing_set_human_agreement.py \
    --cases-root "${CASES_ROOT}" \
    --annotations "${ANNOTATION_NAME}" \
    --prediction-dir "${MODEL_NAME}=${PREDICTION_DIR}:target" \
    --output-dir "${EVALUATION_OUTPUT_DIR}" \
    --nsd-tolerance-mm 1.0 \
    --skip-fleiss

echo "Completed config_002 random-forest evaluation on ${EXPECTED_CASES} cases."
echo "Per-organ summary: ${EVALUATION_OUTPUT_DIR}/per_pair_per_class_summary.csv"
echo "Multiclass summary: ${EVALUATION_OUTPUT_DIR}/per_pair_multiclass_summary.csv"
echo "Overall summary: ${EVALUATION_OUTPUT_DIR}/overall_summary.json"
