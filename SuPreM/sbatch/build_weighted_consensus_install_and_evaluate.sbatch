#!/bin/bash
#SBATCH --job-name=weighted-4mask
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.err

set -euo pipefail

# Regenerate the weighted CURVAS ensemble agreement masks, install them into
# testing_set as agreement_mask.nii.gz, then evaluate annotation_1,
# annotation_2, annotation_3, and agreement_mask together.
#
# Existing testing_set agreement_mask.nii.gz files are moved out of the case
# folders into a timestamped backup directory before the new masks are moved in.
#
# Usage:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/build_weighted_consensus_install_and_evaluate.sbatch
#
# Optional arguments:
#   sbatch sbatch/build_weighted_consensus_install_and_evaluate.sbatch \
#     [PREDICTION_ROOT] [CONSENSUS_OUTPUT_DIR] [CONFIDENCE_DIR] \
#     [CASES_ROOT] [EVAL_OUTPUT_DIR] [EVAL_EXTRA_ARGS...]
#
# Defaults:
#   PREDICTION_ROOT      -> results/CURVAS_INFERENCE
#   CONSENSUS_OUTPUT_DIR -> results/CURVAS_INFERENCE/agreement_masks_weighted_65case
#   CONFIDENCE_DIR       -> results/CURVAS_INFERENCE/agreement_confidence_masks_weighted_65case
#   CASES_ROOT           -> /home/s2347484/Seg/testing_set
#   EVAL_OUTPUT_DIR      -> results/testing_set_human_plus_weighted_agreement_mask_65case
#
# EVAL_EXTRA_ARGS are forwarded to evaluate_testing_set_human_agreement.py,
# for example: --skip-nsd

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
PREDICTION_ROOT="${1:-results/CURVAS_INFERENCE}"
CONSENSUS_OUTPUT_DIR="${2:-results/CURVAS_INFERENCE/agreement_masks_weighted_65case}"
CONFIDENCE_DIR="${3:-results/CURVAS_INFERENCE/agreement_confidence_masks_weighted_65case}"
CASES_ROOT="${4:-/home/s2347484/Seg/testing_set}"
EVAL_OUTPUT_DIR="${5:-results/testing_set_human_plus_weighted_agreement_mask_65case}"

for _ in 1 2 3 4 5; do
    if (( $# > 0 )); then
        shift
    fi
done
EVAL_EXTRA_ARGS=("$@")

make_project_path_absolute() {
    local path="$1"
    if [[ "${path}" == /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s/%s\n' "${PROJECT_DIR}" "${path}"
    fi
}

PREDICTION_ROOT="$(make_project_path_absolute "${PREDICTION_ROOT}")"
CONSENSUS_OUTPUT_DIR="$(make_project_path_absolute "${CONSENSUS_OUTPUT_DIR}")"
CONFIDENCE_DIR="$(make_project_path_absolute "${CONFIDENCE_DIR}")"
CASES_ROOT="$(make_project_path_absolute "${CASES_ROOT}")"
EVAL_OUTPUT_DIR="$(make_project_path_absolute "${EVAL_OUTPUT_DIR}")"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_DIR="${PROJECT_DIR}/results/testing_set_agreement_mask_backups/${RUN_STAMP}"

mkdir -p \
    "${PROJECT_DIR}/slurm_logs" \
    "${CONSENSUS_OUTPUT_DIR}" \
    "${CONFIDENCE_DIR}" \
    "${EVAL_OUTPUT_DIR}"

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

python -c "import nibabel, numpy, scipy, surface_distance, tqdm" || {
    echo "The suprem-h200 environment is missing required dependencies." >&2
    exit 1
}

CONSENSUS_CMD=(
    python -u ensemble_agreement/consensus_agreement_mask.py
    "${PREDICTION_ROOT}"
    --consensus-mode weighted
    --no-clip-fallback
    --model-dir swinunetr_5050
    --model-id swin5050
    --label-space btcv
    --model-dir clip_universal_unet
    --model-id clip_unet
    --label-space suprem
    --model-dir suprem_segresnet
    --model-id segresnet
    --label-space suprem
    --weak-thresholds 0.55 0.80 1.00
    --strong-thresholds 0.65 0.90 1.30
    --output-dir "${CONSENSUS_OUTPUT_DIR}"
    --confidence-dir "${CONFIDENCE_DIR}"
    --overwrite
)

printf 'Prediction root: %s\n' "${PREDICTION_ROOT}"
printf 'Consensus label mask dir: %s\n' "${CONSENSUS_OUTPUT_DIR}"
printf 'Confidence mask dir: %s\n' "${CONFIDENCE_DIR}"
printf 'Testing-set root: %s\n' "${CASES_ROOT}"
printf 'Evaluation output dir: %s\n' "${EVAL_OUTPUT_DIR}"
printf 'Weighted thresholds: weak=(0.55 0.80 1.00), strong=(0.65 0.90 1.30)\n'
printf 'CLIP fallback: disabled\n'

"${CONSENSUS_CMD[@]}"

shopt -s nullglob
NEW_MASKS=("${CONSENSUS_OUTPUT_DIR}"/*.nii.gz)
if (( ${#NEW_MASKS[@]} == 0 )); then
    echo "No generated consensus masks found in ${CONSENSUS_OUTPUT_DIR}." >&2
    exit 1
fi

CASE_DIRS=("${CASES_ROOT}"/*)
if (( ${#CASE_DIRS[@]} == 0 )); then
    echo "No case directories found under ${CASES_ROOT}." >&2
    exit 1
fi

for CASE_DIR in "${CASE_DIRS[@]}"; do
    if [[ ! -d "${CASE_DIR}" ]]; then
        continue
    fi

    CASE_NAME="${CASE_DIR##*/}"
    SOURCE_MASK="${CONSENSUS_OUTPUT_DIR}/${CASE_NAME}.nii.gz"

    if [[ ! -f "${SOURCE_MASK}" ]]; then
        echo "Missing generated consensus mask for case ${CASE_NAME}: ${SOURCE_MASK}" >&2
        exit 1
    fi

    for ANNOTATION in annotation_1.nii.gz annotation_2.nii.gz annotation_3.nii.gz; do
        if [[ ! -f "${CASE_DIR}/${ANNOTATION}" ]]; then
            echo "Missing ${ANNOTATION} for case ${CASE_NAME}: ${CASE_DIR}/${ANNOTATION}" >&2
            exit 1
        fi
    done
done

mkdir -p "${BACKUP_DIR}"
OLD_MASKS=("${CASES_ROOT}"/*/agreement_mask.nii.gz)
if (( ${#OLD_MASKS[@]} > 0 )); then
    printf 'Moving existing testing-set agreement masks to backup: %s\n' "${BACKUP_DIR}"
    for OLD_MASK in "${OLD_MASKS[@]}"; do
        CASE_DIR="${OLD_MASK%/agreement_mask.nii.gz}"
        CASE_NAME="${CASE_DIR##*/}"
        mkdir -p "${BACKUP_DIR}/${CASE_NAME}"
        mv "${OLD_MASK}" "${BACKUP_DIR}/${CASE_NAME}/agreement_mask.nii.gz"
    done
else
    printf 'No existing testing-set agreement masks found to move.\n'
fi

printf 'Moving generated weighted consensus masks into testing_set case folders.\n'
for CASE_DIR in "${CASE_DIRS[@]}"; do
    if [[ ! -d "${CASE_DIR}" ]]; then
        continue
    fi

    CASE_NAME="${CASE_DIR##*/}"
    NEW_MASK="${CONSENSUS_OUTPUT_DIR}/${CASE_NAME}.nii.gz"
    TARGET_MASK="${CASES_ROOT}/${CASE_NAME}/agreement_mask.nii.gz"
    mv "${NEW_MASK}" "${TARGET_MASK}"
done

EVAL_CMD=(
    python -u evaluation/evaluate_testing_set_human_agreement.py
    --output-dir "${EVAL_OUTPUT_DIR}"
    --cases-root "${CASES_ROOT}"
    --annotations
    annotation_1.nii.gz
    annotation_2.nii.gz
    annotation_3.nii.gz
    agreement_mask.nii.gz
)
EVAL_CMD+=("${EVAL_EXTRA_ARGS[@]}")

printf 'Evaluation extra args:'
printf ' %q' "${EVAL_EXTRA_ARGS[@]}"
printf '\n'

"${EVAL_CMD[@]}"

echo "Completed weighted consensus install and four-mask agreement evaluation."
echo "Installed agreement masks under: ${CASES_ROOT}"
echo "Previous agreement-mask backup: ${BACKUP_DIR}"
echo "Evaluation output directory: ${EVAL_OUTPUT_DIR}"
echo "Confidence masks remain in: ${CONFIDENCE_DIR}"
