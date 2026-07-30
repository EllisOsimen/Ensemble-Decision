#!/bin/bash
#SBATCH --job-name=rf-final-002
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

# Train the final CURVAS random-forest stacker on all 20 training cases using
# config_002, selected by five-fold patient-level cross-validation on mean
# foreground Dice. This job saves the fitted model and reproducibility metadata
# but deliberately performs no validation or testing inference.
#
# Selected hyperparameters:
#   n_estimators=100, max_depth=5, min_samples_leaf=1,
#   max_features=all, random_state=42.
#
# Sampling/features match the selection run:
#   - 5,000 voxels per foreground organ per case
#   - 10,000 near-organ background voxels per case, dilation radius 5
#   - 12 one-hot base-prediction features + 3 assigned-label confidences
#   - common inference-validity gating for all three models
#
# Submit after all 20 training cases have complete inference outputs:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/train_final_random_forest_config_002.sbatch
#
# To intentionally replace an existing final model:
#   sbatch --export=ALL,OVERWRITE=1 \
#     sbatch/train_final_random_forest_config_002.sbatch
#
# Optional positional arguments:
#   sbatch sbatch/train_final_random_forest_config_002.sbatch \
#     [TRAIN_PREDICTION_ROOT] [TRAIN_CASES_ROOT] [MODEL_OUTPUT]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${PROJECT_DIR}/.." && pwd)"
TRAIN_PREDICTION_ROOT="${1:-${PROJECT_DIR}/results/all_datasets_curvas_inference_with_confidence/training}"
TRAIN_CASES_ROOT="${2:-${REPO_DIR}/training_set/training_set}"
MODEL_OUTPUT="${3:-${PROJECT_DIR}/results/all_datasets_curvas_inference_with_confidence/random_forest_config_002_final/rf_stacking_config_002.joblib}"
SELECTION_FILE="${PROJECT_DIR}/results/all_datasets_curvas_inference_with_confidence/random_forest_cv_confidence_validity_fixed/best_parameters.json"
OVERWRITE="${OVERWRITE:-0}"
EXPECTED_CASES=20

if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
    echo "OVERWRITE must be 0 or 1; got ${OVERWRITE}" >&2
    exit 1
fi
if [[ ! -d "${TRAIN_PREDICTION_ROOT}" ]]; then
    echo "Training prediction root not found: ${TRAIN_PREDICTION_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${TRAIN_CASES_ROOT}" ]]; then
    echo "Training cases root not found: ${TRAIN_CASES_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${SELECTION_FILE}" ]]; then
    echo "Cross-validation selection file not found: ${SELECTION_FILE}" >&2
    exit 1
fi

MODEL_METADATA="${MODEL_OUTPUT%.joblib}.metadata.json"
if [[ "${OVERWRITE}" != "1" \
    && ( -e "${MODEL_OUTPUT}" || -e "${MODEL_METADATA}" ) ]]; then
    echo "Final model or metadata already exists." >&2
    echo "Model: ${MODEL_OUTPUT}" >&2
    echo "Metadata: ${MODEL_METADATA}" >&2
    echo "Set OVERWRITE=1 only if replacing it intentionally." >&2
    exit 1
fi

MODEL_DIRS=(
    clip_universal_unet
    suprem_segresnet
    swinunetr_5050
)
CONFIDENCE_DIRS=(
    clip_universal_unet_confidence
    suprem_segresnet_confidence
    swinunetr_5050_confidence
)
VALIDITY_DIRS=(
    clip_universal_unet_validity
    suprem_segresnet_validity
    swinunetr_5050_validity
)

mapfile -d '' CASE_DIRS < <(
    find "${TRAIN_CASES_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)
if (( ${#CASE_DIRS[@]} != EXPECTED_CASES )); then
    echo "Expected ${EXPECTED_CASES} CURVAS training cases; found ${#CASE_DIRS[@]}." >&2
    exit 1
fi

# Strict case-wise preflight: do not allow training on an incomplete subset.
for CASE_DIR in "${CASE_DIRS[@]}"; do
    CASE_NAME="$(basename "${CASE_DIR}")"
    TARGET="${CASE_DIR}/annotation_1.nii.gz"
    if [[ ! -f "${TARGET}" ]]; then
        echo "Missing training target: ${TARGET}" >&2
        exit 1
    fi

    for INPUT_DIR in \
        "${MODEL_DIRS[@]}" \
        "${CONFIDENCE_DIRS[@]}" \
        "${VALIDITY_DIRS[@]}"
    do
        INPUT="${TRAIN_PREDICTION_ROOT}/${INPUT_DIR}/${CASE_NAME}.nii.gz"
        if [[ ! -f "${INPUT}" ]]; then
            echo "Missing training input: ${INPUT}" >&2
            exit 1
        fi
    done
done

# Counts catch unexpected extra NIfTI files as well as case-name mismatches.
for INPUT_DIR in \
    "${MODEL_DIRS[@]}" \
    "${CONFIDENCE_DIRS[@]}" \
    "${VALIDITY_DIRS[@]}"
do
    INPUT_COUNT=$(find "${TRAIN_PREDICTION_ROOT}/${INPUT_DIR}" \
        -maxdepth 1 -type f -name '*.nii.gz' | wc -l)
    if (( INPUT_COUNT != EXPECTED_CASES )); then
        echo "Expected ${EXPECTED_CASES} files in ${INPUT_DIR}; found ${INPUT_COUNT}." >&2
        exit 1
    fi
done

mkdir -p "${PROJECT_DIR}/slurm_logs" "$(dirname "${MODEL_OUTPUT}")"

source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-suprem-h200}"
cd "${PROJECT_DIR}"

python -c "import joblib, nibabel, numpy, scipy, sklearn, tqdm" || {
    echo "The suprem-h200 environment is missing required random-forest dependencies." >&2
    exit 1
}

# Assert that the preserved selection artifact still describes config_002.
python -c '
import json
import sys
from pathlib import Path

selection = json.loads(Path(sys.argv[1]).read_text())
expected = {
    "config_id": "config_002",
    "parameters": {
        "max_depth": 5,
        "max_features": "all",
        "min_samples_leaf": 1,
        "n_estimators": 100,
    },
}
if selection.get("config_id") != expected["config_id"]:
    raise SystemExit(f"Unexpected selected configuration: {selection}")
if selection.get("parameters") != expected["parameters"]:
    raise SystemExit(f"Unexpected selected parameters: {selection}")
print(f"Validated cross-validation selection: {selection}")
' "${SELECTION_FILE}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Node: $(hostname)"
echo "Training prediction root: ${TRAIN_PREDICTION_ROOT}"
echo "Training cases root: ${TRAIN_CASES_ROOT}"
echo "Training cases: ${EXPECTED_CASES}"
echo "Selected configuration: config_002"
echo "Model output: ${MODEL_OUTPUT}"
echo "Metadata output: ${MODEL_METADATA}"
echo "Overwrite: ${OVERWRITE}"

python -u ensemble_agreement/random_forest_stacking_consensus.py \
    --train-prediction-root "${TRAIN_PREDICTION_ROOT}" \
    --train-cases-root "${TRAIN_CASES_ROOT}" \
    --model-dir clip_universal_unet \
    --model-id clip_unet \
    --label-space suprem \
    --confidence-dir clip_universal_unet_confidence \
    --validity-dir clip_universal_unet_validity \
    --model-dir suprem_segresnet \
    --model-id segresnet \
    --label-space suprem \
    --confidence-dir suprem_segresnet_confidence \
    --validity-dir suprem_segresnet_validity \
    --model-dir swinunetr_5050 \
    --model-id swin5050 \
    --label-space btcv \
    --confidence-dir swinunetr_5050_confidence \
    --validity-dir swinunetr_5050_validity \
    --target-annotation annotation_1.nii.gz \
    --samples-per-class 5000 \
    --background-samples-per-case 10000 \
    --near-organ-background \
    --dilation-iterations 5 \
    --n-estimators 100 \
    --max-depth 5 \
    --min-samples-leaf 1 \
    --max-features all \
    --random-state 42 \
    --model-output "${MODEL_OUTPUT}" \
    --train-only

echo "Completed final config_002 random-forest training."
echo "Model: ${MODEL_OUTPUT}"
echo "Metadata: ${MODEL_METADATA}"
