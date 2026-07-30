#!/bin/bash
#SBATCH --job-name=rf-stacking
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.err

set -euo pipefail

# Train a supervised voxel-wise random forest stacking model with assigned-label
# confidence features and apply it to the CURVAS validation split.
#
# Usage:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/legacy/random_forest_stacking_consensus.sbatch
#
# Optional arguments:
#   sbatch sbatch/legacy/random_forest_stacking_consensus.sbatch \
#     [TRAIN_PREDICTION_ROOT] [TRAIN_CASES_ROOT] [VAL_PREDICTION_ROOT] \
#     [VAL_CASES_ROOT] [MODEL_OUTPUT] [OUTPUT_DIR] [EXTRA_ARGS...]
#
# Defaults:
#   TRAIN_PREDICTION_ROOT -> results/all_curvas_inference_with_confidence/training
#   TRAIN_CASES_ROOT      -> /home/s2347484/Seg/training_set/training_set
#   VAL_PREDICTION_ROOT   -> results/all_curvas_inference_with_confidence/validation
#   VAL_CASES_ROOT        -> /home/s2347484/Seg/validation_set
#   MODEL_OUTPUT          -> results/all_curvas_inference_with_confidence/random_forest_stacking_with_confidence/rf_stacking.joblib
#   OUTPUT_DIR            -> results/all_curvas_inference_with_confidence/random_forest_stacking_with_confidence/validation_predictions
#
# EXTRA_ARGS are forwarded to random_forest_stacking_consensus.py, e.g.:
#   --samples-per-class 500 --background-samples-per-case 1000
#   --n-estimators 20 --max-depth 8
#   --predict-chunk-size 500000

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
TRAIN_PREDICTION_ROOT="${1:-results/all_curvas_inference_with_confidence/training}"
TRAIN_CASES_ROOT="${2:-/home/s2347484/Seg/training_set/training_set}"
VAL_PREDICTION_ROOT="${3:-results/all_curvas_inference_with_confidence/validation}"
VAL_CASES_ROOT="${4:-/home/s2347484/Seg/validation_set}"
MODEL_OUTPUT="${5:-results/all_curvas_inference_with_confidence/random_forest_stacking_with_confidence/rf_stacking.joblib}"
OUTPUT_DIR="${6:-results/all_curvas_inference_with_confidence/random_forest_stacking_with_confidence/validation_predictions}"

for _ in 1 2 3 4 5 6; do
    if (( $# > 0 )); then
        shift
    fi
done
EXTRA_ARGS=("$@")

make_project_path_absolute() {
    local path="$1"
    if [[ "${path}" == /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s/%s\n' "${PROJECT_DIR}" "${path}"
    fi
}

TRAIN_PREDICTION_ROOT="$(make_project_path_absolute "${TRAIN_PREDICTION_ROOT}")"
TRAIN_CASES_ROOT="$(make_project_path_absolute "${TRAIN_CASES_ROOT}")"
VAL_PREDICTION_ROOT="$(make_project_path_absolute "${VAL_PREDICTION_ROOT}")"
VAL_CASES_ROOT="$(make_project_path_absolute "${VAL_CASES_ROOT}")"
MODEL_OUTPUT="$(make_project_path_absolute "${MODEL_OUTPUT}")"
OUTPUT_DIR="$(make_project_path_absolute "${OUTPUT_DIR}")"

mkdir -p \
    "${PROJECT_DIR}/slurm_logs" \
    "$(dirname "${MODEL_OUTPUT}")" \
    "${OUTPUT_DIR}"

for REQUIRED_DIR in \
    "${TRAIN_PREDICTION_ROOT}" \
    "${TRAIN_CASES_ROOT}" \
    "${VAL_PREDICTION_ROOT}" \
    "${VAL_CASES_ROOT}"
do
    if [[ ! -d "${REQUIRED_DIR}" ]]; then
        echo "Required directory not found: ${REQUIRED_DIR}" >&2
        exit 1
    fi
done

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

python -c "import joblib, nibabel, numpy, scipy, sklearn, tqdm" || {
    echo "The suprem-h200 environment is missing required random-forest stacking dependencies." >&2
    exit 1
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

CMD=(
    python -u ensemble_agreement/random_forest_stacking_consensus.py
    --train-prediction-root "${TRAIN_PREDICTION_ROOT}"
    --train-cases-root "${TRAIN_CASES_ROOT}"
    --val-prediction-root "${VAL_PREDICTION_ROOT}"
    --val-cases-root "${VAL_CASES_ROOT}"
    --model-dir swinunetr_5050
    --model-id swin5050
    --label-space btcv
    --confidence-dir swinunetr_5050_confidence
    --validity-dir swinunetr_5050_validity
    --model-dir clip_universal_unet
    --model-id clip_unet
    --label-space suprem
    --confidence-dir clip_universal_unet_confidence
    --validity-dir clip_universal_unet_validity
    --model-dir suprem_segresnet
    --model-id segresnet
    --label-space suprem
    --confidence-dir suprem_segresnet_confidence
    --validity-dir suprem_segresnet_validity
    --target-annotation annotation_1.nii.gz
    --samples-per-class 5000
    --background-samples-per-case 10000
    --near-organ-background
    --n-estimators 300
    --max-depth 5
    --min-samples-leaf 1
    --max-features sqrt
    --model-output "${MODEL_OUTPUT}"
    --output-dir "${OUTPUT_DIR}"
    --overwrite
)

CMD+=("${EXTRA_ARGS[@]}")

printf 'Node: %s\n' "$(hostname)"
printf 'Training prediction root: %s\n' "${TRAIN_PREDICTION_ROOT}"
printf 'Training cases root: %s\n' "${TRAIN_CASES_ROOT}"
printf 'Validation prediction root: %s\n' "${VAL_PREDICTION_ROOT}"
printf 'Validation cases root: %s\n' "${VAL_CASES_ROOT}"
printf 'Model output: %s\n' "${MODEL_OUTPUT}"
printf 'Validation output dir: %s\n' "${OUTPUT_DIR}"
printf 'Extra args:'
printf ' %q' "${EXTRA_ARGS[@]}"
printf '\n'

"${CMD[@]}"

echo "Completed random forest stacking training and validation inference."
echo "Model: ${MODEL_OUTPUT}"
echo "Metadata: ${MODEL_OUTPUT%.*}.metadata.json"
echo "Validation fused masks: ${OUTPUT_DIR}"
