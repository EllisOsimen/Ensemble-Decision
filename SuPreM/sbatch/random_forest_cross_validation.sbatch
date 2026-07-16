#!/bin/bash
#SBATCH --job-name=rf-cv-tuning
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.err

set -euo pipefail

# Run patient-level five-fold cross-validation and tune the random-forest
# stacking classifier with the three assigned-label confidence maps. The
# separate five-case validation_set is never read.
#
# Default usage:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/random_forest_cross_validation.sbatch
#
# Optional arguments:
#   sbatch sbatch/random_forest_cross_validation.sbatch \
#     [TRAIN_PREDICTION_ROOT] [TRAIN_CASES_ROOT] [OUTPUT_DIR] [EXTRA_ARGS...]
#
# Example smaller smoke run:
#   sbatch sbatch/random_forest_cross_validation.sbatch \
#     results/all_curvas_inference_with_confidence/training \
#     /home/s2347484/Seg/training_set/training_set \
#     results/all_curvas_inference_with_confidence/random_forest_cross_validation_smoke \
#     --n-estimators-grid 10 --max-depth-grid 5 \
#     --min-samples-leaf-grid 1 --max-features-grid sqrt

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
TRAIN_PREDICTION_ROOT="${1:-results/all_curvas_inference_with_confidence/training}"
TRAIN_CASES_ROOT="${2:-/home/s2347484/Seg/training_set/training_set}"
OUTPUT_DIR="${3:-results/all_curvas_inference_with_confidence/random_forest_cross_validation_with_confidence}"

for _ in 1 2 3; do
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
OUTPUT_DIR="$(make_project_path_absolute "${OUTPUT_DIR}")"

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_DIR}"

for REQUIRED_DIR in "${TRAIN_PREDICTION_ROOT}" "${TRAIN_CASES_ROOT}"; do
    if [[ ! -d "${REQUIRED_DIR}" ]]; then
        echo "Required directory not found: ${REQUIRED_DIR}" >&2
        exit 1
    fi
done

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

python -c "import joblib, nibabel, numpy, scipy, sklearn, tqdm" || {
    echo "The suprem-h200 environment is missing required CV dependencies." >&2
    exit 1
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

CMD=(
    python -u ensemble_agreement/random_forest_cross_validation.py
    --train-prediction-root "${TRAIN_PREDICTION_ROOT}"
    --train-cases-root "${TRAIN_CASES_ROOT}"
    --output-dir "${OUTPUT_DIR}"
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
    --n-splits 5
    --random-state 42
    --n-estimators-grid 100 300
    --max-depth-grid 5 10 none
    --min-samples-leaf-grid 1 50
    --max-features-grid sqrt all
    --overwrite
)

CMD+=("${EXTRA_ARGS[@]}")

printf 'Node: %s\n' "$(hostname)"
printf 'Training prediction root: %s\n' "${TRAIN_PREDICTION_ROOT}"
printf 'Training cases root: %s\n' "${TRAIN_CASES_ROOT}"
printf 'Output directory: %s\n' "${OUTPUT_DIR}"
printf 'Extra args:'
printf ' %q' "${EXTRA_ARGS[@]}"
printf '\n'

"${CMD[@]}"

echo "Completed five-fold random-forest cross-validation and tuning."
echo "Best parameters: ${OUTPUT_DIR}/best_parameters.json"
echo "Configuration ranking: ${OUTPUT_DIR}/configuration_summary.csv"
echo "Per-case metrics: ${OUTPUT_DIR}/per_case_metrics.csv"
