#!/bin/bash
#SBATCH --job-name=external-base-eval
#SBATCH --partition=Teaching
#SBATCH --exclude=damnii12
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

# Independently evaluate each external-dataset base model against the single
# combined_mask.nii.gz reference annotation:
#   task 0: CLIP Universal U-Net (native SuPreM/WORD labels)
#   task 1: SuPreM SegResNet      (native SuPreM/WORD labels)
#   task 2: Swin UNETR 5050      (native BTCV labels)
#
# Each task computes per-organ Dice, 1 mm NSD, binary Cohen kappa, and
# whole-map multiclass Cohen kappa. Fleiss kappa is disabled because the
# external dataset has one reference annotation.
#
# Submit with defaults:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/evaluate_external_dataset_base_models.sbatch
#
# Optional positional arguments:
#   sbatch sbatch/evaluate_external_dataset_base_models.sbatch \
#     [PREDICTION_ROOT] [CASES_ROOT] [OUTPUT_ROOT]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${PROJECT_DIR}/.." && pwd)"
PREDICTION_ROOT="${1:-${PROJECT_DIR}/results/external_dataset_inference_with_confidence}"
CASES_ROOT="${2:-${REPO_DIR}/external_dataset}"
OUTPUT_ROOT="${3:-${PROJECT_DIR}/results/external_dataset_base_model_evaluation}"
ANNOTATION_NAME="combined_mask.nii.gz"
EXPECTED_CASES=50

MODEL_NAMES=(clip_unet segresnet swin5050)
MODEL_DIRS=(clip_universal_unet suprem_segresnet swinunetr_5050)
LABEL_SPACES=(suprem suprem btcv)

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array job.}"
if (( TASK_ID < 0 || TASK_ID >= ${#MODEL_NAMES[@]} )); then
    echo "Array task ${TASK_ID} is outside the expected range 0-2." >&2
    exit 1
fi

MODEL_NAME="${MODEL_NAMES[TASK_ID]}"
MODEL_DIR="${PREDICTION_ROOT}/${MODEL_DIRS[TASK_ID]}"
LABEL_SPACE="${LABEL_SPACES[TASK_ID]}"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}"

if [[ ! -d "${CASES_ROOT}" ]]; then
    echo "External cases root not found: ${CASES_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "${MODEL_NAME} prediction directory not found: ${MODEL_DIR}" >&2
    exit 1
fi

mapfile -d '' CASE_DIRS < <(
    find "${CASES_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)
if (( ${#CASE_DIRS[@]} != EXPECTED_CASES )); then
    echo "Expected ${EXPECTED_CASES} external cases; found ${#CASE_DIRS[@]}." >&2
    exit 1
fi

# Strict one-to-one case check prevents evaluation on an accidental subset.
for CASE_DIR in "${CASE_DIRS[@]}"; do
    CASE_NAME="$(basename "${CASE_DIR}")"
    ANNOTATION="${CASE_DIR}/${ANNOTATION_NAME}"
    PREDICTION="${MODEL_DIR}/${CASE_NAME}.nii.gz"

    if [[ ! -f "${ANNOTATION}" ]]; then
        echo "Missing ground truth: ${ANNOTATION}" >&2
        exit 1
    fi
    if [[ ! -f "${PREDICTION}" ]]; then
        echo "Missing ${MODEL_NAME} prediction: ${PREDICTION}" >&2
        exit 1
    fi
done

PREDICTION_COUNT=$(find "${MODEL_DIR}" -maxdepth 1 -type f \
    -name '*.nii.gz' | wc -l)
if (( PREDICTION_COUNT != EXPECTED_CASES )); then
    echo "Expected ${EXPECTED_CASES} ${MODEL_NAME} predictions; found ${PREDICTION_COUNT}." >&2
    exit 1
fi

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_DIR}"

source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-suprem-h200}"
cd "${PROJECT_DIR}"

python -c "import nibabel, numpy, surface_distance, tqdm" || {
    echo "The suprem-h200 environment is missing required evaluation dependencies." >&2
    exit 1
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Array job/task: ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Model: ${MODEL_NAME}"
echo "Native label space: ${LABEL_SPACE}"
echo "Ground truth: ${CASES_ROOT}/<case>/${ANNOTATION_NAME}"
echo "Predictions: ${MODEL_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "NSD tolerance: 1.0 mm"
echo "Fleiss kappa: disabled"

python -u evaluation/evaluate_testing_set_human_agreement.py \
    --cases-root "${CASES_ROOT}" \
    --annotations "${ANNOTATION_NAME}" \
    --prediction-dir "${MODEL_NAME}=${MODEL_DIR}:${LABEL_SPACE}" \
    --output-dir "${OUTPUT_DIR}" \
    --nsd-tolerance-mm 1.0 \
    --skip-fleiss

echo "Completed ${MODEL_NAME} evaluation against ${ANNOTATION_NAME}."
echo "Per-organ summary: ${OUTPUT_DIR}/per_pair_per_class_summary.csv"
echo "Multiclass summary: ${OUTPUT_DIR}/per_pair_multiclass_summary.csv"
