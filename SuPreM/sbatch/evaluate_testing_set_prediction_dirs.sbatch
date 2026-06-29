#!/bin/bash
#SBATCH --job-name=eval-test-3
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.err

set -euo pipefail

# Evaluate prediction directories against testing_set annotations.
#
# Usage:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/evaluate_testing_set_prediction_dirs.sbatch
#
# Optional arguments:
#   sbatch sbatch/evaluate_testing_set_prediction_dirs.sbatch [OUTPUT_DIR] [CASES_ROOT] [MODEL_SPEC...]
#
# Defaults:
#   OUTPUT_DIR -> results/testing_set_prediction_evaluation
#   CASES_ROOT -> /home/s2347484/Seg/testing_set
#   MODEL_SPEC ->
#     swin5050=results/eval_inference/swinunetr_5050:btcv
#     clip_unet=results/eval_inference/clip_universal_unet:suprem
#     segresnet=results/eval_inference/suprem_segresnet:suprem

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
OUTPUT_DIR="${1:-results/testing_set_prediction_evaluation}"
CASES_ROOT="${2:-/home/s2347484/Seg/testing_set}"

if (( $# > 0 )); then
    shift
fi
if (( $# > 0 )); then
    shift
fi

if (( $# > 0 )); then
    MODEL_SPECS=("$@")
else
    MODEL_SPECS=(
        "swin5050=results/eval_inference/swinunetr_5050:btcv"
        "clip_unet=results/eval_inference/clip_universal_unet:suprem"
        "segresnet=results/eval_inference/suprem_segresnet:suprem"
    )
fi

if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
fi

if [[ "${CASES_ROOT}" != /* ]]; then
    CASES_ROOT="${PROJECT_DIR}/${CASES_ROOT}"
fi

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_DIR}"

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

python -c "import nibabel, numpy, surface_distance, tqdm" || {
    echo "The suprem-h200 environment is missing required dependencies." >&2
    exit 1
}

CMD=(
    python -u evaluation/evaluate_testing_set_prediction_dirs.py
    --output-dir "${OUTPUT_DIR}"
    --cases-root "${CASES_ROOT}"
)

for MODEL_SPEC in "${MODEL_SPECS[@]}"; do
    CMD+=(--model "${MODEL_SPEC}")
done

printf 'Model specs:\n'
printf '  %s\n' "${MODEL_SPECS[@]}"
printf 'Cases root: %s\n' "${CASES_ROOT}"
printf 'Output dir: %s\n' "${OUTPUT_DIR}"

"${CMD[@]}"

echo "Completed testing-set evaluation."
echo "Output directory: ${OUTPUT_DIR}"
