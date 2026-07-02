#!/bin/bash
#SBATCH --job-name=curvas-agree
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.err

set -euo pipefail

# Compare CURVAS inference prediction masks with each other.
#
# Usage:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/evaluate_CURVAS_prediction_mask_agreement.sbatch
#
# Optional arguments:
#   sbatch sbatch/evaluate_CURVAS_prediction_mask_agreement.sbatch [OUTPUT_DIR] [MODEL_SPEC...]
#
# Defaults:
#   OUTPUT_DIR -> results/CURVAS_prediction_mask_agreement
#   MODEL_SPEC ->
#     swin5050=results/CURVAS_INFERENCE/swinunetr_5050:btcv
#     clip_unet=results/CURVAS_INFERENCE/clip_universal_unet:suprem
#     segresnet=results/CURVAS_INFERENCE/suprem_segresnet:suprem

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
OUTPUT_DIR="${1:-results/CURVAS_prediction_mask_agreement}"

if (( $# > 0 )); then
    shift
fi

if (( $# > 0 )); then
    MODEL_SPECS=("$@")
else
    MODEL_SPECS=(
        "swin5050=results/CURVAS_INFERENCE/swinunetr_5050:btcv"
        "clip_unet=results/CURVAS_INFERENCE/clip_universal_unet:suprem"
        "segresnet=results/CURVAS_INFERENCE/suprem_segresnet:suprem"
    )
fi

if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
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
    python -u evaluation/evaluate_prediction_mask_agreement.py
    --output-dir "${OUTPUT_DIR}"
)

for MODEL_SPEC in "${MODEL_SPECS[@]}"; do
    CMD+=(--model "${MODEL_SPEC}")
done

printf 'Model specs:\n'
printf '  %s\n' "${MODEL_SPECS[@]}"
printf 'Output dir: %s\n' "${OUTPUT_DIR}"

"${CMD[@]}"

echo "Completed CURVAS prediction-mask agreement evaluation."
echo "Output directory: ${OUTPUT_DIR}"
