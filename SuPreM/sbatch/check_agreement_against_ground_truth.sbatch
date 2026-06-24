#!/bin/bash
#SBATCH --job-name=check_gt_agreement
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.err

set -euo pipefail

# Compare configurable inference-mask directories against WORD labelsTr and
# write CSV summaries. This is CPU-only; it reads NIfTI masks and counts voxels.
#
# Usage:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/check_agreement_against_ground_truth.sbatch
#
# Optional output directory and model specs:
#   sbatch sbatch/check_agreement_against_ground_truth.sbatch \
#     results/my_gt_agreement_csvs \
#     swin5050=results/word_swinunetr_5050:btcv \
#     clip_unet=results/word_clip_universal_unet \
#     segresnet=results/word_three_models/segresnet

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
OUTPUT_DIR="${1:-${PROJECT_DIR}/results/word_swin5050_clipunet_segresnet_gt_agreement}"
if (( $# > 0 )); then
    shift
fi

if (( $# > 0 )); then
    MODEL_SPECS=("$@")
else
    MODEL_SPECS=(
        "swin5050=results/word_swinunetr_5050:btcv"
        "clip_unet=results/word_clip_universal_unet"
        "segresnet=results/word_three_models/segresnet"
    )
fi

if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
fi

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_DIR}"

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

CMD=(
    python -u ensemble_agreement/check_agreement_against_ground_truth.py
    --output-dir "${OUTPUT_DIR}"
)

for MODEL_SPEC in "${MODEL_SPECS[@]}"; do
    CMD+=(--model "${MODEL_SPEC}")
done

printf 'Model specs:\n'
printf '  %s\n' "${MODEL_SPECS[@]}"

"${CMD[@]}"

echo "Completed ground-truth agreement check."
echo "CSV output directory: ${OUTPUT_DIR}"
