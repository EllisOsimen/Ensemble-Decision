#!/bin/bash
#SBATCH --job-name=eval-word-two
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.err

set -euo pipefail

# Evaluate existing Swin 5050 and CLIP Universal U-Net predictions against WORD.
#
# Usage:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/evaluate_word_swin5050_clip_unet.sbatch
#
# Optional output directory:
#   sbatch sbatch/evaluate_word_swin5050_clip_unet.sbatch results/my_word_eval

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
OUTPUT_DIR="${1:-results/word_swin5050_clip_unet_evaluation}"

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

python -u evaluation/evaluate_word_prediction_dirs.py \
    --output-dir "${OUTPUT_DIR}" \
    --model swin5050=results/word_swinunetr_5050:btcv \
    --model clip_unet=results/word_clip_universal_unet

echo "Completed WORD evaluation."
echo "CSV output directory: ${OUTPUT_DIR}"
