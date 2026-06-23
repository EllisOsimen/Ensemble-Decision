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

# Compare all four inference-mask directories against WORD labelsTr and write
# CSV summaries. This is CPU-only; it just reads NIfTI masks and counts voxels.
#
# Usage:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/check_agreement_against_ground_truth.sbatch
#
# Optional output directory:
#   sbatch sbatch/check_agreement_against_ground_truth.sbatch results/my_gt_agreement_csvs

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
OUTPUT_DIR="${1:-${PROJECT_DIR}/results/word_four_model_ground_truth_agreement}"

if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
fi

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_DIR}"

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

python -u ensemble_agreement/check_agreement_against_ground_truth.py \
    --output-dir "${OUTPUT_DIR}"

echo "Completed ground-truth agreement check."
echo "CSV output directory: ${OUTPUT_DIR}"
