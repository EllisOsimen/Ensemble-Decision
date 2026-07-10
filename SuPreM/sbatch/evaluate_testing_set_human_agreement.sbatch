#!/bin/bash
#SBATCH --job-name=human-agreement
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.err

set -euo pipefail

# Compute agreement across testing_set annotations, optionally including
# external prediction directories as extra raters.
#
# Usage:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/evaluate_testing_set_human_agreement.sbatch
#
# Optional arguments:
#   sbatch sbatch/evaluate_testing_set_human_agreement.sbatch [OUTPUT_DIR] [CASES_ROOT] [EXTRA_ARGS...]
#
# Examples:
#   # Full run with foreground NSD:
#   sbatch sbatch/evaluate_testing_set_human_agreement.sbatch
#
#   # Faster Dice/kappa-only run:
#   sbatch sbatch/evaluate_testing_set_human_agreement.sbatch \
#     results/testing_set_human_agreement_no_nsd \
#     /home/s2347484/Seg/testing_set \
#     --skip-nsd
#
#   # Compare human annotations with an external prediction directory:
#   sbatch sbatch/evaluate_testing_set_human_agreement.sbatch \
#     results/testing_set_human_plus_weighted_65case \
#     /home/s2347484/Seg/testing_set \
#     --prediction-dir weighted=results/CURVAS_INFERENCE/agreement_masks_weighted_65case \
#     --skip-nsd

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
OUTPUT_DIR="${1:-results/testing_set_human_agreement}"
CASES_ROOT="${2:-/home/s2347484/Seg/testing_set}"

if (( $# > 0 )); then
    shift
fi
if (( $# > 0 )); then
    shift
fi
EXTRA_ARGS=("$@")

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

python -u evaluation/evaluate_testing_set_human_agreement.py \
    --output-dir "${OUTPUT_DIR}" \
    --cases-root "${CASES_ROOT}" \
    "${EXTRA_ARGS[@]}"

echo "Completed testing-set human agreement evaluation."
echo "Output directory: ${OUTPUT_DIR}"
