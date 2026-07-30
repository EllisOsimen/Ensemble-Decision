#!/bin/bash
#SBATCH --job-name=totalseg-prepare
#SBATCH --partition=Teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

# Build the fixed 50-case TotalSegmentator cohort used for external evaluation.
# CTs and segmentation directories are symlinked to avoid duplicating the
# dataset; combined_mask.nii.gz is generated in CURVAS label space.
#
# Run from SuPreM/:
#   sbatch sbatch/totalsegmentator/prepare_totalsegmentator_external_50.sbatch
#
# Optional arguments:
#   sbatch sbatch/totalsegmentator/prepare_totalsegmentator_external_50.sbatch \
#     [TOTALSEGMENTATOR_ROOT] [OUTPUT_ROOT]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_DIR="$(cd -- "${PROJECT_DIR}/.." && pwd)"
SOURCE_ROOT="${1:-${REPO_DIR}/Totalsegmentator_dataset}"
OUTPUT_ROOT="${2:-${REPO_DIR}/external_dataset}"
CASE_LIST="${SCRIPT_DIR}/totalsegmentator_external_50.txt"
OVERWRITE="${OVERWRITE:-0}"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
    OVERWRITE_ARGS+=(--overwrite)
elif [[ "${OVERWRITE}" != "0" ]]; then
    echo "OVERWRITE must be 0 or 1; got ${OVERWRITE}" >&2
    exit 1
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-suprem-h200}"
cd "${PROJECT_DIR}"

python -u utils/prepare_totalsegmentator_external_subset.py \
    --source-root "${SOURCE_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --case-list "${CASE_LIST}" \
    "${OVERWRITE_ARGS[@]}"
