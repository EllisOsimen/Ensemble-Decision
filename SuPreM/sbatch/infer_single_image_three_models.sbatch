#!/bin/bash
#SBATCH --job-name=suprem-single
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.err

set -euo pipefail

# Usage:
#   sbatch infer_single_image_three_models.sbatch INPUT_CT [OUTPUT_DIR]
#
# Example:
#   sbatch infer_single_image_three_models.sbatch \
#     /path/to/ct.nii.gz \
#     results/single_image

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
INPUT_CT="${1:-}"
OUTPUT_DIR="${2:-results/single_image}"
CHECKPOINT_DIR="${PROJECT_DIR}/pretrained_weights"

if [[ -z "${INPUT_CT}" ]]; then
    echo "Usage: sbatch $0 INPUT_CT [OUTPUT_DIR]" >&2
    exit 2
fi

if [[ ! -f "${INPUT_CT}" ]]; then
    echo "Input CT does not exist: ${INPUT_CT}" >&2
    exit 2
fi

mkdir -p "${PROJECT_DIR}/slurm_logs"

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Input CT: ${INPUT_CT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi

python -c "import nibabel, monai, torch" || {
    echo "The suprem-h200 environment is missing required dependencies." >&2
    exit 1
}

# Checkpoints contain many tensor entries and can load slowly from shared
# storage. Stage all three files in node-local storage before inference.
LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/${USER}/slurm-${SLURM_JOB_ID}}"
LOCAL_CHECKPOINT_DIR="${LOCAL_SCRATCH}/pretrained_weights"
mkdir -p "${LOCAL_CHECKPOINT_DIR}"

for CHECKPOINT in \
    supervised_suprem_unet_2100.pth \
    supervised_suprem_swinunetr_2100.pth \
    supervised_suprem_segresnet_2100.pth
do
    SOURCE="${CHECKPOINT_DIR}/${CHECKPOINT}"
    if [[ ! -f "${SOURCE}" ]]; then
        echo "Missing checkpoint: ${SOURCE}" >&2
        exit 1
    fi
    echo "Staging ${CHECKPOINT}..."
    cp "${SOURCE}" "${LOCAL_CHECKPOINT_DIR}/${CHECKPOINT}"
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u infer_single_image_three_models.py \
    --image "${INPUT_CT}" \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint-dir "${LOCAL_CHECKPOINT_DIR}" \
    --device cuda \
    --sw-batch-size 1

echo "Inference complete."
echo "Results: ${PROJECT_DIR}/${OUTPUT_DIR}"
