#!/bin/bash
#SBATCH --job-name=suprem-word
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
#   sbatch evaluate_word.sbatch [unet|swinunetr|segresnet] [optional_case_limit]
#
# Examples:
#   sbatch evaluate_word.sbatch unet
#   sbatch evaluate_word.sbatch swinunetr 1

BACKBONE="${1:-unet}"
CASE_LIMIT="${2:-}"
PROJECT_DIR="/home/s2347484/Seg/SuPreM"

case "${BACKBONE}" in
    unet)
        CHECKPOINT="pretrained_weights/supervised_suprem_unet_2100.pth"
        ;;
    swinunetr)
        CHECKPOINT="pretrained_weights/supervised_suprem_swinunetr_2100.pth"
        ;;
    segresnet)
        CHECKPOINT="pretrained_weights/supervised_suprem_segresnet_2100.pth"
        ;;
    *)
        echo "Unknown backbone: ${BACKBONE}" >&2
        echo "Choose unet, swinunetr, or segresnet." >&2
        exit 2
        ;;
esac

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Backbone: ${BACKBONE}"
echo "Checkpoint: ${CHECKPOINT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi

# Fail early with a useful message if the evaluation dependencies have not
# yet been installed in the suprem environment.
python -c "import nibabel, surface_distance" || {
    echo "Missing dependencies. Before resubmitting, run:" >&2
    echo "  conda activate suprem-h200" >&2
    echo "  pip install nibabel surface-distance" >&2
    exit 1
}

# PyTorch checkpoints are ZIP archives containing many tensor entries.
# Loading them directly from the shared cluster filesystem can result in slow,
# non-sequential I/O. Copy the selected checkpoint to node-local storage first.
LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/${USER}/slurm-${SLURM_JOB_ID}}"
mkdir -p "${LOCAL_SCRATCH}"
LOCAL_CHECKPOINT="${LOCAL_SCRATCH}/$(basename "${CHECKPOINT}")"
echo "Staging checkpoint to node-local storage: ${LOCAL_CHECKPOINT}"
STAGE_START="${SECONDS}"
cp "${CHECKPOINT}" "${LOCAL_CHECKPOINT}"
echo "Checkpoint staged in $((SECONDS - STAGE_START)) seconds"

OUTPUT_DIR="results/word_${BACKBONE}"
COMMAND=(
    python -u evaluate_word.py
    --device cuda
    --backbone "${BACKBONE}"
    --checkpoint "${LOCAL_CHECKPOINT}"
    --output-dir "${OUTPUT_DIR}"
    --num-workers "${SLURM_CPUS_PER_TASK}"
    # Use one patch at a time so this also fits the 16 GB H200 MIG profile
    # that may be assigned by --gres=gpu:1.
    --sw-batch-size 1
)

if [[ -n "${CASE_LIMIT}" ]]; then
    COMMAND+=(--limit "${CASE_LIMIT}")
fi

echo "Starting evaluation: ${COMMAND[*]}"
"${COMMAND[@]}"
echo "Evaluation complete. Results: ${PROJECT_DIR}/${OUTPUT_DIR}"
