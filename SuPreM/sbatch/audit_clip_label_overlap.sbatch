#!/bin/bash
#SBATCH --job-name=clip-overlap-audit
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
OUTPUT_JSON="${1:-results/clip_overlap_audit/representative_10_cases.json}"

if [[ "${OUTPUT_JSON}" != /* ]]; then
    OUTPUT_JSON="${PROJECT_DIR}/${OUTPUT_JSON}"
fi

mkdir -p "${PROJECT_DIR}/slurm_logs" "$(dirname "${OUTPUT_JSON}")"

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

python -c "import nibabel, monai, numpy, torch" || {
    echo "The suprem-h200 environment is missing required dependencies." >&2
    exit 1
}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u inference/audit_clip_label_overlap.py \
    --image \
        /home/s2347484/Seg/training_set/training_set/UKCHLL001/image.nii.gz \
        /home/s2347484/Seg/training_set/training_set/UKCHLL023/image.nii.gz \
        /home/s2347484/Seg/training_set/training_set/UKCHLL034/image.nii.gz \
        /home/s2347484/Seg/validation_set/UKCHLL049/image.nii.gz \
        /home/s2347484/Seg/validation_set/UKCHLL058/image.nii.gz \
        /home/s2347484/Seg/testing_set/UKCHLL003/image.nii.gz \
        /home/s2347484/Seg/testing_set/UKCHLL032/image.nii.gz \
        /home/s2347484/Seg/testing_set/UKCHLL064/image.nii.gz \
        /home/s2347484/Seg/testing_set/UKCHLL095/image.nii.gz \
        /home/s2347484/Seg/testing_set/UKCHLL112/image.nii.gz \
    --output-json "${OUTPUT_JSON}" \
    --checkpoint pretrained_weights/supervised_clip_driven_universal_unet_2100.pth \
    --device cuda \
    --sw-batch-size 1 \
    --threshold 0.5

echo "Completed CLIP overlap audit."
echo "Report: ${OUTPUT_JSON}"
