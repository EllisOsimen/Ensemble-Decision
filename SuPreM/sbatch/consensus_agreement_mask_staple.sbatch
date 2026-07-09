#!/bin/bash
#SBATCH --job-name=staple-full
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.err

set -euo pipefail

# Full CURVAS per-organ binary STAPLE consensus run.
#
# Usage:
#   cd /home/s2347484/Seg/SuPreM
#   sbatch sbatch/consensus_agreement_mask_staple.sbatch
#
# Extra arguments passed to this script are forwarded to
# consensus_agreement_mask.py. For example:
#   sbatch sbatch/consensus_agreement_mask_staple.sbatch --overwrite

PROJECT_DIR="/home/s2347484/Seg/SuPreM"
EXTRA_ARGS=("$@")

mkdir -p "${PROJECT_DIR}/slurm_logs"

source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200
cd "${PROJECT_DIR}"

python -c "import nibabel, numpy, scipy, tqdm" || {
    echo "The suprem-h200 environment is missing required dependencies." >&2
    exit 1
}

CMD=(
    python -u ensemble_agreement/consensus_agreement_mask.py
    results/CURVAS_INFERENCE
    --consensus-mode staple
    --model-dir swinunetr_5050
    --model-id swin5050
    --label-space btcv
    --model-dir clip_universal_unet
    --model-id clip_unet
    --label-space suprem
    --model-dir suprem_segresnet
    --model-id segresnet
    --label-space suprem
    --output-dir results/CURVAS_INFERENCE/agreement_masks_staple
    --confidence-dir results/CURVAS_INFERENCE/agreement_confidence_masks_staple
    --save-staple-probabilities results/CURVAS_INFERENCE/staple_probabilities
)

CMD+=("${EXTRA_ARGS[@]}")

printf 'Consensus mode: staple\n'
printf 'Output label mask dir: results/CURVAS_INFERENCE/agreement_masks_staple\n'
printf 'Confidence mask dir: results/CURVAS_INFERENCE/agreement_confidence_masks_staple\n'
printf 'Probability dir: results/CURVAS_INFERENCE/staple_probabilities\n'
printf 'Extra args:'
printf ' %q' "${EXTRA_ARGS[@]}"
printf '\n'

"${CMD[@]}"

echo "Completed full STAPLE consensus generation."
