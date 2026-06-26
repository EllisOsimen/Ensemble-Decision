#!/bin/bash
#SBATCH --job-name=3dircadb_to_nifti
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.out
#SBATCH --error=/home/s2347484/Seg/SuPreM/slurm_logs/%x-%j.err

# Convert all 3Dircadb1 cases from DICOM to NIfTI (.nii.gz).
#
# For each case (3Dircadb1.1 … 3Dircadb1.20) this script:
#   1. Unzips LABELLED_DICOM.zip  → runs dcm2niix → Dataset/<case>/ct.nii.gz
#   2. Unzips MASKS_DICOM.zip     → runs dcm2niix per organ subfolder
#                                  → Dataset/<case>/masks/<organ>.nii.gz
#
# Output layout:
#   Dataset/
#     3Dircadb1.1/
#       ct.nii.gz
#       ct.json             (DICOM metadata sidecar — keep it)
#       masks/
#         artery.nii.gz
#         liver.nii.gz
#         ...
#     3Dircadb1.2/
#       ...
#
# Usage (from /home/s2347484/Seg/SuPreM):
#   sbatch sbatch/convert_3dircadb_to_nifti.sbatch
#
# Override paths:
#   sbatch sbatch/convert_3dircadb_to_nifti.sbatch \
#       /custom/3Dircadb1_root /custom/output_root

set -euo pipefail

DATASET_ROOT="${1:-/home/s2347484/Seg/3Dircadb1}"
OUTPUT_ROOT="${2:-/home/s2347484/Seg/Dataset}"

# ── Temp dir cleaned up automatically on exit ──────────────────────────────
TMPDIR_BASE=$(mktemp -d)
cleanup() {
    echo "Removing temp dir: ${TMPDIR_BASE}"
    rm -rf "${TMPDIR_BASE}"
}
trap cleanup EXIT

# ── Conda env ──────────────────────────────────────────────────────────────
source /opt/conda/etc/profile.d/conda.sh
conda activate suprem-h200

mkdir -p "${OUTPUT_ROOT}"

echo "Source : ${DATASET_ROOT}"
echo "Output : ${OUTPUT_ROOT}"
echo ""

# ── Process each case ──────────────────────────────────────────────────────
for CASE_DIR in "${DATASET_ROOT}"/3Dircadb1.*; do
    [[ -d "${CASE_DIR}" ]] || continue
    CASE_NAME=$(basename "${CASE_DIR}")
    echo "=== ${CASE_NAME} ==="

    OUT_CASE="${OUTPUT_ROOT}/${CASE_NAME}"
    MASKS_OUT="${OUT_CASE}/masks"
    mkdir -p "${OUT_CASE}" "${MASKS_OUT}"

    TMP_CASE="${TMPDIR_BASE}/${CASE_NAME}"
    mkdir -p "${TMP_CASE}"

    # ── 1. CT volume (LABELLED_DICOM) ──────────────────────────────────────
    LABELLED_ZIP="${CASE_DIR}/LABELLED_DICOM.zip"
    if [[ -f "${LABELLED_ZIP}" ]]; then
        TMP_CT="${TMP_CASE}/labelled"
        mkdir -p "${TMP_CT}"
        echo "  Unzipping LABELLED_DICOM…"
        unzip -q "${LABELLED_ZIP}" -d "${TMP_CT}"

        echo "  Converting CT with dcm2niix…"
        dcm2niix -z y -f "ct" -o "${OUT_CASE}" "${TMP_CT}/LABELLED_DICOM"
        echo "  -> ${OUT_CASE}/ct.nii.gz"
    else
        echo "  WARNING: LABELLED_DICOM.zip not found — skipping CT for ${CASE_NAME}"
    fi

    # ── 2. Organ masks (MASKS_DICOM) ───────────────────────────────────────
    MASKS_ZIP="${CASE_DIR}/MASKS_DICOM.zip"
    if [[ -f "${MASKS_ZIP}" ]]; then
        TMP_MASKS="${TMP_CASE}/masks"
        mkdir -p "${TMP_MASKS}"
        echo "  Unzipping MASKS_DICOM…"
        unzip -q "${MASKS_ZIP}" -d "${TMP_MASKS}"

        for ORGAN_DIR in "${TMP_MASKS}/MASKS_DICOM"/*/; do
            [[ -d "${ORGAN_DIR}" ]] || continue
            ORGAN=$(basename "${ORGAN_DIR}")
            dcm2niix -z y -f "${ORGAN}" -o "${MASKS_OUT}" "${ORGAN_DIR}"
            echo "  -> ${MASKS_OUT}/${ORGAN}.nii.gz"
        done
    else
        echo "  WARNING: MASKS_DICOM.zip not found — skipping masks for ${CASE_NAME}"
    fi

    echo "  Done: ${OUT_CASE}"
    echo ""
done

echo "All cases converted."
echo "Output directory: ${OUTPUT_ROOT}"
