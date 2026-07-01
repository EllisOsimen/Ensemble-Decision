#!/usr/bin/env python3
"""Copy image geometry onto a label mask without resampling the mask data.

Use this only when the mask voxel array already matches the image voxel array
index-for-index, but the mask NIfTI header/affine is wrong. For example, this
fits UKCHLL007 annotation_2: it has the same shape as the CT and similar label
voxel extents to the other annotations, but its affine places it far away.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_DIR = PROJECT_DIR.parent / "testing_set" / "UKCHLL007"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fix a mask NIfTI header by copying geometry from a reference image. "
            "This does not resample or move voxels inside the array."
        )
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_CASE_DIR / "image.nii.gz",
        help="Reference image with the geometry to copy.",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=DEFAULT_CASE_DIR / "annotation_2.nii.gz",
        help="Input mask whose voxel data should be kept.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CASE_DIR / "annotation_2_corrected.nii.gz",
        help="Output path for the geometry-corrected mask.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    parser.add_argument(
        "--dtype",
        default="uint8",
        help="Integer dtype for the saved labels, e.g. uint8 or int16.",
    )
    return parser.parse_args()


def load_integer_labels(mask_path: Path, dtype: str, atol: float = 1e-3) -> np.ndarray:
    """Load a mask, verify labels are integer-like, and save them compactly."""

    mask_img = nib.load(str(mask_path))
    data = np.asanyarray(mask_img.dataobj)

    if np.issubdtype(data.dtype, np.integer):
        labels = data
    else:
        rounded = np.rint(data)
        if not np.all(np.isclose(data, rounded, rtol=0.0, atol=atol)):
            raise ValueError(f"{mask_path} contains non-integer label values.")
        labels = rounded

    return labels.astype(np.dtype(dtype), copy=False)


def copy_geometry(reference_path: Path, mask_path: Path, output_path: Path, dtype: str, overwrite: bool) -> None:
    reference_img = nib.load(str(reference_path))
    mask_img = nib.load(str(mask_path))

    if reference_img.shape != mask_img.shape:
        raise ValueError(
            "This script only fixes headers for same-shape images. "
            f"Reference shape {reference_img.shape} != mask shape {mask_img.shape}. "
            "Use nearest-neighbour resampling instead."
        )

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {output_path}")

    labels = load_integer_labels(mask_path, dtype)

    # Start from the reference header so spacing, orientation metadata, qform,
    # and sform are all consistent with the CT image.
    header = reference_img.header.copy()
    header.set_data_dtype(labels.dtype)

    corrected = nib.Nifti1Image(labels, reference_img.affine, header)
    qform, qform_code = reference_img.get_qform(coded=True)
    sform, sform_code = reference_img.get_sform(coded=True)
    if qform is not None:
        corrected.set_qform(qform, int(qform_code))
    if sform is not None:
        corrected.set_sform(sform, int(sform_code))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(corrected, str(output_path))

    print(f"Reference: {reference_path}")
    print(f"Mask: {mask_path}")
    print(f"Output: {output_path}")
    print(f"Shape: {labels.shape}")
    print(f"Saved dtype: {labels.dtype}")
    print("Copied reference affine/header geometry without resampling voxel data.")


def main() -> None:
    args = parse_args()
    copy_geometry(
        reference_path=args.reference,
        mask_path=args.mask,
        output_path=args.output,
        dtype=args.dtype,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
