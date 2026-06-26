#!/usr/bin/env python3

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Print unique labels and voxel counts in a NIfTI segmentation mask."
    )
    parser.add_argument("mask", help="Path to the .nii or .nii.gz mask")
    args = parser.parse_args()

    mask_path = Path(args.mask)

    if not mask_path.exists():
        raise FileNotFoundError(f"File not found: {mask_path}")

    img = nib.load(str(mask_path))

    # Load data and round to nearest integer label
    data = img.get_fdata()
    mask_data = np.rint(data).astype(np.int32)

    labels, counts = np.unique(mask_data, return_counts=True)

    print(f"\nMask: {mask_path}")
    print(f"Shape: {mask_data.shape}")
    print(f"Original dtype: {data.dtype}")
    print(f"Rounded dtype: {mask_data.dtype}\n")

    print(f"{'Label':<10}{'Voxel Count'}")
    print("-" * 25)

    for label, count in zip(labels, counts):
        print(f"{label:<10}{count}")


if __name__ == "__main__":
    main()