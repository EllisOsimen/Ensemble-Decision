#!/usr/bin/env python3

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Inspect raw label values inside a NIfTI mask."
    )
    parser.add_argument("mask", help="Path to .nii or .nii.gz mask")
    args = parser.parse_args()

    mask_path = Path(args.mask)

    if not mask_path.exists():
        raise FileNotFoundError(f"File not found: {mask_path}")

    img = nib.load(str(mask_path))

    # Read raw array without forcing float64 unless needed
    data = np.asanyarray(img.dataobj)

    print(f"\nMask: {mask_path}")
    print(f"Shape: {data.shape}")
    print(f"Data dtype: {data.dtype}")
    print(f"Min value: {np.nanmin(data)}")
    print(f"Max value: {np.nanmax(data)}")

    labels, counts = np.unique(data, return_counts=True)

    print("\nRaw unique values:")
    print(f"{'Value':<20}{'Voxel Count'}")
    print("-" * 35)

    for value, count in zip(labels, counts):
        print(f"{str(value):<20}{count}")

    # Also inspect rounded integer labels
    rounded = np.rint(data).astype(np.int32)
    rounded_labels, rounded_counts = np.unique(rounded, return_counts=True)

    print("\nRounded integer labels:")
    print(f"{'Label':<10}{'Voxel Count'}")
    print("-" * 25)

    for label, count in zip(rounded_labels, rounded_counts):
        print(f"{label:<10}{count}")

    # Warn if values are not close to integers
    if not np.allclose(data, rounded):
        print("\nWARNING: Some values are not exact integers.")
        print("This file may not be a clean integer labelmap.")


if __name__ == "__main__":
    main()