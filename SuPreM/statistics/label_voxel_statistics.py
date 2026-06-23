#!/usr/bin/env python3
"""Count voxels and calculate percentages for each label in a NIfTI image."""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Report the number and percentage of voxels belonging to each label "
            "in a .nii or .nii.gz label map."
        )
    )
    parser.add_argument("--image", type=Path, required=True, help="Input label map.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <image_name>_voxel_statistics.csv.",
    )
    parser.add_argument(
        "--exclude-background",
        action="store_true",
        help=(
            "Omit label 0 from the table. Percentages are still calculated using "
            "the total number of image voxels."
        ),
    )
    return parser.parse_args()


def default_csv_path(image_path):
    name = image_path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        raise ValueError(f"Input must end in .nii or .nii.gz: {image_path}")
    return image_path.parent / f"{stem}_voxel_statistics.csv"


def label_statistics(data, exclude_background=False):
    labels, counts = np.unique(data, return_counts=True)
    total_voxels = int(data.size)
    rows = []

    for label, count in zip(labels, counts):
        if exclude_background and label == 0:
            continue
        rows.append(
            {
                "label": label.item(),
                "voxel_count": int(count),
                "percentage_of_total": 100.0 * int(count) / total_voxels,
            }
        )
    return rows, total_voxels


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "voxel_count", "percentage_of_total"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Input label map does not exist: {args.image}")

    image = nib.load(str(args.image))
    data = np.asanyarray(image.dataobj)
    if data.size == 0:
        raise ValueError(f"Input label map is empty: {args.image}")

    rows, total_voxels = label_statistics(data, args.exclude_background)
    output_csv = args.output_csv or default_csv_path(args.image)
    write_csv(output_csv, rows)

    print(f"Image: {args.image}")
    print(f"Shape: {data.shape}")
    print(f"Total voxels: {total_voxels}")
    print()
    print(f"{'Label':>12} {'Voxel count':>15} {'Percentage':>15}")
    print(f"{'-' * 12} {'-' * 15} {'-' * 15}")
    for row in rows:
        print(
            f"{str(row['label']):>12} "
            f"{row['voxel_count']:>15,} "
            f"{row['percentage_of_total']:>14.6f}%"
        )
    print(f"\nSaved CSV: {output_csv}")


if __name__ == "__main__":
    main()
