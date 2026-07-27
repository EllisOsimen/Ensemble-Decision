#!/usr/bin/env python3
"""Calculate per-label voxel statistics for one NIfTI image or a directory."""

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Report the number and percentage of voxels belonging to each label "
            "in one NIfTI label map, or aggregate statistics across a directory."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Input .nii/.nii.gz label map or directory containing label maps.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <image_name>_voxel_statistics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for aggregate CSVs in directory mode. Defaults to an "
            "'voxel_statistics' subdirectory beside the input images."
        ),
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
    """Write the original single-image statistics table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "voxel_count", "percentage_of_total"],
        )
        writer.writeheader()
        writer.writerows(rows)


def nifti_paths(directory):
    """Return all NIfTI files directly inside a directory."""
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and (path.name.endswith(".nii.gz") or path.name.endswith(".nii"))
    )
    if not paths:
        raise FileNotFoundError(f"No .nii or .nii.gz files found in {directory}")
    return paths


def load_label_map(path):
    """Load one non-empty label array."""
    import nibabel as nib

    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.size == 0:
        raise ValueError(f"Input label map is empty: {path}")
    return data


def write_rows(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def directory_statistics(paths, exclude_background=False):
    """Return per-case observations and label-wise mean/population SD."""
    case_data = []
    all_labels = set()

    for path in paths:
        data = load_label_map(path)
        labels, counts = np.unique(data, return_counts=True)
        counts_by_label = {
            label.item(): int(count) for label, count in zip(labels, counts)
        }
        all_labels.update(counts_by_label)
        case_data.append((path.name, int(data.size), counts_by_label))

    if exclude_background:
        all_labels.discard(0)
    labels = sorted(all_labels)

    detailed_rows = []
    counts_by_label = {label: [] for label in labels}
    percentages_by_label = {label: [] for label in labels}
    for case_name, total_voxels, case_counts in case_data:
        for label in labels:
            # A label absent from one case is a real zero observation.
            count = case_counts.get(label, 0)
            percentage = 100.0 * count / total_voxels
            detailed_rows.append(
                {
                    "case": case_name,
                    "label": label,
                    "voxel_count": count,
                    "percentage_of_total": percentage,
                    "total_voxels": total_voxels,
                }
            )
            counts_by_label[label].append(count)
            percentages_by_label[label].append(percentage)

    summary_rows = []
    for label in labels:
        counts = np.asarray(counts_by_label[label], dtype=np.float64)
        percentages = np.asarray(percentages_by_label[label], dtype=np.float64)
        summary_rows.append(
            {
                "label": label,
                "case_count": len(case_data),
                "cases_with_label": int(np.count_nonzero(counts)),
                "mean_voxel_count": float(np.mean(counts)),
                "std_voxel_count": float(np.std(counts, ddof=0)),
                "mean_percentage_of_total": float(np.mean(percentages)),
                "std_percentage_of_total": float(np.std(percentages, ddof=0)),
                "min_voxel_count": int(np.min(counts)),
                "max_voxel_count": int(np.max(counts)),
            }
        )
    return detailed_rows, summary_rows


def run_directory(args):
    if args.output_csv is not None:
        raise ValueError(
            "--output-csv is only valid for one image; use --output-dir "
            "for directory statistics."
        )

    paths = nifti_paths(args.image)
    output_dir = args.output_dir or args.image.parent / (
        f"{args.image.name}_voxel_statistics"
    )
    detailed_path = output_dir / "per_case_per_label.csv"
    summary_path = output_dir / "per_label_summary.csv"
    detailed_rows, summary_rows = directory_statistics(
        paths, args.exclude_background
    )

    write_rows(
        detailed_path,
        detailed_rows,
        ["case", "label", "voxel_count", "percentage_of_total", "total_voxels"],
    )
    write_rows(
        summary_path,
        summary_rows,
        [
            "label",
            "case_count",
            "cases_with_label",
            "mean_voxel_count",
            "std_voxel_count",
            "mean_percentage_of_total",
            "std_percentage_of_total",
            "min_voxel_count",
            "max_voxel_count",
        ],
    )

    print(f"Directory: {args.image}")
    print(f"Images: {len(paths)}")
    print()
    print(
        f"{'Label':>8} {'Cases':>8} {'Mean voxels':>15} "
        f"{'SD voxels':>15} {'Mean %':>12} {'SD %':>12}"
    )
    print(
        f"{'-' * 8} {'-' * 8} {'-' * 15} "
        f"{'-' * 15} {'-' * 12} {'-' * 12}"
    )
    for row in summary_rows:
        print(
            f"{str(row['label']):>8} "
            f"{row['cases_with_label']:>8} "
            f"{row['mean_voxel_count']:>15,.2f} "
            f"{row['std_voxel_count']:>15,.2f} "
            f"{row['mean_percentage_of_total']:>11.6f}% "
            f"{row['std_percentage_of_total']:>11.6f}%"
        )
    print(f"\nSaved detailed observations: {detailed_path}")
    print(f"Saved aggregate summary: {summary_path}")


def run_single_image(args):
    if args.output_dir is not None:
        raise ValueError("--output-dir is only valid when --image is a directory.")

    data = load_label_map(args.image)
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


def main():
    args = parse_args()
    if args.image.is_dir():
        run_directory(args)
    elif args.image.is_file():
        run_single_image(args)
    else:
        raise FileNotFoundError(f"Input path does not exist: {args.image}")


if __name__ == "__main__":
    main()
