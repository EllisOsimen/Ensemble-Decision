#!/usr/bin/env python3
"""Check shape and affine consistency across testing_set annotations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare annotation shapes, voxel sizes, and affine matrices within "
            "each testing_set case."
        )
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=PROJECT_DIR.parent / "testing_set",
        help="Directory containing one subdirectory per case.",
    )
    parser.add_argument(
        "--annotations",
        nargs="+",
        default=["annotation_1.nii.gz", "annotation_2.nii.gz", "annotation_3.nii.gz"],
        help="Annotation filenames inside each case directory.",
    )
    parser.add_argument(
        "--case-name",
        default=None,
        help="Check only one case.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for all comparison rows.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance used by numpy.allclose.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance used by numpy.allclose.",
    )
    return parser.parse_args()


def case_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f"Cases root does not exist: {root}")
    cases = sorted(path for path in root.iterdir() if path.is_dir())
    if not cases:
        raise FileNotFoundError(f"No case directories found in {root}")
    return cases


def affine_summary(affine: np.ndarray) -> str:
    return np.array2string(
        affine,
        precision=6,
        suppress_small=False,
        separator=", ",
    )


def main() -> None:
    args = parse_args()
    cases = case_directories(args.cases_root)
    if args.case_name is not None:
        cases = [path for path in cases if path.name == args.case_name]
        if not cases:
            raise FileNotFoundError(f"{args.case_name} not found in {args.cases_root}")

    rows: list[dict[str, object]] = []
    mismatch_count = 0
    missing_count = 0

    for case_dir in cases:
        images = {}
        for annotation_name in args.annotations:
            path = case_dir / annotation_name
            if not path.is_file():
                missing_count += 1
                rows.append(
                    {
                        "case": case_dir.name,
                        "reference": args.annotations[0],
                        "annotation": annotation_name,
                        "status": "missing",
                        "shape_match": "",
                        "affine_match": "",
                        "zooms_match": "",
                        "max_abs_affine_diff": "",
                        "shape": "",
                        "reference_shape": "",
                        "zooms": "",
                        "reference_zooms": "",
                    }
                )
                continue
            images[annotation_name] = nib.load(str(path))

        if args.annotations[0] not in images:
            continue

        reference_name = args.annotations[0]
        reference = images[reference_name]
        reference_zooms = tuple(float(value) for value in reference.header.get_zooms()[:3])

        for annotation_name in args.annotations:
            if annotation_name not in images:
                continue

            image = images[annotation_name]
            zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
            shape_match = image.shape == reference.shape
            affine_diff = np.abs(image.affine - reference.affine)
            max_abs_affine_diff = float(affine_diff.max())
            affine_match = bool(
                np.allclose(
                    image.affine,
                    reference.affine,
                    rtol=args.rtol,
                    atol=args.atol,
                )
            )
            zooms_match = bool(np.allclose(zooms, reference_zooms, rtol=args.rtol, atol=args.atol))
            status = "ok" if shape_match and affine_match else "mismatch"
            if status == "mismatch":
                mismatch_count += 1

            rows.append(
                {
                    "case": case_dir.name,
                    "reference": reference_name,
                    "annotation": annotation_name,
                    "status": status,
                    "shape_match": shape_match,
                    "affine_match": affine_match,
                    "zooms_match": zooms_match,
                    "max_abs_affine_diff": max_abs_affine_diff,
                    "shape": "x".join(str(value) for value in image.shape),
                    "reference_shape": "x".join(str(value) for value in reference.shape),
                    "zooms": "x".join(f"{value:.8g}" for value in zooms),
                    "reference_zooms": "x".join(f"{value:.8g}" for value in reference_zooms),
                }
            )

    fieldnames = [
        "case",
        "reference",
        "annotation",
        "status",
        "shape_match",
        "affine_match",
        "zooms_match",
        "max_abs_affine_diff",
        "shape",
        "reference_shape",
        "zooms",
        "reference_zooms",
    ]

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    problem_rows = [row for row in rows if row["status"] != "ok"]
    print(f"Cases checked: {len(cases)}")
    print(f"Annotation comparisons: {len(rows)}")
    print(f"Mismatches: {mismatch_count}")
    print(f"Missing annotations: {missing_count}")
    if args.output_csv is not None:
        print(f"CSV: {args.output_csv}")

    if problem_rows:
        print("\nFirst mismatches:")
        for row in problem_rows[:20]:
            print(
                f"{row['case']} {row['annotation']}: "
                f"shape_match={row['shape_match']} "
                f"affine_match={row['affine_match']} "
                f"zooms_match={row['zooms_match']} "
                f"max_abs_affine_diff={row['max_abs_affine_diff']}"
            )

        if args.case_name is not None:
            print("\nAffines for inspected case:")
            case_dir = args.cases_root / args.case_name
            for annotation_name in args.annotations:
                path = case_dir / annotation_name
                if path.is_file():
                    image = nib.load(str(path))
                    print(f"\n{annotation_name}")
                    print(affine_summary(image.affine))


if __name__ == "__main__":
    main()
