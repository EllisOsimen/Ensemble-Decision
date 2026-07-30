#!/usr/bin/env python3
"""Prepare the fixed TotalSegmentator cohort used for external evaluation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np


STRUCTURE_LABELS = {
    "pancreas.nii.gz": 1,
    "kidney_left.nii.gz": 2,
    "kidney_right.nii.gz": 2,
    "kidney_cyst_left.nii.gz": 2,
    "kidney_cyst_right.nii.gz": 2,
    "liver.nii.gz": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a lightweight TotalSegmentator subset with CURVAS labels: "
            "0=background, 1=pancreas, 2=kidney, 3=liver."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-list", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_case_list(path: Path) -> list[str]:
    cases = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(cases) != len(set(cases)):
        raise ValueError(f"Duplicate case IDs in {path}")
    if not cases:
        raise ValueError(f"No case IDs found in {path}")
    return cases


def relative_symlink(source: Path, destination: Path) -> None:
    expected = os.path.relpath(source.resolve(), destination.parent.resolve())
    if destination.is_symlink():
        if os.readlink(destination) != expected:
            raise FileExistsError(f"Incorrect existing symlink: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing path: {destination}")
    destination.symlink_to(expected, target_is_directory=source.is_dir())


def load_binary_mask(path: Path, reference: nib.Nifti1Image) -> np.ndarray:
    image = nib.load(path)
    if image.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {path}: {image.shape} != {reference.shape}")
    if not np.allclose(image.affine, reference.affine, rtol=0.0, atol=1e-4):
        raise ValueError(f"Affine mismatch: {path}")
    return np.asarray(image.dataobj) > 0


def prepare_case(
    source_root: Path,
    output_root: Path,
    case_name: str,
    overwrite: bool,
) -> None:
    source_case = source_root / case_name
    source_ct = source_case / "ct.nii.gz"
    source_segmentations = source_case / "segmentations"
    output_case = output_root / case_name
    output_mask = output_case / "combined_mask.nii.gz"

    required = [source_ct]
    required.extend(source_segmentations / name for name in STRUCTURE_LABELS)
    missing = [path for path in required if not path.is_file()]
    if missing:
        joined = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing inputs for {case_name}:\n  {joined}")

    output_case.mkdir(parents=True, exist_ok=True)
    relative_symlink(source_ct, output_case / "ct.nii.gz")
    relative_symlink(source_segmentations, output_case / "segmentations")

    if output_mask.exists() and not overwrite:
        print(f"{case_name}: combined mask exists; skipping")
        return

    reference = nib.load(source_ct)
    combined = np.zeros(reference.shape, dtype=np.uint8)
    for filename, label in STRUCTURE_LABELS.items():
        mask = load_binary_mask(source_segmentations / filename, reference)
        combined[mask] = label

    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    output_image = nib.Nifti1Image(combined, reference.affine, header)
    output_image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    output_image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(output_image, output_mask)
    counts = {label: int(np.count_nonzero(combined == label)) for label in (1, 2, 3)}
    print(f"{case_name}: saved {output_mask} with foreground counts {counts}")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    case_names = read_case_list(args.case_list)

    if not source_root.is_dir():
        raise FileNotFoundError(f"TotalSegmentator root not found: {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    for case_name in case_names:
        prepare_case(source_root, output_root, case_name, args.overwrite)

    print(f"Prepared {len(case_names)} TotalSegmentator cases under {output_root}")


if __name__ == "__main__":
    main()
