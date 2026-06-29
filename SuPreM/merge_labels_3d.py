#!/usr/bin/env python3
"""
Create one multi-organ labelled NIfTI mask per 3D-IRCADb case.

Expected structure:

Dataset/
  ImagesTr/
    3Dircadb1.1.nii.gz
  3Dircadb1.1/
    masks/
      liver.nii.gz
      livertumor01.nii.gz
      spleen.nii.gz
      leftkidney.nii.gz
      ...

Tumour handling:
  livertumor*.nii.gz is merged into the liver label.
"""

import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np


# -------------------------------------------------------------------------
# Label map for the combined segmentation
# -------------------------------------------------------------------------
# Edit this if you want a different convention.
#
# Important:
#   - livertumorXX is not listed here because it is mapped to "liver".
#   - label 0 is always background.
# -------------------------------------------------------------------------
LABEL_MAP = {
    "liver": 1,
    "spleen": 2,
    "leftkidney": 3,
    "rightkidney": 4,
    "leftlung": 5,
    "rightlung": 6,
    "bone": 7,
    "skin": 8,
    "artery": 9,
    "portalvein": 10,
    "venoussystem": 11,
}


# Broad body-region masks should be written first so smaller, more specific
# organs can overwrite them. Without this, skin can overwrite almost every
# abdominal organ because the skin/body mask surrounds or overlaps them.
LABEL_PRIORITY = {
    "skin": 10,
    "bone": 20,
    "leftlung": 30,
    "rightlung": 30,
    "venoussystem": 40,
    "artery": 50,
    "portalvein": 60,
    "spleen": 70,
    "leftkidney": 70,
    "rightkidney": 70,
    "liver": 80,
}


# Masks matching this pattern are merged into the liver label.
LIVER_TUMOUR_PATTERN = re.compile(r"^livertumor\d*$", re.IGNORECASE)
LIVER_MERGE_NAMES = ["liverkyst"]


def strip_nii_gz_name(path: Path) -> str:
    """
    Convert e.g. livertumor01.nii.gz -> livertumor01.
    """
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def load_binary_mask(mask_path: Path) -> np.ndarray:
    """
    Load a binary mask and return boolean foreground.
    Any non-zero voxel is treated as foreground.
    """
    img = nib.load(str(mask_path))
    data = img.get_fdata()
    return data > 0


def reference_image_path(case_dir: Path, images_dir: Path | None) -> Path | None:
    """
    Find the CT image that should provide shape, affine, and header metadata.

    Some 3D-IRCAD layouts keep the CT beside masks as ct.nii.gz, while this
    workspace keeps CTs in Dataset/ImagesTr/<case>.nii.gz.
    """
    local_candidates = [
        case_dir / "ct.nii.gz",
        case_dir / "image.nii.gz",
    ]
    for path in local_candidates:
        if path.is_file():
            return path

    if images_dir is not None:
        image_candidates = [
            images_dir / f"{case_dir.name}.nii.gz",
            images_dir / f"{case_dir.name}_0000.nii.gz",
        ]
        for path in image_candidates:
            if path.is_file():
                return path

    return None


def target_for_mask(mask_path: Path) -> tuple[str, int] | None:
    """
    Return the combined-mask target name and label for one source mask.
    """
    mask_name = strip_nii_gz_name(mask_path).lower()
    if LIVER_TUMOUR_PATTERN.match(mask_name) or mask_name in LIVER_MERGE_NAMES:
        return "liver", LABEL_MAP["liver"]

    target_label = LABEL_MAP.get(mask_name)
    if target_label is None:
        return None
    return mask_name, target_label


def mask_sort_key(mask_path: Path) -> tuple[int, str]:
    """
    Sort source masks from broad structures to specific organs.
    """
    target = target_for_mask(mask_path)
    if target is None:
        return (999, strip_nii_gz_name(mask_path).lower())
    target_name, _ = target
    return (LABEL_PRIORITY.get(target_name, 100), strip_nii_gz_name(mask_path).lower())


def create_case_mask(
    case_dir: Path,
    output_dir: Path,
    images_dir: Path | None,
    overwrite: bool = False,
) -> None:
    case_name = case_dir.name
    masks_dir = case_dir / "masks"
    ct_path = reference_image_path(case_dir, images_dir)

    if not masks_dir.exists():
        print(f"[SKIP] {case_name}: no masks/ directory found")
        return

    if ct_path is None:
        print(f"[SKIP] {case_name}: no reference CT found")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    out_mask_path = output_dir / f"{case_name}_multiorgan.nii.gz"
    out_labels_path = output_dir / f"{case_name}_labels.json"

    if out_mask_path.exists() and not overwrite:
        print(f"[SKIP] {case_name}: output already exists. Use --overwrite to replace it.")
        return

    ct_img = nib.load(str(ct_path))
    ct_shape = ct_img.shape

    combined = np.zeros(ct_shape, dtype=np.uint8)

    mask_paths = sorted(masks_dir.glob("*.nii.gz"), key=mask_sort_key)

    if not mask_paths:
        print(f"[SKIP] {case_name}: no .nii.gz masks found")
        return

    used_masks = {}
    unknown_masks = []

    print(f"\n[CASE] {case_name}")
    print(f"  [REF] {ct_path}")

    for mask_path in mask_paths:
        mask_name = strip_nii_gz_name(mask_path).lower()
        target = target_for_mask(mask_path)

        if target is None:
            unknown_masks.append(mask_name)
            print(f"  [WARN] Unknown mask ignored: {mask_name}")
            continue
        target_name, target_label = target

        foreground = load_binary_mask(mask_path)

        if foreground.shape != ct_shape:
            raise ValueError(
                f"Shape mismatch in {case_name}: {mask_path.name} has shape "
                f"{foreground.shape}, but ct.nii.gz has shape {ct_shape}"
            )

        voxel_count = int(foreground.sum())

        if voxel_count == 0:
            print(f"  [EMPTY] {mask_name}")
            continue

        # Write the label into the combined mask. The source masks are ordered
        # broad-to-specific, so a later overwrite means a more specific organ is
        # taking precedence over a broader body-region label.
        overlap_count = int(np.logical_and(combined > 0, foreground).sum())
        if overlap_count > 0:
            print(
                f"  [OVERLAP] {mask_name} overlaps existing labels at "
                f"{overlap_count} voxels; overwriting those voxels."
            )

        combined[foreground] = target_label

        used_masks.setdefault(target_name, [])
        used_masks[target_name].append(
            {
                "source_mask": mask_path.name,
                "label": int(target_label),
                "voxel_count": voxel_count,
            }
        )

        if mask_name != target_name:
            print(
                f"  [MERGE] {mask_name} -> {target_name} "
                f"(label {target_label}, {voxel_count} voxels)"
            )
        else:
            print(f"  [ADD] {mask_name} -> label {target_label} ({voxel_count} voxels)")

    # Save combined NIfTI
    out_img = nib.Nifti1Image(
        combined,
        affine=ct_img.affine,
        header=ct_img.header.copy(),
    )
    out_img.set_data_dtype(np.uint8)
    nib.save(out_img, str(out_mask_path))

    # Save label metadata
    metadata = {
        "case": case_name,
        "output_mask": str(out_mask_path),
        "label_map": {"background": 0, **LABEL_MAP},
        "tumour_rule": "livertumor*.nii.gz is merged into liver label",
        "used_masks": used_masks,
        "unknown_masks_ignored": unknown_masks,
    }

    with open(out_labels_path, "w") as f:
        json.dump(metadata, f, indent=2)

    unique, counts = np.unique(combined, return_counts=True)

    print(f"\n  [SAVED] {out_mask_path}")
    print(f"  [SAVED] {out_labels_path}")
    print("  Label voxel counts:")
    for label, count in zip(unique, counts):
        print(f"    {int(label):>3}: {int(count)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3D-IRCADb binary masks into one multi-organ label mask per case."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the Dataset directory containing 3Dircadb1.X folders.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where combined masks will be saved.",
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing case CT images. Defaults to "
            "<dataset-root>/ImagesTr when that directory exists."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output masks.",
    )

    args = parser.parse_args()
    if args.images_dir is None:
        default_images_dir = args.dataset_root / "ImagesTr"
        args.images_dir = default_images_dir if default_images_dir.is_dir() else None

    case_dirs = sorted(
        p for p in args.dataset_root.iterdir()
        if p.is_dir() and p.name.startswith("3Dircadb1.")
    )

    if not case_dirs:
        raise RuntimeError(f"No 3Dircadb1.X case directories found in {args.dataset_root}")

    print(f"Found {len(case_dirs)} cases.")

    for case_dir in case_dirs:
        create_case_mask(
            case_dir=case_dir,
            output_dir=args.output_dir,
            images_dir=args.images_dir,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
