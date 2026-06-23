#!/usr/bin/env python3
"""Check how often four model predictions match WORD ground truth labels.

For every ground-truth voxel, this script asks:

    "How many of the four models predicted the same label as the ground truth?"

It then summarizes those answers per WORD structure. For example, for pancreas
voxels it counts how many pancreas voxels were predicted correctly by exactly
4, 3, 2, 1, or 0 models.

The Swin UNETR 5050 predictions are native BTCV labels, so they are remapped to
the WORD label space before comparison.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


WORD_LABEL_NAMES = {
    0: "background",
    1: "liver",
    2: "spleen",
    3: "left_kidney",
    4: "right_kidney",
    5: "stomach",
    6: "gallbladder",
    7: "esophagus",
    8: "pancreas",
    9: "duodenum",
    10: "colon",
    11: "intestine",
    12: "adrenal",
    13: "rectum",
    14: "bladder",
    15: "head_of_femur_left",
    16: "head_of_femur_right",
}


# Convert native Swin 5050/BTCV labels into the WORD IDs already used by the
# three SuPreM outputs. BTCV-only structures receive reserved values so they
# remain distinct from WORD background rather than being treated as label 0.
BTCV_TO_WORD = {
    0: 0,    # background
    1: 2,    # spleen
    2: 4,    # right kidney
    3: 3,    # left kidney
    4: 6,    # gallbladder
    5: 7,    # esophagus
    6: 1,    # liver
    7: 5,    # stomach
    8: 101,  # aorta: not a WORD label
    9: 102,  # inferior vena cava: not a WORD label
    10: 103, # portal/splenic veins: not a WORD label
    11: 8,   # pancreas
    12: 12,  # right adrenal -> WORD's combined adrenal label
    13: 12,  # left adrenal -> WORD's combined adrenal label
}


CORRECTNESS_COLUMNS = [
    "correct_4_voxels",
    "correct_3_voxels",
    "correct_2_voxels",
    "correct_1_voxels",
    "correct_0_voxels",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare U-Net, SuPreM Swin UNETR, SegResNet, and Swin UNETR 5050 "
            "prediction masks against WORD labelsTr."
        )
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=PROJECT_DIR.parent / "WORD" / "WORD-V0.1.0" / "labelsTr",
        help="Directory containing WORD ground-truth label maps.",
    )
    parser.add_argument(
        "--unet-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "word_three_models" / "unet",
        help="Directory containing WORD-numbered SuPreM U-Net masks.",
    )
    parser.add_argument(
        "--swinunetr-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "word_three_models" / "swinunetr",
        help="Directory containing WORD-numbered SuPreM Swin UNETR masks.",
    )
    parser.add_argument(
        "--segresnet-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "word_three_models" / "segresnet",
        help="Directory containing WORD-numbered SuPreM SegResNet masks.",
    )
    parser.add_argument(
        "--swin5050-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "word_swinunetr_5050",
        help="Directory containing native BTCV Swin UNETR 5050 masks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "word_four_model_ground_truth_agreement",
        help="Directory where CSV summaries will be written.",
    )
    parser.add_argument(
        "--case-name",
        default=None,
        help="Process only this .nii.gz case instead of every matching case.",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Include WORD label 0/background in the CSV summaries.",
    )
    parser.add_argument(
        "--ignore-affine",
        action="store_true",
        help="Only check array shapes, not NIfTI affines.",
    )
    return parser.parse_args()


def label_name(label: int) -> str:
    return WORD_LABEL_NAMES.get(label, f"label_{label}")


def nifti_files(directory: Path, name: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"{name} directory does not exist: {directory}")
    files = {path.name: path for path in directory.glob("*.nii.gz")}
    if not files:
        raise FileNotFoundError(f"No .nii.gz files found in {name} directory: {directory}")
    return files


def matching_cases(args: argparse.Namespace) -> tuple[dict[str, dict[str, Path]], list[str]]:
    """Find case names that exist in all four prediction directories and labelsTr."""
    files = {
        "labels": nifti_files(args.labels_dir, "labelsTr"),
        "unet": nifti_files(args.unet_dir, "U-Net"),
        "swinunetr": nifti_files(args.swinunetr_dir, "SuPreM Swin UNETR"),
        "segresnet": nifti_files(args.segresnet_dir, "SegResNet"),
        "swin5050": nifti_files(args.swin5050_dir, "Swin UNETR 5050"),
    }

    reference_names = set(files["unet"])
    for name in ("swinunetr", "segresnet", "swin5050"):
        names = set(files[name])
        if names != reference_names:
            missing = sorted(reference_names - names)
            extra = sorted(names - reference_names)
            details = []
            if missing:
                details.append(f"missing {len(missing)} case(s), e.g. {missing[:3]}")
            if extra:
                details.append(f"has {len(extra)} extra case(s), e.g. {extra[:3]}")
            raise ValueError(f"Prediction case mismatch for {name}: {'; '.join(details)}")

    missing_labels = sorted(reference_names - set(files["labels"]))
    if missing_labels:
        raise ValueError(
            "Ground-truth labelsTr is missing "
            f"{len(missing_labels)} case(s), e.g. {missing_labels[:3]}"
        )

    case_names = sorted(reference_names)
    if args.case_name is not None:
        if args.case_name not in case_names:
            raise FileNotFoundError(
                f"{args.case_name} is not present in all prediction directories."
            )
        case_names = [args.case_name]

    return files, case_names


def load_integer_mask(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if not np.issubdtype(data.dtype, np.integer):
        if not np.all(np.equal(data, np.round(data))):
            raise ValueError(f"{path} contains non-integer labels.")
        data = np.round(data)
    return image, data.astype(np.int16, copy=False)


def validate_grid(
    case_name: str,
    label_image: nib.Nifti1Image,
    prediction_images: list[nib.Nifti1Image],
    ignore_affine: bool,
) -> None:
    """Make sure voxel indices line up before doing voxel-level comparisons."""
    for index, prediction_image in enumerate(prediction_images, start=1):
        if prediction_image.shape != label_image.shape:
            raise ValueError(
                f"{case_name}: prediction {index} shape {prediction_image.shape} "
                f"does not match labelsTr shape {label_image.shape}"
            )
        if not ignore_affine and not np.allclose(
            prediction_image.affine,
            label_image.affine,
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError(
                f"{case_name}: prediction {index} affine does not match labelsTr. "
                "If you have inspected this and only want array-index comparison, "
                "rerun with --ignore-affine."
            )


def remap_btcv_to_word(btcv: np.ndarray, case_name: str) -> np.ndarray:
    unique_labels = np.unique(btcv).astype(int)
    unknown = sorted(set(unique_labels) - set(BTCV_TO_WORD))
    if unknown:
        raise ValueError(f"{case_name}: unexpected Swin 5050 BTCV label(s): {unknown}")

    max_label = max(int(unique_labels.max()), max(BTCV_TO_WORD))
    lookup = np.zeros(max_label + 1, dtype=np.int16)
    for btcv_label, word_label in BTCV_TO_WORD.items():
        lookup[btcv_label] = word_label
    return lookup[btcv]


def empty_counts() -> dict[str, int]:
    return {
        "total_gt_voxels": 0,
        "correct_4_voxels": 0,
        "correct_3_voxels": 0,
        "correct_2_voxels": 0,
        "correct_1_voxels": 0,
        "correct_0_voxels": 0,
        "suprem3_agree_swin5050_disagrees_voxels": 0,
        "suprem3_correct_swin5050_wrong_voxels": 0,
    }


def add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] += int(value)


def percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def make_row(case_name: str | None, label: int | None, counts: dict[str, int]) -> dict[str, object]:
    total = counts["total_gt_voxels"]
    row: dict[str, object] = {}
    if case_name is not None:
        row["case"] = case_name
    if label is not None:
        row["label"] = label
        row["structure"] = label_name(label)

    row.update(counts)
    for column in CORRECTNESS_COLUMNS:
        row[column.replace("_voxels", "_percent")] = round(
            percent(counts[column], total),
            6,
        )
    row["suprem3_agree_swin5050_disagrees_percent"] = round(
        percent(counts["suprem3_agree_swin5050_disagrees_voxels"], total),
        6,
    )
    row["suprem3_correct_swin5050_wrong_percent"] = round(
        percent(counts["suprem3_correct_swin5050_wrong_voxels"], total),
        6,
    )
    return row


def analyze_case(
    case_name: str,
    paths: dict[str, Path],
    labels_to_report: list[int],
    ignore_affine: bool,
) -> tuple[list[dict[str, object]], dict[str, object], dict[int, dict[str, int]]]:
    label_image, ground_truth = load_integer_mask(paths["labels"])
    unet_image, unet = load_integer_mask(paths["unet"])
    swinunetr_image, swinunetr = load_integer_mask(paths["swinunetr"])
    segresnet_image, segresnet = load_integer_mask(paths["segresnet"])
    swin5050_image, swin5050_native = load_integer_mask(paths["swin5050"])

    validate_grid(
        case_name,
        label_image,
        [unet_image, swinunetr_image, segresnet_image, swin5050_image],
        ignore_affine,
    )

    swin5050 = remap_btcv_to_word(swin5050_native, case_name)
    predictions = (unet, swinunetr, segresnet, swin5050)

    correct_count = np.zeros(ground_truth.shape, dtype=np.uint8)
    for prediction in predictions:
        correct_count += prediction == ground_truth

    suprem3_agree = (unet == swinunetr) & (unet == segresnet)
    suprem3_agree_swin5050_disagrees = suprem3_agree & (swin5050 != unet)
    suprem3_correct_swin5050_wrong = suprem3_agree_swin5050_disagrees & (
        unet == ground_truth
    )

    per_label_rows = []
    per_label_counts = {}
    case_counts = empty_counts()
    max_reported_label = max(labels_to_report)

    included_voxels = np.isin(ground_truth, labels_to_report)
    included_labels = ground_truth[included_voxels].astype(np.int64, copy=False)
    included_correct_counts = correct_count[included_voxels].astype(np.int64, copy=False)

    # Build a compact 2D histogram:
    #   rows    = WORD labels
    #   columns = number of models correct, from 0 through 4
    combined_index = included_labels * 5 + included_correct_counts
    correctness_histogram = np.bincount(
        combined_index,
        minlength=(max_reported_label + 1) * 5,
    ).reshape(max_reported_label + 1, 5)

    suprem3_disagree_counts = np.bincount(
        ground_truth[suprem3_agree_swin5050_disagrees & included_voxels].astype(
            np.int64,
            copy=False,
        ),
        minlength=max_reported_label + 1,
    )
    suprem3_correct_swin_wrong_counts = np.bincount(
        ground_truth[suprem3_correct_swin5050_wrong & included_voxels].astype(
            np.int64,
            copy=False,
        ),
        minlength=max_reported_label + 1,
    )

    for label in labels_to_report:
        counts = empty_counts()
        # Histogram column 0 means zero models correct, column 4 means all four
        # models correct. CSV columns are ordered 4 -> 0 for reading.
        histogram = correctness_histogram[label]
        counts["total_gt_voxels"] = int(histogram.sum())
        counts["correct_4_voxels"] = int(histogram[4])
        counts["correct_3_voxels"] = int(histogram[3])
        counts["correct_2_voxels"] = int(histogram[2])
        counts["correct_1_voxels"] = int(histogram[1])
        counts["correct_0_voxels"] = int(histogram[0])
        counts["suprem3_agree_swin5050_disagrees_voxels"] = int(
            suprem3_disagree_counts[label]
        )
        counts["suprem3_correct_swin5050_wrong_voxels"] = int(
            suprem3_correct_swin_wrong_counts[label]
        )

        add_counts(case_counts, counts)
        per_label_counts[label] = counts
        per_label_rows.append(make_row(case_name, label, counts))

    return per_label_rows, make_row(case_name, None, case_counts), per_label_counts


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    files, case_names = matching_cases(args)

    labels_to_report = sorted(label for label in WORD_LABEL_NAMES if label != 0)
    if args.include_background:
        labels_to_report = [0] + labels_to_report

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_case_per_structure_rows = []
    per_case_rows = []
    aggregate_by_label = defaultdict(empty_counts)
    cases_with_label = defaultdict(int)

    for case_name in tqdm(case_names, desc="Checking agreement vs labelsTr"):
        paths = {
            "labels": files["labels"][case_name],
            "unet": files["unet"][case_name],
            "swinunetr": files["swinunetr"][case_name],
            "segresnet": files["segresnet"][case_name],
            "swin5050": files["swin5050"][case_name],
        }
        label_rows, case_row, per_label_counts = analyze_case(
            case_name,
            paths,
            labels_to_report,
            args.ignore_affine,
        )
        per_case_per_structure_rows.extend(label_rows)
        per_case_rows.append(case_row)

        for label, counts in per_label_counts.items():
            add_counts(aggregate_by_label[label], counts)
            if counts["total_gt_voxels"] > 0:
                cases_with_label[label] += 1

    per_structure_rows = []
    for label in labels_to_report:
        row = make_row(None, label, aggregate_by_label[label])
        row["cases_with_structure"] = cases_with_label[label]
        row["case_count"] = len(case_names)
        per_structure_rows.append(row)

    per_case_per_structure_fields = [
        "case",
        "label",
        "structure",
        "total_gt_voxels",
        *CORRECTNESS_COLUMNS,
        "correct_4_percent",
        "correct_3_percent",
        "correct_2_percent",
        "correct_1_percent",
        "correct_0_percent",
        "suprem3_agree_swin5050_disagrees_voxels",
        "suprem3_agree_swin5050_disagrees_percent",
        "suprem3_correct_swin5050_wrong_voxels",
        "suprem3_correct_swin5050_wrong_percent",
    ]
    per_case_fields = [
        "case",
        "total_gt_voxels",
        *CORRECTNESS_COLUMNS,
        "correct_4_percent",
        "correct_3_percent",
        "correct_2_percent",
        "correct_1_percent",
        "correct_0_percent",
        "suprem3_agree_swin5050_disagrees_voxels",
        "suprem3_agree_swin5050_disagrees_percent",
        "suprem3_correct_swin5050_wrong_voxels",
        "suprem3_correct_swin5050_wrong_percent",
    ]
    per_structure_fields = [
        "label",
        "structure",
        "case_count",
        "cases_with_structure",
        "total_gt_voxels",
        *CORRECTNESS_COLUMNS,
        "correct_4_percent",
        "correct_3_percent",
        "correct_2_percent",
        "correct_1_percent",
        "correct_0_percent",
        "suprem3_agree_swin5050_disagrees_voxels",
        "suprem3_agree_swin5050_disagrees_percent",
        "suprem3_correct_swin5050_wrong_voxels",
        "suprem3_correct_swin5050_wrong_percent",
    ]

    write_csv(
        args.output_dir / "per_case_per_structure.csv",
        per_case_per_structure_rows,
        per_case_per_structure_fields,
    )
    write_csv(args.output_dir / "per_case_summary.csv", per_case_rows, per_case_fields)
    write_csv(
        args.output_dir / "per_structure_summary.csv",
        per_structure_rows,
        per_structure_fields,
    )

    print(f"Processed {len(case_names)} case(s).")
    print(f"Wrote CSV files to: {args.output_dir}")


if __name__ == "__main__":
    main()
