#!/usr/bin/env python3
"""Evaluate prediction directories against testing_set annotations.

This script compares one or more prediction directories against the matching
case folders under a testing_set-style root. The target label space is:

  0 = background
  1 = pancreas
  2 = kidney
  3 = liver

Predictions can be interpreted as:

  - target: already in the 0-3 testing-set label space
  - btcv:   native BTCV labels that are remapped to the 0-3 target space
  - suprem: SuPreM saved label maps (0-16) from this repository, or raw
            32-channel SuPreM masks if provided
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from surface_distance import metrics as surface_distance_metrics
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


TARGET_LABELS = {
    0: "background",
    1: "pancreas",
    2: "kidney",
    3: "liver",
}


# Native BTCV prediction labels mapped into the testing_set label space.
# Structures outside the target task are mapped to background.
BTCV_TO_TARGET = {
    0: 0,
    1: 0,  # spleen
    2: 2,  # right kidney
    3: 2,  # left kidney
    4: 0,  # gallbladder
    5: 0,  # esophagus
    6: 3,  # liver
    7: 0,  # stomach
    8: 0,  # aorta
    9: 0,  # inferior vena cava
    10: 0,  # portal/splenic veins
    11: 1,  # pancreas
    12: 0,  # adrenal
    13: 0,  # adrenal
}


# SuPreM saved label maps in this repository are WORD-numbered label maps.
# The relevant WORD labels are mapped into the 4-class testing_set space.
SUPREM_TO_TARGET = {
    0: 0,
    1: 3,  # liver
    2: 0,  # spleen
    3: 2,  # left kidney
    4: 2,  # right kidney
    5: 0,  # stomach
    6: 0,  # gallbladder
    7: 0,  # esophagus
    8: 1,  # pancreas
    9: 0,  # duodenum
    10: 0,  # colon
    11: 0,  # intestine
    12: 0,  # adrenal
    13: 0,  # rectum
    14: 0,  # bladder
    15: 0,  # head_of_femur_left
    16: 0,  # head_of_femur_right
}


# Raw SuPreM channel indices used by the inference scripts in this repo.
# These are zero-based channel numbers, not label values.
SUPREM_CHANNELS_TO_TARGET = {
    1: (10,),  # pancreas
    2: (1, 2),  # kidney
    3: (5,),   # liver
}


SUPPORTED_LABEL_SPACES = {"target", "btcv", "suprem"}


@dataclass(frozen=True)
class ModelSpec:
    """Command-line description of one prediction directory to evaluate."""

    name: str
    directory: Path
    label_space: str = "suprem"


def parse_model_spec(value: str) -> ModelSpec:
    """Parse NAME=DIR or NAME=DIR:LABEL_SPACE entries from repeated --model flags."""

    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--model must be formatted as NAME=DIR or NAME=DIR:LABEL_SPACE"
        )
    name, rest = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Model name cannot be empty.")

    directory_text = rest
    label_space = "suprem"
    if ":" in rest:
        possible_directory, possible_label_space = rest.rsplit(":", 1)
        if possible_label_space in SUPPORTED_LABEL_SPACES:
            directory_text = possible_directory
            label_space = possible_label_space

    directory = Path(directory_text)
    if not directory.is_absolute():
        directory = PROJECT_DIR / directory
    return ModelSpec(name, directory, label_space)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate prediction directories against testing_set annotations and "
            "write Dice/NSD summaries in the 0=background, 1=pancreas, 2=kidney, "
            "3=liver label space."
        )
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=PROJECT_DIR.parent / "testing_set",
        help="Directory containing one subdirectory per case.",
    )
    parser.add_argument(
        "--annotation-name",
        default="annotation_1.nii.gz",
        help="Annotation filename inside each case directory.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        type=parse_model_spec,
        action="append",
        required=True,
        help=(
            "Model prediction directory. Format: NAME=DIR or NAME=DIR:LABEL_SPACE. "
            "LABEL_SPACE can be target, btcv, or suprem. Repeat once per model."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "testing_set_prediction_evaluation",
        help="Directory where CSV and JSON summaries will be written.",
    )
    parser.add_argument(
        "--case-name",
        default=None,
        help="Process only this case name instead of every matching case.",
    )
    parser.add_argument(
        "--nsd-tolerance-mm",
        type=float,
        default=1.0,
        help="Surface tolerance in millimetres for normalized surface Dice.",
    )
    parser.add_argument(
        "--ignore-affine",
        action="store_true",
        help="Only check array shapes, not NIfTI affines.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows with a stable column order so outputs are easy to compare."""

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_integer_mask(path: Path, atol: float = 1e-3) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load a NIfTI label map and ensure it contains integer-like class IDs."""

    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if not np.issubdtype(data.dtype, np.integer):
        rounded = np.rint(data)
        if not np.all(np.isclose(data, rounded, rtol=0.0, atol=atol)):
            raise ValueError(f"{path} contains non-integer labels.")
        data = rounded
    return image, data.astype(np.int16, copy=False)


def case_directories(root: Path) -> dict[str, Path]:
    """Return testing_set cases keyed by case directory name."""

    if not root.is_dir():
        raise NotADirectoryError(f"Cases root does not exist: {root}")
    cases = {path.name: path for path in root.iterdir() if path.is_dir()}
    if not cases:
        raise FileNotFoundError(f"No case directories found in {root}")
    return cases


def prediction_files(directory: Path) -> dict[str, Path]:
    """Return prediction files keyed by case name without the .nii.gz suffix."""

    if not directory.is_dir():
        raise NotADirectoryError(f"Prediction directory does not exist: {directory}")
    files = {
        path.name[:-7] if path.name.endswith(".nii.gz") else path.stem: path
        for path in directory.glob("*.nii.gz")
    }
    if not files:
        raise FileNotFoundError(f"No .nii.gz predictions found in {directory}")
    return files


def annotation_path(case_dir: Path, annotation_name: str) -> Path:
    """Build and validate the expected annotation path for one case."""

    path = case_dir / annotation_name
    if not path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {path}")
    return path


def validate_grid(case_name, label_image, prediction_image, model, ignore_affine):
    """Confirm prediction and annotation occupy the same voxel grid."""

    # Shape equality checks array dimensions. Affine equality checks that those
    # array indices refer to the same physical coordinates in scanner space.
    if prediction_image.shape != label_image.shape:
        raise ValueError(
            f"{case_name}: {model.name} shape {prediction_image.shape} does not match "
            f"annotation shape {label_image.shape}"
        )
    if not ignore_affine and not np.allclose(
        prediction_image.affine,
        label_image.affine,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError(
            f"{case_name}: {model.name} affine does not match annotation. "
            "Rerun with --ignore-affine only if array-index comparison is intended."
        )


def binary_metrics(prediction, gold, spacing, tolerance_mm):
    """Compute Dice, normalized surface Dice, and voxel confusion counts."""

    # Convert to booleans first so the same metric code works for any label ID.
    prediction = np.asarray(prediction, dtype=bool)
    gold = np.asarray(gold, dtype=bool)
    tp = int(np.logical_and(prediction, gold).sum())
    fp = int(np.logical_and(prediction, ~gold).sum())
    fn = int(np.logical_and(~prediction, gold).sum())
    denominator = 2 * tp + fp + fn
    dsc = (2.0 * tp / denominator) if denominator else math.nan

    if prediction.any() and gold.any():
        # NSD uses physical distances, so voxel spacing from the NIfTI header is
        # required. Empty masks have no surface, so those NSD values are NaN.
        distances = surface_distance_metrics.compute_surface_distances(
            gold,
            prediction,
            spacing_mm=spacing,
        )
        nsd = float(
            surface_distance_metrics.compute_surface_dice_at_tolerance(
                distances,
                tolerance_mm,
            )
        )
    else:
        nsd = math.nan
    return dsc, nsd, tp, fp, fn


def remap_btcv_to_target(prediction: np.ndarray, case_name: str, model_name: str) -> np.ndarray:
    """Convert BTCV label IDs into the testing_set 0-3 label space."""

    unique_labels = np.unique(prediction).astype(int)
    unknown = sorted(set(unique_labels) - set(BTCV_TO_TARGET))
    if unknown:
        raise ValueError(
            f"{case_name}: unexpected BTCV label(s) in {model_name}: {unknown}"
        )

    lookup = np.zeros(max(BTCV_TO_TARGET) + 1, dtype=np.int16)
    for btcv_label, target_label in BTCV_TO_TARGET.items():
        lookup[btcv_label] = target_label
    return lookup[prediction]


def remap_suprem_combined_to_target(prediction: np.ndarray) -> np.ndarray:
    """Convert a saved SuPreM/WORD-style integer label map into target labels."""

    unique_labels = np.unique(prediction).astype(int)
    unknown = sorted(set(unique_labels) - set(SUPREM_TO_TARGET))
    if unknown:
        raise ValueError(f"Unexpected SuPreM label(s): {unknown}")

    lookup = np.zeros(max(SUPREM_TO_TARGET) + 1, dtype=np.int16)
    for suprem_label, target_label in SUPREM_TO_TARGET.items():
        lookup[suprem_label] = target_label
    return lookup[prediction]


def remap_suprem_channels_to_target(prediction: np.ndarray) -> np.ndarray:
    """Convert raw SuPreM channel masks into one 0-3 integer label map."""

    if prediction.ndim != 4:
        raise ValueError(
            "Expected a 4D channel stack for raw SuPreM masks, got shape "
            f"{prediction.shape}"
        )

    target = np.zeros(prediction.shape[1:], dtype=np.int16)
    for target_label, channels in SUPREM_CHANNELS_TO_TARGET.items():
        # Kidney is represented by two SuPreM channels, so logical OR merges
        # left and right kidneys into the testing_set kidney class.
        class_mask = np.logical_or.reduce([prediction[channel] > 0 for channel in channels])
        target[class_mask] = target_label
    return target


def remap_prediction_to_target(prediction: np.ndarray, model: ModelSpec, case_name: str) -> np.ndarray:
    """Normalize any supported model output into the testing_set target labels."""

    if model.label_space == "target":
        unique_labels = np.unique(prediction).astype(int)
        unknown = sorted(set(unique_labels) - set(TARGET_LABELS))
        if unknown:
            raise ValueError(f"{case_name}: unexpected target label(s) in {model.name}: {unknown}")
        return prediction.astype(np.int16, copy=False)

    if model.label_space == "btcv":
        return remap_btcv_to_target(prediction, case_name, model.name)

    if model.label_space == "suprem":
        if prediction.ndim == 4:
            return remap_suprem_channels_to_target(prediction)
        return remap_suprem_combined_to_target(prediction)

    raise ValueError(f"Unsupported label space: {model.label_space}")


def evaluate_model(model: ModelSpec, cases_root: Path, case_dirs: dict[str, Path], args) -> None:
    """Evaluate one model directory and write its CSV/JSON reports."""

    pred_files = prediction_files(model.directory)
    case_names = sorted(set(pred_files) & set(case_dirs))

    if args.case_name is not None:
        if args.case_name not in pred_files:
            raise FileNotFoundError(f"{args.case_name} not found in {model.directory}")
        if args.case_name not in case_dirs:
            raise FileNotFoundError(f"{args.case_name} not found in {cases_root}")
        case_names = [args.case_name]

    missing_predictions = sorted(set(case_dirs) - set(pred_files))
    missing_cases = sorted(set(pred_files) - set(case_dirs))
    if missing_predictions:
        print(
            f"WARNING: {model.name} is missing {len(missing_predictions)} prediction(s), "
            f"e.g. {missing_predictions[:3]}"
        )
    if missing_cases:
        print(
            f"WARNING: {model.name} has {len(missing_cases)} prediction(s) without a "
            f"matching case directory, e.g. {missing_cases[:3]}"
        )

    if not case_names:
        raise FileNotFoundError(
            f"No matching case/prediction pairs found for {model.name} in {model.directory} "
            f"and {cases_root}"
        )

    model_output_dir = args.output_dir / model.name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    case_rows: list[dict[str, object]] = []
    totals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "dsc": [], "nsd": []})

    for case_name in tqdm(case_names, desc=f"Testing-set eval / {model.name}"):
        case_dir = case_dirs[case_name]
        label_image, gold = load_integer_mask(annotation_path(case_dir, args.annotation_name))
        prediction_image, prediction = load_integer_mask(pred_files[case_name])
        validate_grid(case_name, label_image, prediction_image, model, args.ignore_affine)

        # All metric code below assumes the same 0-3 target label space,
        # regardless of how the model originally stored its predictions.
        target_prediction = remap_prediction_to_target(prediction, model, case_name)
        if target_prediction.shape != gold.shape:
            raise ValueError(
                f"{case_name}: {model.name} remapped prediction shape {target_prediction.shape} "
                f"does not match annotation shape {gold.shape}"
            )

        spacing = tuple(float(value) for value in label_image.header.get_zooms()[:3])
        for label_value, class_name in TARGET_LABELS.items():
            # Metrics are calculated one class at a time by turning the integer
            # label map into binary masks for this specific label value.
            pred_mask = target_prediction == label_value
            gold_mask = gold == label_value
            dsc, nsd, tp, fp, fn = binary_metrics(
                pred_mask,
                gold_mask,
                spacing,
                args.nsd_tolerance_mm,
            )
            case_rows.append(
                {
                    "case": case_name,
                    "label": label_value,
                    "class": class_name,
                    "dsc": dsc,
                    "nsd": nsd,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )
            totals[class_name]["tp"] += tp
            totals[class_name]["fp"] += fp
            totals[class_name]["fn"] += fn
            if not math.isnan(dsc):
                totals[class_name]["dsc"].append(dsc)
            if not math.isnan(nsd):
                totals[class_name]["nsd"].append(nsd)

    summary_rows = []
    global_tp = global_fp = global_fn = 0
    all_case_dsc, all_case_nsd = [], []
    foreground_tp = foreground_fp = foreground_fn = 0
    foreground_case_dsc, foreground_case_nsd = [], []

    # In addition to all-label metrics, foreground metrics ignore background so
    # large easy background regions do not hide organ-segmentation errors.
    for label_value, class_name in TARGET_LABELS.items():
        item = totals[class_name]
        denominator = 2 * item["tp"] + item["fp"] + item["fn"]
        summary_rows.append(
            {
                "label": label_value,
                "class": class_name,
                "mean_case_dsc": np.mean(item["dsc"]) if item["dsc"] else math.nan,
                "mean_case_nsd": np.mean(item["nsd"]) if item["nsd"] else math.nan,
                "micro_dsc": 2.0 * item["tp"] / denominator if denominator else math.nan,
                "tp": item["tp"],
                "fp": item["fp"],
                "fn": item["fn"],
                "cases_with_dsc": len(item["dsc"]),
                "cases_with_nsd": len(item["nsd"]),
            }
        )
        global_tp += item["tp"]
        global_fp += item["fp"]
        global_fn += item["fn"]
        all_case_dsc.extend(item["dsc"])
        all_case_nsd.extend(item["nsd"])

        if label_value != 0:
            foreground_tp += item["tp"]
            foreground_fp += item["fp"]
            foreground_fn += item["fn"]
            foreground_case_dsc.extend(item["dsc"])
            foreground_case_nsd.extend(item["nsd"])

    global_denominator = 2 * global_tp + global_fp + global_fn
    foreground_denominator = 2 * foreground_tp + foreground_fp + foreground_fn
    overall = {
        "model": model.name,
        "prediction_dir": str(model.directory),
        "label_space": model.label_space,
        "cases_root": str(cases_root),
        "annotation_name": args.annotation_name,
        "cases": len(case_names),
        "nsd_tolerance_mm": args.nsd_tolerance_mm,
        "macro_case_dsc": float(np.mean(all_case_dsc)) if all_case_dsc else math.nan,
        "macro_case_nsd": float(np.mean(all_case_nsd)) if all_case_nsd else math.nan,
        "micro_dsc": 2.0 * global_tp / global_denominator if global_denominator else math.nan,
        "micro_dsc_foreground": (
            2.0 * foreground_tp / foreground_denominator if foreground_denominator else math.nan
        ),
        "macro_case_dsc_foreground": (
            float(np.mean(foreground_case_dsc)) if foreground_case_dsc else math.nan
        ),
        "macro_case_nsd_foreground": (
            float(np.mean(foreground_case_nsd)) if foreground_case_nsd else math.nan
        ),
        "tp": global_tp,
        "fp": global_fp,
        "fn": global_fn,
    }

    write_csv(
        model_output_dir / "per_case_per_class.csv",
        case_rows,
        ["case", "label", "class", "dsc", "nsd", "tp", "fp", "fn"],
    )
    write_csv(
        model_output_dir / "per_class_summary.csv",
        summary_rows,
        [
            "label",
            "class",
            "mean_case_dsc",
            "mean_case_nsd",
            "micro_dsc",
            "tp",
            "fp",
            "fn",
            "cases_with_dsc",
            "cases_with_nsd",
        ],
    )
    with (model_output_dir / "overall_summary.json").open("w") as handle:
        json.dump(overall, handle, indent=2)

    print(json.dumps(overall, indent=2))


def main() -> None:
    args = parse_args()
    if len(args.models) < 1:
        raise ValueError("Pass at least one --model entry.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_root = args.cases_root
    case_dirs = case_directories(cases_root)

    # Save the exact inputs used for the run beside the metric outputs.
    with (args.output_dir / "run_config.txt").open("w") as handle:
        handle.write(f"cases_root={cases_root}\n")
        handle.write(f"annotation_name={args.annotation_name}\n")
        for model in args.models:
            handle.write(
                f"model={model.name}\tlabel_space={model.label_space}"
                f"\tdirectory={model.directory}\n"
            )

    for model in args.models:
        evaluate_model(model, cases_root, case_dirs, args)


if __name__ == "__main__":
    main()
