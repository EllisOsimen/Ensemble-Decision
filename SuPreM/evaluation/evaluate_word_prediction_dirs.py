#!/usr/bin/env python3
"""Evaluate existing prediction directories against WORD labelsTr."""

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


WORD_LABELS = {
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


# BTCV and WORD use different integer IDs for the same organs. This lookup lets
# a BTCV-style prediction be evaluated against WORD labels without retraining or
# rewriting the prediction files. Values 101-103 are deliberately outside WORD
# so those non-WORD structures never count as a WORD foreground class.
BTCV_TO_WORD = {
    0: 0,
    1: 2,    # spleen
    2: 4,    # right kidney
    3: 3,    # left kidney
    4: 6,    # gallbladder
    5: 7,    # esophagus
    6: 1,    # liver
    7: 5,    # stomach
    8: 101,  # aorta: not in WORD
    9: 102,  # inferior vena cava: not in WORD
    10: 103, # portal/splenic veins: not in WORD
    11: 8,   # pancreas
    12: 12,  # right adrenal -> WORD adrenal
    13: 12,  # left adrenal -> WORD adrenal
}


@dataclass(frozen=True)
class ModelSpec:
    """Command-line description of one prediction directory to evaluate."""

    name: str
    directory: Path
    label_space: str = "word"


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
    label_space = "word"
    if ":" in rest:
        possible_directory, possible_label_space = rest.rsplit(":", 1)
        if possible_label_space in {"word", "btcv"}:
            directory_text = possible_directory
            label_space = possible_label_space
    directory = Path(directory_text)
    if not directory.is_absolute():
        directory = PROJECT_DIR / directory
    return ModelSpec(name, directory, label_space)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate existing WORD/BTCV prediction label maps against WORD "
            "labelsTr and write Dice/NSD summaries."
        )
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=PROJECT_DIR.parent / "WORD" / "WORD-V0.1.0" / "labelsTr",
    )
    parser.add_argument(
        "--model",
        dest="models",
        type=parse_model_spec,
        action="append",
        required=True,
        help="Format: NAME=DIR or NAME=DIR:btcv. Repeat once per model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "word_prediction_dir_evaluation",
    )
    parser.add_argument("--case-name", default=None)
    parser.add_argument("--nsd-tolerance-mm", type=float, default=1.0)
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


def load_integer_mask(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load a NIfTI label map and make sure it contains class IDs, not probabilities."""

    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if not np.issubdtype(data.dtype, np.integer):
        if not np.all(np.equal(data, np.round(data))):
            raise ValueError(f"{path} contains non-integer labels.")
        data = np.round(data)
    return image, data.astype(np.int16, copy=False)


def remap_btcv_to_word(prediction: np.ndarray, case_name: str, model_name: str) -> np.ndarray:
    """Convert a BTCV-numbered prediction into WORD label IDs using a lookup table."""

    unique_labels = np.unique(prediction).astype(int)
    unknown = sorted(set(unique_labels) - set(BTCV_TO_WORD))
    if unknown:
        raise ValueError(
            f"{case_name}: unexpected BTCV label(s) in {model_name}: {unknown}"
        )

    max_label = max(int(unique_labels.max()), max(BTCV_TO_WORD))
    lookup = np.zeros(max_label + 1, dtype=np.int16)
    for btcv_label, word_label in BTCV_TO_WORD.items():
        lookup[btcv_label] = word_label
    return lookup[prediction]


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


def prediction_files(directory: Path) -> dict[str, Path]:
    """Return predictions keyed by filename, matching the WORD labelsTr names."""

    if not directory.is_dir():
        raise NotADirectoryError(f"Prediction directory does not exist: {directory}")
    files = {path.name: path for path in directory.glob("*.nii.gz")}
    if not files:
        raise FileNotFoundError(f"No .nii.gz predictions found in {directory}")
    return files


def labels_files(directory: Path) -> dict[str, Path]:
    """Return gold WORD labels keyed by filename."""

    if not directory.is_dir():
        raise NotADirectoryError(f"labelsTr directory does not exist: {directory}")
    files = {path.name: path for path in directory.glob("*.nii.gz")}
    if not files:
        raise FileNotFoundError(f"No .nii.gz labels found in {directory}")
    return files


def validate_grid(case_name, label_image, prediction_image, model, ignore_affine):
    """Confirm prediction and gold label occupy the same voxel grid."""

    # Shape equality checks array dimensions. Affine equality checks that those
    # array indices refer to the same physical coordinates in scanner space.
    if prediction_image.shape != label_image.shape:
        raise ValueError(
            f"{case_name}: {model.name} shape {prediction_image.shape} "
            f"does not match labelsTr shape {label_image.shape}"
        )
    if not ignore_affine and not np.allclose(
        prediction_image.affine,
        label_image.affine,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError(
            f"{case_name}: {model.name} affine does not match labelsTr. "
            "Rerun with --ignore-affine only if array-index comparison is intended."
        )


def evaluate_model(model: ModelSpec, label_files, args) -> None:
    """Evaluate one model directory and write its CSV/JSON reports."""

    pred_files = prediction_files(model.directory)
    case_names = sorted(pred_files)
    if args.case_name is not None:
        if args.case_name not in pred_files:
            raise FileNotFoundError(f"{args.case_name} not found in {model.directory}")
        case_names = [args.case_name]

    missing_labels = sorted(set(case_names) - set(label_files))
    if missing_labels:
        raise FileNotFoundError(
            f"labelsTr is missing {len(missing_labels)} case(s), e.g. {missing_labels[:3]}"
        )

    model_output_dir = args.output_dir / model.name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    case_rows = []
    totals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "dsc": [], "nsd": []})

    for case_name in tqdm(case_names, desc=f"WORD eval / {model.name}"):
        label_image, gold = load_integer_mask(label_files[case_name])
        prediction_image, prediction = load_integer_mask(pred_files[case_name])
        validate_grid(case_name, label_image, prediction_image, model, args.ignore_affine)
        if model.label_space == "btcv":
            # After this point every prediction is treated as WORD-numbered.
            prediction = remap_btcv_to_word(prediction, case_name, model.name)

        spacing = tuple(float(value) for value in label_image.header.get_zooms()[:3])
        for word_label, class_name in WORD_LABELS.items():
            # Metrics are calculated one organ at a time by turning the integer
            # label map into binary masks for this specific WORD class.
            pred_mask = prediction == word_label
            gold_mask = gold == word_label
            dsc, nsd, tp, fp, fn = binary_metrics(
                pred_mask,
                gold_mask,
                spacing,
                args.nsd_tolerance_mm,
            )
            case_rows.append(
                {
                    "case": case_name,
                    "word_label": word_label,
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

    # mean_case_* averages per-case scores, while micro_dsc recomputes Dice from
    # summed TP/FP/FN counts across all cases. They answer slightly different
    # questions, so both are written.
    summary_rows = []
    global_tp = global_fp = global_fn = 0
    all_case_dsc, all_case_nsd = [], []
    for class_name in WORD_LABELS.values():
        item = totals[class_name]
        denominator = 2 * item["tp"] + item["fp"] + item["fn"]
        summary_rows.append(
            {
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

    global_denominator = 2 * global_tp + global_fp + global_fn
    overall = {
        "model": model.name,
        "prediction_dir": str(model.directory),
        "label_space": model.label_space,
        "cases": len(case_names),
        "nsd_tolerance_mm": args.nsd_tolerance_mm,
        "macro_case_dsc": float(np.mean(all_case_dsc)) if all_case_dsc else math.nan,
        "macro_case_nsd": float(np.mean(all_case_nsd)) if all_case_nsd else math.nan,
        "micro_dsc": 2.0 * global_tp / global_denominator if global_denominator else math.nan,
        "tp": global_tp,
        "fp": global_fp,
        "fn": global_fn,
    }

    write_csv(
        model_output_dir / "per_case_per_class.csv",
        case_rows,
        ["case", "word_label", "class", "dsc", "nsd", "tp", "fp", "fn"],
    )
    write_csv(
        model_output_dir / "per_class_summary.csv",
        summary_rows,
        [
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_files = labels_files(args.labels_dir)
    # Save the exact inputs used for the run beside the metric outputs.
    with (args.output_dir / "run_config.txt").open("w") as handle:
        handle.write(f"labels_dir={args.labels_dir}\n")
        for model in args.models:
            handle.write(
                f"model={model.name}\tlabel_space={model.label_space}"
                f"\tdirectory={model.directory}\n"
            )
    for model in args.models:
        evaluate_model(model, label_files, args)


if __name__ == "__main__":
    main()
