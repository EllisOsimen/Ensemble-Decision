#!/usr/bin/env python3
"""Evaluate 3D Dataset prediction directories against LabelsTr masks.

The 3D Dataset gold labels use this label space:

  0 = background
  1 = liver
  2 = spleen
  3 = leftkidney
  4 = rightkidney
  5 = leftlung
  6 = rightlung
  7 = bone
  8 = skin
  9 = artery
 10 = portalvein
 11 = venoussystem

The three pretrained inference outputs do not all support every structure in
that label space. This script evaluates only the labels supported by all three
default prediction spaces: liver, spleen, leftkidney, and rightkidney.
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


DATASET_LABELS = {
    1: "liver",
    2: "spleen",
    3: "leftkidney",
    4: "rightkidney",
    5: "leftlung",
    6: "rightlung",
    7: "bone",
    8: "skin",
    9: "artery",
    10: "portalvein",
    11: "venoussystem",
}


# Only these structures are present in all three default model output spaces:
#   - SuPreM/CLIP and SegResNet save WORD-numbered labels 0-16.
#   - Swin5050 saves BTCV-numbered labels 0-13.
EVALUATED_LABELS = {
    1: "liver",
    2: "spleen",
    3: "leftkidney",
    4: "rightkidney",
}

IGNORED_LABELS = {
    label: name
    for label, name in DATASET_LABELS.items()
    if label not in EVALUATED_LABELS
}


# Native BTCV labels -> 3D Dataset labels.
BTCV_TO_DATASET = {
    0: 0,
    1: 2,  # spleen
    2: 4,  # right kidney
    3: 3,  # left kidney
    4: 0,  # gallbladder
    5: 0,  # esophagus
    6: 1,  # liver
    7: 0,  # stomach
    8: 0,  # aorta, ignored because not all three models save an artery label
    9: 0,  # inferior vena cava, ignored
    10: 0,  # portal/splenic veins, ignored
    11: 0,  # pancreas
    12: 0,  # right adrenal gland
    13: 0,  # left adrenal gland
}


# Saved SuPreM/CLIP/SegResNet predictions are WORD-numbered label maps.
# WORD labels 1-4 happen to match the requested 3D Dataset labels exactly.
SUPREM_TO_DATASET = {
    0: 0,
    1: 1,  # liver
    2: 2,  # spleen
    3: 3,  # left kidney
    4: 4,  # right kidney
    5: 0,  # stomach
    6: 0,  # gallbladder
    7: 0,  # esophagus
    8: 0,  # pancreas
    9: 0,  # duodenum
    10: 0,  # colon
    11: 0,  # intestine
    12: 0,  # adrenal
    13: 0,  # rectum
    14: 0,  # bladder
    15: 0,  # head_of_femur_left
    16: 0,  # head_of_femur_right
}


SUPPORTED_LABEL_SPACES = {"dataset", "target", "btcv", "suprem"}


@dataclass(frozen=True)
class ModelSpec:
    """Command-line description of one prediction directory to evaluate."""

    name: str
    directory: Path
    label_space: str


DEFAULT_MODELS = (
    ModelSpec(
        "swin5050",
        PROJECT_DIR / "results" / "3d_dataset_inference" / "swinunetr_5050",
        "btcv",
    ),
    ModelSpec(
        "clip_unet",
        PROJECT_DIR / "results" / "3d_dataset_inference" / "clip_universal_unet",
        "suprem",
    ),
    ModelSpec(
        "segresnet",
        PROJECT_DIR / "results" / "3d_dataset_inference" / "suprem_segresnet",
        "suprem",
    ),
)


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
            "Evaluate 3D Dataset predictions against Dataset/LabelsTr, ignoring "
            "dataset structures that are not supported by all three default models."
        )
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=PROJECT_DIR.parent / "Dataset" / "LabelsTr",
        help="Directory containing *_multiorgan.nii.gz gold label maps.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        type=parse_model_spec,
        action="append",
        default=None,
        help=(
            "Prediction directory. Format: NAME=DIR or NAME=DIR:LABEL_SPACE. "
            "LABEL_SPACE can be dataset, target, btcv, or suprem. If omitted, "
            "the three results/3d_dataset_inference model directories are used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "3d_dataset_prediction_evaluation",
        help="Directory where CSV and JSON summaries will be written.",
    )
    parser.add_argument(
        "--case-name",
        default=None,
        help="Process only this case name, e.g. 3Dircadb1.4.",
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


def case_name_from_label_path(path: Path) -> str:
    """Convert 3Dircadb1.4_multiorgan.nii.gz -> 3Dircadb1.4."""

    name = path.name
    if name.endswith("_multiorgan.nii.gz"):
        return name[: -len("_multiorgan.nii.gz")]
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return path.stem


def case_name_from_prediction_path(path: Path) -> str:
    """Convert 3Dircadb1.4.nii.gz -> 3Dircadb1.4."""

    if path.name.endswith(".nii.gz"):
        return path.name[: -len(".nii.gz")]
    return path.stem


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


def label_files(directory: Path) -> dict[str, Path]:
    """Return gold 3D Dataset labels keyed by case name."""

    if not directory.is_dir():
        raise NotADirectoryError(f"Labels directory does not exist: {directory}")
    files = {
        case_name_from_label_path(path): path
        for path in directory.glob("*.nii.gz")
        if path.is_file()
    }
    if not files:
        raise FileNotFoundError(f"No .nii.gz labels found in {directory}")
    return files


def prediction_files(directory: Path) -> dict[str, Path]:
    """Return predictions keyed by case name."""

    if not directory.is_dir():
        raise NotADirectoryError(f"Prediction directory does not exist: {directory}")
    files = {
        case_name_from_prediction_path(path): path
        for path in directory.glob("*.nii.gz")
        if path.is_file()
    }
    if not files:
        raise FileNotFoundError(f"No .nii.gz predictions found in {directory}")
    return files


def validate_grid(case_name, label_image, prediction_image, model, ignore_affine):
    """Confirm prediction and gold label occupy the same voxel grid."""

    if prediction_image.shape != label_image.shape:
        raise ValueError(
            f"{case_name}: {model.name} shape {prediction_image.shape} does not match "
            f"label shape {label_image.shape}"
        )
    if not ignore_affine and not np.allclose(
        prediction_image.affine,
        label_image.affine,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError(
            f"{case_name}: {model.name} affine does not match the gold label. "
            "Rerun with --ignore-affine only if array-index comparison is intended."
        )


def remap_with_lookup(
    prediction: np.ndarray,
    lookup_map: dict[int, int],
    case_name: str,
    model_name: str,
    label_space: str,
) -> np.ndarray:
    """Convert a prediction label map into the 3D Dataset label IDs."""

    unique_labels = np.unique(prediction).astype(int)
    unknown = sorted(set(unique_labels) - set(lookup_map))
    if unknown:
        raise ValueError(
            f"{case_name}: unexpected {label_space} label(s) in {model_name}: {unknown}"
        )

    lookup = np.zeros(max(max(lookup_map), int(unique_labels.max())) + 1, dtype=np.int16)
    for source_label, dataset_label in lookup_map.items():
        lookup[source_label] = dataset_label
    return lookup[prediction]


def remap_prediction_to_dataset(
    prediction: np.ndarray,
    model: ModelSpec,
    case_name: str,
) -> np.ndarray:
    """Normalize any supported prediction space into the 3D Dataset label IDs."""

    if model.label_space in {"dataset", "target"}:
        unique_labels = np.unique(prediction).astype(int)
        allowed = set(DATASET_LABELS) | {0}
        unknown = sorted(set(unique_labels) - allowed)
        if unknown:
            raise ValueError(
                f"{case_name}: unexpected dataset label(s) in {model.name}: {unknown}"
            )
        return prediction.astype(np.int16, copy=False)

    if model.label_space == "btcv":
        return remap_with_lookup(
            prediction,
            BTCV_TO_DATASET,
            case_name,
            model.name,
            model.label_space,
        )

    if model.label_space == "suprem":
        return remap_with_lookup(
            prediction,
            SUPREM_TO_DATASET,
            case_name,
            model.name,
            model.label_space,
        )

    raise ValueError(f"Unsupported label space: {model.label_space}")


def binary_metrics(prediction, gold, spacing, tolerance_mm):
    """Compute Dice, normalized surface Dice, and voxel confusion counts."""

    prediction = np.asarray(prediction, dtype=bool)
    gold = np.asarray(gold, dtype=bool)
    tp = int(np.logical_and(prediction, gold).sum())
    fp = int(np.logical_and(prediction, ~gold).sum())
    fn = int(np.logical_and(~prediction, gold).sum())
    denominator = 2 * tp + fp + fn
    dsc = (2.0 * tp / denominator) if denominator else math.nan

    if prediction.any() and gold.any():
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


def evaluate_model(model: ModelSpec, gold_files: dict[str, Path], args) -> None:
    """Evaluate one model directory and write its CSV/JSON reports."""

    pred_files = prediction_files(model.directory)
    case_names = sorted(set(pred_files) & set(gold_files))

    if args.case_name is not None:
        if args.case_name not in pred_files:
            raise FileNotFoundError(f"{args.case_name} not found in {model.directory}")
        if args.case_name not in gold_files:
            raise FileNotFoundError(f"{args.case_name} not found in {args.labels_dir}")
        case_names = [args.case_name]

    missing_predictions = sorted(set(gold_files) - set(pred_files))
    missing_labels = sorted(set(pred_files) - set(gold_files))
    if missing_predictions:
        print(
            f"WARNING: {model.name} is missing {len(missing_predictions)} prediction(s), "
            f"e.g. {missing_predictions[:3]}"
        )
    if missing_labels:
        print(
            f"WARNING: {model.name} has {len(missing_labels)} prediction(s) without a "
            f"matching gold label, e.g. {missing_labels[:3]}"
        )
    if not case_names:
        raise FileNotFoundError(
            f"No matching label/prediction pairs found for {model.name} in "
            f"{model.directory} and {args.labels_dir}"
        )

    model_output_dir = args.output_dir / model.name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    case_rows: list[dict[str, object]] = []
    totals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "dsc": [], "nsd": []})

    for case_name in tqdm(case_names, desc=f"3D Dataset eval / {model.name}"):
        label_image, gold = load_integer_mask(gold_files[case_name])
        prediction_image, prediction = load_integer_mask(pred_files[case_name])
        validate_grid(case_name, label_image, prediction_image, model, args.ignore_affine)

        dataset_prediction = remap_prediction_to_dataset(prediction, model, case_name)
        if dataset_prediction.shape != gold.shape:
            raise ValueError(
                f"{case_name}: {model.name} remapped prediction shape "
                f"{dataset_prediction.shape} does not match label shape {gold.shape}"
            )

        spacing = tuple(float(value) for value in label_image.header.get_zooms()[:3])
        for label_value, class_name in EVALUATED_LABELS.items():
            pred_mask = dataset_prediction == label_value
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

    for label_value, class_name in EVALUATED_LABELS.items():
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

    global_denominator = 2 * global_tp + global_fp + global_fn
    overall = {
        "model": model.name,
        "prediction_dir": str(model.directory),
        "label_space": model.label_space,
        "labels_dir": str(args.labels_dir),
        "cases": len(case_names),
        "nsd_tolerance_mm": args.nsd_tolerance_mm,
        "evaluated_labels": EVALUATED_LABELS,
        "ignored_labels": IGNORED_LABELS,
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
    args.models = args.models or list(DEFAULT_MODELS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gold_files = label_files(args.labels_dir)

    with (args.output_dir / "run_config.txt").open("w") as handle:
        handle.write(f"labels_dir={args.labels_dir}\n")
        handle.write(f"evaluated_labels={EVALUATED_LABELS}\n")
        handle.write(f"ignored_labels={IGNORED_LABELS}\n")
        for model in args.models:
            handle.write(
                f"model={model.name}\tlabel_space={model.label_space}"
                f"\tdirectory={model.directory}\n"
            )

    for model in args.models:
        evaluate_model(model, gold_files, args)


if __name__ == "__main__":
    main()
