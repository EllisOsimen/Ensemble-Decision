#!/usr/bin/env python3
"""Check how often model predictions match WORD ground-truth labels.

Prediction directories are configurable, so the same script can compare any
set of WORD-numbered models plus optional native BTCV predictions such as the
Swin UNETR 5050 checkpoint.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
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


# Convert native Swin 5050/BTCV labels into WORD IDs. BTCV-only structures use
# reserved values so they do not look like WORD background during comparison.
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


@dataclass(frozen=True)
class ModelSpec:
    name: str
    directory: Path
    label_space: str = "word"


def default_models() -> list[ModelSpec]:
    return [
        ModelSpec("unet", PROJECT_DIR / "results" / "word_three_models" / "unet"),
        ModelSpec(
            "swinunetr",
            PROJECT_DIR / "results" / "word_three_models" / "swinunetr",
        ),
        ModelSpec(
            "segresnet",
            PROJECT_DIR / "results" / "word_three_models" / "segresnet",
        ),
        ModelSpec(
            "swin5050",
            PROJECT_DIR / "results" / "word_swinunetr_5050",
            "btcv",
        ),
    ]


def parse_model_spec(value: str) -> ModelSpec:
    """Parse NAME=DIR or NAME=DIR:LABEL_SPACE."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--model must be formatted as NAME=DIR or NAME=DIR:LABEL_SPACE"
        )
    name, rest = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Model name cannot be empty.")

    label_space = "word"
    directory_text = rest
    if ":" in rest:
        possible_directory, possible_label_space = rest.rsplit(":", 1)
        if possible_label_space in {"word", "btcv"}:
            directory_text = possible_directory
            label_space = possible_label_space
    directory = Path(directory_text)
    if not directory.is_absolute():
        directory = PROJECT_DIR / directory
    return ModelSpec(name=name, directory=directory, label_space=label_space)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare configurable model prediction directories against WORD "
            "labelsTr. Use --model NAME=DIR or --model NAME=DIR:btcv for native "
            "BTCV predictions that must be remapped first."
        )
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=PROJECT_DIR.parent / "WORD" / "WORD-V0.1.0" / "labelsTr",
        help="Directory containing WORD ground-truth label maps.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        type=parse_model_spec,
        action="append",
        default=None,
        help=(
            "Model prediction directory. Format: NAME=DIR or NAME=DIR:LABEL_SPACE. "
            "LABEL_SPACE can be word or btcv. Repeat once per model."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "word_model_ground_truth_agreement",
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


def validate_models(models: list[ModelSpec]) -> None:
    if len(models) < 2:
        raise ValueError("Pass at least two --model entries.")
    names = [model.name for model in models]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"Duplicate model name(s): {duplicate_names}")
    bad_label_spaces = sorted(
        {model.label_space for model in models if model.label_space not in {"word", "btcv"}}
    )
    if bad_label_spaces:
        raise ValueError(f"Unsupported label space(s): {bad_label_spaces}")


def matching_cases(
    labels_dir: Path,
    models: list[ModelSpec],
    case_name: str | None,
) -> tuple[dict[str, Path], dict[str, dict[str, Path]], list[str]]:
    """Find case names that exist in every prediction directory and labelsTr."""
    label_files = nifti_files(labels_dir, "labelsTr")
    files_by_model = {
        model.name: nifti_files(model.directory, model.name)
        for model in models
    }

    reference_model = models[0]
    reference_names = set(files_by_model[reference_model.name])
    for model in models[1:]:
        names = set(files_by_model[model.name])
        if names != reference_names:
            missing = sorted(reference_names - names)
            extra = sorted(names - reference_names)
            details = []
            if missing:
                details.append(f"missing {len(missing)} case(s), e.g. {missing[:3]}")
            if extra:
                details.append(f"has {len(extra)} extra case(s), e.g. {extra[:3]}")
            raise ValueError(
                f"Prediction case mismatch for {model.name}: {'; '.join(details)}"
            )

    missing_labels = sorted(reference_names - set(label_files))
    if missing_labels:
        raise ValueError(
            "Ground-truth labelsTr is missing "
            f"{len(missing_labels)} case(s), e.g. {missing_labels[:3]}"
        )

    case_names = sorted(reference_names)
    if case_name is not None:
        if case_name not in case_names:
            raise FileNotFoundError(
                f"{case_name} is not present in all prediction directories."
            )
        case_names = [case_name]

    return label_files, files_by_model, case_names


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
    models: list[ModelSpec],
    ignore_affine: bool,
) -> None:
    """Make sure voxel indices line up before doing voxel-level comparisons."""
    for model, prediction_image in zip(models, prediction_images):
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
                "If you have inspected this and only want array-index comparison, "
                "rerun with --ignore-affine."
            )


def remap_btcv_to_word(btcv: np.ndarray, case_name: str, model_name: str) -> np.ndarray:
    unique_labels = np.unique(btcv).astype(int)
    unknown = sorted(set(unique_labels) - set(BTCV_TO_WORD))
    if unknown:
        raise ValueError(
            f"{case_name}: unexpected BTCV label(s) in {model_name}: {unknown}"
        )

    max_label = max(int(unique_labels.max()), max(BTCV_TO_WORD))
    lookup = np.zeros(max_label + 1, dtype=np.int16)
    for btcv_label, word_label in BTCV_TO_WORD.items():
        lookup[btcv_label] = word_label
    return lookup[btcv]


def correctness_columns(model_count: int) -> list[str]:
    return [f"correct_{count}_voxels" for count in range(model_count, -1, -1)]


def empty_counts(model_count: int) -> dict[str, int]:
    counts = {"total_gt_voxels": 0}
    for column in correctness_columns(model_count):
        counts[column] = 0
    return counts


def add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] += int(value)


def percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def make_row(
    case_name: str | None,
    label: int | None,
    counts: dict[str, int],
    model_count: int,
) -> dict[str, object]:
    total = counts["total_gt_voxels"]
    row: dict[str, object] = {}
    if case_name is not None:
        row["case"] = case_name
    if label is not None:
        row["label"] = label
        row["structure"] = label_name(label)

    row.update(counts)
    for column in correctness_columns(model_count):
        row[column.replace("_voxels", "_percent")] = round(
            percent(counts[column], total),
            6,
        )
    return row


def analyze_case(
    case_name: str,
    label_path: Path,
    model_paths: dict[str, Path],
    models: list[ModelSpec],
    labels_to_report: list[int],
    ignore_affine: bool,
) -> tuple[list[dict[str, object]], dict[str, object], dict[int, dict[str, int]]]:
    label_image, ground_truth = load_integer_mask(label_path)
    prediction_images = []
    predictions = []
    for model in models:
        prediction_image, prediction = load_integer_mask(model_paths[model.name])
        prediction_images.append(prediction_image)
        if model.label_space == "btcv":
            prediction = remap_btcv_to_word(prediction, case_name, model.name)
        predictions.append(prediction)

    validate_grid(case_name, label_image, prediction_images, models, ignore_affine)

    correct_count = np.zeros(ground_truth.shape, dtype=np.uint8)
    for prediction in predictions:
        correct_count += prediction == ground_truth

    model_count = len(models)
    per_label_rows = []
    per_label_counts = {}
    case_counts = empty_counts(model_count)
    max_reported_label = max(labels_to_report)

    included_voxels = np.isin(ground_truth, labels_to_report)
    included_labels = ground_truth[included_voxels].astype(np.int64, copy=False)
    included_correct_counts = correct_count[included_voxels].astype(
        np.int64,
        copy=False,
    )

    combined_index = included_labels * (model_count + 1) + included_correct_counts
    correctness_histogram = np.bincount(
        combined_index,
        minlength=(max_reported_label + 1) * (model_count + 1),
    ).reshape(max_reported_label + 1, model_count + 1)

    for label in labels_to_report:
        counts = empty_counts(model_count)
        histogram = correctness_histogram[label]
        counts["total_gt_voxels"] = int(histogram.sum())
        for correct_models in range(model_count, -1, -1):
            counts[f"correct_{correct_models}_voxels"] = int(histogram[correct_models])

        add_counts(case_counts, counts)
        per_label_counts[label] = counts
        per_label_rows.append(make_row(case_name, label, counts, model_count))

    return (
        per_label_rows,
        make_row(case_name, None, case_counts, model_count),
        per_label_counts,
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_run_config(path: Path, models: list[ModelSpec], labels_dir: Path) -> None:
    with path.open("w") as file:
        file.write(f"labels_dir={labels_dir}\n")
        for model in models:
            file.write(
                f"model={model.name}\tlabel_space={model.label_space}"
                f"\tdirectory={model.directory}\n"
            )


def main() -> None:
    args = parse_args()
    models = args.models if args.models is not None else default_models()
    validate_models(models)

    label_files, files_by_model, case_names = matching_cases(
        args.labels_dir,
        models,
        args.case_name,
    )

    labels_to_report = sorted(label for label in WORD_LABEL_NAMES if label != 0)
    if args.include_background:
        labels_to_report = [0] + labels_to_report

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_case_per_structure_rows = []
    per_case_rows = []
    model_count = len(models)
    aggregate_by_label = defaultdict(lambda: empty_counts(model_count))
    cases_with_label = defaultdict(int)

    for case_name in tqdm(case_names, desc="Checking agreement vs labelsTr"):
        model_paths = {
            model.name: files_by_model[model.name][case_name]
            for model in models
        }
        label_rows, case_row, per_label_counts = analyze_case(
            case_name,
            label_files[case_name],
            model_paths,
            models,
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
        row = make_row(None, label, aggregate_by_label[label], model_count)
        row["cases_with_structure"] = cases_with_label[label]
        row["case_count"] = len(case_names)
        per_structure_rows.append(row)

    correctness_fields = correctness_columns(model_count)
    correctness_percent_fields = [
        column.replace("_voxels", "_percent") for column in correctness_fields
    ]
    per_case_per_structure_fields = [
        "case",
        "label",
        "structure",
        "total_gt_voxels",
        *correctness_fields,
        *correctness_percent_fields,
    ]
    per_case_fields = [
        "case",
        "total_gt_voxels",
        *correctness_fields,
        *correctness_percent_fields,
    ]
    per_structure_fields = [
        "label",
        "structure",
        "case_count",
        "cases_with_structure",
        "total_gt_voxels",
        *correctness_fields,
        *correctness_percent_fields,
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
    write_run_config(args.output_dir / "run_config.txt", models, args.labels_dir)

    print(f"Processed {len(case_names)} case(s) with {model_count} model(s).")
    print(f"Wrote CSV files to: {args.output_dir}")


if __name__ == "__main__":
    main()
