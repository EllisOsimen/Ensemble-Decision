#!/usr/bin/env python3
"""Patient-level five-fold CV and tuning for the CURVAS stacking forest.

The folds are formed before voxel sampling, so a patient's voxels can never
appear in both the training and held-out portions of a fold.  The separate
``validation_set`` is deliberately not accepted by this script: it remains
untouched for the final model evaluation.

Example
-------
python -u ensemble_agreement/random_forest_cross_validation.py \
  --near-organ-background \
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

import random_forest_stacking_consensus as stacking


DEFAULT_OUTPUT_DIR = (
    stacking.ALL_CURVAS_INFERENCE_ROOT / "random_forest_cross_validation_with_confidence"
)
ORGAN_NAMES = {1: "pancreas", 2: "kidney", 3: "liver"}
PATTERN_COUNT = 4 ** 3


@dataclass(frozen=True)
class Fold:
    number: int
    train_ids: tuple[str, ...]
    holdout_ids: tuple[str, ...]


@dataclass(frozen=True)
class ConfidenceFeatureCounts:
    """Exact target counts for each observed label/confidence feature row."""

    feature_keys: np.ndarray
    target_counts: np.ndarray
    invalid_target_counts: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run patient-level five-fold cross-validation and tune the CURVAS "
            "random-forest stacking classifier."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-prediction-root",
        type=Path,
        default=stacking.DEFAULT_TRAIN_PREDICTION_ROOT,
    )
    parser.add_argument(
        "--train-cases-root",
        type=Path,
        default=stacking.DEFAULT_TRAIN_CASES_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-dir",
        dest="model_dirs",
        type=Path,
        nargs="+",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--model-id",
        dest="model_ids",
        choices=sorted(stacking.SUPPORTED_MODEL_IDS),
        nargs="+",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--label-space",
        dest="label_spaces",
        choices=sorted(stacking.SUPPORTED_LABEL_SPACES),
        nargs="+",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--confidence-dir",
        dest="confidence_dirs",
        type=Path,
        nargs="+",
        action="append",
        default=None,
        help=(
            "Optional confidence directory for each model, matching --model-dir "
            "order. Pass all three to tune the 15-feature confidence model."
        ),
    )
    parser.add_argument(
        "--validity-dir",
        dest="validity_dirs",
        type=Path,
        nargs="+",
        action="append",
        default=None,
        help=(
            "Optional binary inference-validity directory for each model. Pass "
            "all three to exclude crop-exterior voxels from forest fitting and "
            "assign them deterministic background during held-out scoring."
        ),
    )
    parser.add_argument("--target-annotation", default="annotation_1.nii.gz")
    parser.add_argument("--samples-per-class", type=int, default=5000)
    parser.add_argument("--background-samples-per-case", type=int, default=10000)
    parser.add_argument("--near-organ-background", action="store_true")
    parser.add_argument("--dilation-iterations", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--predict-chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--n-estimators-grid",
        type=int,
        nargs="+",
        default=[100, 300],
    )
    parser.add_argument(
        "--max-depth-grid",
        type=stacking.parse_optional_max_depth,
        nargs="+",
        default=[5, 10, None],
    )
    parser.add_argument(
        "--min-samples-leaf-grid",
        type=int,
        nargs="+",
        default=[1, 50],
    )
    parser.add_argument(
        "--max-features-grid",
        choices=("sqrt", "all"),
        nargs="+",
        default=["sqrt", "all"],
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Exclude cases with missing/invalid inputs before creating folds.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if args.samples_per_class <= 0 or args.background_samples_per_case <= 0:
        raise ValueError("Voxel sampling limits must be positive.")
    if args.predict_chunk_size <= 0:
        raise ValueError("--predict-chunk-size must be positive.")
    if any(value <= 0 for value in args.n_estimators_grid):
        raise ValueError("Every n_estimators value must be positive.")
    if any(value <= 0 for value in args.min_samples_leaf_grid):
        raise ValueError("Every min_samples_leaf value must be positive.")


def make_folds(
    case_ids: list[str] | tuple[str, ...],
    n_splits: int,
    random_state: int,
) -> list[Fold]:
    """Make deterministic shuffled folds without requiring sklearn at import."""

    ordered_ids = np.asarray(sorted(case_ids), dtype=object)
    if n_splits > len(ordered_ids):
        raise ValueError(
            f"Cannot create {n_splits} folds from only {len(ordered_ids)} cases."
        )
    shuffled_indices = np.arange(len(ordered_ids))
    np.random.RandomState(random_state).shuffle(shuffled_indices)

    folds: list[Fold] = []
    for number, holdout_indices in enumerate(
        np.array_split(shuffled_indices, n_splits),
        start=1,
    ):
        holdout_set = set(int(index) for index in holdout_indices)
        train_ids = tuple(
            str(case_id)
            for index, case_id in enumerate(ordered_ids)
            if index not in holdout_set
        )
        holdout_ids = tuple(
            str(ordered_ids[index]) for index in sorted(holdout_set)
        )
        folds.append(Fold(number, train_ids, holdout_ids))
    return folds


def parameter_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    """Return a stable, de-duplicated Cartesian parameter grid."""

    configurations = []
    seen = set()
    for values in product(
        args.n_estimators_grid,
        args.max_depth_grid,
        args.min_samples_leaf_grid,
        args.max_features_grid,
    ):
        if values in seen:
            continue
        seen.add(values)
        configurations.append(
            {
                "n_estimators": values[0],
                "max_depth": values[1],
                "min_samples_leaf": values[2],
                "max_features": values[3],
            }
        )
    return configurations


def pattern_feature_matrix() -> np.ndarray:
    """One-hot features for all 64 possible triples of model decisions."""

    codes = np.arange(PATTERN_COUNT, dtype=np.uint8)
    predictions = [codes // 16, (codes // 4) % 4, codes % 4]
    return stacking.one_hot_encode_predictions(predictions)


def build_case_pattern_counts(
    case_id: str,
    case_dir: Path,
    prediction_root: Path,
    specs: list[stacking.ModelSpec],
    target_annotation: str,
) -> np.ndarray:
    """Count target labels for each base-model prediction combination."""

    target_path = case_dir / target_annotation
    target, target_image = stacking.load_curvas_target(target_path, case_id)
    predictions, reference_image, prediction_paths = stacking.load_case_predictions(
        prediction_root,
        specs,
        case_id,
    )
    stacking.validate_image_grid(
        case_id,
        target_image,
        target_path,
        reference_image,
        prediction_paths[0],
        role="cross-validation target",
    )

    pattern_codes = (
        predictions[0].reshape(-1).astype(np.int16) * 16
        + predictions[1].reshape(-1).astype(np.int16) * 4
        + predictions[2].reshape(-1).astype(np.int16)
    )
    joint_codes = target.reshape(-1).astype(np.int16) * PATTERN_COUNT + pattern_codes
    return np.bincount(
        joint_codes,
        minlength=len(stacking.CURVAS_LABELS) * PATTERN_COUNT,
    ).reshape(len(stacking.CURVAS_LABELS), PATTERN_COUNT)


def load_scaled_uint8_confidence_codes(
    prediction_root: Path,
    spec: stacking.ModelSpec,
    case_id: str,
    expected_shape: tuple[int, ...],
    expected_affine: np.ndarray,
) -> tuple[np.ndarray, Path]:
    """Load stored confidence bytes directly to keep CV memory practical."""

    if spec.confidence_dir is None:
        raise RuntimeError(f"No confidence directory configured for {spec.model_id}.")
    path = stacking.discover_confidence_file(
        prediction_root,
        spec.confidence_dir,
        case_id,
    )
    if path is None:
        confidence_root = stacking.resolve_path(spec.confidence_dir, prediction_root)
        raise FileNotFoundError(
            f"Missing confidence map for case {case_id} in {confidence_root}"
        )

    image = stacking.nib.load(str(path))
    if image.shape != expected_shape:
        raise ValueError(
            f"Confidence shape mismatch for case {case_id}: {path} has "
            f"{image.shape}, expected {expected_shape}."
        )
    if not np.allclose(
        image.affine,
        expected_affine,
        rtol=stacking.AFFINE_RTOL,
        atol=stacking.AFFINE_ATOL,
    ):
        raise ValueError(f"Confidence affine mismatch for case {case_id}: {path}.")
    slope = float(image.dataobj.slope)
    intercept = float(image.dataobj.inter)
    if image.get_data_dtype() != np.dtype(np.uint8) or not np.isclose(
        slope,
        1.0 / 255.0,
        rtol=1e-5,
        atol=1e-8,
    ) or not np.isclose(intercept, 0.0, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"{path} is not a scaled uint8 confidence map. Confidence CV "
            "currently requires the default uint8 storage mode."
        )
    return np.asarray(image.dataobj.get_unscaled(), dtype=np.uint8), path


def build_case_confidence_counts(
    case_id: str,
    case_dir: Path,
    prediction_root: Path,
    specs: list[stacking.ModelSpec],
    target_annotation: str,
) -> ConfidenceFeatureCounts:
    """Compress exact voxel counts by observed labels and uint8 confidences.

    Three 2-bit labels and three 8-bit confidences fit in one 30-bit key. This
    preserves every observed feature row while avoiding a dense table over the
    more than one billion theoretically possible combinations.
    """

    target_path = case_dir / target_annotation
    target, target_image = stacking.load_curvas_target(target_path, case_id)
    predictions, reference_image, prediction_paths = stacking.load_case_predictions(
        prediction_root,
        specs,
        case_id,
    )
    stacking.validate_image_grid(
        case_id,
        target_image,
        target_path,
        reference_image,
        prediction_paths[0],
        role="cross-validation target",
    )

    confidence_codes = [
        load_scaled_uint8_confidence_codes(
            prediction_root,
            spec,
            case_id,
            target.shape,
            reference_image.affine,
        )[0]
        for spec in specs
    ]
    flat_predictions = [prediction.reshape(-1).astype(np.uint32) for prediction in predictions]
    confidence_codes = [codes.reshape(-1).astype(np.uint32) for codes in confidence_codes]

    feature_keys = (
        flat_predictions[0]
        | (flat_predictions[1] << 2)
        | (flat_predictions[2] << 4)
        | (confidence_codes[0] << 6)
        | (confidence_codes[1] << 14)
        | (confidence_codes[2] << 22)
    )
    target_flat = target.reshape(-1).astype(np.uint64)
    invalid_target_counts = np.zeros(len(stacking.CURVAS_LABELS), dtype=np.int64)
    validity_mask, _ = stacking.load_common_validity_mask(
        prediction_root,
        specs,
        case_id,
        target.shape,
        reference_image.affine,
    )
    if validity_mask is not None:
        valid_flat = validity_mask.reshape(-1)
        invalid_target_counts = np.bincount(
            target_flat[~valid_flat].astype(np.intp, copy=False),
            minlength=len(stacking.CURVAS_LABELS),
        ).astype(np.int64, copy=False)
        feature_keys = feature_keys[valid_flat]
        target_flat = target_flat[valid_flat]
    joint_keys = (feature_keys.astype(np.uint64) << 2) | target_flat
    unique_joint, joint_counts = np.unique(joint_keys, return_counts=True)

    observed_feature_keys = (unique_joint >> 2).astype(np.uint32)
    observed_targets = (unique_joint & 0b11).astype(np.intp)
    unique_features, inverse = np.unique(observed_feature_keys, return_inverse=True)
    target_counts = np.zeros(
        (unique_features.size, len(stacking.CURVAS_LABELS)),
        dtype=np.int64,
    )
    np.add.at(target_counts, (inverse, observed_targets), joint_counts)
    return ConfidenceFeatureCounts(
        unique_features,
        target_counts,
        invalid_target_counts,
    )


def feature_matrix_from_keys(feature_keys: np.ndarray) -> np.ndarray:
    """Decode packed confidence-CV keys into the forest's 15 input columns."""

    keys = np.asarray(feature_keys, dtype=np.uint32)
    predictions = [
        (keys & 0b11).astype(np.uint8),
        ((keys >> 2) & 0b11).astype(np.uint8),
        ((keys >> 4) & 0b11).astype(np.uint8),
    ]
    confidences = [
        ((keys >> 6) & 0xFF).astype(np.float32) / 255.0,
        ((keys >> 14) & 0xFF).astype(np.float32) / 255.0,
        ((keys >> 22) & 0xFF).astype(np.float32) / 255.0,
    ]
    return stacking.one_hot_encode_predictions(predictions, confidences)


def dice_scores_from_confidence_counts(
    table: ConfidenceFeatureCounts,
    classifier,
    predict_chunk_size: int,
) -> dict[str, float]:
    """Calculate exact Dice by weighting every observed feature combination."""

    predicted_totals = np.zeros(len(stacking.CURVAS_LABELS), dtype=np.int64)
    true_positives = np.zeros(len(stacking.CURVAS_LABELS), dtype=np.int64)
    gold_totals = table.target_counts.sum(axis=0) + table.invalid_target_counts
    # Invalid crop-exterior voxels bypass the forest and are fixed background.
    predicted_totals[0] = int(table.invalid_target_counts.sum())
    true_positives[0] = int(table.invalid_target_counts[0])

    for start in range(0, table.feature_keys.size, predict_chunk_size):
        stop = min(start + predict_chunk_size, table.feature_keys.size)
        features = feature_matrix_from_keys(table.feature_keys[start:stop])
        predicted = classifier.predict(features).astype(np.intp, copy=False)
        counts = table.target_counts[start:stop]
        row_totals = counts.sum(axis=1)
        predicted_totals += np.bincount(
            predicted,
            weights=row_totals,
            minlength=len(stacking.CURVAS_LABELS),
        ).astype(np.int64)
        for label in stacking.CURVAS_LABELS:
            selected = predicted == label
            if np.any(selected):
                true_positives[label] += int(counts[selected, label].sum())

    scores: dict[str, float] = {}
    organ_scores = []
    for label, name in ORGAN_NAMES.items():
        denominator = int(predicted_totals[label] + gold_totals[label])
        score = (
            2.0 * int(true_positives[label]) / denominator
            if denominator
            else math.nan
        )
        scores[f"{name}_dice"] = score
        organ_scores.append(score)
    scores["mean_foreground_dice"] = float(np.nanmean(organ_scores))
    return scores


def dice_scores_from_pattern_counts(
    counts: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, float]:
    """Calculate exact per-organ Dice without materializing a prediction mask."""

    if counts.shape != (len(stacking.CURVAS_LABELS), PATTERN_COUNT):
        raise ValueError(f"Expected a (4, 64) count table; got {counts.shape}.")
    if predicted_labels.shape != (PATTERN_COUNT,):
        raise ValueError("Expected one predicted label for each of the 64 patterns.")

    scores: dict[str, float] = {}
    organ_scores = []
    for label, name in ORGAN_NAMES.items():
        predicted_patterns = predicted_labels == label
        true_positive = int(counts[label, predicted_patterns].sum())
        predicted_positive = int(counts[:, predicted_patterns].sum())
        gold_positive = int(counts[label, :].sum())
        denominator = predicted_positive + gold_positive
        score = (2.0 * true_positive / denominator) if denominator else math.nan
        scores[f"{name}_dice"] = score
        organ_scores.append(score)
    scores["mean_foreground_dice"] = float(np.nanmean(organ_scores))
    return scores


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    parameters: dict[str, object],
    random_state: int,
):
    max_features = (
        None if parameters["max_features"] == "all" else parameters["max_features"]
    )
    classifier = stacking.RandomForestClassifier(
        n_estimators=parameters["n_estimators"],
        max_depth=parameters["max_depth"],
        min_samples_leaf=parameters["min_samples_leaf"],
        max_features=max_features,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    classifier.fit(X, y)
    return classifier


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.nanmean(array))


def finite_std(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.std(finite, ddof=1)) if finite.size > 1 else math.nan


def summarise_configuration(
    config_id: str,
    parameters: dict[str, object],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    summary: dict[str, object] = {"config_id": config_id, **parameters}
    for metric in (
        "pancreas_dice",
        "kidney_dice",
        "liver_dice",
        "mean_foreground_dice",
    ):
        values = [float(row[metric]) for row in rows]
        summary[f"mean_{metric}"] = finite_mean(values)
        summary[f"std_{metric}"] = finite_std(values)
    summary["evaluated_cases"] = len(rows)
    return summary


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    tracked_outputs = (
        "fold_manifest.json",
        "run_config.json",
        "per_case_metrics.csv",
        "configuration_summary.csv",
        "best_parameters.json",
    )
    existing = [
        output_dir / name
        for name in tracked_outputs
        if (output_dir / name).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Cross-validation output already exists: {existing[0]}. "
            "Pass --overwrite to replace the reports."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    if stacking.RUNTIME_IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing runtime dependency for random forest cross-validation: "
            f"{stacking.RUNTIME_IMPORT_ERROR}. Install nibabel, scikit-learn, "
            "and joblib in the Python environment used for this script."
        )

    prepare_output_dir(args.output_dir, args.overwrite)
    specs = stacking.resolve_model_specs(args)
    use_confidence = stacking.uses_confidence(specs)
    use_validity = stacking.uses_validity(specs)
    discovered_cases = stacking.discover_case_dirs(
        args.train_cases_root,
        args.target_annotation,
    )
    if not discovered_cases:
        raise RuntimeError(f"No training cases found under {args.train_cases_root}.")

    print(f"Discovered training cases: {len(discovered_cases)}")
    print(f"Confidence features enabled: {use_confidence}")
    print(f"Validity gating enabled: {use_validity}")
    print("Validating inputs and caching per-case evaluation tables...")
    case_counts: dict[str, np.ndarray | ConfidenceFeatureCounts] = {}
    excluded_cases: dict[str, str] = {}
    for case_id, case_dir in sorted(discovered_cases.items()):
        print(f"  Validating {case_id}", flush=True)
        try:
            if use_confidence:
                case_counts[case_id] = build_case_confidence_counts(
                    case_id,
                    case_dir,
                    args.train_prediction_root,
                    specs,
                    args.target_annotation,
                )
            else:
                case_counts[case_id] = build_case_pattern_counts(
                    case_id,
                    case_dir,
                    args.train_prediction_root,
                    specs,
                    args.target_annotation,
                )
        except Exception as exc:
            if not args.skip_missing:
                raise
            excluded_cases[case_id] = str(exc)
            print(f"WARNING: excluding {case_id}: {exc}")

    folds = make_folds(sorted(case_counts), args.n_splits, args.random_state)
    configurations = parameter_grid(args)
    print(f"Usable cases: {len(case_counts)}")
    print(f"Folds: {len(folds)}")
    print(f"Hyperparameter configurations: {len(configurations)}")
    print(f"Total forest fits: {len(folds) * len(configurations)}")

    fold_manifest = {
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "folds": [
            {
                "fold": fold.number,
                "train_cases": list(fold.train_ids),
                "holdout_cases": list(fold.holdout_ids),
            }
            for fold in folds
        ],
    }
    run_config = {
        "train_cases_root": str(args.train_cases_root),
        "train_prediction_root": str(args.train_prediction_root),
        "output_dir": str(args.output_dir),
        "target_annotation": args.target_annotation,
        "model_dirs": [str(spec.model_dir) for spec in specs],
        "model_ids": [spec.model_id for spec in specs],
        "label_spaces": [spec.label_space for spec in specs],
        "confidence_dirs": [
            str(spec.confidence_dir) if spec.confidence_dir is not None else None
            for spec in specs
        ],
        "validity_dirs": [
            str(spec.validity_dir) if spec.validity_dir is not None else None
            for spec in specs
        ],
        "uses_confidence": use_confidence,
        "uses_validity": use_validity,
        "invalid_voxel_policy": "deterministic background; forest bypassed",
        "feature_names": stacking.feature_names(specs),
        "samples_per_class": args.samples_per_class,
        "background_samples_per_case": args.background_samples_per_case,
        "near_organ_background": args.near_organ_background,
        "dilation_iterations": args.dilation_iterations,
        "predict_chunk_size": args.predict_chunk_size,
        "random_state": args.random_state,
        "primary_metric": "mean patient-level foreground Dice",
        "parameter_grid": configurations,
        "discovered_cases": len(discovered_cases),
        "usable_cases": len(case_counts),
        "excluded_cases": excluded_cases,
    }
    write_json(args.output_dir / "fold_manifest.json", fold_manifest)
    write_json(args.output_dir / "run_config.json", run_config)

    print("Sampling fold-specific training matrices...")
    fold_training_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fold in folds:
        train_cases = {case_id: discovered_cases[case_id] for case_id in fold.train_ids}
        X, y, counts, usable, skipped = stacking.collect_training_samples(
            train_cases,
            args.train_prediction_root,
            specs,
            args,
        )
        if usable != len(train_cases) or skipped:
            raise RuntimeError(
                f"Fold {fold.number} used {usable}/{len(train_cases)} training cases; "
                "inputs changed after validation."
            )
        fold_training_data[fold.number] = (X, y)
        print(
            f"  Fold {fold.number}: X={X.shape}, class_counts={counts}, "
            f"holdout={list(fold.holdout_ids)}"
        )

    all_pattern_features = None if use_confidence else pattern_feature_matrix()
    per_case_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for config_number, parameters in enumerate(configurations, start=1):
        config_id = f"config_{config_number:03d}"
        print(f"Evaluating {config_id}/{len(configurations):03d}: {parameters}")
        config_rows: list[dict[str, object]] = []
        for fold in folds:
            X, y = fold_training_data[fold.number]
            classifier = train_classifier(X, y, parameters, args.random_state)
            predicted_labels = (
                None
                if all_pattern_features is None
                else classifier.predict(all_pattern_features).astype(np.uint8)
            )
            for case_id in fold.holdout_ids:
                case_table = case_counts[case_id]
                if use_confidence:
                    if not isinstance(case_table, ConfidenceFeatureCounts):
                        raise RuntimeError("Expected confidence feature counts.")
                    metrics = dice_scores_from_confidence_counts(
                        case_table,
                        classifier,
                        args.predict_chunk_size,
                    )
                else:
                    if isinstance(case_table, ConfidenceFeatureCounts):
                        raise RuntimeError("Expected categorical pattern counts.")
                    if predicted_labels is None:
                        raise RuntimeError("Missing categorical pattern predictions.")
                    metrics = dice_scores_from_pattern_counts(
                        case_table,
                        predicted_labels,
                    )
                row = {
                    "config_id": config_id,
                    "fold": fold.number,
                    "case_id": case_id,
                    **parameters,
                    **metrics,
                }
                config_rows.append(row)
                per_case_rows.append(row)

        summary = summarise_configuration(config_id, parameters, config_rows)
        summary_rows.append(summary)
        print(
            "  Mean foreground Dice: "
            f"{float(summary['mean_mean_foreground_dice']):.6f}"
        )

    primary_column = "mean_mean_foreground_dice"
    best_summary = max(summary_rows, key=lambda row: float(row[primary_column]))
    best_parameters = {
        "selection_metric": "mean patient-level foreground Dice",
        "selection_score": best_summary[primary_column],
        "config_id": best_summary["config_id"],
        "parameters": {
            key: best_summary[key]
            for key in (
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "max_features",
            )
        },
    }

    write_csv(args.output_dir / "per_case_metrics.csv", per_case_rows)
    write_csv(args.output_dir / "configuration_summary.csv", summary_rows)
    write_json(args.output_dir / "best_parameters.json", best_parameters)

    print(f"Best configuration: {best_parameters['config_id']}")
    print(f"Best parameters: {best_parameters['parameters']}")
    print(f"Selection score: {float(best_parameters['selection_score']):.6f}")
    print(f"Reports written to: {args.output_dir}")


if __name__ == "__main__":
    main()
