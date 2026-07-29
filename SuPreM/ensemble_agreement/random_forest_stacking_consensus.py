#!/usr/bin/env python3
"""Train a voxel-wise random forest stacking baseline for CURVAS consensus.

The classifier learns to fuse three model prediction masks into the CURVAS
target label space:

  0 = background
  1 = pancreas
  2 = kidney
  3 = liver

In stacking, the three segmentation models are the "base" models and the
random forest is the "meta" model.  For each voxel, the forest sees the three
base-model decisions (encoded as 12 binary features) and, when requested, one
continuous confidence feature per model.  It learns which CURVAS label best
matches a human annotation.  The high-level workflow is:

  1. Remap every base model's native labels into the common CURVAS labels.
  2. Sample labelled voxels from the training scans.
  3. Train the forest to map the three predictions to the human label.
  4. Apply that learned mapping to every voxel in each validation scan.

Example using the local train/validation inference split:

python -u ensemble_agreement/random_forest_stacking_consensus.py \
  --train-prediction-root /home/s2347484/Seg/SuPreM/results/all_curvas_inference_with_confidence/training \
  --train-cases-root /home/s2347484/Seg/training_set/training_set \
  --val-prediction-root /home/s2347484/Seg/SuPreM/results/all_curvas_inference_with_confidence/validation \
  --val-cases-root /home/s2347484/Seg/validation_set \
  --model-dir swinunetr_5050 \
  --model-id swin5050 \
  --label-space btcv \
  --confidence-dir swinunetr_5050_confidence \
  --model-dir clip_universal_unet \
  --model-id clip_unet \
  --label-space suprem \
  --confidence-dir clip_universal_unet_confidence \
  --model-dir suprem_segresnet \
  --model-id segresnet \
  --label-space suprem \
  --confidence-dir suprem_segresnet_confidence \
  --target-annotation annotation_1.nii.gz \
  --samples-per-class 5000 \
  --background-samples-per-case 10000 \
  --near-organ-background \
  --model-output /home/s2347484/Seg/SuPreM/results/train_validation_inference/random_forest_stacking/rf_stacking.joblib \
  --output-dir /home/s2347484/Seg/SuPreM/results/train_validation_inference/random_forest_stacking/predictions \
  --overwrite
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

try:
    import joblib
    import nibabel as nib
    from sklearn.ensemble import RandomForestClassifier
except ImportError as exc:  # Let --help and lightweight imports still work.
    joblib = None
    nib = None
    RandomForestClassifier = None
    RUNTIME_IMPORT_ERROR = exc
else:
    RUNTIME_IMPORT_ERROR = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SEG_ROOT = PROJECT_DIR.parent
TRAIN_VAL_ROOT = PROJECT_DIR / "results" / "train_validation_inference"
ALL_CURVAS_INFERENCE_ROOT = (
    PROJECT_DIR / "results" / "all_curvas_inference_with_confidence"
)

DEFAULT_TRAIN_PREDICTION_ROOT = ALL_CURVAS_INFERENCE_ROOT / "training"
DEFAULT_VAL_PREDICTION_ROOT = ALL_CURVAS_INFERENCE_ROOT / "validation"
DEFAULT_TRAIN_CASES_ROOT = SEG_ROOT / "training_set" / "training_set"
DEFAULT_VAL_CASES_ROOT = SEG_ROOT / "validation_set"
DEFAULT_MODEL_OUTPUT = (
    ALL_CURVAS_INFERENCE_ROOT
    / "random_forest_stacking_with_confidence"
    / "rf_stacking.joblib"
)
DEFAULT_OUTPUT_DIR = (
    ALL_CURVAS_INFERENCE_ROOT
    / "random_forest_stacking_with_confidence"
    / "validation_predictions"
)

CURVAS_LABELS = (0, 1, 2, 3)
ORGAN_LABELS = (1, 2, 3)
# Feature columns must always use this order.  User-supplied model arguments
# are reordered to match it in resolve_model_specs(), so training and inference
# cannot silently attach a prediction to the wrong set of columns.
FEATURE_MODEL_IDS = ("clip_unet", "segresnet", "swin5050")
SUPPORTED_MODEL_IDS = set(FEATURE_MODEL_IDS)
SUPPORTED_LABEL_SPACES = {"target", "btcv", "suprem", "word"}
AFFINE_RTOL = 1e-5
AFFINE_ATOL = 1e-5

LABEL_FEATURE_NAMES = [
    "clip_bg",
    "clip_pancreas",
    "clip_kidney",
    "clip_liver",
    "segresnet_bg",
    "segresnet_pancreas",
    "segresnet_kidney",
    "segresnet_liver",
    "swin_bg",
    "swin_pancreas",
    "swin_kidney",
    "swin_liver",
]
CONFIDENCE_FEATURE_NAMES = [
    "clip_confidence",
    "segresnet_confidence",
    "swin_confidence",
]
# Kept as an alias for code that imports the original label-only feature list.
FEATURE_NAMES = LABEL_FEATURE_NAMES

# Copied from consensus_agreement_mask.py so this script remains standalone.
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


@dataclass(frozen=True)
class ModelSpec:
    model_dir: Path
    model_id: str
    label_space: str
    confidence_dir: Path | None = None
    validity_dir: Path | None = None


def flatten_repeated(values: list[list] | None) -> list | None:
    if values is None:
        return None
    return [item for group in values for item in group]


def parse_optional_max_depth(value: str) -> int | None:
    """Accept a positive tree depth or ``none`` for unrestricted trees."""

    if value.lower() in {"none", "unlimited"}:
        return None
    depth = int(value)
    if depth <= 0:
        raise argparse.ArgumentTypeError("max depth must be positive or 'none'.")
    return depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised random forest stacking for CURVAS consensus masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-prediction-root", type=Path, default=DEFAULT_TRAIN_PREDICTION_ROOT)
    parser.add_argument("--train-cases-root", type=Path, default=DEFAULT_TRAIN_CASES_ROOT)
    parser.add_argument("--val-prediction-root", type=Path, default=DEFAULT_VAL_PREDICTION_ROOT)
    parser.add_argument("--val-cases-root", type=Path, default=DEFAULT_VAL_CASES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument(
        "--model-dir",
        dest="model_dirs",
        type=Path,
        nargs="+",
        action="append",
        default=None,
        help="Prediction directory for one model. Relative paths are under the prediction root.",
    )
    parser.add_argument(
        "--model-id",
        dest="model_ids",
        choices=sorted(SUPPORTED_MODEL_IDS),
        nargs="+",
        action="append",
        default=None,
        help="Model ID matching --model-dir order.",
    )
    parser.add_argument(
        "--label-space",
        dest="label_spaces",
        choices=sorted(SUPPORTED_LABEL_SPACES),
        nargs="+",
        action="append",
        default=None,
        help="Label space matching --model-dir order.",
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
            "order. Pass exactly three directories to add one confidence feature "
            "per model; omit this argument for the 12-feature label-only model."
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
            "Optional binary inference-validity directory for each model, "
            "matching --model-dir order. Pass all three together with the "
            "confidence maps so exterior crop padding bypasses the forest."
        ),
    )
    parser.add_argument("--target-annotation", default="annotation_1.nii.gz")
    parser.add_argument("--samples-per-class", type=int, default=5000)
    parser.add_argument("--background-samples-per-case", type=int, default=10000)
    parser.add_argument("--near-organ-background", action="store_true")
    parser.add_argument("--dilation-iterations", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=parse_optional_max_depth, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument(
        "--max-features",
        choices=("sqrt", "all"),
        default="sqrt",
        help="Features considered per split; 'all' passes None to scikit-learn.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--predict-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Fit and save the random forest without running validation inference.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path


def default_model_dirs() -> list[Path]:
    return [
        Path("clip_universal_unet"),
        Path("suprem_segresnet"),
        Path("swinunetr_5050"),
    ]


def default_model_ids() -> list[str]:
    return ["clip_unet", "segresnet", "swin5050"]


def default_label_spaces() -> list[str]:
    return ["suprem", "suprem", "btcv"]


def resolve_model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    """Validate the three base models and return them in feature-column order."""

    model_dirs = flatten_repeated(args.model_dirs) or default_model_dirs()
    model_ids = flatten_repeated(args.model_ids) or default_model_ids()
    label_spaces = flatten_repeated(args.label_spaces) or default_label_spaces()
    confidence_dirs = flatten_repeated(getattr(args, "confidence_dirs", None))
    validity_dirs = flatten_repeated(getattr(args, "validity_dirs", None))

    if not (len(model_dirs) == len(model_ids) == len(label_spaces) == 3):
        raise ValueError(
            "Pass exactly three --model-dir, --model-id, and --label-space values."
        )
    if set(model_ids) != SUPPORTED_MODEL_IDS:
        raise ValueError(
            "Expected model IDs clip_unet, segresnet, and swin5050 exactly once; "
            f"got {model_ids}."
        )
    if confidence_dirs is not None and len(confidence_dirs) != 3:
        raise ValueError(
            "Pass exactly three --confidence-dir values, one for each base model, "
            "or omit --confidence-dir for label-only features."
        )
    if confidence_dirs is None:
        confidence_dirs = [None, None, None]
    if validity_dirs is not None and len(validity_dirs) != 3:
        raise ValueError(
            "Pass exactly three --validity-dir values, one for each base model, "
            "or omit --validity-dir."
        )
    if validity_dirs is None:
        validity_dirs = [None, None, None]

    specs = [
        ModelSpec(
            Path(model_dir),
            model_id,
            label_space,
            Path(confidence_dir) if confidence_dir is not None else None,
            Path(validity_dir) if validity_dir is not None else None,
        )
        for model_dir, model_id, label_space, confidence_dir, validity_dir in zip(
            model_dirs,
            model_ids,
            label_spaces,
            confidence_dirs,
            validity_dirs,
        )
    ]
    by_id = {spec.model_id: spec for spec in specs}
    return [by_id[model_id] for model_id in FEATURE_MODEL_IDS]


def uses_confidence(specs: list[ModelSpec]) -> bool:
    """Return whether all three model specifications include confidence maps."""

    configured = [spec.confidence_dir is not None for spec in specs]
    if any(configured) and not all(configured):
        raise ValueError("Confidence directories must be configured for all three models.")
    return all(configured)


def uses_validity(specs: list[ModelSpec]) -> bool:
    """Return whether all models include binary inference-validity masks."""

    configured = [spec.validity_dir is not None for spec in specs]
    if any(configured) and not all(configured):
        raise ValueError("Validity directories must be configured for all three models.")
    enabled = all(configured)
    if enabled and not uses_confidence(specs):
        raise ValueError("Validity masks currently require all three confidence maps.")
    return enabled


def feature_names(specs: list[ModelSpec]) -> list[str]:
    names = list(LABEL_FEATURE_NAMES)
    if uses_confidence(specs):
        names.extend(CONFIDENCE_FEATURE_NAMES)
    return names


def allowed_labels_for_space(label_space: str) -> set[int]:
    """Return the complete native-label vocabulary for a configured space."""

    label_space = label_space.lower()
    if label_space == "target":
        return set(CURVAS_LABELS)
    if label_space == "btcv":
        return set(BTCV_TO_TARGET)
    if label_space in {"suprem", "word"}:
        return set(SUPREM_TO_TARGET)
    raise ValueError(f"Unsupported label space: {label_space}")


def validate_label_values(
    labels: np.ndarray,
    allowed: set[int],
    source: str,
) -> None:
    """Reject labels outside an explicitly declared source vocabulary."""

    observed = {int(label) for label in np.unique(labels)}
    unexpected = sorted(observed - allowed)
    if unexpected:
        raise ValueError(
            f"{source} contains unexpected labels {unexpected}; "
            f"allowed labels are {sorted(allowed)}."
        )


def remap_to_curvas(
    labels: np.ndarray,
    label_space: str,
    source: str = "Label map",
) -> np.ndarray:
    """Collapse a model's native classes into background/pancreas/kidney/liver."""

    label_space = label_space.lower()
    if label_space == "target":
        validate_label_values(labels, allowed_labels_for_space(label_space), source)
        remapped = labels
    elif label_space == "btcv":
        remapped = map_labels(labels, BTCV_TO_TARGET, source)
    elif label_space in {"suprem", "word"}:
        remapped = map_labels(labels, SUPREM_TO_TARGET, source)
    else:
        raise ValueError(f"Unsupported label space: {label_space}")
    return np.asarray(remapped, dtype=np.uint8)


def map_labels(
    labels: np.ndarray,
    mapping: dict[int, int],
    source: str = "Label map",
) -> np.ndarray:
    """Map native labels only after confirming every value is supported."""

    labels = np.asarray(labels)
    validate_label_values(labels, set(mapping), source)
    remapped = np.zeros(labels.shape, dtype=np.uint8)
    for source_label, target_label in mapping.items():
        remapped[labels == source_label] = target_label
    return remapped


def one_hot_encode_predictions(
    predictions: list[np.ndarray],
    confidences: list[np.ndarray] | None = None,
) -> np.ndarray:
    """One-hot encode clip, segresnet, and swin labels into 12 binary features.

    Each model contributes four columns, one per CURVAS class.  For example, a
    voxel predicted as pancreas (1) by clip, kidney (2) by segresnet, and liver
    (3) by swin becomes::

        [0,1,0,0 | 0,0,1,0 | 0,0,0,1]

    This representation treats class IDs as categories, rather than incorrectly
    suggesting that liver (3) is numerically "larger" than pancreas (1).
    """

    if len(predictions) != 3:
        raise ValueError("Expected exactly three prediction arrays.")

    flattened = [np.asarray(prediction, dtype=np.uint8).reshape(-1) for prediction in predictions]
    n_voxels = flattened[0].shape[0]
    if any(prediction.shape[0] != n_voxels for prediction in flattened):
        raise ValueError("All prediction arrays must contain the same number of voxels.")

    # Set exactly one of each model's four columns to 1 for every voxel.
    features = np.zeros((n_voxels, 12), dtype=np.uint8)
    for model_index, labels in enumerate(flattened):
        if np.any((labels < 0) | (labels > 3)):
            bad = np.unique(labels[(labels < 0) | (labels > 3)])
            raise ValueError(f"Predictions must be in CURVAS label space 0-3; got {bad[:10]}.")
        row_indices = np.arange(n_voxels)
        features[row_indices, model_index * 4 + labels] = 1 # turns the model prediction into a one-hot encoded feature vector
    if confidences is None:
        return features
    if len(confidences) != 3:
        raise ValueError("Expected exactly three confidence arrays.")

    confidence_columns = []
    for confidence in confidences:
        column = np.asarray(confidence, dtype=np.float32).reshape(-1)
        if column.shape[0] != n_voxels:
            raise ValueError("Confidence and prediction arrays must contain the same voxels.")
        if not np.all(np.isfinite(column)):
            raise ValueError("Confidence arrays contain NaN or infinite values.")
        if np.any(column < -1e-4) or np.any(column > 1.0 + 1e-4):
            raise ValueError("Confidence values must lie in the range [0, 1].")
        confidence_columns.append(np.clip(column, 0.0, 1.0))

    return np.concatenate(
        [features.astype(np.float32), np.column_stack(confidence_columns)],
        axis=1,
    )


def is_nifti_file(path: Path) -> bool:
    return path.is_file() and (path.name.endswith(".nii.gz") or path.name.endswith(".nii"))


def is_obvious_non_label_file(path: Path) -> bool:
    """Heuristic to exclude probability-like outputs from the hard-label search."""
    name = path.name.lower()
    blocked = ("prob", "proba", "confidence", "conf", "softmax", "logit", "uncertainty")
    return any(token in name for token in blocked)


def discover_case_dirs(cases_root: Path, target_annotation: str) -> dict[str, Path]:
    """Find case directories at the root or up to two directory levels below it."""

    if not cases_root.is_dir():
        raise NotADirectoryError(f"Cases root does not exist: {cases_root}")

    case_dirs: dict[str, Path] = {}
    if (cases_root / target_annotation).is_file():
        case_dirs[cases_root.name] = cases_root

    for child in sorted(cases_root.iterdir()):
        if child.is_dir() and (child / target_annotation).is_file():
            case_dirs[child.name] = child

    if not case_dirs:
        for child in sorted(cases_root.iterdir()):
            if not child.is_dir():
                continue
            for grandchild in sorted(child.iterdir()):
                if grandchild.is_dir() and (grandchild / target_annotation).is_file():
                    case_dirs[grandchild.name] = grandchild

    return case_dirs


def discover_prediction_file(prediction_root: Path, model_dir: Path, case_id: str) -> Path | None:
    """Find a case's hard-label NIfTI while excluding probability-like outputs."""

    model_root = resolve_path(model_dir, prediction_root)
    if not model_root.is_dir():
        return None

    exact_candidates = [
        model_root / f"{case_id}.nii.gz",
        model_root / f"{case_id}.nii",
    ]
    for candidate in exact_candidates:
        if is_nifti_file(candidate) and not is_obvious_non_label_file(candidate):
            return candidate

    nested_dir = model_root / case_id
    if nested_dir.is_dir():
        nested_files = sorted(
            path for path in nested_dir.rglob("*") if is_nifti_file(path) and not is_obvious_non_label_file(path)
        )
        if nested_files:
            return nested_files[0]

    loose_files = sorted(
        path
        for path in model_root.rglob(f"{case_id}*")
        if is_nifti_file(path) and not is_obvious_non_label_file(path)
    )
    return loose_files[0] if loose_files else None


def discover_confidence_file(
    prediction_root: Path,
    confidence_dir: Path,
    case_id: str,
) -> Path | None:
    """Find a case confidence map in its explicitly configured directory."""

    confidence_root = resolve_path(confidence_dir, prediction_root)
    if not confidence_root.is_dir():
        return None

    for candidate in (
        confidence_root / f"{case_id}.nii.gz",
        confidence_root / f"{case_id}.nii",
    ):
        if is_nifti_file(candidate):
            return candidate

    nested_dir = confidence_root / case_id
    if nested_dir.is_dir():
        nested_files = sorted(path for path in nested_dir.rglob("*") if is_nifti_file(path))
        if nested_files:
            return nested_files[0]

    loose_files = sorted(
        path
        for path in confidence_root.rglob(f"{case_id}*")
        if is_nifti_file(path)
    )
    return loose_files[0] if loose_files else None


def discover_validity_file(
    prediction_root: Path,
    validity_dir: Path,
    case_id: str,
) -> Path | None:
    """Find a case validity mask in its explicitly configured directory."""

    return discover_confidence_file(prediction_root, validity_dir, case_id)


def load_label_image(
    path: Path,
    atol: float = 1e-3,
) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load an integer label map without truncating scaled NIfTI values.

    NIfTI integer images may use a slope/intercept when nibabel decodes them.
    Values intended to be 1, 2, and 3 can consequently be represented as
    0.9999999998, 1.9999999995, and 2.9999999993. Casting those values directly
    to an integer shifts every foreground class down by one, so floating-point
    data must be checked and rounded first.
    """

    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if not np.issubdtype(data.dtype, np.integer):
        rounded = np.rint(data)
        if not np.all(np.isclose(data, rounded, rtol=0.0, atol=atol)):
            raise ValueError(f"{path} contains non-integer label values.") # If not integer raise error
        data = rounded
    data = data.astype(np.int16, copy=False)
    return data, image


def validate_image_grid(
    case_id: str,
    candidate_image: nib.Nifti1Image,
    candidate_path: Path,
    reference_image: nib.Nifti1Image,
    reference_path: Path,
    role: str,
) -> None:
    """Require two NIfTI images to describe the same voxel and world grid."""

    if candidate_image.shape != reference_image.shape:
        raise ValueError(
            f"{case_id}: {role} shape mismatch: {candidate_path} has "
            f"{candidate_image.shape}, but reference {reference_path} has "
            f"{reference_image.shape}."
        )
    if not np.allclose(
        candidate_image.affine,
        reference_image.affine,
        rtol=AFFINE_RTOL,
        atol=AFFINE_ATOL,
    ):
        maximum_difference = float(
            np.max(np.abs(candidate_image.affine - reference_image.affine))
        )
        raise ValueError(
            f"{case_id}: {role} affine mismatch: {candidate_path} does not "
            f"match reference {reference_path} "
            f"(maximum absolute difference {maximum_difference:.6g})."
        )


def load_curvas_target(
    target_path: Path,
    case_id: str,
) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load a 3-D CURVAS target and require labels 0, 1, 2, and 3 only."""

    target, target_image = load_label_image(target_path)
    if len(target_image.shape) != 3:
        raise ValueError(
            f"{case_id}: target {target_path} must be 3-D; "
            f"got shape {target_image.shape}."
        )
    target = remap_to_curvas(
        target,
        "target",
        source=f"{case_id}: target {target_path}",
    )
    return target, target_image


def load_confidence_image(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load a scaled NIfTI confidence map as finite float32 values in [0, 1]."""

    image = nib.load(str(path))
    confidence = np.asarray(image.dataobj, dtype=np.float32)
    if not np.all(np.isfinite(confidence)):
        raise ValueError(f"{path} contains NaN or infinite confidence values.")
    if np.any(confidence < -1e-4) or np.any(confidence > 1.0 + 1e-4):
        minimum = float(np.min(confidence))
        maximum = float(np.max(confidence))
        raise ValueError(
            f"{path} confidence range [{minimum}, {maximum}] is outside [0, 1]."
        )
    return np.clip(confidence, 0.0, 1.0), image


def load_validity_image(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load a binary inference-validity map as a boolean array."""

    values, image = load_label_image(path)
    unexpected = sorted(set(np.unique(values).astype(int)) - {0, 1})
    if unexpected:
        raise ValueError(f"{path} contains unexpected validity values {unexpected}.")
    return values.astype(bool, copy=False), image


def validate_training_class_coverage(labels: np.ndarray) -> None:
    """Fail before fitting if the sampled targets omit a CURVAS class."""

    observed = {int(label) for label in np.unique(labels)}
    expected = set(CURVAS_LABELS)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing expected classes {missing}")
        if unexpected:
            details.append(f"found unexpected classes {unexpected}")
        raise RuntimeError(
            "Invalid sampled training-label coverage: "
            + "; ".join(details)
            + f". Observed classes: {sorted(observed)}."
        )


def load_remapped_prediction(
    prediction_root: Path,
    spec: ModelSpec,
    case_id: str,
) -> tuple[np.ndarray, nib.Nifti1Image, Path]:
    """Load a single model's prediction and remap it to the CURVAS label space."""
    prediction_path = discover_prediction_file(prediction_root, spec.model_dir, case_id)
    if prediction_path is None:
        model_root = resolve_path(spec.model_dir, prediction_root)
        raise FileNotFoundError(f"Missing prediction for case {case_id} in {model_root}")
    labels, image = load_label_image(prediction_path)
    if len(image.shape) != 3:
        raise ValueError(
            f"{case_id}: prediction {prediction_path} must be 3-D; "
            f"got shape {image.shape}."
        )
    remapped = remap_to_curvas(
        labels,
        spec.label_space,
        source=(
            f"{case_id}: {spec.model_id} prediction {prediction_path} "
            f"in {spec.label_space} label space"
        ),
    )
    return remapped, image, prediction_path


def load_case_predictions(
    prediction_root: Path,
    specs: list[ModelSpec],
    case_id: str,
) -> tuple[list[np.ndarray], nib.Nifti1Image, list[Path]]:
    """Load all three remapped predictions and ensure their voxel grids match."""

    predictions: list[np.ndarray] = []
    paths: list[Path] = []
    reference_image: nib.Nifti1Image | None = None
    reference_path: Path | None = None

    for spec in specs:
        prediction, image, path = load_remapped_prediction(prediction_root, spec, case_id)
        if reference_image is None:
            reference_image = image
            reference_path = path
        else:
            if reference_path is None:  # Kept explicit for static type checkers.
                raise RuntimeError(f"No reference prediction path for case {case_id}.")
            validate_image_grid(
                case_id,
                image,
                path,
                reference_image,
                reference_path,
                role=f"{spec.model_id} prediction",
            )
        predictions.append(prediction)
        paths.append(path)

    if reference_image is None:
        raise RuntimeError(f"No predictions loaded for case {case_id}.")
    return predictions, reference_image, paths


def load_common_validity_mask(
    prediction_root: Path,
    specs: list[ModelSpec],
    case_id: str,
    expected_shape: tuple[int, ...],
    expected_affine: np.ndarray,
) -> tuple[np.ndarray | None, list[Path]]:
    """Load matching per-model validity masks and return their common mask."""

    if not uses_validity(specs):
        return None, []

    validity_masks: list[np.ndarray] = []
    validity_paths: list[Path] = []
    for spec in specs:
        if spec.validity_dir is None:  # Guarded by uses_validity().
            raise RuntimeError(f"No validity directory configured for {spec.model_id}.")
        validity_path = discover_validity_file(
            prediction_root,
            spec.validity_dir,
            case_id,
        )
        if validity_path is None:
            validity_root = resolve_path(spec.validity_dir, prediction_root)
            raise FileNotFoundError(
                f"Missing validity mask for case {case_id} in {validity_root}"
            )
        validity, validity_image = load_validity_image(validity_path)
        if validity.shape != expected_shape:
            raise ValueError(
                f"Validity shape mismatch for case {case_id}: {validity_path} "
                f"has {validity.shape}, expected {expected_shape}."
            )
        if not np.allclose(
            validity_image.affine,
            expected_affine,
            rtol=AFFINE_RTOL,
            atol=AFFINE_ATOL,
        ):
            raise ValueError(
                f"Validity affine mismatch for case {case_id}: {validity_path}."
            )
        validity_masks.append(validity)
        validity_paths.append(validity_path)

    common = validity_masks[0]
    for spec, validity in zip(specs[1:], validity_masks[1:]):
        if not np.array_equal(validity, common):
            raise ValueError(
                f"Validity masks disagree for case {case_id}; "
                f"{spec.model_id} does not match {specs[0].model_id}."
            )
    return common, validity_paths


def load_case_inputs(
    prediction_root: Path,
    specs: list[ModelSpec],
    case_id: str,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray] | None,
    np.ndarray | None,
    nib.Nifti1Image,
    list[Path],
    list[Path],
    list[Path],
]:
    """Load aligned predictions, confidences, and optional validity masks."""

    predictions, reference_image, prediction_paths = load_case_predictions(
        prediction_root,
        specs,
        case_id,
    )
    confidences: list[np.ndarray] | None = [] if uses_confidence(specs) else None
    confidence_paths: list[Path] = []
    if confidences is not None:
        for spec, prediction in zip(specs, predictions):
            if spec.confidence_dir is None:  # Guarded by uses_confidence().
                raise RuntimeError(f"No confidence directory configured for {spec.model_id}.")
            confidence_path = discover_confidence_file(
                prediction_root,
                spec.confidence_dir,
                case_id,
            )
            if confidence_path is None:
                confidence_root = resolve_path(spec.confidence_dir, prediction_root)
                raise FileNotFoundError(
                    f"Missing confidence map for case {case_id} in {confidence_root}"
                )
            confidence, confidence_image = load_confidence_image(confidence_path)
            if confidence.shape != prediction.shape:
                raise ValueError(
                    f"Confidence shape mismatch for case {case_id}: {confidence_path} "
                    f"has {confidence.shape}, expected {prediction.shape}."
                )
            if not np.allclose(
                confidence_image.affine,
                reference_image.affine,
                rtol=AFFINE_RTOL,
                atol=AFFINE_ATOL,
            ):
                raise ValueError(
                    f"Confidence affine mismatch for case {case_id}: {confidence_path}."
                )
            confidences.append(confidence)
            confidence_paths.append(confidence_path)

    validity_mask, validity_paths = load_common_validity_mask(
        prediction_root,
        specs,
        case_id,
        predictions[0].shape,
        reference_image.affine,
    )

    return (
        predictions,
        confidences,
        validity_mask,
        reference_image,
        prediction_paths,
        confidence_paths,
        validity_paths,
    )


def sample_flat_indices(indices: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly sample up to max_samples from a flat array of voxel indices."""
    if max_samples <= 0 or indices.size == 0:
        return np.empty((0,), dtype=np.int64)
    sample_size = min(max_samples, indices.size)
    return rng.choice(indices, size=sample_size, replace=False) # Same voxel cant be sampled twice without replacement


def sample_background_indices(
    target: np.ndarray,
    predictions: list[np.ndarray],
    validity_mask: np.ndarray | None,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose background examples, optionally concentrating on difficult borders.

    Most voxels in an abdominal scan are easy background.  Sampling uniformly
    can therefore teach the forest little beyond "predict background".  With
    --near-organ-background, priority goes to background near any predicted
    organ, where false positives and boundary disagreements actually occur.
    """

    target_flat = target.reshape(-1)

    if args.near_organ_background:
        model_foreground_union = np.zeros(target.shape, dtype=bool)
        for prediction in predictions:
            model_foreground_union |= prediction > 0
        # Dilation creates a band around the union of all predicted organs.
        near_foreground = binary_dilation(
            model_foreground_union,
            iterations=args.dilation_iterations,
        )
        candidate_background = (target == 0) & near_foreground
        if validity_mask is not None:
            candidate_background &= validity_mask
        near_indices = np.flatnonzero(candidate_background.reshape(-1))
        selected = sample_flat_indices(near_indices, args.background_samples_per_case, rng)
        # If the boundary band is too small, fill the quota from all remaining
        # true-background voxels rather than reducing this case's sample count.
        remaining_needed = args.background_samples_per_case - selected.size
        if remaining_needed <= 0:
            return selected

        eligible_background = target == 0
        if validity_mask is not None:
            eligible_background &= validity_mask
        all_background = np.flatnonzero(eligible_background.reshape(-1))
        if selected.size:
            supplement_pool = all_background[~np.isin(all_background, selected, assume_unique=False)]
        else:
            supplement_pool = all_background
        supplement = sample_flat_indices(supplement_pool, remaining_needed, rng)
        return np.concatenate([selected, supplement])

    eligible_background = target == 0
    if validity_mask is not None:
        eligible_background &= validity_mask
    background_indices = np.flatnonzero(eligible_background.reshape(-1))
    return sample_flat_indices(background_indices, args.background_samples_per_case, rng)


def collect_training_samples(
    train_cases: dict[str, Path],
    prediction_root: Path,
    specs: list[ModelSpec],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], int, int]:
    """Build the meta-model's feature matrix X and human-label vector y."""

    rng = np.random.default_rng(args.random_state)
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    skipped_cases = 0
    usable_cases = 0

    iterator = sorted(train_cases.items())
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Training cases") #Progress bar for training cases

    for case_id, case_dir in iterator:
        target_path = case_dir / args.target_annotation
        try:
            # Load and validate the human target before pairing voxel indices.
            target, target_image = load_curvas_target(target_path, case_id)
            (
                predictions,
                confidences,
                validity_mask,
                reference_image,
                prediction_paths,
                _,
                _,
            ) = load_case_inputs(
                prediction_root,
                specs,
                case_id,
            )
            validate_image_grid(
                case_id,
                target_image,
                target_path,
                reference_image,
                prediction_paths[0],
                role="training target",
            )
        except Exception as exc:
            if not args.skip_missing:
                raise
            skipped_cases += 1
            print(f"WARNING: skipping training case {case_id}: {exc}")
            continue

        # Flattening preserves voxel alignment: index i refers to the same
        # physical grid location in the target and all three predictions.
        target_flat = target.reshape(-1) # 3d mask converted to 1d array
        if validity_mask is not None:
            outside_foreground = (target > 0) & ~validity_mask
            if np.any(outside_foreground):
                counts = {
                    str(int(label)): int(count)
                    for label, count in zip(
                        *np.unique(target[outside_foreground], return_counts=True)
                    )
                }
                raise ValueError(
                    f"{case_id}: target foreground lies outside the common "
                    f"inference-validity mask: {counts}."
                )
        sampled_indices: list[np.ndarray] = []
        # Cap each organ separately so large organs (especially liver) do not
        # overwhelm the smaller pancreas class during training.
        for label in ORGAN_LABELS:
            eligible = target == label
            if validity_mask is not None:
                eligible &= validity_mask
            label_indices = np.flatnonzero(eligible.reshape(-1))
            sampled_indices.append(sample_flat_indices(label_indices, args.samples_per_class, rng))
        sampled_indices.append(
            sample_background_indices(
                target,
                predictions,
                validity_mask,
                args,
                rng,
            )
        )

        case_indices = np.concatenate(sampled_indices)
        if case_indices.size == 0:
            if not args.skip_missing:
                raise ValueError(f"Training case {case_id}: no voxels were sampled.")
            skipped_cases += 1
            print(f"WARNING: skipping training case {case_id}: no voxels sampled.")
            continue

        labels = target_flat[case_indices].astype(np.uint8, copy=False)
        # Use the same sampled indices for every base model and the target.
        prediction_samples = [
            prediction.reshape(-1)[case_indices].astype(np.uint8, copy=False)
            for prediction in predictions
        ]
        confidence_samples = None
        if confidences is not None:
            confidence_samples = [
                confidence.reshape(-1)[case_indices].astype(np.float32, copy=False)
                for confidence in confidences
            ]
        feature_blocks.append(
            one_hot_encode_predictions(prediction_samples, confidence_samples)
        )
        label_blocks.append(labels)
        usable_cases += 1

    if not feature_blocks:
        raise RuntimeError("No usable training samples were collected.")

    X = np.concatenate(feature_blocks, axis=0) # each row is a voxel, each column is a feature (one-hot encoded predictions from the three models)
    y = np.concatenate(label_blocks, axis=0)
    validate_training_class_coverage(y)
    sample_counts = {str(label): int(count) for label, count in sorted(Counter(y.tolist()).items())}
    return X, y, sample_counts, usable_cases, skipped_cases


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
) -> RandomForestClassifier:
    """Fit the second-level model that learns how to combine base predictions."""

    max_features = None if args.max_features == "all" else args.max_features
    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=max_features,
        # Weight rare target classes more heavily in each tree's loss.
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.random_state,
    )
    classifier.fit(X, y)
    return classifier


def confidence_from_predictions(predicted: np.ndarray, max_probability: np.ndarray) -> np.ndarray:
    """Convert forest vote probability to the project's 1-5 confidence codes.

    Codes 1-4 are increasingly uncertain foreground predictions.  Background
    always receives code 5, so this is not a symmetric uncertainty map.
    """

    confidence = np.full(predicted.shape, 5, dtype=np.uint8)
    foreground = predicted > 0
    confidence[foreground & (max_probability >= 0.90)] = 1
    confidence[foreground & (max_probability >= 0.70) & (max_probability < 0.90)] = 2
    confidence[foreground & (max_probability >= 0.50) & (max_probability < 0.70)] = 3
    confidence[foreground & (max_probability < 0.50)] = 4
    return confidence


def save_nifti_like(data: np.ndarray, reference_image: nib.Nifti1Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = reference_image.header.copy()
    header.set_data_dtype(data.dtype)
    output_image = nib.Nifti1Image(data, reference_image.affine, header)
    nib.save(output_image, str(output_path))


def run_validation_inference(
    classifier: RandomForestClassifier,
    val_cases: dict[str, Path],
    prediction_root: Path,
    specs: list[ModelSpec],
    args: argparse.Namespace,
) -> None:
    """Fuse each validation case and write its consensus and confidence masks."""

    args.output_dir.mkdir(parents=True, exist_ok=True)

    iterator = sorted(val_cases)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Validation cases")

    for case_id in iterator:
        case_output_dir = args.output_dir / case_id
        agreement_path = case_output_dir / "agreement_mask.nii.gz"
        confidence_path = case_output_dir / "confidence_mask.nii.gz"
        if not args.overwrite and (agreement_path.exists() or confidence_path.exists()):
            print(f"Skipping {case_id}: output exists and --overwrite was not provided.")
            continue

        print(f"Running validation inference for {case_id}")
        try:
            (
                predictions,
                confidences,
                validity_mask,
                reference_image,
                _,
                _,
                _,
            ) = load_case_inputs(
                prediction_root,
                specs,
                case_id,
            )
        except Exception:
            if args.skip_missing:
                print(f"WARNING: skipping validation case {case_id}: missing or invalid predictions.")
                continue
            raise

        # The forest operates on independent voxel rows, so flatten the spatial
        # arrays now and restore their 3-D shape after prediction.
        flat_predictions = [prediction.reshape(-1) for prediction in predictions]
        flat_confidences = (
            [confidence.reshape(-1) for confidence in confidences]
            if confidences is not None
            else None
        )
        n_voxels = flat_predictions[0].shape[0]
        # Exterior crop voxels bypass the forest and remain deterministic
        # background with the project's background confidence code 5.
        predicted_flat = np.zeros(n_voxels, dtype=np.uint8)
        confidence_flat = np.full(n_voxels, 5, dtype=np.uint8)
        flat_validity = (
            validity_mask.reshape(-1) if validity_mask is not None else None
        )

        # A full scan can contain hundreds of millions of feature values.
        # Chunking limits peak memory without changing any voxel's prediction.
        for start in range(0, n_voxels, args.predict_chunk_size):
            stop = min(start + args.predict_chunk_size, n_voxels)
            chunk_validity = (
                flat_validity[start:stop]
                if flat_validity is not None
                else slice(None)
            )
            if flat_validity is not None and not np.any(chunk_validity):
                continue
            chunk_predictions = [
                prediction[start:stop][chunk_validity]
                for prediction in flat_predictions
            ]
            chunk_confidences = (
                [
                    confidence[start:stop][chunk_validity]
                    for confidence in flat_confidences
                ]
                if flat_confidences is not None
                else None
            )
            X_chunk = one_hot_encode_predictions(
                chunk_predictions,
                chunk_confidences,
            )
            predicted_chunk = classifier.predict(X_chunk).astype(np.uint8, copy=False)
            # predict_proba averages the trees' class probabilities; the
            # winning class's probability drives the confidence code.
            probabilities = classifier.predict_proba(X_chunk)
            max_probability = np.max(probabilities, axis=1)
            predicted_block = predicted_flat[start:stop]
            confidence_block = confidence_flat[start:stop]
            predicted_block[chunk_validity] = predicted_chunk
            confidence_block[chunk_validity] = confidence_from_predictions(
                predicted_chunk,
                max_probability,
            )

        predicted = predicted_flat.reshape(predictions[0].shape)
        confidence = confidence_flat.reshape(predictions[0].shape)
        save_nifti_like(predicted, reference_image, agreement_path)
        save_nifti_like(confidence, reference_image, confidence_path)


def metadata_path_for_model(model_output: Path) -> Path:
    return model_output.with_suffix(".metadata.json")


def save_model_and_metadata(
    classifier: RandomForestClassifier,
    specs: list[ModelSpec],
    sample_counts: dict[str, int],
    train_cases_count: int,
    usable_training_cases: int,
    skipped_training_cases: int,
    val_cases_count: int,
    args: argparse.Namespace,
) -> Path:
    """Persist the fitted forest plus enough context to reproduce its features."""

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, args.model_output)

    metadata = {
        "train_prediction_root": str(args.train_prediction_root),
        "train_cases_root": str(args.train_cases_root),
        "val_prediction_root": str(args.val_prediction_root),
        "val_cases_root": str(args.val_cases_root),
        "model_dirs": [str(spec.model_dir) for spec in specs],
        "model_ids": [spec.model_id for spec in specs],
        "label_spaces": [spec.label_space for spec in specs],
        "confidence_dirs": [
            str(spec.confidence_dir) if spec.confidence_dir is not None else None
            for spec in specs
        ],
        "uses_confidence": uses_confidence(specs),
        "validity_dirs": [
            str(spec.validity_dir) if spec.validity_dir is not None else None
            for spec in specs
        ],
        "uses_validity": uses_validity(specs),
        "invalid_voxel_policy": "deterministic background; forest bypassed",
        "target_annotation": args.target_annotation,
        "samples_per_class": args.samples_per_class,
        "background_samples_per_case": args.background_samples_per_case,
        "near_organ_background": bool(args.near_organ_background),
        "dilation_iterations": args.dilation_iterations,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "random_state": args.random_state,
        "feature_names": feature_names(specs),
        "class_labels": [int(label) for label in classifier.classes_],
        "training_sample_counts": sample_counts,
        "discovered_training_cases": train_cases_count,
        "usable_training_cases": usable_training_cases,
        "skipped_training_cases": skipped_training_cases,
        "discovered_validation_cases": val_cases_count,
        "validation_inference_run": not args.train_only,
    }
    metadata_path = metadata_path_for_model(args.model_output)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata_path


def print_feature_importances(
    classifier: RandomForestClassifier,
    specs: list[ModelSpec],
) -> None:
    print("Feature importances:")
    for name, importance in sorted(
        zip(feature_names(specs), classifier.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {name}: {importance:.6f}")


def main() -> None:
    args = parse_args()
    if RUNTIME_IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing runtime dependency for random forest stacking: "
            f"{RUNTIME_IMPORT_ERROR}. Install nibabel, scikit-learn, and joblib "
            "in the Python environment used for this script."
        )

    specs = resolve_model_specs(args)

    train_cases = discover_case_dirs(args.train_cases_root, args.target_annotation)
    val_cases = (
        {}
        if args.train_only
        else discover_case_dirs(args.val_cases_root, args.target_annotation)
    )

    print(f"Number of discovered training cases: {len(train_cases)}")
    print(f"Number of discovered validation cases: {len(val_cases)}")
    print("Model configuration in feature order:")
    for spec in specs:
        confidence_description = (
            str(resolve_path(spec.confidence_dir, args.train_prediction_root))
            if spec.confidence_dir is not None
            else "disabled"
        )
        validity_description = (
            str(resolve_path(spec.validity_dir, args.train_prediction_root))
            if spec.validity_dir is not None
            else "disabled"
        )
        print(
            f"  {spec.model_id}: dir={resolve_path(spec.model_dir, args.train_prediction_root)} "
            f"label_space={spec.label_space} confidence={confidence_description} "
            f"validity={validity_description}"
        )
    print(f"Confidence features enabled: {uses_confidence(specs)}")
    print(f"Validity gating enabled: {uses_validity(specs)}")

    X, y, sample_counts, usable_training_cases, skipped_training_cases = collect_training_samples(
        train_cases,
        args.train_prediction_root,
        specs,
        args,
    )

    print(f"Number of usable training cases: {usable_training_cases}")
    print(f"Number of skipped training cases: {skipped_training_cases}")
    print(f"Sample counts per class: {sample_counts}")
    print(f"Training matrix shape: {X.shape}")
    print(
        "Random forest parameters: "
        f"n_estimators={args.n_estimators}, max_depth={args.max_depth}, "
        f"min_samples_leaf={args.min_samples_leaf}, max_features={args.max_features}, "
        f"class_weight=balanced, n_jobs=-1, random_state={args.random_state}"
    )

    classifier = train_random_forest(X, y, args)
    metadata_path = save_model_and_metadata(
        classifier,
        specs,
        sample_counts,
        len(train_cases),
        usable_training_cases,
        skipped_training_cases,
        len(val_cases),
        args,
    )

    print(f"Model output path: {args.model_output}")
    print(f"Metadata output path: {metadata_path}")
    if args.train_only:
        print("Validation inference: skipped (--train-only)")
    else:
        print(f"Validation output directory: {args.output_dir}")
    print_feature_importances(classifier, specs)

    if not args.train_only:
        run_validation_inference(
            classifier,
            val_cases,
            args.val_prediction_root,
            specs,
            args,
        )


if __name__ == "__main__":
    main()
