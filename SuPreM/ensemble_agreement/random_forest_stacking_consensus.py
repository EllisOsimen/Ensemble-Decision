#!/usr/bin/env python3
"""Train a voxel-wise random forest stacking baseline for CURVAS consensus.

The classifier learns to fuse three model prediction masks into the CURVAS
target label space:

  0 = background
  1 = pancreas
  2 = kidney
  3 = liver

In stacking, the three segmentation models are the "base" models and the
random forest is the "meta" model.  For each voxel, the forest sees only the
three base-model decisions (encoded as 12 binary features) and learns which
CURVAS label best matches a human annotation.  The high-level workflow is:

  1. Remap every base model's native labels into the common CURVAS labels.
  2. Sample labelled voxels from the training scans.
  3. Train the forest to map the three predictions to the human label.
  4. Apply that learned mapping to every voxel in each validation scan.

Example using the local train/validation inference split:

python -u ensemble_agreement/random_forest_stacking_consensus.py \
  --train-prediction-root /home/s2347484/Seg/SuPreM/results/train_validation_inference/training \
  --train-cases-root /home/s2347484/Seg/training_set/training_set \
  --val-prediction-root /home/s2347484/Seg/SuPreM/results/train_validation_inference/validation \
  --val-cases-root /home/s2347484/Seg/validation_set \
  --model-dir swinunetr_5050 \
  --model-id swin5050 \
  --label-space btcv \
  --model-dir clip_universal_unet \
  --model-id clip_unet \
  --label-space suprem \
  --model-dir suprem_segresnet \
  --model-id segresnet \
  --label-space suprem \
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

DEFAULT_TRAIN_PREDICTION_ROOT = TRAIN_VAL_ROOT / "training"
DEFAULT_VAL_PREDICTION_ROOT = TRAIN_VAL_ROOT / "validation"
DEFAULT_TRAIN_CASES_ROOT = SEG_ROOT / "training_set" / "training_set"
DEFAULT_VAL_CASES_ROOT = SEG_ROOT / "validation_set"
DEFAULT_MODEL_OUTPUT = TRAIN_VAL_ROOT / "random_forest_stacking" / "rf_stacking.joblib"
DEFAULT_OUTPUT_DIR = TRAIN_VAL_ROOT / "random_forest_stacking" / "predictions"

CURVAS_LABELS = (0, 1, 2, 3)
ORGAN_LABELS = (1, 2, 3)
# Feature columns must always use this order.  User-supplied model arguments
# are reordered to match it in resolve_model_specs(), so training and inference
# cannot silently attach a prediction to the wrong set of columns.
FEATURE_MODEL_IDS = ("clip_unet", "segresnet", "swin5050")
SUPPORTED_MODEL_IDS = set(FEATURE_MODEL_IDS)
SUPPORTED_LABEL_SPACES = {"target", "btcv", "suprem", "word"}

FEATURE_NAMES = [
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

    if not (len(model_dirs) == len(model_ids) == len(label_spaces) == 3):
        raise ValueError(
            "Pass exactly three --model-dir, --model-id, and --label-space values."
        )
    if set(model_ids) != SUPPORTED_MODEL_IDS:
        raise ValueError(
            "Expected model IDs clip_unet, segresnet, and swin5050 exactly once; "
            f"got {model_ids}."
        )

    specs = [
        ModelSpec(Path(model_dir), model_id, label_space)
        for model_dir, model_id, label_space in zip(model_dirs, model_ids, label_spaces)
    ]
    by_id = {spec.model_id: spec for spec in specs}
    return [by_id[model_id] for model_id in FEATURE_MODEL_IDS]


def remap_to_curvas(labels: np.ndarray, label_space: str) -> np.ndarray:
    """Collapse a model's native classes into background/pancreas/kidney/liver."""

    label_space = label_space.lower()
    if label_space == "target":
        remapped = labels
    elif label_space == "btcv":
        remapped = map_labels(labels, BTCV_TO_TARGET)
    elif label_space in {"suprem", "word"}:
        remapped = map_labels(labels, SUPREM_TO_TARGET)
    else:
        raise ValueError(f"Unsupported label space: {label_space}")
    return np.asarray(remapped, dtype=np.uint8)


def map_labels(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    labels = np.asarray(labels)
    remapped = np.zeros(labels.shape, dtype=np.uint8)
    for source, target in mapping.items():
        remapped[labels == source] = target
    return remapped


def one_hot_encode_predictions(predictions: list[np.ndarray]) -> np.ndarray:
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
    return features


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
    return remap_to_curvas(labels, spec.label_space), image, prediction_path


def load_case_predictions(
    prediction_root: Path,
    specs: list[ModelSpec],
    case_id: str,
) -> tuple[list[np.ndarray], nib.Nifti1Image, list[Path]]:
    """Load all three remapped predictions and ensure their voxel grids match."""

    predictions: list[np.ndarray] = []
    paths: list[Path] = []
    reference_image: nib.Nifti1Image | None = None
    reference_shape: tuple[int, ...] | None = None

    for spec in specs:
        prediction, image, path = load_remapped_prediction(prediction_root, spec, case_id)
        if reference_shape is None:
            #First prediction sets the reference shape and image for the case, just shape not affine etc.
            reference_shape = prediction.shape
            reference_image = image
        elif prediction.shape != reference_shape:
            raise ValueError(
                f"Prediction shape mismatch for case {case_id}: {path} has "
                f"{prediction.shape}, expected {reference_shape}."
            )
        predictions.append(prediction)
        paths.append(path)

    if reference_image is None:
        raise RuntimeError(f"No predictions loaded for case {case_id}.")
    return predictions, reference_image, paths


def sample_flat_indices(indices: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly sample up to max_samples from a flat array of voxel indices."""
    if max_samples <= 0 or indices.size == 0:
        return np.empty((0,), dtype=np.int64)
    sample_size = min(max_samples, indices.size)
    return rng.choice(indices, size=sample_size, replace=False) # Same voxel cant be sampled twice without replacement


def sample_background_indices(
    target: np.ndarray,
    predictions: list[np.ndarray],
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
        near_indices = np.flatnonzero(candidate_background.reshape(-1))
        selected = sample_flat_indices(near_indices, args.background_samples_per_case, rng)
        # If the boundary band is too small, fill the quota from all remaining
        # true-background voxels rather than reducing this case's sample count.
        remaining_needed = args.background_samples_per_case - selected.size
        if remaining_needed <= 0:
            return selected

        all_background = np.flatnonzero(target_flat == 0)
        if selected.size:
            supplement_pool = all_background[~np.isin(all_background, selected, assume_unique=False)]
        else:
            supplement_pool = all_background
        supplement = sample_flat_indices(supplement_pool, remaining_needed, rng)
        return np.concatenate([selected, supplement])

    background_indices = np.flatnonzero(target_flat == 0)
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
            #Loads the target label image and remaps it to CURVAS label space, then loads the predictions for the case
            target, _ = load_label_image(target_path)
            target = remap_to_curvas(target, "target")
            predictions, _, _ = load_case_predictions(prediction_root, specs, case_id)
        except Exception as exc:
            skipped_cases += 1
            print(f"WARNING: skipping training case {case_id}: {exc}")
            continue

        if any(prediction.shape != target.shape for prediction in predictions): # check if shapes match
            skipped_cases += 1
            shapes = [prediction.shape for prediction in predictions]
            print(
                f"WARNING: skipping training case {case_id}: prediction shapes {shapes} "
                f"do not match target shape {target.shape}."
            )
            continue

        # Flattening preserves voxel alignment: index i refers to the same
        # physical grid location in the target and all three predictions.
        target_flat = target.reshape(-1) # 3d mask converted to 1d array
        sampled_indices: list[np.ndarray] = []
        # Cap each organ separately so large organs (especially liver) do not
        # overwhelm the smaller pancreas class during training.
        for label in ORGAN_LABELS:
            label_indices = np.flatnonzero(target_flat == label)
            sampled_indices.append(sample_flat_indices(label_indices, args.samples_per_class, rng))
        sampled_indices.append(sample_background_indices(target, predictions, args, rng)) # Sample background indices based on the target and predictions

        case_indices = np.concatenate(sampled_indices)
        if case_indices.size == 0:
            skipped_cases += 1
            print(f"WARNING: skipping training case {case_id}: no voxels sampled.")
            continue

        labels = target_flat[case_indices].astype(np.uint8, copy=False)
        # Use the same sampled indices for every base model and the target.
        prediction_samples = [
            prediction.reshape(-1)[case_indices].astype(np.uint8, copy=False)
            for prediction in predictions
        ]
        feature_blocks.append(one_hot_encode_predictions(prediction_samples))
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
            predictions, reference_image, _ = load_case_predictions(prediction_root, specs, case_id)
        except Exception:
            if args.skip_missing:
                print(f"WARNING: skipping validation case {case_id}: missing or invalid predictions.")
                continue
            raise

        # The forest operates on independent voxel rows, so flatten the spatial
        # arrays now and restore their 3-D shape after prediction.
        flat_predictions = [prediction.reshape(-1) for prediction in predictions]
        n_voxels = flat_predictions[0].shape[0]
        predicted_flat = np.empty(n_voxels, dtype=np.uint8)
        confidence_flat = np.empty(n_voxels, dtype=np.uint8)

        # A full scan can contain hundreds of millions of feature values.
        # Chunking limits peak memory without changing any voxel's prediction.
        for start in range(0, n_voxels, args.predict_chunk_size):
            stop = min(start + args.predict_chunk_size, n_voxels)
            chunk_predictions = [prediction[start:stop] for prediction in flat_predictions]
            X_chunk = one_hot_encode_predictions(chunk_predictions)
            predicted_chunk = classifier.predict(X_chunk).astype(np.uint8, copy=False)
            # predict_proba averages the trees' class probabilities; the
            # winning class's probability drives the confidence code.
            probabilities = classifier.predict_proba(X_chunk)
            max_probability = np.max(probabilities, axis=1)
            predicted_flat[start:stop] = predicted_chunk
            confidence_flat[start:stop] = confidence_from_predictions(
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
        "feature_names": FEATURE_NAMES,
        "class_labels": [int(label) for label in classifier.classes_],
        "training_sample_counts": sample_counts,
        "discovered_training_cases": train_cases_count,
        "usable_training_cases": usable_training_cases,
        "skipped_training_cases": skipped_training_cases,
        "discovered_validation_cases": val_cases_count,
    }
    metadata_path = metadata_path_for_model(args.model_output)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata_path


def print_feature_importances(classifier: RandomForestClassifier) -> None:
    print("Feature importances:")
    for name, importance in sorted(
        zip(FEATURE_NAMES, classifier.feature_importances_),
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
    val_cases = discover_case_dirs(args.val_cases_root, args.target_annotation)

    print(f"Number of discovered training cases: {len(train_cases)}")
    print(f"Number of discovered validation cases: {len(val_cases)}")
    print("Model configuration in feature order:")
    for spec in specs:
        print(
            f"  {spec.model_id}: dir={resolve_path(spec.model_dir, args.train_prediction_root)} "
            f"label_space={spec.label_space}"
        )

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
    print(f"Validation output directory: {args.output_dir}")
    print_feature_importances(classifier)

    run_validation_inference(
        classifier,
        val_cases,
        args.val_prediction_root,
        specs,
        args,
    )


if __name__ == "__main__":
    main()
