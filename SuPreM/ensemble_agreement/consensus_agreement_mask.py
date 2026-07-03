#!/usr/bin/env python3
"""Create consensus segmentation masks from three matching prediction folders.

The consensus behavior follows ``Agreement.MD``:

  * 3 identical labels -> that label
  * 2 matching organ labels -> that organ label
  * 2 matching background labels -> background, unless the single organ vote is
    connected to a stronger consensus region of the same organ
  * all labels different -> choose a locally supported label, otherwise mark
    uncertain or fall back to CLIP Universal U-Net
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_PREDICTION_ROOT = PROJECT_DIR / "results" / "CURVAS_INFERENCE"
DEFAULT_UNCERTAIN_LABEL = 255
SUPPORTED_LABEL_SPACES = {"target", "btcv", "suprem"}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one consensus agreement mask for every NIfTI prediction "
            "filename present in all three model subdirectories."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "prediction_root",
        type=Path,
        nargs="?",
        default=DEFAULT_PREDICTION_ROOT,
        help=(
            "Directory containing exactly three prediction subdirectories, unless "
            "--model-dir is provided."
        ),
    )
    parser.add_argument(
        "--model-dir",
        dest="model_dirs",
        type=Path,
        action="append",
        default=None,
        help=(
            "Prediction directory for one model. Repeat exactly three times. "
            "Relative paths are resolved under prediction_root."
        ),
    )
    parser.add_argument(
        "--label-space",
        dest="label_spaces",
        choices=sorted(SUPPORTED_LABEL_SPACES),
        action="append",
        default=None,
        help=(
            "Label space for one --model-dir, in the same order. Use target for "
            "already-remapped 0-3 masks, btcv for Swin/BTCV labels, and suprem "
            "for SuPreM/WORD labels. If omitted, all models are treated as target."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for consensus masks. If omitted, writes to "
            "prediction_root/agreement_masks."
        ),
    )
    parser.add_argument(
        "--case-name",
        default=None,
        help=(
            "Process only this case. Accepts either the full filename, e.g. "
            "UKCHLL003.nii.gz, or the stem, e.g. UKCHLL003."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if the three model directories do not contain identical NIfTI names.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing consensus mask.",
    )
    parser.add_argument(
        "--ignore-affine",
        action="store_true",
        help="Only check array shapes, not NIfTI affines.",
    )
    parser.add_argument(
        "--uncertain-label",
        type=int,
        default=DEFAULT_UNCERTAIN_LABEL,
        help=(
            "Output value for all-disagree voxels without strong local evidence "
            "when CLIP fallback is disabled or unavailable."
        ),
    )
    parser.add_argument(
        "--clip-fallback-dir",
        type=Path,
        default=None,
        help=(
            "Model directory to use when consensus and local support abstain. "
            "The directory must be one of the three prediction directories. "
            "If omitted, a directory with 'clip' in its name is detected automatically."
        ),
    )
    parser.add_argument(
        "--no-clip-fallback",
        action="store_true",
        help="Keep unresolved all-disagree voxels as --uncertain-label instead of using CLIP fallback.",
    )
    parser.add_argument(
        "--local-radius",
        type=int,
        default=1,
        help=(
            "Radius, in voxels, used for all-disagree local decisions. "
            "The default 1 means a 3x3x3 neighborhood."
        ),
    )
    parser.add_argument(
        "--min-local-margin",
        type=float,
        default=3.0,
        help=(
            "Minimum local support gap required between the best and second-best "
            "label in all-disagree voxels."
        ),
    )
    parser.add_argument(
        "--min-local-support",
        type=float,
        default=3.0,
        help="Minimum local support required for an all-disagree local decision.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("W1", "W2", "W3"),
        help=(
            "Model weights used only for all-disagree local decisions. The order "
            "matches the discovered or explicitly supplied model directories."
        ),
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help=(
            "Connectivity for the '[0,0,organ] unless connected to organ' rule. "
            "In 3D, 3 means 26-neighbor connectivity."
        ),
    )
    parser.add_argument(
        "--confidence-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for confidence maps with labels "
            "1=high, 2=medium, 3=medium-low, 4=low, 5=very-low/uncertain."
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path


def output_dir_from_args(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return resolve_path(args.output_dir, PROJECT_DIR)
    return args.prediction_root / "agreement_masks"


def is_nifti_file(path: Path) -> bool:
    return path.is_file() and (path.name.endswith(".nii") or path.name.endswith(".nii.gz"))


def discover_model_dirs(
    prediction_root: Path,
    explicit_model_dirs: list[Path] | None,
    output_dir: Path,
    confidence_dir: Path | None,
) -> list[Path]:
    if explicit_model_dirs is not None:
        if len(explicit_model_dirs) != 3:
            raise ValueError("Pass exactly three --model-dir values.")
        return [resolve_path(path, prediction_root) for path in explicit_model_dirs]

    if not prediction_root.is_dir():
        raise NotADirectoryError(f"Prediction root does not exist: {prediction_root}")

    ignored = {output_dir.resolve(), (prediction_root / "agreement_masks").resolve()}
    if confidence_dir is not None:
        ignored.add(confidence_dir.resolve())

    model_dirs = [
        path
        for path in prediction_root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.resolve() not in ignored
    ]
    model_dirs = sorted(model_dirs)
    if len(model_dirs) != 3:
        names = ", ".join(path.name for path in model_dirs) or "none"
        raise ValueError(
            f"Expected exactly three prediction subdirectories in {prediction_root}, "
            f"found {len(model_dirs)} ({names}). Use --model-dir to choose them explicitly."
        )
    return model_dirs


def clip_fallback_index(
    model_dirs: list[Path],
    explicit_clip_fallback_dir: Path | None,
    prediction_root: Path,
    disabled: bool,
) -> int | None:
    if disabled:
        return None

    if explicit_clip_fallback_dir is not None:
        fallback_dir = resolve_path(explicit_clip_fallback_dir, prediction_root).resolve()
        matches = [
            index
            for index, model_dir in enumerate(model_dirs)
            if model_dir.resolve() == fallback_dir
        ]
        if not matches:
            raise ValueError(
                "--clip-fallback-dir must match one of the three prediction directories."
            )
        return matches[0]

    candidates = [
        index
        for index, model_dir in enumerate(model_dirs)
        if "clip" in model_dir.name.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(model_dirs[index].name for index in candidates)
        raise ValueError(
            f"Multiple CLIP-like model directories found ({names}); pass "
            "--clip-fallback-dir to choose one."
        )
    return None


def nifti_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Prediction directory does not exist: {directory}")
    files = {path.name: path for path in directory.iterdir() if is_nifti_file(path)}
    if not files:
        raise FileNotFoundError(f"No .nii or .nii.gz files found in {directory}")
    return files


def matching_cases(
    model_dirs: list[Path],
    case_name: str | None,
    strict: bool,
) -> tuple[list[dict[str, Path]], list[str]]:
    files_by_model = [nifti_files(directory) for directory in model_dirs]
    name_sets = [set(files) for files in files_by_model]
    common_names = set.intersection(*name_sets)

    if strict:
        reference_names = name_sets[0]
        for directory, names in zip(model_dirs[1:], name_sets[1:]):
            if names != reference_names:
                missing = sorted(reference_names - names)
                unexpected = sorted(names - reference_names)
                details = []
                if missing:
                    details.append(f"missing {len(missing)} case(s), e.g. {missing[:3]}")
                if unexpected:
                    details.append(f"has {len(unexpected)} extra case(s), e.g. {unexpected[:3]}")
                raise ValueError(f"Case mismatch in {directory}: {'; '.join(details)}")
    else:
        for directory, names in zip(model_dirs, name_sets):
            skipped = sorted(names - common_names)
            if skipped:
                print(
                    f"Skipping {len(skipped)} unmatched case(s) from {directory}: "
                    f"{skipped[:3]}"
                )

    if not common_names:
        raise FileNotFoundError("No matching NIfTI filenames were found across all three directories.")

    case_names = sorted(common_names)
    if case_name is not None:
        candidates = [case_name]
        if not case_name.endswith(".nii") and not case_name.endswith(".nii.gz"):
            candidates.append(f"{case_name}.nii.gz")
            candidates.append(f"{case_name}.nii")
        matched_name = next((name for name in candidates if name in common_names), None)
        if matched_name is None:
            raise FileNotFoundError(
                f"{case_name} is not present in all three prediction directories."
            )
        case_names = [matched_name]

    return files_by_model, case_names


def load_integer_mask(path: Path):
    nib = import_nibabel()

    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if not np.issubdtype(data.dtype, np.integer):
        rounded = np.rint(data)
        if not np.all(np.isclose(data, rounded, rtol=0.0, atol=1e-3)):
            raise ValueError(f"{path} contains non-integer labels.")
        data = rounded
    return image, data.astype(np.int16, copy=False)


def remap_lookup(prediction: np.ndarray, lookup: dict[int, int], case_name: str, model_name: str) -> np.ndarray:
    unique_labels = np.unique(prediction).astype(int)
    unknown = sorted(set(unique_labels) - set(lookup))
    if unknown:
        raise ValueError(f"{case_name}: unexpected label(s) in {model_name}: {unknown}")

    table = np.zeros(max(lookup) + 1, dtype=np.int16)
    for source_label, target_label in lookup.items():
        table[source_label] = target_label
    return table[prediction]


def remap_prediction_to_target(
    prediction: np.ndarray,
    label_space: str,
    case_name: str,
    model_name: str,
) -> np.ndarray:
    """Normalize one model prediction into 0=bg, 1=pancreas, 2=kidney, 3=liver."""

    if label_space == "target":
        unique_labels = np.unique(prediction).astype(int)
        unknown = sorted(set(unique_labels) - {0, 1, 2, 3})
        if unknown:
            raise ValueError(f"{case_name}: unexpected target label(s) in {model_name}: {unknown}")
        return prediction.astype(np.int16, copy=False)
    if label_space == "btcv":
        return remap_lookup(prediction, BTCV_TO_TARGET, case_name, model_name)
    if label_space == "suprem":
        return remap_lookup(prediction, SUPREM_TO_TARGET, case_name, model_name)
    raise ValueError(f"Unsupported label space for {model_name}: {label_space}")


def validate_spatial_grid(
    case_name: str,
    images: list[object],
    ignore_affine: bool,
) -> None:
    reference = images[0]
    for model_index, image in enumerate(images[1:], start=2):
        if image.shape != reference.shape:
            raise ValueError(
                f"{case_name}: model {model_index} shape {image.shape} does not "
                f"match model 1 shape {reference.shape}"
            )
        if not ignore_affine and not np.allclose(
            image.affine,
            reference.affine,
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError(
                f"{case_name}: model {model_index} affine does not match model 1"
            )


def make_structure(ndim: int, connectivity: int) -> np.ndarray:
    return ndimage.generate_binary_structure(ndim, min(connectivity, ndim))


def pairwise_majority(first: np.ndarray, second: np.ndarray, third: np.ndarray):
    first_second = first == second
    first_third = first == third
    second_third = second == third

    pair_agrees = first_second | first_third | second_third
    pair_label = np.zeros(first.shape, dtype=first.dtype)
    pair_label[first_second] = first[first_second]
    pair_label[first_third] = first[first_third]
    pair_label[second_third] = second[second_third]

    unanimous = first_second & first_third
    all_disagree = ~pair_agrees
    return unanimous, pair_agrees, pair_label, all_disagree


def connected_single_organ_votes(
    consensus: np.ndarray,
    confidence: np.ndarray,
    single_organ_label: np.ndarray,
    two_background: np.ndarray,
    connectivity: int,
) -> None:
    structure = make_structure(consensus.ndim, connectivity)
    candidate_labels = sorted(
        int(label)
        for label in np.unique(single_organ_label[two_background])
        if int(label) != 0
    )

    confident_organ = confidence <= 2
    for label in candidate_labels:
        candidates = two_background & (single_organ_label == label)
        confident_same_label = confident_organ & (consensus == label)
        if not candidates.any() or not confident_same_label.any():
            continue

        components, _ = ndimage.label(candidates | confident_same_label, structure=structure)
        connected_ids = np.unique(components[confident_same_label])
        connected_ids = connected_ids[connected_ids != 0]
        if connected_ids.size == 0:
            continue

        keep = candidates & np.isin(components, connected_ids)
        consensus[keep] = label
        confidence[keep] = 3


def local_label_decisions(
    predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
    all_disagree: np.ndarray,
    weights: tuple[float, float, float],
    local_radius: int,
    min_local_margin: float,
    min_local_support: float,
) -> tuple[np.ndarray, np.ndarray]:
    if local_radius < 0:
        raise ValueError("--local-radius must be non-negative.")
    if len(weights) != len(predictions):
        raise ValueError("Number of weights must match number of predictions.")
    if len({prediction.shape for prediction in predictions}) != 1:
        raise ValueError("All predictions must have the same shape.")

    output_shape = predictions[0].shape
    label_values = set()
    for prediction in predictions:
        label_values.update(int(label) for label in np.unique(prediction))
    labels = sorted(label_values)
    if not labels or not all_disagree.any():
        return np.zeros(output_shape, dtype=np.int16), np.zeros(output_shape, dtype=bool)

    footprint = np.ones(
        (2 * local_radius + 1,) * predictions[0].ndim,
        dtype=np.float32,
    )

    # Keep only the best and second-best label scores instead of stacking one
    # full-volume support map per label. This preserves the decision rule but
    # avoids a large support_stack allocation for every case.
    top_score = np.full(output_shape, -np.inf, dtype=np.float32)
    second_score = np.full(output_shape, -np.inf, dtype=np.float32)
    top_labels = np.zeros(output_shape, dtype=np.int16)

    for label in labels:
        support = np.zeros(output_shape, dtype=np.float32)
        present = np.zeros(output_shape, dtype=bool)
        for prediction, weight in zip(predictions, weights):
            label_mask = prediction == label
            present |= label_mask
            label_votes = label_mask.astype(np.float32, copy=False) * weight
            support += ndimage.convolve(label_votes, footprint, mode="constant", cval=0.0)

        support[~present] = -np.inf
        better = support > top_score
        second_score[better] = top_score[better]
        top_score[better] = support[better]
        top_labels[better] = label

        second_better = (~better) & (support > second_score)
        second_score[second_better] = support[second_better]

    accepted = (
        all_disagree
        & np.isfinite(top_score)
        & (top_score >= min_local_support)
        & ((top_score - second_score) >= min_local_margin)
    )
    return top_labels, accepted


def expanded_mask_slices(mask: np.ndarray, radius: int) -> tuple[slice, ...] | None:
    """Return the bounding box of true voxels, padded by local radius."""

    coordinates = np.where(mask)
    if len(coordinates) == 0 or coordinates[0].size == 0:
        return None

    slices = []
    for axis_coordinates, axis_size in zip(coordinates, mask.shape):
        start = max(int(axis_coordinates.min()) - radius, 0)
        stop = min(int(axis_coordinates.max()) + radius + 1, axis_size)
        slices.append(slice(start, stop))
    return tuple(slices)


def consensus_agreement_mask(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    *,
    uncertain_label: int = DEFAULT_UNCERTAIN_LABEL,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    local_radius: int = 1,
    min_local_margin: float = 3.0,
    min_local_support: float = 3.0,
    connectivity: int = 3,
    clip_fallback: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    shapes = {array.shape for array in (first, second, third)}
    if len(shapes) != 1:
        raise ValueError(f"Prediction shapes differ: {sorted(shapes)}")
    if clip_fallback is not None and clip_fallback.shape != first.shape:
        raise ValueError(
            f"CLIP fallback shape {clip_fallback.shape} does not match prediction shape {first.shape}"
        )

    unanimous, pair_agrees, pair_label, all_disagree = pairwise_majority(first, second, third)

    consensus_dtype = (
        np.int16
        if np.iinfo(np.int16).min <= uncertain_label <= np.iinfo(np.int16).max
        else np.int32
    )
    consensus = np.full(first.shape, uncertain_label, dtype=consensus_dtype)
    confidence = np.full(first.shape, 5, dtype=np.uint8)

    consensus[unanimous] = first[unanimous]
    confidence[unanimous] = 1

    two_agree = pair_agrees & ~unanimous
    two_agree_organ = two_agree & (pair_label != 0)
    consensus[two_agree_organ] = pair_label[two_agree_organ]
    confidence[two_agree_organ] = 2

    two_background = two_agree & (pair_label == 0)
    consensus[two_background] = 0
    confidence[two_background] = 3
    single_organ_label = first + second + third
    connected_single_organ_votes(
        consensus,
        confidence,
        single_organ_label,
        two_background,
        connectivity,
    )

    local_slices = expanded_mask_slices(all_disagree, local_radius)
    local_accept = None
    if local_slices is not None:
        local_labels, local_accept = local_label_decisions(
            (first[local_slices], second[local_slices], third[local_slices]),
            all_disagree[local_slices],
            weights,
            local_radius,
            min_local_margin,
            min_local_support,
        )
        consensus_crop = consensus[local_slices]
        confidence_crop = confidence[local_slices]
        consensus_crop[local_accept] = local_labels[local_accept]
        confidence_crop[local_accept] = 4

    if clip_fallback is not None:
        unresolved = all_disagree.copy()
        if local_slices is not None and local_accept is not None:
            unresolved_crop = unresolved[local_slices]
            unresolved_crop[local_accept] = False
        consensus[unresolved] = clip_fallback[unresolved]

    return consensus, confidence


def output_dtype(data: np.ndarray) -> np.dtype:
    data_min = int(data.min())
    data_max = int(data.max())
    if 0 <= data_min and data_max <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if np.iinfo(np.int16).min <= data_min and data_max <= np.iinfo(np.int16).max:
        return np.dtype(np.int16)
    return np.dtype(np.int32)


def save_mask(
    data: np.ndarray,
    reference: object,
    output_path: Path,
    description: str,
) -> None:
    nib = import_nibabel()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = output_dtype(data)
    saved_data = data.astype(dtype, copy=False)

    header = reference.header.copy()
    header.set_data_dtype(dtype)
    header["cal_min"] = int(saved_data.min())
    header["cal_max"] = int(saved_data.max())
    header["descrip"] = description[:79]

    output = nib.Nifti1Image(saved_data, reference.affine, header)
    qform, qform_code = reference.get_qform(coded=True)
    sform, sform_code = reference.get_sform(coded=True)
    if qform is not None:
        output.set_qform(qform, int(qform_code))
    if sform is not None:
        output.set_sform(sform, int(sform_code))
    nib.save(output, str(output_path))


def import_nibabel():
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nibabel is required to read and write NIfTI masks. Install nibabel "
            "in the Python environment used to run this script."
        ) from exc
    return nib


def main() -> None:
    args = parse_args()
    args.prediction_root = resolve_path(args.prediction_root, PROJECT_DIR)
    output_dir = output_dir_from_args(args)
    confidence_dir = (
        resolve_path(args.confidence_dir, PROJECT_DIR)
        if args.confidence_dir is not None
        else None
    )

    model_dirs = discover_model_dirs(
        args.prediction_root,
        args.model_dirs,
        output_dir,
        confidence_dir,
    )
    if args.label_spaces is None:
        label_spaces = ["target"] * len(model_dirs)
    elif len(args.label_spaces) != len(model_dirs):
        raise ValueError("Pass one --label-space value for each model directory.")
    else:
        label_spaces = args.label_spaces

    fallback_index = clip_fallback_index(
        model_dirs,
        args.clip_fallback_dir,
        args.prediction_root,
        args.no_clip_fallback,
    )
    print("Using prediction directories:")
    for index, (directory, weight, label_space) in enumerate(zip(model_dirs, args.weights, label_spaces)):
        fallback_note = " + CLIP fallback" if index == fallback_index else ""
        print(f"  {directory} (label_space={label_space}, local weight={weight:g}){fallback_note}")
    if fallback_index is None:
        print(
            "CLIP fallback disabled or not auto-detected; unresolved voxels "
            f"will remain {args.uncertain_label}."
        )

    files_by_model, case_names = matching_cases(model_dirs, args.case_name, args.strict)
    output_dir.mkdir(parents=True, exist_ok=True)
    if confidence_dir is not None:
        confidence_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    skipped_count = 0
    for case_name in tqdm(case_names, desc="Consensus agreement"):
        output_path = output_dir / case_name
        confidence_path = confidence_dir / case_name if confidence_dir is not None else None
        if output_path.is_file() and not args.overwrite:
            print(f"Output already exists; skipping: {output_path}")
            skipped_count += 1
            continue

        loaded = [load_integer_mask(files[case_name]) for files in files_by_model]
        images = [image for image, _ in loaded]
        predictions = [
            remap_prediction_to_target(data, label_space, case_name, model_dir.name)
            for (model_dir, label_space, (_image, data)) in zip(model_dirs, label_spaces, loaded)
        ]
        validate_spatial_grid(case_name, images, args.ignore_affine)

        consensus, confidence = consensus_agreement_mask(
            *predictions,
            uncertain_label=args.uncertain_label,
            weights=tuple(args.weights),
            local_radius=args.local_radius,
            min_local_margin=args.min_local_margin,
            min_local_support=args.min_local_support,
            connectivity=args.connectivity,
            clip_fallback=predictions[fallback_index] if fallback_index is not None else None,
        )
        save_mask(
            consensus,
            images[0],
            output_path,
            (
                f"Consensus mask; clip fallback={model_dirs[fallback_index].name}"
                if fallback_index is not None
                else f"Consensus mask; uncertain={args.uncertain_label}"
            ),
        )
        if confidence_path is not None:
            save_mask(
                confidence,
                images[0],
                confidence_path,
                "Confidence: 1 high, 2 med, 3 med-low, 4 low, 5 very-low",
            )
        saved_count += 1

    print(
        f"Saved {saved_count} consensus mask(s) to {output_dir}"
        + (f"; skipped {skipped_count} existing mask(s)" if skipped_count else "")
    )
    if confidence_dir is not None:
        print(f"Saved matching confidence map(s) to {confidence_dir}")


if __name__ == "__main__":
    main()
