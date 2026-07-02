#!/usr/bin/env python3
"""Create consensus segmentation masks from three matching prediction folders.

The consensus behavior follows ``Agreement.MD``:

  * 3 identical labels -> that label
  * 2 matching organ labels -> that organ label
  * 2 matching background labels -> background, unless the single organ vote is
    connected to a stronger consensus region of the same organ
  * all labels different -> choose a locally supported label, otherwise mark
    uncertain
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
        help="Output value for all-disagree voxels without strong local evidence.",
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
    return image, data.astype(np.int32, copy=False)


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
    pair_label = np.zeros(first.shape, dtype=np.int32)
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
    predictions: np.ndarray,
    all_disagree: np.ndarray,
    weights: tuple[float, float, float],
    local_radius: int,
    min_local_margin: float,
    min_local_support: float,
) -> tuple[np.ndarray, np.ndarray]:
    if local_radius < 0:
        raise ValueError("--local-radius must be non-negative.")
    if len(weights) != predictions.shape[0]:
        raise ValueError("Number of weights must match number of predictions.")

    labels = sorted(int(label) for label in np.unique(predictions))
    if not labels or not all_disagree.any():
        return np.zeros(predictions.shape[1:], dtype=np.int32), np.zeros(
            predictions.shape[1:],
            dtype=bool,
        )

    footprint = np.ones(
        (2 * local_radius + 1,) * (predictions.ndim - 1),
        dtype=np.float32,
    )
    support_maps = []
    present_maps = []
    for label in labels:
        support = np.zeros(predictions.shape[1:], dtype=np.float32)
        for prediction, weight in zip(predictions, weights):
            label_votes = (prediction == label).astype(np.float32, copy=False) * weight
            support += ndimage.convolve(label_votes, footprint, mode="constant", cval=0.0)
        support_maps.append(support)
        present_maps.append(np.any(predictions == label, axis=0))

    support_stack = np.stack(support_maps, axis=0)
    present_stack = np.stack(present_maps, axis=0)
    candidate_scores = np.where(present_stack, support_stack, -np.inf)

    top_index = np.argmax(candidate_scores, axis=0)
    top_score = np.take_along_axis(candidate_scores, top_index[None, ...], axis=0)[0]
    if len(labels) == 1:
        second_score = np.full(top_score.shape, -np.inf, dtype=np.float32)
    else:
        second_score = np.partition(candidate_scores, -2, axis=0)[-2]

    top_labels = np.asarray(labels, dtype=np.int32)[top_index]
    accepted = (
        all_disagree
        & np.isfinite(top_score)
        & (top_score >= min_local_support)
        & ((top_score - second_score) >= min_local_margin)
    )
    return top_labels, accepted


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
) -> tuple[np.ndarray, np.ndarray]:
    shapes = {array.shape for array in (first, second, third)}
    if len(shapes) != 1:
        raise ValueError(f"Prediction shapes differ: {sorted(shapes)}")

    unanimous, pair_agrees, pair_label, all_disagree = pairwise_majority(first, second, third)

    consensus = np.full(first.shape, uncertain_label, dtype=np.int32)
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

    predictions = np.stack((first, second, third), axis=0)
    local_labels, local_accept = local_label_decisions(
        predictions,
        all_disagree,
        weights,
        local_radius,
        min_local_margin,
        min_local_support,
    )
    consensus[local_accept] = local_labels[local_accept]
    confidence[local_accept] = 4

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
    print("Using prediction directories:")
    for directory, weight in zip(model_dirs, args.weights):
        print(f"  {directory} (local weight={weight:g})")

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
        predictions = [data for _, data in loaded]
        validate_spatial_grid(case_name, images, args.ignore_affine)

        consensus, confidence = consensus_agreement_mask(
            *predictions,
            uncertain_label=args.uncertain_label,
            weights=tuple(args.weights),
            local_radius=args.local_radius,
            min_local_margin=args.min_local_margin,
            min_local_support=args.min_local_support,
            connectivity=args.connectivity,
        )
        save_mask(
            consensus,
            images[0],
            output_path,
            f"Consensus mask; uncertain={args.uncertain_label}",
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
