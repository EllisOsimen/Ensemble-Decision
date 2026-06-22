#!/usr/bin/env python3
"""Create voxel-wise agreement maps from three NIfTI label predictions."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare three directories of identically named .nii.gz label maps. "
            "Output voxels are 1 when all models predict the same label, 2 when "
            "exactly two models agree, and 3 when all three predictions differ."
        )
    )
    parser.add_argument(
        "prediction_dirs",
        type=Path,
        nargs=3,
        metavar="PREDICTION_DIR",
        help="Directory containing one model's .nii.gz label predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which to save the agreement maps.",
    )
    return parser.parse_args()


def nifti_files(directory):
    if not directory.is_dir():
        raise NotADirectoryError(f"Prediction directory does not exist: {directory}")
    files = {path.name: path for path in directory.glob("*.nii.gz")}
    if not files:
        raise FileNotFoundError(f"No .nii.gz files found in {directory}")
    return files


def matching_cases(prediction_dirs):
    files_by_model = [nifti_files(directory) for directory in prediction_dirs]
    reference_names = set(files_by_model[0])

    for directory, files in zip(prediction_dirs[1:], files_by_model[1:]):
        names = set(files)
        if names != reference_names:
            missing = sorted(reference_names - names)
            unexpected = sorted(names - reference_names)
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected: {', '.join(unexpected)}")
            raise ValueError(f"Case mismatch in {directory} ({'; '.join(details)})")

    return files_by_model, sorted(reference_names)


def agreement_labels(first, second, third):
    if first.shape != second.shape or first.shape != third.shape:
        raise ValueError(
            f"Prediction shapes differ: {first.shape}, {second.shape}, {third.shape}"
        )

    unanimous = (first == second) & (second == third)
    pair_agreement = (first == second) | (first == third) | (second == third)

    agreement = np.full(first.shape, 3, dtype=np.uint8)
    agreement[pair_agreement] = 2
    agreement[unanimous] = 1
    return agreement


def validate_spatial_grid(case_name, images):
    reference = images[0]
    for model_index, image in enumerate(images[1:], start=2):
        if image.shape != reference.shape:
            raise ValueError(
                f"{case_name}: model {model_index} shape {image.shape} does not "
                f"match model 1 shape {reference.shape}"
            )
        if not np.allclose(image.affine, reference.affine, rtol=1e-5, atol=1e-5):
            raise ValueError(
                f"{case_name}: model {model_index} affine does not match model 1"
            )


def save_agreement(agreement, reference, output_path):
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    header["cal_min"] = 1
    header["cal_max"] = 3
    header["descrip"] = "Model agreement: 1=all, 2=two, 3=none"

    output = nib.Nifti1Image(agreement, reference.affine, header)
    qform, qform_code = reference.get_qform(coded=True)
    sform, sform_code = reference.get_sform(coded=True)
    if qform is not None:
        output.set_qform(qform, int(qform_code))
    if sform is not None:
        output.set_sform(sform, int(sform_code))
    nib.save(output, str(output_path))


def main():
    args = parse_args()
    files_by_model, case_names = matching_cases(args.prediction_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for case_name in tqdm(case_names, desc="Ensemble agreement"):
        images = [nib.load(str(files[case_name])) for files in files_by_model]
        validate_spatial_grid(case_name, images)
        predictions = [np.asanyarray(image.dataobj) for image in images]
        agreement = agreement_labels(*predictions)
        save_agreement(agreement, images[0], args.output_dir / case_name)

    print(f"Saved {len(case_names)} agreement map(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
