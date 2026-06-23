#!/usr/bin/env python3
"""Create one four-level agreement map from four model predictions."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent

# Convert native Swin 5050/BTCV labels into the WORD IDs already used by the
# three SuPreM outputs. BTCV-only structures receive reserved values so they
# remain distinct from WORD background rather than being mistaken for label 0.
BTCV_TO_WORD = {
    0: 0,   # background
    1: 2,   # spleen
    2: 4,   # right kidney
    3: 3,   # left kidney
    4: 6,   # gallbladder
    5: 7,   # esophagus
    6: 1,   # liver
    7: 5,   # stomach
    11: 8,  # pancreas
    12: 12, # right adrenal -> WORD's combined adrenal label
    13: 12, # left adrenal -> WORD's combined adrenal label
    8: 101, # aorta: absent from the saved WORD-numbered SuPreM maps
    9: 102, # inferior vena cava: absent from the saved WORD maps
    10: 103,# portal/splenic veins: absent from the saved WORD maps
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare U-Net, Swin UNETR, SegResNet, and Swin 5050 predictions. "
            "Output labels are 1=all four agree, 2=three agree, "
            "3=two agree, and 4=all four disagree."
        )
    )
    parser.add_argument(
        "--unet-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "word_three_models" / "unet",
    )
    parser.add_argument(
        "--swinunetr-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "word_three_models" / "swinunetr",
    )
    parser.add_argument(
        "--segresnet-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "word_three_models" / "segresnet",
    )
    parser.add_argument(
        "--swin5050-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "word_swinunetr_5050",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "word_four_model_agreement",
    )
    parser.add_argument(
        "--case-name",
        default=None,
        help="Process only this .nii.gz filename instead of every matching case.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing agreement map.",
    )
    return parser.parse_args()


def nifti_files(directory):
    if not directory.is_dir():
        raise NotADirectoryError(f"Prediction directory does not exist: {directory}")
    files = {path.name: path for path in directory.glob("*.nii.gz")}
    if not files:
        raise FileNotFoundError(f"No .nii.gz files found in {directory}")
    return files


def matching_cases(model_dirs):
    """Require one identically named prediction from every model."""
    files_by_model = [nifti_files(directory) for directory in model_dirs]
    reference_names = set(files_by_model[0])

    for directory, files in zip(model_dirs[1:], files_by_model[1:]):
        names = set(files)
        if names != reference_names:
            missing = sorted(reference_names - names)
            unexpected = sorted(names - reference_names)
            details = []
            if missing:
                details.append(f"missing {len(missing)} case(s)")
            if unexpected:
                details.append(f"has {len(unexpected)} additional case(s)")
            raise ValueError(
                f"Case mismatch in {directory}: {', '.join(details)}. "
                "Finish inference for all models before creating the ensemble."
            )

    return files_by_model, sorted(reference_names)


def validate_spatial_grid(case_name, images):
    """Ensure corresponding voxel indices represent the same physical location."""
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


def remap_btcv_to_word(btcv):
    """Map BTCV anatomy to WORD IDs or reserved non-background values."""
    if not np.issubdtype(btcv.dtype, np.integer):
        if not np.all(np.equal(btcv, np.round(btcv))):
            raise ValueError("Swin 5050 prediction contains non-integer labels.")
        btcv = np.round(btcv).astype(np.int16)

    unknown = sorted(
        set(np.unique(btcv).astype(int))
        - set(BTCV_TO_WORD)
    )
    if unknown:
        raise ValueError(f"Unexpected Swin 5050 BTCV label(s): {unknown}")

    remapped = np.zeros(btcv.shape, dtype=np.uint8)
    for btcv_label, word_label in BTCV_TO_WORD.items():
        remapped[btcv == btcv_label] = word_label
    return remapped


def four_model_agreement(first, second, third, fourth):
    """Assign agreement levels from the largest number of matching votes."""
    shapes = {array.shape for array in (first, second, third, fourth)}
    if len(shapes) != 1:
        raise ValueError(f"Prediction shapes differ: {sorted(shapes)}")

    # Count how many of the four predictions match each model's label. The
    # largest count is the winning vote size at that voxel:
    #   4 -> unanimous
    #   3 -> three-versus-one
    #   2 -> two-versus-two or two-versus-one-versus-one
    #   1 -> four different labels
    predictions = np.stack((first, second, third, fourth), axis=0)
    vote_counts = np.stack(
        [(predictions == prediction).sum(axis=0) for prediction in predictions],
        axis=0,
    )
    largest_vote = vote_counts.max(axis=0)

    # Convert largest vote size 4,3,2,1 into agreement label 1,2,3,4.
    return (5 - largest_vote).astype(np.uint8)


def save_map(data, reference, output_path, description, cal_min, cal_max):
    """Save a uint8 map while preserving the first model's NIfTI geometry."""
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    header["cal_min"] = cal_min
    header["cal_max"] = cal_max
    header["descrip"] = description
    output = nib.Nifti1Image(data.astype(np.uint8), reference.affine, header)

    qform, qform_code = reference.get_qform(coded=True)
    sform, sform_code = reference.get_sform(coded=True)
    if qform is not None:
        output.set_qform(qform, int(qform_code))
    if sform is not None:
        output.set_sform(sform, int(sform_code))
    nib.save(output, str(output_path))


def main():
    args = parse_args()
    model_dirs = [
        args.unet_dir,
        args.swinunetr_dir,
        args.segresnet_dir,
        args.swin5050_dir,
    ]
    files_by_model, case_names = matching_cases(model_dirs)
    if args.case_name is not None:
        if args.case_name not in case_names:
            raise FileNotFoundError(
                f"Case {args.case_name} is not present in all four model directories."
            )
        case_names = [args.case_name]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for case_name in tqdm(case_names, desc="Four-model agreement"):
        output_path = args.output_dir / case_name
        if output_path.is_file() and not args.overwrite:
            print(f"Output already exists; skipping: {output_path}")
            continue

        images = [nib.load(str(files[case_name])) for files in files_by_model]
        validate_spatial_grid(case_name, images)

        suprem_predictions = [
            np.asanyarray(image.dataobj) for image in images[:3]
        ]
        swin5050 = np.asanyarray(images[3].dataobj)
        remapped_swin5050 = remap_btcv_to_word(swin5050)

        agreement = four_model_agreement(
            *suprem_predictions,
            remapped_swin5050,
        )
        save_map(
            agreement,
            images[0],
            output_path,
            "Agreement: 1=four, 2=three, 3=two, 4=none",
            1,
            4,
        )
        saved_count += 1

    print(f"Saved {saved_count} agreement map(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
