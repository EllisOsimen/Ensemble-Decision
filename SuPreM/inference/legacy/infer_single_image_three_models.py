#!/usr/bin/env python3
"""Run all three downloaded SuPreM models on one CT NIfTI image."""

import argparse
import gc
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

from ensemble_agreement import agreement_labels, save_agreement
from evaluate_word import WORD_TO_SUPREM, invert_prediction, load_model


SCRIPT_DIR = Path(__file__).resolve().parent

# Checkpoints are run in dictionary insertion order. Only one model is kept on
# the GPU at a time, which keeps memory use lower than loading all three models
# simultaneously.
MODEL_CHECKPOINTS = {
    "unet": "supervised_suprem_unet_2100.pth",
    "swinunetr": "supervised_suprem_swinunetr_2100.pth",
    "segresnet": "supervised_suprem_segresnet_2100.pth",
}


def parse_args():
    """Define and read the command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the SuPreM U-Net, Swin UNETR, and SegResNet checkpoints on one "
            "CT image and save all three WORD-numbered predictions plus their "
            "voxel-wise agreement map."
        )
    )
    parser.add_argument("--image", type=Path, required=True, help="Input CT .nii.gz file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=SCRIPT_DIR / "pretrained_weights",
        help="Directory containing the three downloaded SuPreM checkpoints.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roi-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.5, 1.5, 1.5))
    parser.add_argument("--overlap", type=float, default=0.75)
    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=1,
        help="Sliding-window patches evaluated together; 1 uses the least GPU memory.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun all three models and replace existing outputs.",
    )
    return parser.parse_args()


def output_name(image_path):
    """Return a compressed NIfTI filename for the model outputs."""
    if image_path.name.endswith(".nii.gz"):
        return image_path.name
    if image_path.suffix == ".nii":
        return f"{image_path.stem}.nii.gz"
    raise ValueError(f"Input image must end in .nii or .nii.gz: {image_path}")


def make_loader(image_path, spacing):
    """Create the one-image loader and record invertible preprocessing steps."""

    # These transforms match evaluate_word.py. MONAI records orientation,
    # spacing, and cropping operations so invert_prediction() can later return
    # each model output to the original CT grid.
    transforms = Compose(
        [
            # Load the NIfTI data and affine/header metadata.
            LoadImaged(keys=["image"]),
            # Convert a 3D array into the channel-first shape [1, D, H, W].
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            # Give every model a consistent anatomical orientation.
            Orientationd(keys=["image"], axcodes="RAS"),
            # Resample to the voxel spacing used during SuPreM inference.
            Spacingd(keys=["image"], pixdim=tuple(spacing), mode="bilinear"),
            # Clip the useful abdominal CT window and normalize it to [0, 1].
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # Remove empty margins to reduce the volume processed by the model.
            CropForegroundd(keys=["image"], source_key="image"),
        ]
    )
    loader = DataLoader(
        Dataset(data=[{"image": str(image_path)}], transform=transforms),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    return loader, transforms


def combine_word_labels(prediction):
    """Convert 32 independent SuPreM masks into one WORD-numbered label map."""

    # `prediction` has shape [32, D, H, W]. The combined output has one integer
    # label per voxel, with 0 reserved for background.
    combined = np.zeros(prediction.shape[1:], dtype=np.uint8)
    for word_label, (_, channels) in WORD_TO_SUPREM.items():
        # Most WORD classes map to one SuPreM channel. WORD's adrenal class maps
        # to two channels, so logical OR merges the left and right glands.
        class_mask = np.logical_or.reduce(
            [prediction[channel] > 0 for channel in channels]
        )

        # SuPreM channels are independent and can overlap. If that happens, a
        # class processed later in WORD_TO_SUPREM overwrites the earlier label.
        combined[class_mask] = word_label
    return combined


def save_prediction(prediction, reference, output_path):
    """Save a uint8 label map using the input CT's spatial metadata."""

    # Copying the header and affine keeps the segmentation aligned with the CT.
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(prediction, reference.affine, header)

    # Preserve the explicit scanner/world transforms when they are present.
    qform, qform_code = reference.get_qform(coded=True)
    sform, sform_code = reference.get_sform(coded=True)
    if qform is not None:
        output.set_qform(qform, int(qform_code))
    if sform is not None:
        output.set_sform(sform, int(sform_code))
    nib.save(output, str(output_path))


def validate_inputs(args, required_backbones):
    """Fail early for invalid paths and inference parameters."""
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image does not exist: {args.image}")
    if not 0.0 <= args.overlap < 1.0:
        raise ValueError("--overlap must be at least 0 and less than 1.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    # Only models with missing outputs need their checkpoints. A fully complete
    # case can therefore resume without loading or even checking model files.
    checkpoints = {
        backbone: args.checkpoint_dir / filename
        for backbone, filename in MODEL_CHECKPOINTS.items()
        if backbone in required_backbones
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {', '.join(missing)}")
    return checkpoints


def load_existing_prediction(path, reference, backbone):
    """Load and validate one completed WORD-numbered model prediction."""
    image = nib.load(str(path))
    if image.shape != reference.shape:
        raise ValueError(
            f"{backbone}: existing output shape {image.shape} does not match "
            f"input shape {reference.shape}: {path}"
        )
    if not np.allclose(image.affine, reference.affine, rtol=1e-5, atol=1e-5):
        raise ValueError(f"{backbone}: existing output affine does not match: {path}")

    prediction = np.asanyarray(image.dataobj)
    if not np.all(np.equal(prediction, np.round(prediction))):
        raise ValueError(f"{backbone}: existing output has non-integer labels: {path}")
    return np.round(prediction).astype(np.uint8)


def main():
    # STEP 1 — Identify completed and missing outputs before loading any model.
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image does not exist: {args.image}")
    case_name = output_name(args.image)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        backbone: args.output_dir / backbone / case_name
        for backbone in MODEL_CHECKPOINTS
    }
    agreement_path = args.output_dir / "agreement" / case_name

    if args.overwrite:
        missing_backbones = list(MODEL_CHECKPOINTS)
    else:
        missing_backbones = [
            backbone
            for backbone, output_path in output_paths.items()
            if not output_path.is_file()
        ]

    if not missing_backbones and agreement_path.is_file() and not args.overwrite:
        print(f"All model outputs and agreement map already exist; skipping {case_name}")
        return

    # STEP 2 — Load the untouched CT as the spatial reference for every output.
    reference = nib.load(str(args.image))
    if len(reference.shape) != 3:
        raise ValueError(f"Expected a 3D CT image, got shape {reference.shape}")

    # Reuse every valid existing prediction instead of rerunning its model.
    combined_predictions = {}
    if not args.overwrite:
        for backbone, output_path in output_paths.items():
            if output_path.is_file():
                combined_predictions[backbone] = load_existing_prediction(
                    output_path, reference, backbone
                )
                print(f"Reusing existing {backbone} prediction: {output_path}")

    # STEP 3 — Preprocess only when at least one model still needs inference.
    if missing_backbones:
        checkpoints = validate_inputs(args, missing_backbones)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable; pass --device cpu."
            )
        loader, transforms = make_loader(args.image, args.spacing)
        batch = next(iter(loader))
        image = batch["image"].to(device)
        print(f"Input: {args.image}")
        print(
            f"Original shape: {reference.shape}; "
            f"preprocessed shape: {tuple(image.shape)}"
        )

    # STEP 4 — Run U-Net, Swin UNETR, and SegResNet sequentially.
    for backbone in missing_backbones:
        checkpoint = checkpoints[backbone]
        print(f"\nRunning {backbone}...")
        args.backbone = backbone
        args.checkpoint = checkpoint
        model = load_model(args, device)

        with torch.no_grad():
            # Process overlapping 3D patches instead of placing the entire
            # volume through the network at once. Gaussian blending reduces
            # seams where neighboring patches overlap.
            logits = sliding_window_inference(
                image,
                roi_size=tuple(args.roi_size),
                sw_batch_size=args.sw_batch_size,
                predictor=model,
                overlap=args.overlap,
                mode="gaussian",
            )

            # Each of the 32 channels is an independent binary structure.
            # Sigmoid converts logits to probabilities, then the configured
            # threshold (0.5 by default) produces 0/1 masks.
            masks = torch.sigmoid(logits).ge(args.threshold).to(torch.uint8).cpu()

        # STEP 5 — Undo cropping, resampling, and orientation so the prediction
        # returns to exactly the same grid as the original input CT.
        prediction = invert_prediction(batch, transforms, masks)
        if prediction.shape[1:] != reference.shape:
            raise RuntimeError(
                f"{backbone}: restored prediction shape {prediction.shape[1:]} "
                f"does not match input shape {reference.shape}"
            )

        # STEP 6 — Convert the 32 SuPreM masks to one WORD-numbered label map
        # and save it under a directory named after the current backbone.
        combined = combine_word_labels(prediction)
        output_path = output_paths[backbone]
        output_path.parent.mkdir(exist_ok=True)
        save_prediction(combined, reference, output_path)
        combined_predictions[backbone] = combined
        print(f"Saved: {output_path}")

        # Release this model before loading the next architecture. The combined
        # NumPy label map is retained for the final agreement calculation.
        del model, logits, masks, prediction
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # STEP 7 — Compare the three WORD maps voxel by voxel:
    #   1 = all three labels match
    #   2 = exactly two labels match
    #   3 = all three labels differ
    ordered_predictions = [
        combined_predictions[backbone] for backbone in MODEL_CHECKPOINTS
    ]
    agreement = agreement_labels(*ordered_predictions)
    agreement_path.parent.mkdir(exist_ok=True)
    save_agreement(agreement, reference, agreement_path)
    print(f"\nSaved agreement map: {agreement_path}")


if __name__ == "__main__":
    main()
