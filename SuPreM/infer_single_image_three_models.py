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
MODEL_CHECKPOINTS = {
    "unet": "supervised_suprem_unet_2100.pth",
    "swinunetr": "supervised_suprem_swinunetr_2100.pth",
    "segresnet": "supervised_suprem_segresnet_2100.pth",
}


def parse_args():
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
    return parser.parse_args()


def output_name(image_path):
    if image_path.name.endswith(".nii.gz"):
        return image_path.name
    if image_path.suffix == ".nii":
        return f"{image_path.stem}.nii.gz"
    raise ValueError(f"Input image must end in .nii or .nii.gz: {image_path}")


def make_loader(image_path, spacing):
    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=tuple(spacing), mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
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
    combined = np.zeros(prediction.shape[1:], dtype=np.uint8)
    for word_label, (_, channels) in WORD_TO_SUPREM.items():
        class_mask = np.logical_or.reduce(
            [prediction[channel] > 0 for channel in channels]
        )
        combined[class_mask] = word_label
    return combined


def save_prediction(prediction, reference, output_path):
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(prediction, reference.affine, header)
    qform, qform_code = reference.get_qform(coded=True)
    sform, sform_code = reference.get_sform(coded=True)
    if qform is not None:
        output.set_qform(qform, int(qform_code))
    if sform is not None:
        output.set_sform(sform, int(sform_code))
    nib.save(output, str(output_path))


def validate_inputs(args):
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image does not exist: {args.image}")
    if not 0.0 <= args.overlap < 1.0:
        raise ValueError("--overlap must be at least 0 and less than 1.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    checkpoints = {
        backbone: args.checkpoint_dir / filename
        for backbone, filename in MODEL_CHECKPOINTS.items()
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {', '.join(missing)}")
    return checkpoints


def main():
    args = parse_args()
    checkpoints = validate_inputs(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu.")

    reference = nib.load(str(args.image))
    if len(reference.shape) != 3:
        raise ValueError(f"Expected a 3D CT image, got shape {reference.shape}")

    case_name = output_name(args.image)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loader, transforms = make_loader(args.image, args.spacing)
    batch = next(iter(loader))
    image = batch["image"].to(device)
    print(f"Input: {args.image}")
    print(f"Original shape: {reference.shape}; preprocessed shape: {tuple(image.shape)}")

    combined_predictions = []
    for backbone, checkpoint in checkpoints.items():
        print(f"\nRunning {backbone}...")
        args.backbone = backbone
        args.checkpoint = checkpoint
        model = load_model(args, device)

        with torch.no_grad():
            logits = sliding_window_inference(
                image,
                roi_size=tuple(args.roi_size),
                sw_batch_size=args.sw_batch_size,
                predictor=model,
                overlap=args.overlap,
                mode="gaussian",
            )
            masks = torch.sigmoid(logits).ge(args.threshold).to(torch.uint8).cpu()

        prediction = invert_prediction(batch, transforms, masks)
        if prediction.shape[1:] != reference.shape:
            raise RuntimeError(
                f"{backbone}: restored prediction shape {prediction.shape[1:]} "
                f"does not match input shape {reference.shape}"
            )

        combined = combine_word_labels(prediction)
        model_dir = args.output_dir / backbone
        model_dir.mkdir(exist_ok=True)
        output_path = model_dir / case_name
        save_prediction(combined, reference, output_path)
        combined_predictions.append(combined)
        print(f"Saved: {output_path}")

        del model, logits, masks, prediction
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    agreement = agreement_labels(*combined_predictions)
    agreement_dir = args.output_dir / "agreement"
    agreement_dir.mkdir(exist_ok=True)
    agreement_path = agreement_dir / case_name
    save_agreement(agreement, reference, agreement_path)
    print(f"\nSaved agreement map: {agreement_path}")


if __name__ == "__main__":
    main()
