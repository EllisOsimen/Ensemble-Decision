#!/usr/bin/env python3
"""Run the CLIP-driven Universal U-Net checkpoint on one CT NIfTI image."""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset, MetaTensor, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "direct_inference"))
from model.Universal_model import Universal_model  # noqa: E402


WORD_TO_UNIVERSAL = {
    1: ("liver", (5,)),
    2: ("spleen", (0,)),
    3: ("left_kidney", (2,)),
    4: ("right_kidney", (1,)),
    5: ("stomach", (6,)),
    6: ("gallbladder", (3,)),
    7: ("esophagus", (4,)),
    8: ("pancreas", (10,)),
    9: ("duodenum", (13,)),
    10: ("colon", (17,)),
    11: ("intestine", (18,)),
    12: ("adrenal", (11, 12)),
    13: ("rectum", (19,)),
    14: ("bladder", (20,)),
    15: ("head_of_femur_left", (22,)),
    16: ("head_of_femur_right", (23,)),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run supervised_clip_driven_universal_unet_2100.pth on one CT and "
            "save one combined label map with WORD labels 0-16."
        )
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_DIR
        / "pretrained_weights"
        / "supervised_clip_driven_universal_unet_2100.pth",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roi-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.5, 1.5, 1.5))
    parser.add_argument("--overlap", type=float, default=0.75)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output instead of treating it as complete.",
    )
    return parser.parse_args()


def checkpoint_state(checkpoint):
    for key in ("net", "state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            return checkpoint[key]
    return checkpoint


def strip_module_prefix(state):
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(args, device):
    model = Universal_model(
        img_size=tuple(args.roi_size),
        in_channels=1,
        out_channels=32,
        backbone="unet",
        encoding="word_embedding",
    )
    raw = load_checkpoint(args.checkpoint)
    state = strip_module_prefix(checkpoint_state(raw))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


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


def invert_prediction(batch, transforms, masks):
    item = decollate_batch(batch)[0]
    source = item["image"]
    item["prediction"] = MetaTensor(
        masks[0],
        meta=deepcopy(source.meta),
        applied_operations=deepcopy(source.applied_operations),
    )
    inverter = Compose(
        [
            Invertd(
                keys="prediction",
                transform=transforms,
                orig_keys="image",
                nearest_interp=True,
                to_tensor=False,
            )
        ]
    )
    return np.asarray(inverter(item)["prediction"])


def combine_word_labels(masks):
    """Convert 32 independent binary masks to one WORD-numbered label map."""
    combined = np.zeros(masks.shape[1:], dtype=np.uint8)
    for word_label, (_, channels) in WORD_TO_UNIVERSAL.items():
        class_mask = np.logical_or.reduce([masks[channel] > 0 for channel in channels])
        combined[class_mask] = word_label
    return combined


def save_prediction(prediction, reference, output_path):
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    header["cal_min"] = 0
    header["cal_max"] = 16
    header["descrip"] = "CLIP Universal U-Net WORD labels 0-16"
    output = nib.Nifti1Image(prediction.astype(np.uint8), reference.affine, header)

    qform, qform_code = reference.get_qform(coded=True)
    sform, sform_code = reference.get_sform(coded=True)
    if qform is not None:
        output.set_qform(qform, int(qform_code))
    if sform is not None:
        output.set_sform(sform, int(sform_code))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, str(output_path))


def main():
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Input CT does not exist: {args.image}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    if args.output.exists() and not args.overwrite:
        print(f"Output already exists; skipping: {args.output}")
        return
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if not 0.0 <= args.overlap < 1.0:
        raise ValueError("--overlap must be at least 0 and less than 1.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu.")

    reference = nib.load(str(args.image))
    if len(reference.shape) != 3:
        raise ValueError(f"Expected a 3D CT image, got shape {reference.shape}")

    loader, transforms = make_loader(args.image, args.spacing)
    batch = next(iter(loader))
    image = batch["image"].to(device)
    print(f"Input: {args.image}")
    print(f"Original shape: {reference.shape}")
    print(f"Preprocessed shape: {tuple(image.shape)}")

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

    restored = invert_prediction(batch, transforms, masks)
    if restored.shape[1:] != reference.shape:
        raise RuntimeError(
            f"Restored shape {restored.shape[1:]} does not match input {reference.shape}"
        )

    combined = combine_word_labels(restored)
    save_prediction(combined, reference, args.output)
    present = np.unique(combined).astype(int).tolist()
    print(f"Saved prediction: {args.output}")
    print(f"Labels present: {present}")


if __name__ == "__main__":
    main()
