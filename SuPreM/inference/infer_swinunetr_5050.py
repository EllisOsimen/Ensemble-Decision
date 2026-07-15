#!/usr/bin/env python3
"""Run the Tang et al. Swin UNETR BTCV model on one CT NIfTI image."""

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
sys.path.insert(0, str(PROJECT_DIR / "benchmark_backbones"))
from model.SwinUNETR import SwinUNETR  # noqa: E402


BTCV_LABELS = {
    0: "background",
    1: "spleen",
    2: "right_kidney",
    3: "left_kidney",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior_vena_cava",
    10: "portal_and_splenic_veins",
    11: "pancreas",
    12: "right_adrenal_gland",
    13: "left_adrenal_gland",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the 14-channel Tang et al. Swin UNETR checkpoint on one CT. "
            "The saved label map uses native BTCV labels 0-13."
        )
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confidence-output",
        type=Path,
        default=None,
        help=(
            "Optional 3-D map containing the softmax probability of the BTCV "
            "label assigned at each voxel."
        ),
    )
    parser.add_argument(
        "--confidence-storage",
        choices=("uint8", "float32"),
        default="uint8",
        help="Store confidence as scaled uint8 or full float32.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_DIR
        / "pretrained_weights"
        / "self_supervised_nv_swin_unetr_5050.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roi-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.5, 1.5, 1.5))
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output instead of treating it as complete.",
    )
    return parser.parse_args()


def make_loader(image_path, spacing):
    """Standardize the CT while retaining transforms needed for inversion."""
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


def load_model(args, device):
    """Build the 14-class BTCV architecture and load the complete checkpoint."""
    model = SwinUNETR(
        img_size=tuple(args.roi_size),
        in_channels=1,
        out_channels=len(BTCV_LABELS),
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def invert_prediction(batch, transforms, prediction, nearest_interp=True):
    """Return the predicted label map to the input CT's original spatial grid."""
    item = decollate_batch(batch)[0]
    source = item["image"]
    item["prediction"] = MetaTensor(
        prediction[0],
        meta=deepcopy(source.meta),
        applied_operations=deepcopy(source.applied_operations),
    )
    inverter = Compose(
        [
            Invertd(
                keys="prediction",
                transform=transforms,
                orig_keys="image",
                nearest_interp=nearest_interp,
                to_tensor=False,
            )
        ]
    )
    return np.asarray(inverter(item)["prediction"])


def save_prediction(prediction, reference, output_path):
    """Save uint8 BTCV labels while preserving the original NIfTI geometry."""
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    header["cal_min"] = 0
    header["cal_max"] = 13
    header["descrip"] = "Tang Swin UNETR BTCV labels 0-13"
    output = nib.Nifti1Image(prediction.astype(np.uint8), reference.affine, header)

    qform, qform_code = reference.get_qform(coded=True)
    sform, sform_code = reference.get_sform(coded=True)
    if qform is not None:
        output.set_qform(qform, int(qform_code))
    if sform is not None:
        output.set_sform(sform, int(sform_code))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, str(output_path))


def save_confidence(confidence, reference, output_path, storage):
    """Save assigned-label confidence in [0, 1], optionally quantized."""

    confidence = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
    header = reference.header.copy()
    header["cal_min"] = 0
    header["cal_max"] = 1
    header["descrip"] = "Swin assigned BTCV-label confidence [0,1]"
    if storage == "uint8":
        stored = np.rint(confidence * 255.0).astype(np.uint8)
        header.set_data_dtype(np.uint8)
        output = nib.Nifti1Image(stored, reference.affine, header)
        output.header.set_slope_inter(1.0 / 255.0, 0.0)
    else:
        header.set_data_dtype(np.float32)
        output = nib.Nifti1Image(confidence, reference.affine, header)

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
    requested_outputs = [args.output]
    if args.confidence_output is not None:
        requested_outputs.append(args.confidence_output)
    if all(path.exists() for path in requested_outputs) and not args.overwrite:
        print(f"All requested outputs already exist; skipping: {args.output}")
        return

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
        # The 14 outputs are mutually exclusive BTCV classes.
        probabilities = torch.softmax(logits, dim=1)
        assigned_confidence, prediction = torch.max(
            probabilities,
            dim=1,
            keepdim=True,
        )
        prediction = prediction.to(torch.uint8).cpu()
        assigned_confidence = assigned_confidence.to(torch.float32).cpu()

    restored = invert_prediction(batch, transforms, prediction, nearest_interp=True)
    restored = np.squeeze(restored, axis=0)
    if restored.shape != reference.shape:
        raise RuntimeError(
            f"Restored shape {restored.shape} does not match input {reference.shape}"
        )

    save_prediction(restored, reference, args.output)
    if args.confidence_output is not None:
        restored_confidence = invert_prediction(
            batch,
            transforms,
            assigned_confidence,
            nearest_interp=True,
        )
        restored_confidence = np.squeeze(restored_confidence, axis=0)
        if restored_confidence.shape != reference.shape:
            raise RuntimeError(
                "Restored confidence shape "
                f"{restored_confidence.shape} does not match input {reference.shape}"
            )
        save_confidence(
            restored_confidence,
            reference,
            args.confidence_output,
            args.confidence_storage,
        )
        print(f"Saved assigned-label confidence: {args.confidence_output}")

    print(f"Saved BTCV prediction: {args.output}")
    print(f"Labels present: {np.unique(restored).astype(int).tolist()}")


if __name__ == "__main__":
    main()
