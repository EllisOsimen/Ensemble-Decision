#!/usr/bin/env python3
"""Run the Tang et al. Swin UNETR BTCV model on one CT NIfTI image.

The model checkpoint predicts native BTCV labels 0-13.

This script saves a CURVAS-style organ-only label map:

    0 = background
    1 = pancreas
    2 = kidney       # BTCV right_kidney + left_kidney merged
    3 = liver

All other BTCV predictions are discarded as background.
"""

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
sys.path.insert(0, str(SCRIPT_DIR / "benchmark_backbones"))

from model.SwinUNETR import SwinUNETR  # noqa: E402


# Native output labels of the Tang et al. BTCV Swin UNETR checkpoint.
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


# Desired CURVAS output labels.
CURVAS_LABELS = {
    0: "background",
    1: "pancreas",
    2: "kidney",
    3: "liver",
}


# Mapping from BTCV argmax labels to CURVAS labels.
BTCV_TO_CURVAS = {
    11: 1,  # pancreas
    2: 2,   # right kidney -> kidney
    3: 2,   # left kidney -> kidney
    6: 3,   # liver
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the 14-channel Tang et al. Swin UNETR BTCV checkpoint on one CT, "
            "then remap predictions to CURVAS labels: "
            "0 background, 1 pancreas, 2 kidney, 3 liver."
        )
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input CT image, e.g. ct.nii.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output CURVAS-style prediction, e.g. pred_curvas.nii.gz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SCRIPT_DIR
        / "pretrained_weights"
        / "self_supervised_nv_swin_unetr_5050.pt",
        help="Path to Tang et al. Swin UNETR BTCV checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use: cuda or cpu.",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        default=(96, 96, 96),
        help="Sliding-window ROI size.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=(1.5, 1.5, 1.5),
        help="Preprocessing spacing used before inference.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Sliding-window overlap.",
    )
    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=1,
        help="Sliding-window batch size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output instead of skipping.",
    )
    parser.add_argument(
        "--mode",
        choices=["all_then_remap", "selected_only"],
        default="all_then_remap",
        help=(
            "all_then_remap: argmax over all BTCV classes, then keep only pancreas/kidney/liver. "
            "selected_only: argmax only over background, pancreas, kidney, liver logits. "
            "Default is all_then_remap."
        ),
    )

    return parser.parse_args()


def make_loader(image_path, spacing):
    """Standardize the CT while retaining transforms needed for inversion."""
    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"), # Ensures a channel first [D, H, W] to [C, D, H, W] shape
            Orientationd(keys=["image"], axcodes="RAS"), # Makes sure the orientation is consistent
            Spacingd(keys=["image"], pixdim=tuple(spacing), mode="bilinear"), # standarises voxel spacing
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ), # clips intensities to a range and scales to [0,1]
            CropForegroundd(keys=["image"], source_key="image"), # crops the foreground region
        ]
    )

    loader = DataLoader(
        Dataset(data=[{"image": str(image_path)}], transform=transforms),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    # This function returns both the loader and the transforms so that we can apply the inverse transforms later to map the prediction back to the original image space.
    # Important to note that without it prediction could be in a different space than the original image.

    return loader, transforms


def load_model(args, device):
    """Build the 14-class BTCV architecture and load the checkpoint."""
    model = SwinUNETR(
        img_size=tuple(args.roi_size), # Image patch size to process at a time, 96x96x96 in this case (DEFAULT)
        in_channels=1, # 1 channel for CT images
        out_channels=len(BTCV_LABELS), # Number of output classes
        feature_size=48, # Base feature size
        drop_rate=0.0, # Dropout rate
        attn_drop_rate=0.0, # Attention dropout rate
        dropout_path_rate=0.0, # Stochastic depth rate
        use_checkpoint=False, # Whether to use gradient checkpointing
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]

    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def btcv_prediction_to_curvas(prediction):
    """Convert BTCV label map to CURVAS label map.

    Input:
        prediction: torch.Tensor with BTCV labels, shape [B, 1, D, H, W]

    Output:
        curvas: torch.Tensor with CURVAS labels, shape [B, 1, D, H, W]
    """
    curvas = torch.zeros_like(prediction, dtype=torch.uint8) # create empty tensor of same shape as prediction to hold CURVAS labels

    for btcv_label, curvas_label in BTCV_TO_CURVAS.items():
        curvas[prediction == btcv_label] = curvas_label # map each BTCV label to the corresponding CURVAS label

    return curvas


def logits_to_curvas_selected_only(logits):
    """Infer only background, pancreas, kidney, and liver from BTCV logits.
    This is before any argmax is applied, so we can select only the relevant channels for CURVAS.
    This compares only these effective channels:

        CURVAS 0 = BTCV background logit
        CURVAS 1 = BTCV pancreas logit
        CURVAS 2 = max(BTCV right kidney, BTCV left kidney)
        CURVAS 3 = BTCV liver logit

    Output:
        torch.Tensor of CURVAS labels, shape [B, 1, D, H, W]
    """
    background_logit = logits[:, 0] # background logit is the first channel of the logits tensor
    pancreas_logit = logits[:, 11] # pancreas logit is the 12th channel of the logits tensor
    kidney_logit = torch.maximum(logits[:, 2], logits[:, 3]) # kidney logit is the maximum of the 3rd and 4th channels (right and left kidney)
    liver_logit = logits[:, 6] # liver logit is the 7th channel of the logits tensor

    selected_logits = torch.stack(
        [
            background_logit,
            pancreas_logit,
            kidney_logit,
            liver_logit,
        ],
        dim=1,
    )

    curvas = torch.argmax(selected_logits, dim=1, keepdim=True) # this is then used to get the voxel label
    return curvas.to(torch.uint8)


def invert_prediction(batch, transforms, prediction):
    """Return the predicted label map to the input CT's original spatial grid."""
    item = decollate_batch(batch)[0] # extracts the first item from the batch, which is a dictionary containing the image and its metadata
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
                nearest_interp=True,
                to_tensor=False,
            )
        ]
    )

    return np.asarray(inverter(item)["prediction"])


def save_prediction(prediction, reference, output_path):
    """Save uint8 CURVAS labels while preserving original NIfTI geometry."""
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    header["cal_min"] = 0
    header["cal_max"] = 3
    header["descrip"] = "CURVAS labels: 0 background, 1 pancreas, 2 kidney, 3 liver"

    output = nib.Nifti1Image(
        prediction.astype(np.uint8),
        reference.affine,
        header,
    )

    qform, qform_code = reference.get_qform(coded=True)
    sform, sform_code = reference.get_sform(coded=True)

    if qform is not None:
        output.set_qform(qform, int(qform_code))
    if sform is not None:
        output.set_sform(sform, int(sform_code))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, str(output_path))


def print_label_counts(label_map):
    labels, counts = np.unique(label_map, return_counts=True)

    print("\nCURVAS labels present:")
    print(f"{'Label':<10}{'Name':<15}{'Voxel Count'}")
    print("-" * 40)

    for label, count in zip(labels.astype(int), counts):
        name = CURVAS_LABELS.get(label, "unknown")
        print(f"{label:<10}{name:<15}{count}")


def main():
    args = parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Input CT does not exist: {args.image}")

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    if args.output.exists() and not args.overwrite:
        print(f"Output already exists; skipping: {args.output}")
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
    print(f"Output: {args.output}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mode: {args.mode}")
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
        # The logits are the raw output of the model before applying argmax to get discrete labels.

        #This selected mode represents how the CURVAS labels are derived from the BTCV logits. The "all_then_remap" mode first computes the argmax over all 14 BTCV classes and then remaps the relevant classes to CURVAS labels. The "selected_only" mode directly computes the argmax over only the relevant BTCV classes that correspond to CURVAS labels.
        if args.mode == "all_then_remap":
            # First get the normal BTCV prediction over all 14 classes.
            btcv_prediction = torch.argmax(logits, dim=1, keepdim=True).to(torch.uint8)

            # Then keep only pancreas, kidneys, and liver in CURVAS label space.
            curvas_prediction = btcv_prediction_to_curvas(btcv_prediction)

            btcv_labels = torch.unique(btcv_prediction).detach().cpu().numpy().astype(int)
            print(f"BTCV labels predicted before remapping: {btcv_labels.tolist()}")

        elif args.mode == "selected_only":
            # Directly choose between background, pancreas, kidney, and liver only.
            curvas_prediction = logits_to_curvas_selected_only(logits)

        else:
            raise ValueError(f"Unsupported mode: {args.mode}")

        curvas_prediction = curvas_prediction.cpu()

    #This inversion is needed in order to map the predicted CURVAS labels back to the original CT image space, which may have different orientation, spacing, and cropping compared to the preprocessed input used for inference. The invert_prediction function uses the stored transforms to reverse these preprocessing steps.
    restored = invert_prediction(batch, transforms, curvas_prediction)
    restored = np.squeeze(restored, axis=0) # gets rid of the channel dimension, resulting in a 3D array with shape [D, H, W]

    # The inversion may return near-integer floats, so round before saving.
    restored = np.rint(restored).astype(np.uint8)

    if restored.shape != reference.shape:
        raise RuntimeError(
            f"Restored shape {restored.shape} does not match input {reference.shape}"
        )

    save_prediction(restored, reference, args.output)

    print(f"\nSaved CURVAS prediction: {args.output}")
    print_label_counts(restored)


if __name__ == "__main__":
    main()