#!/usr/bin/env python3
"""Run SuPreM inference and evaluate it on the WORD training set."""

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset, MetaTensor, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
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
from surface_distance import metrics as surface_distance_metrics
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DIRECT_INFERENCE_DIR = SCRIPT_DIR / "direct_inference"
sys.path.insert(0, str(DIRECT_INFERENCE_DIR))
from model.Universal_model import Universal_model  # noqa: E402


# STEP 1 — Define how WORD classes correspond to SuPreM output channels.
#
# WORD uses one integer-valued label map, whereas SuPreM predicts one mask per
# structure. These IDs cannot be compared directly because their class orders
# differ. The channel numbers below are zero-based Python indices.
# WORD label -> (name, one or more zero-based SuPreM output channels).
# SuPreM channels are independent binary masks. WORD combines the two adrenal
# glands into one class, hence the two-channel mapping for label 12.
WORD_TO_SUPREM = {
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
    """Collect paths, model settings, inference settings, and output options."""

    parser = argparse.ArgumentParser(
        description="Evaluate a downloaded SuPreM checkpoint on WORD imagesTr/labelsTr."
    )
    parser.add_argument(
        "--word-root",
        type=Path,
        default=SCRIPT_DIR.parent / "WORD" / "WORD-V0.1.0",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--backbone",
        choices=("unet", "swinunetr", "segresnet"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roi-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.5, 1.5, 1.5))
    parser.add_argument("--overlap", type=float, default=0.75)
    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=4,
        help="Number of sliding-window patches evaluated together on the GPU.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--nsd-tolerance-mm",
        type=float,
        default=1.0,
        help="Surface tolerance in millimetres for normalized surface Dice.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N cases (useful for a smoke test).",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save a WORD-numbered combined prediction for every case.",
    )
    return parser.parse_args()


def checkpoint_state(checkpoint):
    """Extract the tensor dictionary from common SuPreM checkpoint formats."""

    for key in ("net", "state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            return checkpoint[key]
    return checkpoint


def strip_module_prefix(state):
    """Remove DistributedDataParallel's 'module.' prefix from checkpoint keys."""

    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def load_model(args, device):
    # STEP 2 — Construct the requested architecture and load its checkpoint.
    #
    # The --backbone argument must match the checkpoint file. All downloaded
    # SuPreM models used here produce 32 foreground-structure logits.
    if args.backbone == "segresnet":
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=1,
            out_channels=32,
            dropout_prob=0.0,
        )
    else:
        model = Universal_model(
            img_size=tuple(args.roi_size),
            in_channels=1,
            out_channels=32,
            backbone=args.backbone,
            encoding="word_embedding",
        )

    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    started = time.perf_counter()
    raw = torch.load(args.checkpoint, map_location="cpu")
    state = strip_module_prefix(checkpoint_state(raw))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        # Older SuPreM releases saved a DistributedDataParallel state under
        # "net" and the direct-inference script loaded it by insertion order.
        target = model.state_dict()
        if len(target) != len(state):
            raise RuntimeError(
                f"Checkpoint/model mismatch ({len(state)} versus {len(target)} tensors)."
            ) from error
        if any(a.shape != b.shape for a, b in zip(target.values(), state.values())):
            raise RuntimeError("Checkpoint tensors do not match this model.") from error
        model.load_state_dict(dict(zip(target.keys(), state.values())), strict=True)

    model = model.to(device).eval()
    print(f"Model loaded in {time.perf_counter() - started:.1f} seconds", flush=True)
    return model


def make_loader(args):
    # STEP 3 — Pair every imagesTr CT with its identically named labelsTr file.
    image_dir = args.word_root / "imagesTr"
    label_dir = args.word_root / "labelsTr"
    image_paths = sorted(image_dir.glob("*.nii.gz"))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No .nii.gz images found in {image_dir}")

    data = []
    for image_path in image_paths:
        label_path = label_dir / image_path.name
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing gold label: {label_path}")
        data.append(
            {"image": str(image_path), "label_path": str(label_path), "case": image_path.name}
        )

    # STEP 4 — Apply the preprocessing used by SuPreM inference.
    #
    # The image is reoriented, resampled, intensity-normalized and cropped.
    # Only the CT is transformed here; the gold label remains on its original
    # grid and the prediction is transformed back to that grid after inference.
    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            # Add the single CT modality as an explicit channel dimension.
            # EnsureChannelFirstd replaces the removed MONAI 0.9 AddChanneld.
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=tuple(args.spacing), mode="bilinear"),
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
        Dataset(data=data, transform=transforms),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return loader, transforms


def invert_prediction(batch, transforms, prediction):
    # STEP 7 — Undo orientation, spacing and cropping transformations.
    #
    # This returns the 32 predicted masks to the original WORD NIfTI grid so
    # voxel-level comparison with labelsTr is spatially valid.
    item = decollate_batch(batch)[0]
    source_image = item["image"]

    # Model outputs are ordinary tensors and do not automatically retain the
    # invertible-transform history recorded on the input MetaTensor. Attach a
    # copy of that history so modern MONAI can undo crop, spacing and orientation.
    item["prediction"] = MetaTensor(
        prediction[0],
        meta=deepcopy(source_image.meta),
        applied_operations=deepcopy(source_image.applied_operations),
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


def binary_metrics(prediction, gold, spacing, tolerance_mm):
    # STEP 9 — Compare one predicted class mask with one WORD gold class mask.
    #
    # TP, FP and FN are voxel counts. DSC follows directly from these counts.
    # NSD is computed in physical millimetres using the NIfTI voxel spacing.
    prediction = np.asarray(prediction, dtype=bool)
    gold = np.asarray(gold, dtype=bool)
    tp = int(np.logical_and(prediction, gold).sum())
    fp = int(np.logical_and(prediction, ~gold).sum())
    fn = int(np.logical_and(~prediction, gold).sum())
    denominator = 2 * tp + fp + fn
    dsc = (2.0 * tp / denominator) if denominator else math.nan

    if prediction.any() and gold.any():
        distances = surface_distance_metrics.compute_surface_distances(
            gold, prediction, spacing_mm=spacing
        )
        nsd = float(
            surface_distance_metrics.compute_surface_dice_at_tolerance(
                distances, tolerance_mm
            )
        )
    else:
        # The DeepMind implementation has undefined surface distances when
        # either mask is empty. NaN keeps these cases out of macro NSD.
        nsd = math.nan
    return dsc, nsd, tp, fp, fn


def write_csv(path, rows, fieldnames):
    """Write rows with a stable column order so outputs are easy to compare."""

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args):
    """Run inference on WORD cases, compare to labelsTr, and write reports."""

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; pass --device cpu.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "predictions"
    if args.save_predictions:
        prediction_dir.mkdir(exist_ok=True)

    model = load_model(args, device)
    loader, transforms = make_loader(args)
    case_rows = []
    totals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "dsc": [], "nsd": []})

    # Process one complete 3D WORD scan at a time.
    for batch in tqdm(loader, desc=f"WORD / {args.backbone}"):
        case_started = time.perf_counter()
        image = batch["image"].to(device)
        print(
            f"{batch['case'][0]}: preprocessed shape={tuple(image.shape)}",
            flush=True,
        )
        with torch.no_grad():
            # STEP 5 — Run full-volume inference using overlapping 3D patches.
            #
            # Sliding-window inference limits GPU memory use. Gaussian blending
            # combines overlapping patch predictions into 32 full-volume logits.
            logits = sliding_window_inference(
                image,
                roi_size=tuple(args.roi_size),
                sw_batch_size=args.sw_batch_size,
                predictor=model,
                overlap=args.overlap,
                mode="gaussian",
            )

            # STEP 6 — Convert each structure's logits into a binary voxel mask.
            #
            # Sigmoid gives an independent probability for every structure at
            # every voxel. Thresholding at 0.5 by default produces values 0/1.
            # This is not an argmax: multiple channels may be positive at a voxel.
            masks = torch.sigmoid(logits).ge(args.threshold).to(torch.uint8).cpu()
        print(
            f"{batch['case'][0]}: inference completed in "
            f"{time.perf_counter() - case_started:.1f} seconds",
            flush=True,
        )

        prediction = invert_prediction(batch, transforms, masks)

        # STEP 8 — Load the untouched WORD gold label and its physical spacing.
        case_name = batch["case"][0]
        label_path = Path(batch["label_path"][0])
        gold_image = nib.load(str(label_path))
        gold = np.asanyarray(gold_image.dataobj)
        spacing = tuple(float(value) for value in gold_image.header.get_zooms()[:3])
        if prediction.shape[1:] != gold.shape:
            raise RuntimeError(
                f"{case_name}: prediction shape {prediction.shape[1:]} != gold shape {gold.shape}"
            )

        combined = np.zeros(gold.shape, dtype=np.uint8)
        for word_label, (class_name, channels) in WORD_TO_SUPREM.items():
            # Select the SuPreM mask corresponding to this WORD structure.
            # For WORD's adrenal class, logical OR merges the separate right
            # and left SuPreM adrenal channels before evaluation.
            pred_mask = np.logical_or.reduce([prediction[channel] > 0 for channel in channels])
            gold_mask = gold == word_label
            dsc, nsd, tp, fp, fn = binary_metrics(
                pred_mask, gold_mask, spacing, args.nsd_tolerance_mm
            )
            case_rows.append(
                {
                    "case": case_name,
                    "word_label": word_label,
                    "class": class_name,
                    "dsc": dsc,
                    "nsd": nsd,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )
            totals[class_name]["tp"] += tp
            totals[class_name]["fp"] += fp
            totals[class_name]["fn"] += fn
            if not math.isnan(dsc):
                totals[class_name]["dsc"].append(dsc)
            if not math.isnan(nsd):
                totals[class_name]["nsd"].append(nsd)
            combined[pred_mask] = word_label

        if args.save_predictions:
            # Optional inspection output. Metrics above use the independent
            # masks, because one integer label map cannot preserve overlaps.
            nib.save(
                nib.Nifti1Image(combined, gold_image.affine, gold_image.header),
                str(prediction_dir / case_name),
            )
        print(
            f"{case_name}: inference and metrics completed in "
            f"{time.perf_counter() - case_started:.1f} seconds",
            flush=True,
        )

    # STEP 10 — Aggregate case results into per-class and overall summaries.
    summary_rows = []
    global_tp = global_fp = global_fn = 0
    all_case_dsc, all_case_nsd = [], []
    for _, (class_name, _) in WORD_TO_SUPREM.items():
        item = totals[class_name]
        denominator = 2 * item["tp"] + item["fp"] + item["fn"]
        summary_rows.append(
            {
                "class": class_name,
                "mean_case_dsc": np.mean(item["dsc"]) if item["dsc"] else math.nan,
                "mean_case_nsd": np.mean(item["nsd"]) if item["nsd"] else math.nan,
                "micro_dsc": 2.0 * item["tp"] / denominator if denominator else math.nan,
                "tp": item["tp"],
                "fp": item["fp"],
                "fn": item["fn"],
                "cases_with_dsc": len(item["dsc"]),
                "cases_with_nsd": len(item["nsd"]),
            }
        )
        global_tp += item["tp"]
        global_fp += item["fp"]
        global_fn += item["fn"]
        all_case_dsc.extend(item["dsc"])
        all_case_nsd.extend(item["nsd"])

    global_denominator = 2 * global_tp + global_fp + global_fn
    overall = {
        "backbone": args.backbone,
        "checkpoint": str(args.checkpoint),
        "cases": len(loader.dataset),
        "nsd_tolerance_mm": args.nsd_tolerance_mm,
        "macro_case_dsc": float(np.mean(all_case_dsc)),
        "macro_case_nsd": float(np.mean(all_case_nsd)),
        "micro_dsc": 2.0 * global_tp / global_denominator,
        "tp": global_tp,
        "fp": global_fp,
        "fn": global_fn,
    }

    # STEP 11 — Write detailed and aggregate reports to the output directory.
    write_csv(
        args.output_dir / "per_case_per_class.csv",
        case_rows,
        ["case", "word_label", "class", "dsc", "nsd", "tp", "fp", "fn"],
    )
    write_csv(
        args.output_dir / "per_class_summary.csv",
        summary_rows,
        [
            "class",
            "mean_case_dsc",
            "mean_case_nsd",
            "micro_dsc",
            "tp",
            "fp",
            "fn",
            "cases_with_dsc",
            "cases_with_nsd",
        ],
    )
    with (args.output_dir / "overall_summary.json").open("w") as handle:
        json.dump(overall, handle, indent=2)
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    evaluate(parse_args())
