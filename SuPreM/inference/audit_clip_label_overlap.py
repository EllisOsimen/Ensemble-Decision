#!/usr/bin/env python3
"""Audit overlapping CLIP WORD predictions without writing segmentation masks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from infer_clip_universal_unet import (
    PROJECT_DIR,
    WORD_TO_UNIVERSAL,
    load_model,
    make_loader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun CLIP inference and measure voxels where multiple grouped "
            "WORD-label sigmoid scores exceed the assignment threshold."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_DIR
            / "pretrained_weights"
            / "supervised_clip_driven_universal_unet_2100.pth"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roi-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.5, 1.5, 1.5))
    parser.add_argument("--overlap", type=float, default=0.75)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def pair_counts_from_bitmasks(bitmasks: np.ndarray) -> Counter[tuple[int, int]]:
    """Count every WORD-label pair present in each multi-positive voxel."""

    pair_counts: Counter[tuple[int, int]] = Counter()
    if bitmasks.size == 0:
        return pair_counts

    patterns, counts = np.unique(bitmasks.astype(np.uint16), return_counts=True)
    for pattern, count in zip(patterns.tolist(), counts.tolist()):
        labels = [index + 1 for index in range(16) if pattern & (1 << index)]
        for first, second in combinations(labels, 2):
            pair_counts[(first, second)] += int(count)
    return pair_counts


def audit_case(
    image_path: Path,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], Counter[tuple[int, int]]]:
    loader, _ = make_loader(image_path, args.spacing)
    batch = next(iter(loader))
    image = batch["image"].to(device)

    with torch.no_grad():
        logits = sliding_window_inference(
            image,
            roi_size=tuple(args.roi_size),
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.overlap,
            mode="gaussian",
        )
        probabilities = logits.sigmoid_()

        spatial_shape = tuple(int(value) for value in probabilities.shape[2:])
        state_shape = (probabilities.shape[0], 1, *spatial_shape)
        positive_count = torch.zeros(state_shape, dtype=torch.uint8, device=device)
        positive_bits = torch.zeros(state_shape, dtype=torch.int32, device=device)
        overwrite_label = torch.zeros(state_shape, dtype=torch.uint8, device=device)
        maximum_label = torch.zeros(state_shape, dtype=torch.uint8, device=device)
        maximum_probability = torch.zeros(
            state_shape,
            dtype=probabilities.dtype,
            device=device,
        )

        for word_label, (_, channels) in WORD_TO_UNIVERSAL.items():
            class_probability = probabilities[:, list(channels)].amax(
                dim=1,
                keepdim=True,
            )
            class_positive = class_probability.ge(args.threshold)
            positive_count.add_(class_positive.to(torch.uint8))
            positive_bits.bitwise_or_(
                class_positive.to(torch.int32) << (word_label - 1)
            )

            # This reproduces the current later-WORD-label overwrite rule.
            overwrite_label = torch.where(
                class_positive,
                word_label,
                overwrite_label,
            )

            # This represents the proposed maximum-score assignment rule.
            better = class_probability > maximum_probability
            maximum_probability = torch.where(
                better,
                class_probability,
                maximum_probability,
            )
            maximum_label = torch.where(better, word_label, maximum_label)

        foreground = maximum_probability.ge(args.threshold)
        maximum_assignment = torch.where(
            foreground,
            maximum_label,
            torch.zeros_like(maximum_label),
        )
        multi_positive = positive_count.ge(2)
        changed = overwrite_label.ne(maximum_assignment)

        total_voxels = int(positive_count.numel())
        foreground_voxels = int(foreground.sum().item())
        multi_positive_voxels = int(multi_positive.sum().item())
        changed_voxels = int(changed.sum().item())
        positive_histogram = (
            torch.bincount(
                positive_count.reshape(-1).to(torch.int64),
                minlength=17,
            )
            .cpu()
            .tolist()
        )
        overlap_bitmasks = positive_bits[multi_positive].cpu().numpy()

    pair_counts = pair_counts_from_bitmasks(overlap_bitmasks)
    case_id = image_path.parent.name
    report = {
        "case_id": case_id,
        "image": str(image_path),
        "preprocessed_shape": list(spatial_shape),
        "total_preprocessed_voxels": total_voxels,
        "foreground_voxels": foreground_voxels,
        "multi_positive_voxels": multi_positive_voxels,
        "multi_positive_fraction_all": safe_fraction(
            multi_positive_voxels,
            total_voxels,
        ),
        "multi_positive_fraction_foreground": safe_fraction(
            multi_positive_voxels,
            foreground_voxels,
        ),
        "overwrite_vs_max_changed_voxels": changed_voxels,
        "overwrite_vs_max_changed_fraction_all": safe_fraction(
            changed_voxels,
            total_voxels,
        ),
        "overwrite_vs_max_changed_fraction_foreground": safe_fraction(
            changed_voxels,
            foreground_voxels,
        ),
        "positive_word_label_count_histogram": {
            str(index): int(count)
            for index, count in enumerate(positive_histogram)
            if count
        },
        "overlap_pair_counts": {
            f"{first}:{second}": count
            for (first, second), count in pair_counts.most_common()
        },
    }

    del (
        image,
        batch,
        probabilities,
        logits,
        positive_count,
        positive_bits,
        overwrite_label,
        maximum_label,
        maximum_probability,
        foreground,
        maximum_assignment,
        multi_positive,
        changed,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report, pair_counts


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if not 0.0 <= args.overlap < 1.0:
        raise ValueError("--overlap must be at least 0 and less than 1.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    missing_images = [path for path in args.image if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(f"Input CT does not exist: {missing_images[0]}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu.")

    model = load_model(args, device)
    case_reports = []
    aggregate_pairs: Counter[tuple[int, int]] = Counter()
    for index, image_path in enumerate(args.image, start=1):
        print(f"Auditing {index}/{len(args.image)}: {image_path}", flush=True)
        report, pair_counts = audit_case(image_path, model, args, device)
        case_reports.append(report)
        aggregate_pairs.update(pair_counts)
        print(
            "  multi-positive foreground: "
            f"{100.0 * float(report['multi_positive_fraction_foreground']):.6f}%"
        )
        print(
            "  overwrite-vs-max changed foreground: "
            f"{100.0 * float(report['overwrite_vs_max_changed_fraction_foreground']):.6f}%"
        )

    total_voxels = sum(int(report["total_preprocessed_voxels"]) for report in case_reports)
    foreground_voxels = sum(int(report["foreground_voxels"]) for report in case_reports)
    multi_positive_voxels = sum(
        int(report["multi_positive_voxels"]) for report in case_reports
    )
    changed_voxels = sum(
        int(report["overwrite_vs_max_changed_voxels"]) for report in case_reports
    )
    output = {
        "settings": {
            "checkpoint": str(args.checkpoint),
            "threshold": args.threshold,
            "roi_size": list(args.roi_size),
            "spacing": list(args.spacing),
            "sliding_window_overlap": args.overlap,
            "sw_batch_size": args.sw_batch_size,
            "counting_grid": "preprocessed cropped model-inference grid",
        },
        "aggregate": {
            "case_count": len(case_reports),
            "total_preprocessed_voxels": total_voxels,
            "foreground_voxels": foreground_voxels,
            "multi_positive_voxels": multi_positive_voxels,
            "multi_positive_fraction_all": safe_fraction(
                multi_positive_voxels,
                total_voxels,
            ),
            "multi_positive_fraction_foreground": safe_fraction(
                multi_positive_voxels,
                foreground_voxels,
            ),
            "overwrite_vs_max_changed_voxels": changed_voxels,
            "overwrite_vs_max_changed_fraction_all": safe_fraction(
                changed_voxels,
                total_voxels,
            ),
            "overwrite_vs_max_changed_fraction_foreground": safe_fraction(
                changed_voxels,
                foreground_voxels,
            ),
            "overlap_pair_counts": {
                f"{first}:{second}": count
                for (first, second), count in aggregate_pairs.most_common()
            },
        },
        "word_labels": {
            str(label): name
            for label, (name, _) in WORD_TO_UNIVERSAL.items()
        },
        "cases": case_reports,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Audit report: {args.output_json}")


if __name__ == "__main__":
    main()
