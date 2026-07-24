#!/usr/bin/env python3
"""Compute expert-versus-method confusion matrices with four workers.

Rows represent one expert annotation and columns represent one fusion method.
The labels are:

    0 = background, 1 = pancreas, 2 = kidney, 3 = liver

The script streams compressed NIfTI files in bounded-memory chunks, so it can
summarize large CT label maps without loading both full volumes at once.
"""

import argparse
import gzip
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CASES_ROOT = REPOSITORY_ROOT / "testing_set"
EXCLUDED_CASES = {"UKCHLL082"}
NUMBER_OF_LABELS = 4
CHUNK_VOXELS = 4_000_000
WORKERS = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute a row-normalized confusion matrix between one expert "
            "annotation and one fusion method across the testing set."
        )
    )
    parser.add_argument(
        "--expert-annotation",
        required=True,
        help="Expert annotation filename used for matrix rows.",
    )
    parser.add_argument(
        "--method-name",
        required=True,
        help="Short method name used in logs, e.g. unweighted or random_forest.",
    )
    parser.add_argument(
        "--method-root",
        type=Path,
        required=True,
        help="Root directory containing method masks.",
    )
    parser.add_argument(
        "--method-layout",
        choices=("flat", "nested-agreement-mask"),
        default="flat",
        help=(
            "flat expects <method-root>/<case>.nii.gz; nested-agreement-mask "
            "expects <method-root>/<case>/agreement_mask.nii.gz."
        ),
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=CASES_ROOT,
        help="Directory containing testing-set case folders.",
    )
    parser.add_argument(
        "--exclude-case",
        action="append",
        default=sorted(EXCLUDED_CASES),
        help="Case ID to exclude. Can be passed more than once.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help="Number of worker processes.",
    )
    return parser.parse_args()


def method_mask_path(method_root: Path, method_layout: str, case_id: str) -> Path:
    if method_layout == "flat":
        return method_root / f"{case_id}.nii.gz"
    if method_layout == "nested-agreement-mask":
        return method_root / case_id / "agreement_mask.nii.gz"
    raise ValueError(f"Unsupported method layout: {method_layout}")


def scaled_labels(values: np.ndarray, proxy) -> np.ndarray:
    slope = 1.0 if proxy.slope is None else proxy.slope
    inter = 0.0 if proxy.inter is None else proxy.inter
    return np.rint(values * slope + inter).astype(np.uint8)


def process_case(task):
    """Return raw and row-normalized expert-versus-method counts."""

    case_dir, expert_annotation, method_name, method_root, method_layout = task
    expert_path = case_dir / expert_annotation
    method_path = method_mask_path(method_root, method_layout, case_dir.name)
    paths = [expert_path, method_path]

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{case_dir.name}: mask not found: {path}")

    images = [nib.load(str(path)) for path in paths]
    if images[0].shape != images[1].shape:
        raise RuntimeError(
            f"{case_dir.name}: shape mismatch: "
            f"{images[0].shape} versus {images[1].shape}"
        )
    if not np.allclose(images[0].affine, images[1].affine, rtol=1e-5, atol=1e-5):
        raise RuntimeError(f"{case_dir.name}: affine mismatch")

    proxies = [image.dataobj for image in images]
    dtypes = [np.dtype(proxy.dtype) for proxy in proxies]
    remaining = int(np.prod(images[0].shape))
    confusion = np.zeros((NUMBER_OF_LABELS, NUMBER_OF_LABELS), dtype=np.int64)

    with gzip.open(expert_path, "rb") as expert_file, gzip.open(
        method_path, "rb"
    ) as method_file:
        expert_file.seek(proxies[0].offset)
        method_file.seek(proxies[1].offset)

        while remaining:
            count = min(remaining, CHUNK_VOXELS)
            arrays = []

            for stream, dtype, proxy in zip((expert_file, method_file), dtypes, proxies):
                expected_bytes = count * dtype.itemsize
                raw = stream.read(expected_bytes)
                if len(raw) != expected_bytes:
                    raise RuntimeError(f"{case_dir.name}: truncated NIfTI data")

                stored_values = np.frombuffer(raw, dtype=dtype)
                arrays.append(scaled_labels(stored_values, proxy))

            expert, method = arrays
            if expert.max(initial=0) >= NUMBER_OF_LABELS:
                raise RuntimeError(f"{case_dir.name}: unexpected expert label")
            if method.max(initial=0) >= NUMBER_OF_LABELS:
                raise RuntimeError(f"{case_dir.name}: unexpected {method_name} label")

            encoded_pairs = expert * NUMBER_OF_LABELS + method
            confusion += np.bincount(
                encoded_pairs,
                minlength=NUMBER_OF_LABELS**2,
            ).reshape(NUMBER_OF_LABELS, NUMBER_OF_LABELS)
            remaining -= count

    row_totals = confusion.sum(axis=1, keepdims=True)
    if np.any(row_totals == 0):
        raise RuntimeError(f"{case_dir.name}: empty expert class")

    return case_dir.name, confusion, confusion / row_totals


def main():
    args = parse_args()
    excluded_cases = set(args.exclude_case)

    print(
        f"COMPARISON rows={args.expert_annotation} columns={args.method_name}",
        flush=True,
    )
    print(
        "EXCLUDED_CASES "
        + (", ".join(sorted(excluded_cases)) if excluded_cases else "none"),
        flush=True,
    )
    print(f"METHOD_ROOT {args.method_root}", flush=True)
    print(f"METHOD_LAYOUT {args.method_layout}", flush=True)

    if not args.cases_root.is_dir():
        raise NotADirectoryError(f"Testing-set directory not found: {args.cases_root}")
    if not args.method_root.is_dir():
        raise NotADirectoryError(f"Method directory not found: {args.method_root}")

    case_dirs = sorted(
        path for path in args.cases_root.iterdir()
        if path.is_dir() and path.name not in excluded_cases
    )
    if not case_dirs:
        raise RuntimeError(f"No testing cases found in {args.cases_root}")

    tasks = [
        (
            case_dir,
            args.expert_annotation,
            args.method_name,
            args.method_root,
            args.method_layout,
        )
        for case_dir in case_dirs
    ]

    results = []
    with Pool(processes=args.workers) as pool:
        for index, result in enumerate(pool.imap_unordered(process_case, tasks), start=1):
            results.append(result)
            print(f"{index}/{len(case_dirs)} {result[0]}", flush=True)

    pooled = sum(
        (result[1] for result in results),
        start=np.zeros((NUMBER_OF_LABELS, NUMBER_OF_LABELS), dtype=np.int64),
    )
    pooled_normalized = pooled / pooled.sum(axis=1, keepdims=True)
    normalized = np.stack([result[2] for result in results])
    mean_patient = normalized.mean(axis=0)
    sd_patient = normalized.std(axis=0, ddof=1)

    np.set_printoptions(precision=8, suppress=True)
    print(f"RESULT rows {args.expert_annotation}")
    print(f"RESULT columns {args.method_name}")
    print("RESULT label_order background pancreas kidney liver")
    print("RESULT pooled_counts")
    print(pooled)
    print("RESULT pooled_row_percent")
    print(pooled_normalized * 100)
    print("RESULT mean_patient_row_percent")
    print(mean_patient * 100)
    print("RESULT sd_patient_percentage_points")
    print(sd_patient * 100)
    print("RESULT row_sums")
    print((mean_patient * 100).sum(axis=1))


if __name__ == "__main__":
    main()
