#!/usr/bin/env python3
"""Compute one expert versus unweighted-consensus confusion matrices.

Rows represent ``EXPERT_ANNOTATION`` and columns represent the unweighted
consensus labels. The labels are:

    0 = background, 1 = pancreas, 2 = kidney, 3 = liver

Four worker processes analyse separate testing cases. Each worker reads the
compressed NIfTI masks in bounded-memory chunks and returns both raw counts
and a row-normalized matrix for its case. The parent process then prints:

    pooled_row_percent:
        All voxel counts are combined before row normalization, so cases with
        more voxels belonging to a class have greater influence on that row.
    mean_patient_row_percent:
        Each case is row-normalized before averaging, so all cases contribute
        equally to the final matrix.
"""

import gzip
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CASES_ROOT = REPOSITORY_ROOT / "testing_set"
UNWEIGHTED_ROOT = (
    REPOSITORY_ROOT
    / "SuPreM/results/legacy/CURVAS_INFERENCE/agreement_masks_target"
)
EXPERT_ANNOTATION = "annotation_3.nii.gz"

NUMBER_OF_LABELS = 4
CHUNK_VOXELS = 4_000_000
WORKERS = 4


def process_case(case_dir: Path):
    """Return raw and row-normalized expert-versus-unweighted counts."""

    # The first path supplies the matrix rows and the second supplies columns.
    expert_path = case_dir / EXPERT_ANNOTATION
    unweighted_path = UNWEIGHTED_ROOT / f"{case_dir.name}.nii.gz"
    paths = [expert_path, unweighted_path]

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{case_dir.name}: mask not found: {path}")

    images = [nib.load(str(path)) for path in paths]

    # Both masks must describe the same voxel grid for a direct comparison.
    if images[0].shape != images[1].shape:
        raise RuntimeError(
            f"{case_dir.name}: shape mismatch: "
            f"{images[0].shape} versus {images[1].shape}"
        )
    if not np.allclose(images[0].affine, images[1].affine, rtol=1e-5, atol=1e-5):
        raise RuntimeError(f"{case_dir.name}: affine mismatch")

    # Proxies expose the on-disk data format without loading both full volumes.
    proxies = [image.dataobj for image in images]
    dtypes = [np.dtype(proxy.dtype) for proxy in proxies]
    remaining = int(np.prod(images[0].shape))
    confusion = np.zeros((NUMBER_OF_LABELS, NUMBER_OF_LABELS), dtype=np.int64)

    # Read matching chunks from the two compressed NIfTI files. Seeking to each
    # proxy's offset skips its NIfTI header and reaches the voxel data.
    with gzip.open(expert_path, "rb") as expert_file, gzip.open(
        unweighted_path, "rb"
    ) as unweighted_file:
        expert_file.seek(proxies[0].offset)
        unweighted_file.seek(proxies[1].offset)

        while remaining:
            count = min(remaining, CHUNK_VOXELS)
            arrays = []

            for stream, dtype, proxy in zip(
                (expert_file, unweighted_file), dtypes, proxies
            ):
                expected_bytes = count * dtype.itemsize
                raw = stream.read(expected_bytes)
                if len(raw) != expected_bytes:
                    raise RuntimeError(f"{case_dir.name}: truncated NIfTI data")

                stored_values = np.frombuffer(raw, dtype=dtype)
                labels = np.rint(
                    stored_values * proxy.slope + proxy.inter
                ).astype(np.uint8)
                arrays.append(labels)

            expert, unweighted = arrays
            if expert.max(initial=0) >= NUMBER_OF_LABELS:
                raise RuntimeError(f"{case_dir.name}: unexpected expert label")
            if unweighted.max(initial=0) >= NUMBER_OF_LABELS:
                raise RuntimeError(
                    f"{case_dir.name}: unexpected unweighted-consensus label"
                )

            # Encode each pair as (expert_label * 4 + method_label). After
            # bincount and reshape, confusion[i, j] is the number of voxels
            # labelled i by the selected expert and j by the unweighted method.
            encoded_pairs = expert * NUMBER_OF_LABELS + unweighted
            confusion += np.bincount(
                encoded_pairs,
                minlength=NUMBER_OF_LABELS**2,
            ).reshape(NUMBER_OF_LABELS, NUMBER_OF_LABELS)
            remaining -= count

    # Each normalized row answers: of the voxels the selected expert called
    # this class, what proportion did the unweighted method assign to each
    # output class?
    row_totals = confusion.sum(axis=1, keepdims=True)
    if np.any(row_totals == 0):
        raise RuntimeError(f"{case_dir.name}: empty expert class")

    return case_dir.name, confusion, confusion / row_totals


if __name__ == "__main__":
    print(
        f"COMPARISON rows={EXPERT_ANNOTATION} columns=unweighted_consensus",
        flush=True,
    )

    if not CASES_ROOT.is_dir():
        raise NotADirectoryError(f"Testing-set directory not found: {CASES_ROOT}")
    if not UNWEIGHTED_ROOT.is_dir():
        raise NotADirectoryError(
            f"Unweighted prediction directory not found: {UNWEIGHTED_ROOT}"
        )

    case_dirs = sorted(path for path in CASES_ROOT.iterdir() if path.is_dir())
    if not case_dirs:
        raise RuntimeError(f"No testing cases found in {CASES_ROOT}")

    results = []
    with Pool(processes=WORKERS) as pool:
        for index, result in enumerate(
            pool.imap_unordered(process_case, case_dirs),
            start=1,
        ):
            results.append(result)
            print(f"{index}/{len(case_dirs)} {result[0]}", flush=True)

    # Add raw matrices before normalizing for the voxel-weighted pooled result.
    pooled = sum(
        (result[1] for result in results),
        start=np.zeros((NUMBER_OF_LABELS, NUMBER_OF_LABELS), dtype=np.int64),
    )
    pooled_normalized = pooled / pooled.sum(axis=1, keepdims=True)

    # Average the normalized case matrices for the equal-patient-weight result.
    normalized = np.stack([result[2] for result in results])
    mean_patient = normalized.mean(axis=0)
    sd_patient = normalized.std(axis=0, ddof=1)

    np.set_printoptions(precision=8, suppress=True)
    print(f"RESULT rows {EXPERT_ANNOTATION}")
    print("RESULT columns unweighted_consensus")
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
