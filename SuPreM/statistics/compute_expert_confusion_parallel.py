#!/usr/bin/env python3
"""Compute pairwise expert confusion matrices with four workers.

Rows represent ``ROW_ANNOTATION`` and columns represent
``COLUMN_ANNOTATION``. Each worker streams one case in bounded-memory chunks;
the parent process prints pooled and equal-patient-weighted confusion-matrix
summaries across all cases.

The label indices used in both the rows and columns are:
    0 = background, 1 = pancreas, 2 = kidney, 3 = liver

The script produces two main versions of the confusion matrix:
    pooled_row_percent:
        Adds voxel counts from all patients before normalizing each row.
        Patients with more voxels for a class therefore have more influence.
    mean_patient_row_percent:
        Normalizes each patient's matrix first and then averages the matrices.
        Every patient therefore contributes equally to the final result.
"""

from multiprocessing import Pool
from pathlib import Path
import gzip

import nibabel as nib
import numpy as np


# This file is SuPreM/statistics/<script>.py, so parents[2] is the repository
# root. Building the path this way lets the script run from any directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPOSITORY_ROOT / "testing_set"
ROW_ANNOTATION = "annotation_2.nii.gz"
COLUMN_ANNOTATION = "annotation_3.nii.gz"


def process_case(case_dir: Path):
    """Process one patient and return raw and row-normalized confusion counts.

    A multiprocessing worker calls this function for one patient directory.
    The returned tuple contains:
        1. the patient directory name, for progress reporting;
        2. a 4 x 4 matrix of raw voxel counts;
        3. that matrix normalized so every row sums to one.
    """

    # The first annotation is represented by matrix rows, while the second
    # annotation supplies the comparison labels in the columns.
    paths = [case_dir / ROW_ANNOTATION, case_dir / COLUMN_ANNOTATION]
    images = [nib.load(str(path)) for path in paths]

    # A voxel-by-voxel comparison is meaningful only when both annotations
    # have the same dimensions and describe the same physical image grid.
    if images[0].shape != images[1].shape:
        raise RuntimeError(f"{case_dir.name}: shape mismatch")
    if not np.allclose(images[0].affine, images[1].affine, rtol=1e-5, atol=1e-5):
        raise RuntimeError(f"{case_dir.name}: affine mismatch")

    # Nibabel proxies contain information about the on-disk array without
    # loading the entire 3-D annotation into memory. This matters when four
    # patients are being processed at the same time.
    proxies = [image.dataobj for image in images]
    dtypes = [np.dtype(proxy.dtype) for proxy in proxies] # how each file stores voxel values
    remaining = int(np.prod(images[0].shape)) # total number of voxels to process
    confusion = np.zeros((4, 4), dtype=np.int64)

    # Read the compressed NIfTI files directly. At most four million voxels
    # from each annotation are held in memory in any iteration.
    with gzip.open(paths[0], "rb") as first_file, gzip.open(paths[1], "rb") as second_file:
        # Skip the NIfTI headers and move to the beginning of the image data.
        first_file.seek(proxies[0].offset)
        second_file.seek(proxies[1].offset)
        while remaining:
            count = min(remaining, 4_000_000)
            arrays = []
            for stream, dtype, proxy in zip((first_file, second_file), dtypes, proxies):
                raw = stream.read(count * dtype.itemsize)
                if len(raw) != count * dtype.itemsize:
                    raise RuntimeError(f"{case_dir.name}: truncated NIfTI data")
                values = np.frombuffer(raw, dtype=dtype)

                # Apply the NIfTI scale and intercept, then round to the
                # integer segmentation labels 0, 1, 2, and 3.
                arrays.append(np.rint(values * proxy.slope + proxy.inter).astype(np.uint8))
            first, second = arrays
            if first.max(initial=0) > 3 or second.max(initial=0) > 3:
                raise RuntimeError(f"{case_dir.name}: unexpected label")

            # Encode every label pair as (row label * 4 + column label). Bincount
            # then counts all 16 possible pairs at once. After reshaping:
            # confusion[i, j] = number of voxels labelled i by the row
            # annotation and j by the column annotation.
            confusion += np.bincount(first * 4 + second, minlength=16).reshape(4, 4)
            remaining -= count

    # Dividing by the row totals answers: "Of all voxels that the row annotator
    # called this class, what percentage did the column annotator assign to
    # each class?"
    row_totals = confusion.sum(axis=1, keepdims=True)
    if np.any(row_totals == 0):
        raise RuntimeError(f"{case_dir.name}: empty row-annotation class")
    return case_dir.name, confusion, confusion / row_totals


if __name__ == "__main__":
    print(
        f"COMPARISON rows={ROW_ANNOTATION} columns={COLUMN_ANNOTATION}",
        flush=True,
    )

    # Treat every directory in testing_set as one patient/case.
    case_dirs = sorted(path for path in ROOT.iterdir() if path.is_dir())
    results = []

    # Four worker processes analyse different patients concurrently.
    # imap_unordered reports a result as soon as any worker finishes, so the
    # progress messages may not follow the alphabetical patient order.
    #result includes the patient directory name, raw confusion matrix, and normalized confusion matrix.
    with Pool(processes=4) as pool:
        for index, result in enumerate(pool.imap_unordered(process_case, case_dirs), start=1):
            results.append(result)
            print(f"{index}/{len(case_dirs)} {result[0]}", flush=True)

    # Pooled result: add raw voxel counts over all patients, then normalize.
    # A patient with more voxels belonging to a class has more influence on
    # that class's row.
    pooled = sum((result[1] for result in results), start=np.zeros((4, 4), dtype=np.int64))

    # Mean-patient result: stack the already normalized patient matrices and
    # average them. This gives every patient equal weight, regardless of scan
    # size or the number of voxels belonging to an organ.
    normalized = np.stack([result[2] for result in results])
    mean_patient = normalized.mean(axis=0)

    # Sample standard deviation (ddof=1) shows how much the row percentages
    # vary between patients. It is reported in percentage points below.
    sd_patient = normalized.std(axis=0, ddof=1)
    pooled_normalized = pooled / pooled.sum(axis=1, keepdims=True)

    # Display percentages rather than fractions. row_sums is a quick check
    # that every row of the mean-patient matrix totals approximately 100%.
    np.set_printoptions(precision=8, suppress=True)
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
