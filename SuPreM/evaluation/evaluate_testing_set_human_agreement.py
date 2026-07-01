#!/usr/bin/env python3
"""Measure human-human agreement across testing_set annotations.

The testing_set label space is:

  0 = background
  1 = pancreas
  2 = kidney
  3 = liver

The script treats each annotation file as a human rater. It computes:

  - pairwise per-class Dice, NSD, and binary Cohen kappa
  - pairwise whole-label-map Cohen kappa and exact agreement
  - three-rater Fleiss kappa across all annotations
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
from surface_distance import metrics as surface_distance_metrics
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

TARGET_LABELS = {
    0: "background",
    1: "pancreas",
    2: "kidney",
    3: "liver",
}


def parse_args() -> argparse.Namespace:
    """Collect paths and metric options from the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute human inter-rater agreement for testing_set annotations "
            "annotation_1.nii.gz, annotation_2.nii.gz, and annotation_3.nii.gz."
        )
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=PROJECT_DIR.parent / "testing_set",
        help="Directory containing one subdirectory per case.",
    )
    parser.add_argument(
        "--annotations",
        nargs="+",
        default=["annotation_1.nii.gz", "annotation_2.nii.gz", "annotation_3.nii.gz"],
        help="Annotation filenames inside each case directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "testing_set_human_agreement",
        help="Directory where agreement CSV and JSON summaries will be written.",
    )
    parser.add_argument(
        "--case-name",
        default=None,
        help="Process only this case name instead of every case.",
    )
    parser.add_argument(
        "--exclude-case",
        action="append",
        default=[],
        help="Case name to exclude from the run. Can be passed multiple times.",
    )
    parser.add_argument(
        "--nsd-tolerance-mm",
        type=float,
        default=1.0,
        help="Surface tolerance in millimetres for normalized surface Dice.",
    )
    parser.add_argument(
        "--ignore-affine",
        action="store_true",
        help="Only check array shapes, not NIfTI affines.",
    )
    parser.add_argument(
        "--include-background-nsd",
        action="store_true",
        help="Also compute NSD for background. This is slow and usually not informative.",
    )
    parser.add_argument(
        "--skip-nsd",
        action="store_true",
        help="Skip all NSD calculations and write NaN in NSD columns.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write CSVs with stable column order for easier comparison between runs."""

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_integer_mask(path: Path, atol: float = 1e-3) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load a NIfTI label map and verify it uses the expected integer labels."""

    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)

    # Some tools save integer labels in floating-point images. Accept those only
    # if every value is very close to an integer, then round them back.
    if not np.issubdtype(data.dtype, np.integer):
        rounded = np.rint(data)
        if not np.all(np.isclose(data, rounded, rtol=0.0, atol=atol)):
            raise ValueError(f"{path} contains non-integer labels.")
        data = rounded
    data = data.astype(np.int16, copy=False)

    # Human annotations should already be in the testing-set 0-3 label space.
    unknown = sorted(set(np.unique(data).astype(int)) - set(TARGET_LABELS))
    if unknown:
        raise ValueError(f"{path} contains unexpected label(s): {unknown}")
    return image, data


def case_directories(root: Path) -> dict[str, Path]:
    """Return case folders keyed by case name, e.g. UKCHLL003 -> path."""

    if not root.is_dir():
        raise NotADirectoryError(f"Cases root does not exist: {root}")
    cases = {path.name: path for path in root.iterdir() if path.is_dir()}
    if not cases:
        raise FileNotFoundError(f"No case directories found in {root}")
    return cases


def validate_grid(case_name: str, reference_image, moving_image, moving_name: str, ignore_affine: bool) -> None:
    """Check that two annotations can be compared voxel-by-voxel."""

    # Shape equality means both arrays have the same voxel index range.
    if moving_image.shape != reference_image.shape:
        raise ValueError(
            f"{case_name}: {moving_name} shape {moving_image.shape} does not match "
            f"reference annotation shape {reference_image.shape}"
        )
    # Affine equality means those voxel indices point to the same physical CT
    # locations. If this fails, direct Dice/kappa/NSD can be misleading.
    if not ignore_affine and not np.allclose(
        moving_image.affine,
        reference_image.affine,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError(
            f"{case_name}: {moving_name} affine does not match the reference annotation. "
            "Rerun with --ignore-affine only if array-index comparison is intended."
        )


def binary_counts_from_confusion(confusion: np.ndarray, label_value: int):
    """Convert a multiclass confusion matrix into one-vs-rest counts.

    For a selected label, e.g. pancreas, this treats the problem as:
    pancreas vs not-pancreas. Those counts feed Dice and binary Cohen kappa.
    """

    tp = int(confusion[label_value, label_value])
    fp = int(confusion[label_value, :].sum() - tp)
    fn = int(confusion[:, label_value].sum() - tp)
    tn = int(confusion.sum() - tp - fp - fn)

    denominator = 2 * tp + fp + fn
    dsc = (2.0 * tp / denominator) if denominator else math.nan
    kappa = cohen_kappa_from_counts(np.array([[tn, fp], [fn, tp]], dtype=np.int64))
    return dsc, kappa, tp, tn, fp, fn


def binary_nsd(first, second, spacing, tolerance_mm):
    """Compute normalized surface Dice for one binary class mask pair."""

    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)

    # NSD needs a surface from both masks. Empty masks have no boundary, so the
    # metric is not defined and is recorded as NaN.
    if first.any() and second.any():
        distances = surface_distance_metrics.compute_surface_distances(
            second,
            first,
            spacing_mm=spacing,
        )
        nsd = float(
            surface_distance_metrics.compute_surface_dice_at_tolerance(
                distances,
                tolerance_mm,
            )
        )
    else:
        nsd = math.nan
    return nsd


def cohen_kappa_from_counts(confusion: np.ndarray) -> float:
    """Compute Cohen's kappa from a confusion matrix.

    Kappa measures observed agreement after subtracting agreement expected by
    chance from the row/column label frequencies.
    """

    total = int(confusion.sum())
    if total == 0:
        return math.nan
    observed = float(np.trace(confusion) / total)
    row_totals = confusion.sum(axis=1)
    col_totals = confusion.sum(axis=0)
    expected = float(np.dot(row_totals, col_totals) / (total * total))
    if math.isclose(1.0, expected):
        return 1.0 if math.isclose(observed, 1.0) else math.nan
    return (observed - expected) / (1.0 - expected)


def multiclass_pair_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Compute whole-label-map agreement for one pair of annotators."""

    confusion = pair_confusion(first, second)

    # Exact agreement is simply the fraction of voxels where labels match. It is
    # useful, but background-heavy volumes can make it look very high.
    exact = float(np.trace(confusion) / confusion.sum())

    # Foreground metrics focus on voxels where at least one annotator marked an
    # organ, reducing the dominance of easy background agreement.
    foreground_union = np.logical_or(first != 0, second != 0)
    if foreground_union.any():
        foreground_exact = float((first[foreground_union] == second[foreground_union]).mean())
        foreground_confusion = confusion[1:, 1:]
        foreground_kappa = cohen_kappa_from_counts(foreground_confusion)
    else:
        foreground_exact = math.nan
        foreground_kappa = math.nan

    return {
        "multiclass_kappa": cohen_kappa_from_counts(confusion),
        "exact_agreement": exact,
        "foreground_multiclass_kappa": foreground_kappa,
        "foreground_exact_agreement": foreground_exact,
    }


def pair_confusion(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Build a 4x4 confusion matrix between two integer label maps.

    Rows correspond to labels from the first annotator and columns to labels
    from the second annotator. np.bincount keeps this fast on large CT volumes.
    """

    encoded = (first.astype(np.int16, copy=False) * len(TARGET_LABELS)) + second
    return np.bincount(encoded.ravel(), minlength=len(TARGET_LABELS) ** 2).reshape(
        len(TARGET_LABELS),
        len(TARGET_LABELS),
    )


def fleiss_kappa(annotations: list[np.ndarray], labels: list[int], foreground_union_only: bool = False) -> float:
    """Compute multi-rater Fleiss kappa across all annotations.

    Cohen kappa is for two raters. Fleiss kappa is the corresponding chance-
    corrected agreement measure for 3+ raters.
    """

    if len(annotations) < 2:
        return math.nan

    # The testing set normally has exactly three annotations. This branch avoids
    # stacking a large 3 x X x Y x Z array and computes the observed pairwise
    # agreement directly, which is much lighter on memory.
    if len(annotations) == 3:
        first, second, third = annotations
        include = None

        # Foreground Fleiss kappa ignores voxels where all annotators marked
        # background, which is more informative for organ segmentation.
        if foreground_union_only:
            include = np.logical_or.reduce((first != 0, second != 0, third != 0))
            n_items = int(include.sum())
            if n_items == 0:
                return math.nan
        else:
            n_items = first.size

        agree_pairs = (
            (first == second).astype(np.uint8)
            + (first == third).astype(np.uint8)
            + (second == third).astype(np.uint8)
        )

        # With 3 raters there are 3 possible annotator pairs per voxel. The mean
        # number of agreeing pairs, divided by 3, is Fleiss' observed agreement.
        if include is not None:
            p_observed = float(agree_pairs[include].mean() / 3.0)
        else:
            p_observed = float(agree_pairs.mean() / 3.0)

        # Expected agreement is based on how often each label is used overall.
        category_totals = np.zeros(len(labels), dtype=np.float64)
        arrays = [array[include] if include is not None else array.ravel() for array in annotations]
        for array in arrays:
            category_totals += np.bincount(array, minlength=len(labels))[: len(labels)]

        proportions = category_totals / (n_items * len(annotations))
        p_expected = float(np.sum(proportions * proportions))
        if math.isclose(1.0, p_expected):
            return 1.0 if math.isclose(p_observed, 1.0) else math.nan
        return (p_observed - p_expected) / (1.0 - p_expected)

    # Generic implementation for any number of annotators. It is less memory
    # efficient, but keeps the function correct if more annotations are passed.
    stacked = np.stack(annotations, axis=0)
    if foreground_union_only:
        include = np.any(stacked != 0, axis=0)
        if not include.any():
            return math.nan
        stacked = stacked[:, include]
    else:
        stacked = stacked.reshape(len(annotations), -1)

    n_raters = stacked.shape[0]
    n_items = stacked.shape[1]
    if n_items == 0:
        return math.nan

    pair_agreements = np.zeros(n_items, dtype=np.float64)
    category_totals = np.zeros(len(labels), dtype=np.float64)
    for label_index, label in enumerate(labels):
        counts = (stacked == label).sum(axis=0)
        pair_agreements += counts * (counts - 1)
        category_totals[label_index] = counts.sum()

    p_observed = float(np.mean(pair_agreements / (n_raters * (n_raters - 1))))
    proportions = category_totals / (n_items * n_raters)
    p_expected = float(np.sum(proportions * proportions))
    if math.isclose(1.0, p_expected):
        return 1.0 if math.isclose(p_observed, 1.0) else math.nan
    return (p_observed - p_expected) / (1.0 - p_expected)


def summarize_pair_class_totals(totals) -> list[dict[str, object]]:
    """Collapse per-case pair/class metrics into one summary table."""

    rows = []
    for pair_name in sorted(totals):
        for label_value, class_name in TARGET_LABELS.items():
            item = totals[pair_name][class_name]
            denominator = 2 * item["tp"] + item["fp"] + item["fn"]

            # Rebuild the aggregate one-vs-rest confusion matrix so we can
            # report micro kappa over all voxels/cases for this pair/class.
            confusion = np.array(
                [[item["tn"], item["fp"]], [item["fn"], item["tp"]]],
                dtype=np.int64,
            )
            rows.append(
                {
                    "pair": pair_name,
                    "label": label_value,
                    "class": class_name,
                    "mean_case_dsc": np.mean(item["dsc"]) if item["dsc"] else math.nan,
                    "mean_case_nsd": np.mean(item["nsd"]) if item["nsd"] else math.nan,
                    "mean_case_binary_kappa": (
                        np.mean(item["binary_kappa"]) if item["binary_kappa"] else math.nan
                    ),
                    "micro_dsc": 2.0 * item["tp"] / denominator if denominator else math.nan,
                    "micro_binary_kappa": cohen_kappa_from_counts(confusion),
                    "tp": item["tp"],
                    "tn": item["tn"],
                    "fp": item["fp"],
                    "fn": item["fn"],
                    "cases_with_dsc": len(item["dsc"]),
                    "cases_with_nsd": len(item["nsd"]),
                    "cases_with_binary_kappa": len(item["binary_kappa"]),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    if len(args.annotations) < 2:
        raise ValueError("Pass at least two annotation filenames.")

    # Decide which cases are in this run before loading any large NIfTI arrays.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = case_directories(args.cases_root)
    case_names = sorted(case_dirs)
    if args.case_name is not None:
        if args.case_name not in case_dirs:
            raise FileNotFoundError(f"{args.case_name} not found in {args.cases_root}")
        case_names = [args.case_name]
    excluded_cases = set(args.exclude_case)
    case_names = [case_name for case_name in case_names if case_name not in excluded_cases]
    if not case_names:
        raise ValueError("No cases left to process after applying --case-name/--exclude-case.")

    # Detailed rows keep per-case information. The totals dictionaries collect
    # enough information to build mean-case and micro summaries at the end.
    pair_rows: list[dict[str, object]] = []
    pair_multiclass_rows: list[dict[str, object]] = []
    fleiss_rows: list[dict[str, object]] = []
    pair_totals = defaultdict(
        lambda: defaultdict(
            lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "dsc": [], "nsd": [], "binary_kappa": []}
        )
    )
    pair_multiclass_totals = defaultdict(lambda: defaultdict(list))
    fleiss_totals = defaultdict(list)

    # Save the exact run settings next to the outputs so old CSVs remain
    # interpretable later.
    with (args.output_dir / "run_config.txt").open("w") as handle:
        handle.write(f"cases_root={args.cases_root}\n")
        handle.write(f"annotations={','.join(args.annotations)}\n")
        handle.write(f"excluded_cases={','.join(sorted(excluded_cases))}\n")
        handle.write(f"nsd_tolerance_mm={args.nsd_tolerance_mm}\n")
        handle.write(f"ignore_affine={args.ignore_affine}\n")
        handle.write(f"include_background_nsd={args.include_background_nsd}\n")
        handle.write(f"skip_nsd={args.skip_nsd}\n")

    for case_name in tqdm(case_names, desc="Human agreement"):
        case_dir = case_dirs[case_name]
        loaded = []

        # Load all requested annotations for this case. Each annotation is one
        # rater's integer label map.
        for annotation_name in args.annotations:
            path = case_dir / annotation_name
            if not path.is_file():
                raise FileNotFoundError(f"Missing annotation file: {path}")
            loaded.append((annotation_name, *load_integer_mask(path)))

        # Use the first annotation as the reference grid. All other annotations
        # must match this grid unless --ignore-affine is explicitly used.
        reference_image = loaded[0][1]
        for annotation_name, image, _mask in loaded[1:]:
            validate_grid(case_name, reference_image, image, annotation_name, args.ignore_affine)

        spacing = tuple(float(value) for value in reference_image.header.get_zooms()[:3])
        masks_by_name = {annotation_name: mask for annotation_name, _image, mask in loaded}

        # Compare each annotator pair: 1 vs 2, 1 vs 3, and 2 vs 3.
        for first_name, second_name in combinations(args.annotations, 2):
            first = masks_by_name[first_name]
            second = masks_by_name[second_name]
            pair_name = f"{first_name[:-7]}_vs_{second_name[:-7]}"
            confusion = pair_confusion(first, second)

            # Whole-map pairwise metrics: multiclass kappa and exact agreement.
            multiclass = multiclass_pair_metrics(first, second)
            pair_multiclass_rows.append({"case": case_name, "pair": pair_name, **multiclass})
            for key, value in multiclass.items():
                if not math.isnan(value):
                    pair_multiclass_totals[pair_name][key].append(value)

            # Per-class metrics turn each label into a binary one-vs-rest mask.
            for label_value, class_name in TARGET_LABELS.items():
                dsc, kappa, tp, tn, fp, fn = binary_counts_from_confusion(confusion, label_value)

                # Background NSD is skipped by default because it is slow and
                # not usually meaningful. Foreground NSD can also be skipped for
                # fast kappa/Dice-only runs.
                if not args.skip_nsd and (label_value != 0 or args.include_background_nsd):
                    nsd = binary_nsd(
                        first == label_value,
                        second == label_value,
                        spacing,
                        args.nsd_tolerance_mm,
                    )
                else:
                    nsd = math.nan
                pair_rows.append(
                    {
                        "case": case_name,
                        "pair": pair_name,
                        "rater_a": first_name,
                        "rater_b": second_name,
                        "label": label_value,
                        "class": class_name,
                        "dsc": dsc,
                        "nsd": nsd,
                        "binary_kappa": kappa,
                        "tp": tp,
                        "tn": tn,
                        "fp": fp,
                        "fn": fn,
                    }
                )

                item = pair_totals[pair_name][class_name]
                item["tp"] += tp
                item["tn"] += tn
                item["fp"] += fp
                item["fn"] += fn
                if not math.isnan(dsc):
                    item["dsc"].append(dsc)
                if not math.isnan(nsd):
                    item["nsd"].append(nsd)
                if not math.isnan(kappa):
                    item["binary_kappa"].append(kappa)

        # Fleiss kappa evaluates all human annotations together, not pairwise.
        annotations = [masks_by_name[name] for name in args.annotations]
        case_fleiss = {
            "case": case_name,
            "fleiss_kappa": fleiss_kappa(annotations, sorted(TARGET_LABELS)),
            "foreground_fleiss_kappa": fleiss_kappa(
                annotations,
                sorted(TARGET_LABELS),
                foreground_union_only=True,
            ),
        }
        fleiss_rows.append(case_fleiss)
        for key in ["fleiss_kappa", "foreground_fleiss_kappa"]:
            if not math.isnan(case_fleiss[key]):
                fleiss_totals[key].append(case_fleiss[key])

    # Build the run-level summary tables after all cases have contributed rows.
    pair_summary_rows = summarize_pair_class_totals(pair_totals)
    pair_multiclass_summary_rows = []
    for pair_name in sorted(pair_multiclass_totals):
        row = {"pair": pair_name}
        for key in [
            "multiclass_kappa",
            "exact_agreement",
            "foreground_multiclass_kappa",
            "foreground_exact_agreement",
        ]:
            values = pair_multiclass_totals[pair_name][key]
            row[f"mean_case_{key}"] = np.mean(values) if values else math.nan
            row[f"cases_with_{key}"] = len(values)
        pair_multiclass_summary_rows.append(row)

    # Overall JSON is intentionally compact: it highlights the multi-rater
    # agreement numbers most useful as a human-human baseline.
    overall = {
        "cases_root": str(args.cases_root),
        "annotations": args.annotations,
        "excluded_cases": sorted(excluded_cases),
        "cases": len(case_names),
        "nsd_tolerance_mm": args.nsd_tolerance_mm,
        "mean_case_fleiss_kappa": (
            float(np.mean(fleiss_totals["fleiss_kappa"])) if fleiss_totals["fleiss_kappa"] else math.nan
        ),
        "mean_case_foreground_fleiss_kappa": (
            float(np.mean(fleiss_totals["foreground_fleiss_kappa"]))
            if fleiss_totals["foreground_fleiss_kappa"]
            else math.nan
        ),
        "cases_with_fleiss_kappa": len(fleiss_totals["fleiss_kappa"]),
        "cases_with_foreground_fleiss_kappa": len(fleiss_totals["foreground_fleiss_kappa"]),
    }

    write_csv(
        args.output_dir / "per_case_pair_per_class.csv",
        pair_rows,
        [
            "case",
            "pair",
            "rater_a",
            "rater_b",
            "label",
            "class",
            "dsc",
            "nsd",
            "binary_kappa",
            "tp",
            "tn",
            "fp",
            "fn",
        ],
    )
    write_csv(
        args.output_dir / "per_pair_per_class_summary.csv",
        pair_summary_rows,
        [
            "pair",
            "label",
            "class",
            "mean_case_dsc",
            "mean_case_nsd",
            "mean_case_binary_kappa",
            "micro_dsc",
            "micro_binary_kappa",
            "tp",
            "tn",
            "fp",
            "fn",
            "cases_with_dsc",
            "cases_with_nsd",
            "cases_with_binary_kappa",
        ],
    )
    write_csv(
        args.output_dir / "per_case_pair_multiclass.csv",
        pair_multiclass_rows,
        [
            "case",
            "pair",
            "multiclass_kappa",
            "exact_agreement",
            "foreground_multiclass_kappa",
            "foreground_exact_agreement",
        ],
    )
    write_csv(
        args.output_dir / "per_pair_multiclass_summary.csv",
        pair_multiclass_summary_rows,
        [
            "pair",
            "mean_case_multiclass_kappa",
            "cases_with_multiclass_kappa",
            "mean_case_exact_agreement",
            "cases_with_exact_agreement",
            "mean_case_foreground_multiclass_kappa",
            "cases_with_foreground_multiclass_kappa",
            "mean_case_foreground_exact_agreement",
            "cases_with_foreground_exact_agreement",
        ],
    )
    write_csv(
        args.output_dir / "per_case_fleiss_kappa.csv",
        fleiss_rows,
        ["case", "fleiss_kappa", "foreground_fleiss_kappa"],
    )
    with (args.output_dir / "overall_summary.json").open("w") as handle:
        json.dump(overall, handle, indent=2)

    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
