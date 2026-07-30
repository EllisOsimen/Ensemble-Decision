#!/usr/bin/env python3
"""Measure agreement between saved prediction mask directories.

This is the model-output analogue of evaluate_testing_set_human_agreement.py.
Each --model argument is treated as one rater, but predictions can first be
remapped from their native label spaces into the testing_set target labels:

  0 = background
  1 = pancreas
  2 = kidney
  3 = liver
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from tqdm import tqdm

from evaluate_testing_set_human_agreement import (
    TARGET_LABELS,
    binary_counts_from_confusion,
    binary_nsd,
    cohen_kappa_from_counts,
    fleiss_kappa,
    pair_confusion,
    summarize_pair_class_totals,
    validate_grid,
    write_csv,
)
from evaluate_testing_set_prediction_dirs import (
    load_integer_mask,
    parse_model_spec,
    prediction_files,
    remap_prediction_to_target,
)


# PROJECT_DIR is the SuPreM folder. Relative prediction/output paths passed on
# the command line are resolved against this directory by the helper functions.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    """Parse the model directories and metric options.

    A model spec looks like:

      name=path/to/predictions:label_space

    The label_space part matters because different inference scripts save
    different integer IDs for the same anatomy. For example, BTCV uses liver=6
    while the saved SuPreM/WORD-style maps use liver=1. Everything is remapped
    later into the shared 0-3 target space before metrics are computed.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Compare saved prediction mask directories with each other after "
            "normalizing them into the testing_set 0-3 label space."
        )
    )
    parser.add_argument(
        "--model",
        dest="models",
        type=parse_model_spec,
        action="append",
        required=True,
        help=(
            "Prediction directory. Format: NAME=DIR or NAME=DIR:LABEL_SPACE. "
            "LABEL_SPACE can be target, btcv, or suprem. Repeat once per model."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "prediction_mask_agreement",
        help="Directory where agreement CSV and JSON summaries will be written.",
    )
    parser.add_argument(
        "--case-name",
        default=None,
        help="Process only this case name instead of every common prediction.",
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
    parser.add_argument(
        "--skip-fleiss",
        action="store_true",
        help="Skip all-model Fleiss kappa. Pairwise Dice/kappa/exact agreement are still written.",
    )
    return parser.parse_args()


def common_case_names(files_by_model, case_name: str | None, excluded_cases: set[str]) -> list[str]:
    """Return case IDs that exist in every prediction directory.

    Agreement only makes sense when every model has a mask for the same case.
    Taking the intersection avoids accidentally comparing a partial set. If
    --case-name is passed, this also verifies that the requested case is present
    for all models.
    """

    common = set.intersection(*(set(files) for files in files_by_model.values()))
    common -= excluded_cases
    if case_name is not None:
        if case_name not in common:
            raise FileNotFoundError(f"{case_name} is not present in every prediction directory.")
        common = {case_name}
    if not common:
        raise ValueError("No common prediction cases left to process.")
    return sorted(common)


def multiclass_metrics_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    """Compute whole-label-map agreement from an existing 4x4 confusion matrix.

    The confusion matrix rows are labels from the first model and columns are
    labels from the second model. Reusing this matrix is much faster than
    repeatedly scanning the full 3D mask, which is important for CURVAS volumes.
    """

    # Exact agreement is the fraction of all voxels, including background, where
    # both masks assign the same label.
    exact = float(np.trace(confusion) / confusion.sum())

    # Foreground exact agreement ignores the easy background-background matches.
    # The denominator is the foreground union: any voxel where at least one model
    # predicted pancreas, kidney, or liver.
    foreground_union_total = int(confusion.sum() - confusion[0, 0])

    # Foreground kappa is computed only on voxels where both models predicted a
    # foreground class. This mirrors the older human-agreement script's behavior.
    foreground_confusion = confusion[1:, 1:]
    foreground_total = int(foreground_confusion.sum())
    if foreground_union_total:
        foreground_exact = float((np.trace(confusion) - confusion[0, 0]) / foreground_union_total)
    else:
        foreground_exact = math.nan
    if foreground_total:
        foreground_kappa = cohen_kappa_from_counts(foreground_confusion)
    else:
        foreground_kappa = math.nan

    return {
        "multiclass_kappa": cohen_kappa_from_counts(confusion),
        "exact_agreement": exact,
        "foreground_multiclass_kappa": foreground_kappa,
        "foreground_exact_agreement": foreground_exact,
    }


def main() -> None:
    args = parse_args()
    if len(args.models) < 2:
        raise ValueError("Pass at least two --model entries.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build one filename lookup per model:
    #   model name -> {case name -> /path/to/UKCHLLxxx.nii.gz}
    # The case name is the filename without .nii.gz.
    files_by_model = {model.name: prediction_files(model.directory) for model in args.models}
    model_by_name = {model.name: model for model in args.models}
    excluded_cases = set(args.exclude_case)
    case_names = common_case_names(files_by_model, args.case_name, excluded_cases)
    model_names = [model.name for model in args.models]

    # These row lists become the detailed CSVs. The totals dictionaries collect
    # enough counts and per-case metric values to build summary CSVs at the end.
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

    # Save the exact inputs/options used for the run, so the CSVs are still
    # interpretable later even if the default sbatch arguments change.
    with (args.output_dir / "run_config.txt").open("w") as handle:
        handle.write("models=\n")
        for model in args.models:
            handle.write(f"  {model.name}={model.directory}:{model.label_space}\n")
        handle.write(f"excluded_cases={','.join(sorted(excluded_cases))}\n")
        handle.write(f"nsd_tolerance_mm={args.nsd_tolerance_mm}\n")
        handle.write(f"ignore_affine={args.ignore_affine}\n")
        handle.write(f"include_background_nsd={args.include_background_nsd}\n")
        handle.write(f"skip_nsd={args.skip_nsd}\n")
        handle.write(f"skip_fleiss={args.skip_fleiss}\n")

    for case_name in tqdm(case_names, desc="Prediction agreement"):
        loaded = {}
        reference_image = None

        # Load each model's mask for this case and normalize it into:
        #   0 background, 1 pancreas, 2 kidney, 3 liver
        # After this point, all models can be compared voxel-by-voxel because
        # the label IDs now mean the same thing.
        for model_name in model_names:
            model = model_by_name[model_name]
            image, mask = load_integer_mask(files_by_model[model_name][case_name])
            mask = remap_prediction_to_target(mask, model, case_name)
            if reference_image is None:
                reference_image = image
            else:
                # The arrays must have the same shape, and normally the same
                # affine, otherwise voxel index i,j,k may refer to different
                # physical CT locations in the two files.
                validate_grid(case_name, reference_image, image, model_name, args.ignore_affine)
            loaded[model_name] = mask

        # Physical voxel spacing is needed only for NSD. It comes from the NIfTI
        # header of the reference mask.
        spacing = tuple(float(value) for value in reference_image.header.get_zooms()[:3])

        # Compare every pair of models for this case:
        #   swin vs clip, swin vs segresnet, clip vs segresnet, etc.
        for first_name, second_name in combinations(model_names, 2):
            first = loaded[first_name]
            second = loaded[second_name]
            pair_name = f"{first_name}_vs_{second_name}"

            # One 4x4 confusion matrix powers both the whole-map metrics and
            # the per-class one-vs-rest metrics below.
            confusion = pair_confusion(first, second)

            # Whole-label-map metrics look at the full multiclass segmentation
            # rather than one organ at a time.
            multiclass = multiclass_metrics_from_confusion(confusion)
            pair_multiclass_rows.append({"case": case_name, "pair": pair_name, **multiclass})
            for key, value in multiclass.items():
                if not math.isnan(value):
                    pair_multiclass_totals[pair_name][key].append(value)

            # Per-class metrics treat each organ as binary:
            #   current class vs every other label
            # For example, liver Dice uses liver as positive and everything
            # else, including background and other organs, as negative.
            for label_value, class_name in TARGET_LABELS.items():
                dsc, kappa, tp, tn, fp, fn = binary_counts_from_confusion(confusion, label_value)
                if not args.skip_nsd and (label_value != 0 or args.include_background_nsd):
                    nsd = binary_nsd(
                        first == label_value,
                        second == label_value,
                        spacing,
                        args.nsd_tolerance_mm,
                    )
                else:
                    nsd = math.nan

                # Detailed per-case, per-pair, per-class row.
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

                # Accumulate counts for micro-averaged summaries, and keep the
                # per-case metric values for mean-case summaries.
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

        # Fleiss kappa is an all-rater agreement statistic. It is useful if you
        # want one number across 3+ models, but it is expensive on these large
        # volumes, so the CURVAS sbatch skips it by default.
        if len(model_names) >= 3 and not args.skip_fleiss:
            masks = [loaded[name] for name in model_names]
            case_fleiss = {
                "case": case_name,
                "fleiss_kappa": fleiss_kappa(masks, sorted(TARGET_LABELS)),
                "foreground_fleiss_kappa": fleiss_kappa(
                    masks,
                    sorted(TARGET_LABELS),
                    foreground_union_only=True,
                ),
            }
            fleiss_rows.append(case_fleiss)
            for key in ["fleiss_kappa", "foreground_fleiss_kappa"]:
                if not math.isnan(case_fleiss[key]):
                    fleiss_totals[key].append(case_fleiss[key])

    # Summaries are created after all cases have contributed their rows.
    # pair_summary_rows includes per-class mean-case and micro metrics.
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

    # The compact JSON is mostly useful for quick logging/checking. The CSVs
    # contain the richer pairwise results.
    overall = {
        "models": [f"{model.name}={model.directory}:{model.label_space}" for model in args.models],
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

    # CSV naming:
    # - per_case_* files are detailed rows for every case.
    # - summary files aggregate across all cases in the run.
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
