#!/usr/bin/env python3
"""Summarize base-model human agreement CSVs after excluding invalid cases.

This script does not rerun model inference or reload NIfTI masks. It reads the
per-case CSV files already produced for the base models, drops the requested
case IDs, and recomputes the report-style agreement table.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = (
    REPOSITORY_ROOT
    / "SuPreM"
    / "results"
    / "base_model_human_annotator_evaluation"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "SuPreM"
    / "results"
    / "base_model_human_annotator_evaluation_exclude_UKCHLL082_from_csv"
)

BASE_MODELS = [
    {"name": "CLIP Universal U-Net", "result_dir": INPUT_ROOT / "clip_unet", "token": "clip_unet"},
    {"name": "SuPreM SegResNet", "result_dir": INPUT_ROOT / "segresnet", "token": "segresnet"},
    {"name": "Swin UNETR", "result_dir": INPUT_ROOT / "swin5050", "token": "swin5050"},
]

ORGAN_LABELS = {
    "pancreas": "1",
    "kidney": "2",
    "liver": "3",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute base-model agreement summary metrics from existing "
            "per-case CSV files after excluding invalid cases."
        )
    )
    parser.add_argument(
        "--exclude-case",
        action="append",
        default=["UKCHLL082"],
        help=(
            "Case ID to exclude. Can be passed more than once. Defaults to "
            "UKCHLL082."
        ),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=INPUT_ROOT,
        help="Directory containing one base-model evaluation folder per model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the recomputed summary files are written.",
    )
    return parser.parse_args()


def read_csv(path: Path, excluded_cases: set[str]) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row.get("case") not in excluded_cases]


def numeric(values: list[str]) -> list[float]:
    return [float(value) for value in values if value not in ("", "nan", "NaN")]


def mean_and_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot summarize an empty value list.")
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def group_pair_means(
    rows: list[dict[str, str]],
    metric_column: str,
    model_token: str,
) -> list[dict[str, object]]:
    values_by_pair = defaultdict(list)
    for row in rows:
        pair = row["pair"]
        if model_token not in pair:
            continue
        value = row.get(metric_column, "")
        if value in ("", "nan", "NaN"):
            continue
        values_by_pair[pair].append(float(value))

    if not values_by_pair:
        raise ValueError(f"No rows found for model token: {model_token}")

    summaries = []
    for pair, values in sorted(values_by_pair.items()):
        pair_mean, pair_sd = mean_and_sd(values)
        summaries.append(
            {
                "pair": pair,
                "n_cases": len(values),
                "mean": pair_mean,
                "case_sd": pair_sd,
            }
        )
    return summaries


def summarize_pair_metric(
    rows: list[dict[str, str]],
    metric_column: str,
    model_token: str,
) -> tuple[float, float, list[dict[str, object]]]:
    pair_summaries = group_pair_means(rows, metric_column, model_token)
    pair_means = [float(row["mean"]) for row in pair_summaries]
    overall_mean, pair_sd = mean_and_sd(pair_means)
    return overall_mean, pair_sd, pair_summaries


def summarize_organ_metric(
    rows: list[dict[str, str]],
    label: str,
    metric_column: str,
    model_token: str,
) -> tuple[float, float, list[dict[str, object]]]:
    organ_rows = [row for row in rows if row["label"] == label]
    return summarize_pair_metric(organ_rows, metric_column, model_token)


def format_value(value: float) -> str:
    return f"{value:.3f}"


def format_mean_pm_sd(value: float, sd: float) -> str:
    return f"{value:.3f} $\\pm$ {sd:.3f}"


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_table(rows: list[dict[str, object]], excluded_cases: set[str]) -> str:
    body = []
    for row in rows:
        body.append(
            " & ".join(
                [
                    str(row["model"]),
                    format_value(float(row["foreground_fleiss_kappa_mean"])),
                    format_mean_pm_sd(
                        float(row["pancreas_kappa_mean"]),
                        float(row["pancreas_kappa_pair_sd"]),
                    ),
                    format_mean_pm_sd(
                        float(row["kidney_kappa_mean"]),
                        float(row["kidney_kappa_pair_sd"]),
                    ),
                    format_mean_pm_sd(
                        float(row["liver_kappa_mean"]),
                        float(row["liver_kappa_pair_sd"]),
                    ),
                    format_mean_pm_sd(
                        float(row["multiclass_cohen_kappa_mean"]),
                        float(row["multiclass_cohen_kappa_pair_sd"]),
                    ),
                ]
            )
            + r" \\"
        )

    excluded = ", ".join(sorted(excluded_cases))
    return "\n".join(
        [
            r"\begin{table}[h]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"\textbf{Base model}",
            r"& \textbf{Foreground Fleiss' $\kappa$}",
            r"& \textbf{Pancreas $\kappa$}",
            r"& \textbf{Kidney $\kappa$}",
            r"& \textbf{Liver $\kappa$}",
            r"& \textbf{Multiclass Cohen's $\kappa$} \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Base-model agreement metrics after excluding "
                + excluded
                + r". Pairwise metrics are reported as mean $\pm$ standard "
                + r"deviation across the three model-expert comparisons.}"
            ),
            r"\label{tab:base_model_kappa_comparison_excluding_invalid_case}",
            r"\end{table}",
            "",
        ]
    )


def main():
    args = parse_args()
    excluded_cases = set(args.exclude_case)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [
        {
            **model,
            "result_dir": args.input_root / model["result_dir"].name,
        }
        for model in BASE_MODELS
    ]

    table_rows = []
    detail_rows = []

    for model in models:
        result_dir = model["result_dir"]
        model_name = model["name"]
        token = model["token"]

        fleiss_rows = read_csv(result_dir / "per_case_fleiss_kappa.csv", excluded_cases)
        multiclass_rows = read_csv(
            result_dir / "per_case_pair_multiclass.csv", excluded_cases
        )
        per_class_rows = read_csv(
            result_dir / "per_case_pair_per_class.csv", excluded_cases
        )

        foreground_fleiss_values = numeric(
            [row["foreground_fleiss_kappa"] for row in fleiss_rows]
        )
        fleiss_values = numeric([row["fleiss_kappa"] for row in fleiss_rows])
        foreground_fleiss_mean, foreground_fleiss_sd = mean_and_sd(
            foreground_fleiss_values
        )
        fleiss_mean, fleiss_sd = mean_and_sd(fleiss_values)

        multiclass_mean, multiclass_pair_sd, multiclass_pairs = (
            summarize_pair_metric(multiclass_rows, "multiclass_kappa", token)
        )
        organ_summaries = {}
        for organ, label in ORGAN_LABELS.items():
            organ_summaries[organ] = summarize_organ_metric(
                per_class_rows,
                label,
                "binary_kappa",
                token,
            )

        table_rows.append(
            {
                "model": model_name,
                "model_token": token,
                "cases": len(fleiss_rows),
                "excluded_cases": ";".join(sorted(excluded_cases)),
                "fleiss_kappa_mean": fleiss_mean,
                "fleiss_kappa_case_sd": fleiss_sd,
                "foreground_fleiss_kappa_mean": foreground_fleiss_mean,
                "foreground_fleiss_kappa_case_sd": foreground_fleiss_sd,
                "pancreas_kappa_mean": organ_summaries["pancreas"][0],
                "pancreas_kappa_pair_sd": organ_summaries["pancreas"][1],
                "kidney_kappa_mean": organ_summaries["kidney"][0],
                "kidney_kappa_pair_sd": organ_summaries["kidney"][1],
                "liver_kappa_mean": organ_summaries["liver"][0],
                "liver_kappa_pair_sd": organ_summaries["liver"][1],
                "multiclass_cohen_kappa_mean": multiclass_mean,
                "multiclass_cohen_kappa_pair_sd": multiclass_pair_sd,
            }
        )

        for metric_name, summaries in [
            ("multiclass_cohen_kappa", multiclass_pairs),
            ("pancreas_binary_kappa", organ_summaries["pancreas"][2]),
            ("kidney_binary_kappa", organ_summaries["kidney"][2]),
            ("liver_binary_kappa", organ_summaries["liver"][2]),
        ]:
            for summary in summaries:
                detail_rows.append(
                    {
                        "model": model_name,
                        "metric": metric_name,
                        "pair": summary["pair"],
                        "n_cases": summary["n_cases"],
                        "mean": summary["mean"],
                        "case_sd": summary["case_sd"],
                    }
                )

    table_path = output_dir / "base_model_agreement_table_excluding_cases.csv"
    details_path = output_dir / "base_model_pair_level_details_excluding_cases.csv"
    latex_path = output_dir / "base_model_agreement_table_excluding_cases.tex"

    table_fields = [
        "model",
        "model_token",
        "cases",
        "excluded_cases",
        "fleiss_kappa_mean",
        "fleiss_kappa_case_sd",
        "foreground_fleiss_kappa_mean",
        "foreground_fleiss_kappa_case_sd",
        "pancreas_kappa_mean",
        "pancreas_kappa_pair_sd",
        "kidney_kappa_mean",
        "kidney_kappa_pair_sd",
        "liver_kappa_mean",
        "liver_kappa_pair_sd",
        "multiclass_cohen_kappa_mean",
        "multiclass_cohen_kappa_pair_sd",
    ]
    detail_fields = ["model", "metric", "pair", "n_cases", "mean", "case_sd"]

    write_rows(table_path, table_rows, table_fields)
    write_rows(details_path, detail_rows, detail_fields)
    latex_path.write_text(latex_table(table_rows, excluded_cases))

    print(f"Excluded cases: {', '.join(sorted(excluded_cases))}")
    print(f"Wrote: {table_path}")
    print(f"Wrote: {details_path}")
    print(f"Wrote: {latex_path}")
    print()
    for row in table_rows:
        print(
            f"{row['model']}: foreground Fleiss={row['foreground_fleiss_kappa_mean']:.6f}, "
            f"multiclass Cohen={row['multiclass_cohen_kappa_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
