#!/usr/bin/env python3
"""Summarize existing agreement CSVs after excluding invalid cases.

This script does not rerun inference or reload NIfTI masks. It reads the
per-case CSV files already produced by the agreement evaluation scripts,
drops the requested case IDs, and recomputes the report table metrics.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "SuPreM"
    / "results"
    / "testing_set_agreement_summary_exclude_UKCHLL082_from_csv"
)

METHODS = [
    {
        "name": "Human baseline",
        "result_dir": REPOSITORY_ROOT / "SuPreM/results/testing_set_human_agreement_all_65",
        "method_pair_token": None,
    },
    {
        "name": "Unweighted",
        "result_dir": REPOSITORY_ROOT
        / "SuPreM/results/legacy/testing_set_human_plus_agreement_mask",
        "method_pair_token": "agreement_mask",
    },
    {
        "name": "Weighted",
        "result_dir": REPOSITORY_ROOT
        / "SuPreM/results/all_curvas_inference_with_confidence"
        / "weighted_consensus_training_weights/testing/human_annotator_evaluation",
        "method_pair_token": "weighted_training_consensus",
    },
    {
        "name": "STAPLE",
        "result_dir": REPOSITORY_ROOT
        / "SuPreM/results/legacy/testing_set_human_plus_staple_agreement_mask",
        "method_pair_token": "staple",
    },
    {
        "name": "Random Forest",
        "result_dir": REPOSITORY_ROOT
        / "SuPreM/results/all_curvas_inference_with_confidence"
        / "random_forest_config_002_final/human_annotator_evaluation",
        "method_pair_token": "random_forest_config_002",
    },
]

ORGAN_LABELS = {
    "pancreas": "1",
    "kidney": "2",
    "liver": "3",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute agreement summary metrics from existing per-case CSV "
            "files after excluding invalid cases."
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the recomputed summary CSV and LaTeX table are written.",
    )
    return parser.parse_args()


def read_csv(path: Path, excluded_cases: set[str]) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            row
            for row in reader
            if row.get("case") not in excluded_cases
        ]


def numeric(values: list[str]) -> list[float]:
    return [float(value) for value in values if value not in ("", "nan", "NaN")]


def mean_and_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot summarize an empty value list.")
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def pair_is_human(pair: str) -> bool:
    return (
        pair.startswith("annotation_")
        and "_vs_annotation_" in pair
        and pair.count("annotation_") == 2
    )


def group_pair_means(
    rows: list[dict[str, str]],
    metric_column: str,
    method_pair_token: str | None,
) -> list[dict[str, object]]:
    values_by_pair = defaultdict(list)
    for row in rows:
        pair = row["pair"]
        if method_pair_token is None:
            if not pair_is_human(pair):
                continue
        elif method_pair_token not in pair:
            continue
        values_by_pair[pair].append(float(row[metric_column]))

    if not values_by_pair:
        raise ValueError(f"No rows found for method token: {method_pair_token}")

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
    method_pair_token: str | None,
) -> tuple[float, float, list[dict[str, object]]]:
    pair_summaries = group_pair_means(rows, metric_column, method_pair_token)
    pair_means = [float(row["mean"]) for row in pair_summaries]
    overall_mean, pair_sd = mean_and_sd(pair_means)
    return overall_mean, pair_sd, pair_summaries


def summarize_organ_metric(
    rows: list[dict[str, str]],
    label: str,
    method_pair_token: str | None,
) -> tuple[float, float, list[dict[str, object]]]:
    organ_rows = [row for row in rows if row["label"] == label]
    return summarize_pair_metric(organ_rows, "binary_kappa", method_pair_token)


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
                    str(row["method"]),
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
            r"\textbf{Method}",
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
                r"\caption{Agreement metrics after excluding "
                + excluded
                + r". Pairwise metrics are reported as mean $\pm$ standard "
                + r"deviation across the three relevant expert comparisons.}"
            ),
            r"\label{tab:kappa_comparison_excluding_invalid_case}",
            r"\end{table}",
            "",
        ]
    )


def main():
    args = parse_args()
    excluded_cases = set(args.exclude_case)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    table_rows = []
    detail_rows = []

    for method in METHODS:
        result_dir = method["result_dir"]
        method_name = method["name"]
        token = method["method_pair_token"]

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
                per_class_rows, label, token
            )

        table_rows.append(
            {
                "method": method_name,
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
                        "method": method_name,
                        "metric": metric_name,
                        "pair": summary["pair"],
                        "n_cases": summary["n_cases"],
                        "mean": summary["mean"],
                        "case_sd": summary["case_sd"],
                    }
                )

    table_path = output_dir / "agreement_table_excluding_cases.csv"
    details_path = output_dir / "pair_level_details_excluding_cases.csv"
    latex_path = output_dir / "agreement_table_excluding_cases.tex"

    table_fields = [
        "method",
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
    detail_fields = ["method", "metric", "pair", "n_cases", "mean", "case_sd"]

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
            f"{row['method']}: foreground Fleiss={row['foreground_fleiss_kappa_mean']:.6f}, "
            f"multiclass Cohen={row['multiclass_cohen_kappa_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
