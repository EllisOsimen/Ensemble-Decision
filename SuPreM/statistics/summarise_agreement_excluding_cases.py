#!/usr/bin/env python3
"""Recompute agreement summaries after excluding one or more test cases.

The source CSVs are the canonical 65-case evaluations. By default, the script
excludes UKCHLL082 and writes both the method and base-model sensitivity tables
under ``results/results_64_testing_set``. It does not rerun inference or load
NIfTI masks.
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPOSITORY_ROOT / "SuPreM" / "results"
DEFAULT_INPUT_ROOT = RESULTS_ROOT / "results_65_testing_set"
DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "results_64_testing_set"
DEFAULT_EXCLUDED_CASE = "UKCHLL082"

ORGAN_LABELS = {
    "pancreas": "1",
    "kidney": "2",
    "liver": "3",
}
TABLE_FIELDS = [
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


@dataclass(frozen=True)
class Evaluation:
    name: str
    relative_dir: Path
    pair_token: str | None


@dataclass(frozen=True)
class EvaluationGroup:
    key: str
    display_field: str
    evaluations: tuple[Evaluation, ...]
    output_dir_name: str
    table_filename: str
    details_filename: str
    latex_filename: str
    latex_heading: str
    latex_caption_prefix: str
    latex_caption_suffix: str
    latex_label: str
    include_token: bool = False


METHOD_GROUP = EvaluationGroup(
    key="methods",
    display_field="method",
    evaluations=(
        Evaluation("Human baseline", Path("human_annotator_evaluation"), None),
        Evaluation(
            "Unweighted",
            Path("unweighted_human_annotator_evaluation"),
            "agreement_mask",
        ),
        Evaluation(
            "Weighted",
            Path("weighted_human_annotator_evaluation"),
            "weighted_training_consensus",
        ),
        Evaluation(
            "STAPLE",
            Path("staple_human_annotator_evaluation"),
            "staple",
        ),
        Evaluation(
            "Random Forest",
            Path("random_forest_human_annotator_evaluation"),
            "random_forest_config_002",
        ),
    ),
    output_dir_name="testing_set_agreement_summary_exclude_UKCHLL082_from_csv",
    table_filename="agreement_table_excluding_cases.csv",
    details_filename="pair_level_details_excluding_cases.csv",
    latex_filename="agreement_table_excluding_cases.tex",
    latex_heading="Method",
    latex_caption_prefix="Agreement metrics after excluding ",
    latex_caption_suffix=(
        "Pairwise metrics are reported as mean $\\pm$ standard deviation "
        "across the three relevant expert comparisons."
    ),
    latex_label="tab:kappa_comparison_excluding_invalid_case",
)

BASE_MODEL_GROUP = EvaluationGroup(
    key="base-models",
    display_field="model",
    evaluations=(
        Evaluation(
            "CLIP Universal U-Net",
            Path("base_model_human_annotator_evaluation/clip_unet"),
            "clip_unet",
        ),
        Evaluation(
            "SuPreM SegResNet",
            Path("base_model_human_annotator_evaluation/segresnet"),
            "segresnet",
        ),
        Evaluation(
            "Swin UNETR",
            Path("base_model_human_annotator_evaluation/swin5050"),
            "swin5050",
        ),
    ),
    output_dir_name=(
        "base_model_human_annotator_evaluation_exclude_UKCHLL082_from_csv"
    ),
    table_filename="base_model_agreement_table_excluding_cases.csv",
    details_filename="base_model_pair_level_details_excluding_cases.csv",
    latex_filename="base_model_agreement_table_excluding_cases.tex",
    latex_heading="Base model",
    latex_caption_prefix="Base-model agreement metrics after excluding ",
    latex_caption_suffix=(
        "Pairwise metrics are reported as mean $\\pm$ standard deviation "
        "across the three model-expert comparisons."
    ),
    latex_label="tab:base_model_kappa_comparison_excluding_invalid_case",
    include_token=True,
)

GROUPS = (METHOD_GROUP, BASE_MODEL_GROUP)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute method and base-model agreement tables from canonical "
            "per-case CSVs after excluding one or more cases."
        )
    )
    parser.add_argument(
        "--exclude-case",
        action="append",
        default=[DEFAULT_EXCLUDED_CASE],
        help=(
            "Case ID to exclude. May be repeated. Defaults to UKCHLL082."
        ),
    )
    parser.add_argument(
        "--group",
        choices=("all", "methods", "base-models"),
        default="all",
        help="Summary group to generate. Defaults to both groups.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Canonical evaluation root containing the 65-case CSVs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for the recomputed sensitivity summaries.",
    )
    return parser.parse_args()


def read_csv(path: Path, excluded_cases: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("case") not in excluded_cases
        ]


def numeric(values: list[str]) -> list[float]:
    return [float(value) for value in values if value not in ("", "nan", "NaN")]


def mean_and_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot summarize an empty value list.")
    return (values[0], 0.0) if len(values) == 1 else (mean(values), stdev(values))


def pair_is_human(pair: str) -> bool:
    return (
        pair.startswith("annotation_")
        and "_vs_annotation_" in pair
        and pair.count("annotation_") == 2
    )


def group_pair_means(
    rows: list[dict[str, str]],
    metric_column: str,
    pair_token: str | None,
) -> list[dict[str, object]]:
    values_by_pair: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        pair = row["pair"]
        if pair_token is None:
            if not pair_is_human(pair):
                continue
        elif pair_token not in pair:
            continue

        value = row.get(metric_column, "")
        if value in ("", "nan", "NaN"):
            continue
        values_by_pair[pair].append(float(value))

    if not values_by_pair:
        raise ValueError(f"No rows found for pair token: {pair_token}")

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
    pair_token: str | None,
) -> tuple[float, float, list[dict[str, object]]]:
    pair_summaries = group_pair_means(rows, metric_column, pair_token)
    overall_mean, pair_sd = mean_and_sd(
        [float(row["mean"]) for row in pair_summaries]
    )
    return overall_mean, pair_sd, pair_summaries


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: object) -> str:
    return f"{float(value):.3f}"


def format_mean_pm_sd(value: object, sd: object) -> str:
    return f"{float(value):.3f} $\\pm$ {float(sd):.3f}"


def latex_table(
    rows: list[dict[str, object]],
    excluded_cases: set[str],
    group: EvaluationGroup,
) -> str:
    body = []
    for row in rows:
        body.append(
            " & ".join(
                [
                    str(row[group.display_field]),
                    format_value(row["foreground_fleiss_kappa_mean"]),
                    format_mean_pm_sd(
                        row["pancreas_kappa_mean"], row["pancreas_kappa_pair_sd"]
                    ),
                    format_mean_pm_sd(
                        row["kidney_kappa_mean"], row["kidney_kappa_pair_sd"]
                    ),
                    format_mean_pm_sd(
                        row["liver_kappa_mean"], row["liver_kappa_pair_sd"]
                    ),
                    format_mean_pm_sd(
                        row["multiclass_cohen_kappa_mean"],
                        row["multiclass_cohen_kappa_pair_sd"],
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
            rf"\textbf{{{group.latex_heading}}}",
            r"& \textbf{Foreground Fleiss' $\kappa$}",
            r"& \textbf{Pancreas $\kappa$}",
            r"& \textbf{Kidney $\kappa$}",
            r"& \textbf{Liver $\kappa$}",
            "& \\textbf{Multiclass Cohen's $\\kappa$} \\\\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{"
                + group.latex_caption_prefix
                + excluded
                + ". "
                + group.latex_caption_suffix
                + "}"
            ),
            rf"\label{{{group.latex_label}}}",
            r"\end{table}",
            "",
        ]
    )


def summarize_evaluation(
    evaluation: Evaluation,
    input_root: Path,
    excluded_cases: set[str],
) -> tuple[dict[str, object], list[tuple[str, list[dict[str, object]]]]]:
    result_dir = input_root / evaluation.relative_dir
    fleiss_rows = read_csv(result_dir / "per_case_fleiss_kappa.csv", excluded_cases)
    multiclass_rows = read_csv(
        result_dir / "per_case_pair_multiclass.csv", excluded_cases
    )
    per_class_rows = read_csv(
        result_dir / "per_case_pair_per_class.csv", excluded_cases
    )

    fleiss_mean, fleiss_sd = mean_and_sd(
        numeric([row["fleiss_kappa"] for row in fleiss_rows])
    )
    foreground_mean, foreground_sd = mean_and_sd(
        numeric([row["foreground_fleiss_kappa"] for row in fleiss_rows])
    )
    multiclass_mean, multiclass_pair_sd, multiclass_pairs = summarize_pair_metric(
        multiclass_rows, "multiclass_kappa", evaluation.pair_token
    )

    organs = {}
    for organ, label in ORGAN_LABELS.items():
        organ_rows = [row for row in per_class_rows if row["label"] == label]
        organs[organ] = summarize_pair_metric(
            organ_rows, "binary_kappa", evaluation.pair_token
        )

    row = {
        "cases": len(fleiss_rows),
        "excluded_cases": ";".join(sorted(excluded_cases)),
        "fleiss_kappa_mean": fleiss_mean,
        "fleiss_kappa_case_sd": fleiss_sd,
        "foreground_fleiss_kappa_mean": foreground_mean,
        "foreground_fleiss_kappa_case_sd": foreground_sd,
        "pancreas_kappa_mean": organs["pancreas"][0],
        "pancreas_kappa_pair_sd": organs["pancreas"][1],
        "kidney_kappa_mean": organs["kidney"][0],
        "kidney_kappa_pair_sd": organs["kidney"][1],
        "liver_kappa_mean": organs["liver"][0],
        "liver_kappa_pair_sd": organs["liver"][1],
        "multiclass_cohen_kappa_mean": multiclass_mean,
        "multiclass_cohen_kappa_pair_sd": multiclass_pair_sd,
    }
    detail_groups = [
        ("multiclass_cohen_kappa", multiclass_pairs),
        ("pancreas_binary_kappa", organs["pancreas"][2]),
        ("kidney_binary_kappa", organs["kidney"][2]),
        ("liver_binary_kappa", organs["liver"][2]),
    ]
    return row, detail_groups


def summarize_group(
    group: EvaluationGroup,
    input_root: Path,
    output_root: Path,
    excluded_cases: set[str],
):
    table_rows = []
    detail_rows = []

    for evaluation in group.evaluations:
        row, detail_groups = summarize_evaluation(
            evaluation, input_root, excluded_cases
        )
        row = {group.display_field: evaluation.name, **row}
        if group.include_token:
            row = {
                group.display_field: evaluation.name,
                "model_token": evaluation.pair_token,
                **{
                    key: value
                    for key, value in row.items()
                    if key != group.display_field
                },
            }
        table_rows.append(row)

        for metric_name, summaries in detail_groups:
            for summary in summaries:
                detail_rows.append(
                    {
                        group.display_field: evaluation.name,
                        "metric": metric_name,
                        **summary,
                    }
                )

    output_dir = output_root / group.output_dir_name
    table_path = output_dir / group.table_filename
    details_path = output_dir / group.details_filename
    latex_path = output_dir / group.latex_filename
    table_fields = [group.display_field]
    if group.include_token:
        table_fields.append("model_token")
    table_fields.extend(TABLE_FIELDS)
    detail_fields = [
        group.display_field,
        "metric",
        "pair",
        "n_cases",
        "mean",
        "case_sd",
    ]

    write_rows(table_path, table_rows, table_fields)
    write_rows(details_path, detail_rows, detail_fields)
    output_dir.mkdir(parents=True, exist_ok=True)
    latex_path.write_text(
        latex_table(table_rows, excluded_cases, group), encoding="utf-8"
    )

    print(f"{group.latex_heading} summary:")
    print(f"  {table_path}")
    print(f"  {details_path}")
    print(f"  {latex_path}")


def main():
    args = parse_args()
    excluded_cases = set(args.exclude_case)
    selected_groups = (
        GROUPS
        if args.group == "all"
        else tuple(group for group in GROUPS if group.key == args.group)
    )

    if not args.input_root.is_dir():
        raise NotADirectoryError(f"Input root not found: {args.input_root}")

    print(f"Input: {args.input_root}")
    print(f"Excluded cases: {', '.join(sorted(excluded_cases))}")
    for group in selected_groups:
        summarize_group(group, args.input_root, args.output_root, excluded_cases)


if __name__ == "__main__":
    main()
