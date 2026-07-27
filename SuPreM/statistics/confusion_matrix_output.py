"""Structured output helpers for confusion-matrix scripts."""

import csv
import json
from pathlib import Path


LABEL_NAMES = ("background", "pancreas", "kidney", "liver")


def write_matrix_csv(path: Path, matrix, integer: bool = False):
    """Write a labelled square matrix as CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_label", *LABEL_NAMES])
        for label, values in zip(LABEL_NAMES, matrix):
            if integer:
                values = [int(value) for value in values]
            writer.writerow([label, *values])


def write_confusion_outputs(
    output_dir: Path,
    metadata: dict[str, object],
    pooled_counts,
    pooled_row_percent,
    mean_patient_row_percent,
    sd_patient_percentage_points,
):
    """Write metadata and the four reported matrices to a new output folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {**metadata, "label_order": list(LABEL_NAMES)}
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_matrix_csv(output_dir / "pooled_counts.csv", pooled_counts, integer=True)
    write_matrix_csv(output_dir / "pooled_row_percent.csv", pooled_row_percent)
    write_matrix_csv(
        output_dir / "mean_patient_row_percent.csv", mean_patient_row_percent
    )
    write_matrix_csv(
        output_dir / "sd_patient_percentage_points.csv",
        sd_patient_percentage_points,
    )
