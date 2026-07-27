# Statistics utilities

These scripts calculate descriptive label statistics, confusion matrices, and
the 64-case sensitivity summaries reported in `SuPreM/results/`.

## Sensitivity summaries

`summarize_agreement_excluding_cases.py` reads the canonical per-case CSVs from
`results/results_65_testing_set/`, excludes `UKCHLL082` by default, and writes
both the method and base-model tables under `results/results_64_testing_set/`.
It does not rerun inference.

For each configured human, consensus-method, and base-model evaluation, it
reads exactly these files:

- `per_case_fleiss_kappa.csv`
- `per_case_pair_multiclass.csv`
- `per_case_pair_per_class.csv`

Rows whose `case` value is `UKCHLL082` are removed before all summary metrics
are recalculated, leaving 64 cases. Existing aggregate files such as
`overall_summary.json` and `per_pair_*_summary.csv` are deliberately not used,
because they already contain values calculated from all 65 cases.

From the repository root, run:

```bash
python SuPreM/statistics/summarize_agreement_excluding_cases.py
```

Use `--group methods` or `--group base-models` to generate only one table set.
An additional exclusion is added to the default `UKCHLL082` exclusion as
follows:

```bash
python SuPreM/statistics/summarize_agreement_excluding_cases.py \
  --exclude-case UKCHLL007
```

The default command prints output similar to:

```text
Input: .../SuPreM/results/results_65_testing_set
Excluded cases: UKCHLL082
Method summary:
  .../testing_set_agreement_summary_exclude_UKCHLL082_from_csv/agreement_table_excluding_cases.csv
  .../testing_set_agreement_summary_exclude_UKCHLL082_from_csv/pair_level_details_excluding_cases.csv
  .../testing_set_agreement_summary_exclude_UKCHLL082_from_csv/agreement_table_excluding_cases.tex
Base model summary:
  .../base_model_human_annotator_evaluation_exclude_UKCHLL082_from_csv/base_model_agreement_table_excluding_cases.csv
  .../base_model_human_annotator_evaluation_exclude_UKCHLL082_from_csv/base_model_pair_level_details_excluding_cases.csv
  .../base_model_human_annotator_evaluation_exclude_UKCHLL082_from_csv/base_model_agreement_table_excluding_cases.tex
```

The main method CSV contains one row per human baseline or consensus method.
For example, its first columns are:

```text
method,cases,excluded_cases,...
Human baseline,64,UKCHLL082,...
Unweighted,64,UKCHLL082,...
```

## Confusion matrices

- `compute_expert_confusion_parallel.py` compares two expert annotations.
- `compute_method_vs_expert_confusion_parallel.py` compares one expert with a
  model or consensus mask.
- `confusion_matrix_output.py` is their shared structured-output helper and is
  not run directly.

Both comparison commands exclude `UKCHLL082` by default and print progress and
results to stdout. Pass `--output-dir` to also save labelled CSV matrices and a
`metadata.json` file. For example:

```bash
python SuPreM/statistics/compute_expert_confusion_parallel.py \
  --row-annotation annotation_1.nii.gz \
  --column-annotation annotation_2.nii.gz \
  --output-dir SuPreM/results/results_64_testing_set/confusion_matrices/annotation_1_vs_annotation_2
```

```bash
python SuPreM/statistics/compute_method_vs_expert_confusion_parallel.py \
  --expert-annotation annotation_1.nii.gz \
  --method-name weighted \
  --method-root SuPreM/results/all_datasets_curvas_inference_with_confidence/weighted_consensus_training_masks/testing/consensus_masks \
  --output-dir SuPreM/results/results_64_testing_set/confusion_matrices/annotation_1_vs_weighted
```

The reported matrices are:

- `pooled_counts.csv`: voxel counts pooled across cases.
- `pooled_row_percent.csv`: pooled counts normalized within each expert row.
- `mean_patient_row_percent.csv`: row-normalized matrices averaged with equal
  weight per case.
- `sd_patient_percentage_points.csv`: between-case sample standard deviations.

## Label volumes

`label_voxel_statistics.py` reports per-label voxel counts and percentages for
one NIfTI label map or a directory of maps. Use `--help` on any executable
script for all arguments.
