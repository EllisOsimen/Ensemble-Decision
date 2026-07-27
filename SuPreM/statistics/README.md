# Statistics utilities

These scripts calculate descriptive label statistics, confusion matrices, and
the 64-case sensitivity summaries reported in `SuPreM/results/`.

## Sensitivity summaries

`summarize_agreement_excluding_cases.py` reads the canonical per-case CSVs from
`results/results_65_testing_set/`, excludes `UKCHLL082` by default, and writes
both the method and base-model tables under `results/results_64_testing_set/`.
It does not rerun inference.

From the repository root, run:

```bash
python SuPreM/statistics/summarize_agreement_excluding_cases.py
```

Use `--group methods` or `--group base-models` to generate only one table set.
Repeat `--exclude-case` when additional cases must be excluded.

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
