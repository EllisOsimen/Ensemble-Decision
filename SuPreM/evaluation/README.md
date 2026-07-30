# Evaluation

The main evaluation workflow compares the three testing-set expert annotations
with each other and, optionally, with a model or consensus prediction.

## Testing-set human agreement

Use `evaluate_testing_set_human_agreement.py` for the human baseline and the
main method-performance comparisons.

Each case directory must contain:

```text
testing_set/UKCHLL003/
├── annotation_1.nii.gz
├── annotation_2.nii.gz
└── annotation_3.nii.gz
```

All masks are compared in the following target label space:

```text
0 background
1 pancreas
2 kidney
3 liver
```

### Human baseline

From the repository root, run:

```bash
python SuPreM/evaluation/evaluate_testing_set_human_agreement.py
```

This evaluates all three expert pairs and all three experts together across the
65 testing cases. The default output directory is:

```text
SuPreM/results/results_65_testing_set/human_annotator_evaluation/
```

### Comparing a method with the experts

Add a prediction directory as another rater:

```bash
python SuPreM/evaluation/evaluate_testing_set_human_agreement.py \
  --prediction-dir weighted=results/all_datasets_curvas_inference_with_confidence/weighted_consensus_training_masks/testing/consensus_masks \
  --output-dir SuPreM/results/results_65_testing_set/weighted_human_annotator_evaluation
```

Relative prediction paths are resolved from `SuPreM/`. Prediction masks may be
stored as either:

```text
<prediction-dir>/UKCHLL003.nii.gz
<prediction-dir>/UKCHLL003/agreement_mask.nii.gz
```

Predictions are assumed to use target labels 0--3. Append `:suprem` or `:btcv`
to the prediction specification when native model labels must be remapped:

```text
--prediction-dir clip_unet=path/to/predictions:suprem
--prediction-dir swin_unetr=path/to/predictions:btcv
```

Missing predictions stop the run by default, ensuring that methods are not
silently compared using different case subsets. Use
`--allow-missing-predictions` only when a common-case analysis is intentional.

### Excluding cases

To evaluate directly without the incomplete case `UKCHLL082`, add:

```text
--exclude-case UKCHLL082
```

The main 64-case tables in this repository are instead recalculated from the
65-case per-case CSVs. That workflow is documented in
`SuPreM/statistics/README.md`.

## Main metrics

- **Fleiss' kappa:** agreement across all requested raters. A prediction is
  included as an additional rater when supplied.
- **Multiclass Cohen's kappa:** whole-label-map agreement for each rater pair.
- **Binary Cohen's kappa:** one-versus-rest agreement for each organ and rater
  pair.
- **Dice:** spatial overlap for each organ and rater pair.
- **NSD:** boundary agreement within the requested tolerance, 1 mm by default.

Both full-volume and foreground-focused agreement are reported because the
large amount of shared background can make full-volume scores appear very
high.

## Outputs

Each run writes:

```text
run_config.txt
overall_summary.json
per_case_fleiss_kappa.csv
per_case_pair_multiclass.csv
per_pair_multiclass_summary.csv
per_case_pair_per_class.csv
per_pair_per_class_summary.csv
```

The most useful files for performance comparison are:

- `overall_summary.json` for mean case-wise Fleiss' kappa.
- `per_pair_multiclass_summary.csv` for model--expert whole-map kappa.
- `per_pair_per_class_summary.csv` for model--expert organ Dice, NSD, and
  binary kappa.
- The `per_case_*` files for case-level inspection and exclusion analyses.
- `run_config.txt` for the exact sources, cases, exclusions, and options.

The command prints the overall summary when complete. Selected fields from a
human-only run look like:

```json
{
  "metric_implementation_version": 2,
  "sources": [
    "annotation_1.nii.gz",
    "annotation_2.nii.gz",
    "annotation_3.nii.gz"
  ],
  "excluded_cases": [],
  "cases": 65,
  "mean_case_fleiss_kappa": 0.969,
  "mean_case_foreground_fleiss_kappa": 0.863
}
```

Metric implementation version 2 preserves the project's original organ-only
definition of foreground multiclass kappa and prevents integer overflow in
pooled micro kappa. The FP/FN columns use rater A as the reference and rater B
as the comparison.

## Batch evaluation

The three base models can be evaluated against all experts with:

```bash
sbatch SuPreM/sbatch/curvas/evaluate_base_models_human_annotators.sbatch
```

Each array task writes one model's results under
`results/results_65_testing_set/base_model_human_annotator_evaluation/`.

## Other utilities

- `check_testing_set_annotation_affines.py`: checks annotation shape, spacing,
  and affine consistency.
- `evaluate_testing_set_prediction_dirs.py`: evaluates saved predictions
  against one selected expert annotation.
- `evaluate_prediction_mask_agreement.py`: compares prediction masks with one
  another.
- `evaluate_word.py` and `evaluate_word_prediction_dirs.py`: WORD inference and
  evaluation.
- `evaluate_3d_dataset_prediction_dirs.py`: evaluates supported labels in the
  3D dataset.

Use `python <script> --help` for the complete arguments. NIfTI-based evaluation
requires the dependencies declared in `SuPreM/requirements.txt`.
