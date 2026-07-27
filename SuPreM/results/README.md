# Results

This directory contains the final testing results and the intermediate outputs
used to produce them.

## Main results

- `results_65_testing_set/` contains the primary analysis of all 65 testing
  cases. Start with `testing_set_human_annotator_evaluation_overall_summary.csv`
  for the main method comparison; the subdirectories contain detailed results
  for the human baseline, base models, and consensus methods.
- `results_64_testing_set/` contains the results after excluding
  the incomplete case `UKCHLL082`. It provides summary tables for the consensus
  methods and base models in CSV and LaTeX formats, plus labelled expert and
  method confusion matrices.

## Supporting outputs

- `train_validation_inference/` contains training and validation predictions,
  cross-validation outputs, and model-selection results. Used to choose hyperparameters.
- `all_datasets_curvas_inference_with_confidence/` contains model predictions,
  confidence and validity outputs, and consensus-generation results across the
  dataset splits.
- `legacy/` contains older or superseded outputs retained for provenance; these
  should not be used as the main reported results.

Within each evaluation directory, `overall_summary.json` gives the aggregate
metrics, while `per_case_*` and `per_pair_*` files provide the corresponding
case-level and annotator-pair details. `run_config.txt` records how an evaluation
was produced.
