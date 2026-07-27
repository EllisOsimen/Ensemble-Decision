# Ensemble agreement

This directory contains the code used to combine the predictions from CLIP
U-Net, SegResNet, and Swin UNETR 5050. Before fusion, all predictions are
mapped to the common CURVAS label space: `0` background, `1` pancreas,
`2` kidney, and `3` liver.

## Reported ensemble methods

| Method | How it was run |
| --- | --- |
| Unweighted | `consensus_agreement_mask.py` in its default `legacy` mode (no `--consensus-mode` argument). It used the original majority/local-agreement rule and CLIP U-Net as the fallback for unresolved voxels. |
| Weighted | `consensus_agreement_mask.py --consensus-mode weighted`, launched by `sbatch/run_testing_weighted_consensus.sbatch`. Organ-specific model weights came from mean patient-level Dice on the 20 training cases against `annotation_1.nii.gz`; the weak and strong thresholds are frozen in the batch script. |
| STAPLE | `consensus_agreement_mask.py --consensus-mode staple`. STAPLE was run independently for pancreas, kidney, and liver using the default probability threshold `0.5` and inter-organ margin `0.1`, after which the organ results were combined into one mask. |
| Random forest | Hyperparameters were selected with `random_forest_cross_validation.py` using five-fold patient-level cross-validation on the 20 training cases. `random_forest_stacking_consensus.py` was then trained on all 20 cases and applied to all 65 testing cases through `sbatch/train_final_random_forest_config_002.sbatch`. |

The unweighted, weighted, and STAPLE methods therefore use the same core
script with different fusion rules. The random forest is a supervised stacker
and has a separate training workflow.

## Final random-forest configuration

Cross-validation selected `config_002` using mean patient-level foreground
Dice:

```text
n_estimators=100
max_depth=5
min_samples_leaf=1
max_features=all
random_state=42
```

The forest used each model's one-hot predicted label, assigned-label
confidence, and validity mask. Its training target was
`annotation_1.nii.gz`; testing annotations were not used during training or
prediction.

## Main outputs

The masks used for the reported 65-case comparisons are retained under:

```text
results/legacy/CURVAS_INFERENCE/agreement_masks_target/          # unweighted
results/all_datasets_curvas_inference_with_confidence/
  weighted_consensus_training_masks/testing/consensus_masks/     # weighted
results/legacy/CURVAS_INFERENCE/agreement_masks_staple/          # STAPLE
results/all_datasets_curvas_inference_with_confidence/
  random_forest_config_002_final/testing_predictions/            # random forest
```

Some saved run configurations contain the earlier directory name
`all_curvas_inference_with_confidence`; these files were produced before that
directory was renamed to `all_datasets_curvas_inference_with_confidence`.

All four methods were compared with the three testing-set experts using
`evaluation/evaluate_testing_set_human_agreement.py`. The final evaluation
tables are in `results/results_65_testing_set/`, with the corresponding
case-82-excluded summaries in `results/results_64_testing_set/`.

Use `python <script> --help` for all command-line options. The older WORD
agreement experiments are kept separately in `Legacy_WORD_experiments/` and
are not part of these four reported methods. See `agreement.md` for the exact
unweighted and weighted voting cases.
