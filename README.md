This is a project designed for my IPAB summer internship

Supervisor: Eleonora D'Arnese

# CURVAS Ensemble Agreement Experiments

This repository evaluates four ways to combine three CT segmentation models on
the CURVAS pancreas, kidney, and liver task. The base predictions are produced
by CLIP Universal U-Net, SuPreM SegResNet, and Swin UNETR 5050, then remapped
to the common CURVAS label space: `0` background, `1` pancreas, `2` kidney,
and `3` liver.

The dataset contains 65 held-out testing cases. The final reported comparison
uses 64 of them because `UKCHLL082` was excluded owing to incomplete pancreas
and liver annotations. Each consensus mask was evaluated jointly with the
three independent human annotations using Fleiss' kappa; foreground Fleiss'
kappa excludes the dominant background class. The normalized surface Dice
tolerance used by the evaluator was 1 mm.

## Ensemble Approaches And Results

| Method | Technique | What it checks | Mean case Fleiss' kappa | Mean foreground Fleiss' kappa |
| --- | --- | --- | ---: | ---: |
| **Unweighted consensus** | The non-learned baseline. It applies the legacy three-model agreement rule: preserve unanimous and two-model organ votes, treat two background votes conservatively, and resolve fully discordant voxels using local support or the CLIP U-Net fallback. | Whether simple model agreement alone can improve robustness, without using training-derived reliability or labels. | 0.9440 | 0.7902 |
| **Weighted consensus** | An organ-aware deterministic fusion. Each model's pancreas, kidney, and liver vote is weighted by its mean patient-level Dice on the 20 training cases against `annotation_1.nii.gz`. Weak/strong thresholds reject unreliable isolated votes while retaining connected evidence and all two-model agreements. | Whether known, organ-specific differences in model reliability improve over the unweighted baseline without needing a voxel-level classifier. | 0.9447 | 0.7910 |
| **STAPLE** | Per-organ binary STAPLE fusion. The procedure estimates each base segmenter's sensitivity/specificity and combines their masks into posterior probabilities before resolving the multi-organ label. | Whether an established probabilistic label-fusion method provides a better consensus than the hand-designed unweighted rule. | 0.9553 | 0.8018 |
| **Random-forest stacking** | A supervised, voxel-wise classifier fitted on the 20 training patients. It receives 12 one-hot base-label features plus the three assigned-label confidence values; identical validity masks prevent crop-exterior voxels from being learned as foreground. The final `config_002` uses 100 trees, maximum depth 5, all features per split, and `random_state=42`. | Whether training labels and calibrated model evidence can learn complementary error patterns that deterministic fusion misses. Testing annotations are not used to train or infer the masks. | **0.9575** | **0.8120** |

These values are recorded in
[`agreement_table_excluding_cases.csv`](SuPreM/results/results_64_testing_set/testing_set_agreement_summary_exclude_UKCHLL082_from_csv/agreement_table_excluding_cases.csv).
All four methods covered the same 64 cases, with only `UKCHLL082` excluded.
Results should be interpreted as agreement with the three human annotators,
rather than as a single-reference segmentation score.

## Required Repository Layout

The inference and Slurm scripts use this
layout by default; alternative locations can be supplied as positional
arguments to the relevant `.sbatch` scripts.

```text
Ensemble-Decision/
├── SuPreM/
│   ├── ensemble_agreement/             # fusion, stacking, and agreement code
│   ├── evaluation/                     # human-annotator agreement evaluator
│   ├── inference/                      # the three base-model inference scripts
│   ├── pretrained_weights/             # model checkpoints (not versioned)
│   ├── results/                        # generated inference/fusion/evaluation outputs
│   ├── sbatch/                         # reproducible Slurm entry points
│   └── requirements.txt
├── training_set/
│   └── training_set/
│       └── UKCHLL###/
│           ├── image.nii.gz
│           ├── annotation_1.nii.gz
│           ├── annotation_2.nii.gz
│           └── annotation_3.nii.gz
├── validation_set/
│   └── UKCHLL###/                      # same per-case files; 5 cases
├── testing_set/
│   └── UKCHLL###/                      # same per-case files; 65 cases
├── Dataset/                            # source 3D-IRCADb data and derived training data
└── WORD/                               # optional WORD benchmark data; separate from CURVAS
```

The CURVAS experiment uses 20 training cases, 5 validation cases, and 65
testing cases. Model inference and the canonical per-case evaluation cover all
65 testing scans; the final summary excludes `UKCHLL082`, leaving 64 cases.
Every per-case NIfTI image and annotation must have the same voxel shape and
affine transform. The inference array accepts either `image.nii.gz` or
`ct.nii.gz` as the CT filename, but the layout above matches the current CURVAS
data. `annotation_1.nii.gz` is the training target for the random forest and
for deriving weighted-consensus weights; testing annotations are used only for
evaluation.

Place the required checkpoints under `SuPreM/pretrained_weights/`:

```text
supervised_clip_driven_universal_unet_2100.pth
supervised_suprem_segresnet_2100.pth
self_supervised_nv_swin_unetr_5050.pt
```

Generated model outputs are currently organised under the following root:

```text
SuPreM/results/all_datasets_curvas_inference_with_confidence/
├── training|validation|testing/
│   ├── clip_universal_unet/
│   ├── clip_universal_unet_confidence/
│   ├── clip_universal_unet_validity/
│   ├── suprem_segresnet/
│   ├── suprem_segresnet_confidence/
│   ├── suprem_segresnet_validity/
│   ├── swinunetr_5050/
│   ├── swinunetr_5050_confidence/
│   └── swinunetr_5050_validity/
├── weighted_consensus_training_masks/
└── random_forest_config_002_final/
```

Some saved configurations and Slurm wrappers retain the earlier directory
name `all_curvas_inference_with_confidence`; update those machine-specific
paths to `all_datasets_curvas_inference_with_confidence` before rerunning them.

Hard-label, confidence, and validity maps use one identically named NIfTI file
per case. Confidence values are stored as scaled `uint8`; validity maps are
binary and identify voxels evaluated by every base model. Do not mix case IDs,
NIfTI grids, or affine transforms between these directories.

## Reproduce The Experiment

The Slurm scripts assume a GPU-capable environment named `suprem-h200`; create
an equivalent environment with the dependencies below, then adapt the conda
environment name and Slurm resource directives if your cluster differs.

```bash
cd /path/to/Ensemble-Decision/SuPreM
pip install -r requirements.txt
```

1. Generate the base predictions, confidences, and validity masks for all 90
  CURVAS cases:

  ```bash
  cd /path/to/Ensemble-Decision/SuPreM
  sbatch sbatch/infer_all_curvas_three_models_with_confidence.sbatch
  ```

2. Run deterministic fusion. The checked-in wrapper recreates the final
  training-weighted test masks:

  ```bash
  sbatch sbatch/run_testing_weighted_consensus.sbatch
  ```

  `ensemble_agreement/consensus_agreement_mask.py` implements all three
  non-RF modes: `legacy` for unweighted consensus, `weighted` for the
  training-derived method, and `staple` for per-organ STAPLE. Supply the three
  prediction directories with label spaces `suprem`, `suprem`, and `btcv` in
  CLIP, SegResNet, Swin order; STAPLE additionally requires the corresponding
  three validity directories.

3. Select and train the learned ensemble, then produce the held-out test
  masks. Five-fold CV operates at patient level, so voxels from an individual
  patient never appear in both a fitting and held-out fold.

  ```bash
  sbatch sbatch/random_forest_cross_validation.sbatch
  sbatch sbatch/train_final_random_forest_config_002.sbatch
  ```

4. Evaluate all 65 fused testing masks with the three expert annotations. The
  random forest and weighted wrappers are provided directly:

  ```bash
  sbatch sbatch/evaluate_final_random_forest_human_annotators.sbatch
  sbatch sbatch/evaluate_weighted_consensus_human_annotators.sbatch
  ```

5. From the repository root, regenerate the final 64-case method and base-model
  tables by excluding `UKCHLL082` from the canonical per-case CSVs:

  ```bash
  python SuPreM/statistics/summarise_agreement_excluding_cases.py
  ```

The evaluator writes the canonical 65-case per-case CSVs and aggregate files.
The final command does not rerun inference or evaluation: it removes
`UKCHLL082` from the per-case CSVs and recalculates the reported 64-case means
and standard deviations under `SuPreM/results/results_64_testing_set/`.
