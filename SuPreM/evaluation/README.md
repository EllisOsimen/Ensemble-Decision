# Evaluation Scripts Overview

The evaluation scripts measure how well saved predictions, freshly generated
predictions, or human annotations agree with a reference.

The general pattern is:

```text
parse evaluation inputs
find matching cases/predictions/labels
load NIfTI masks
validate shape/affine alignment
normalize label spaces
compute per-case/per-class metrics
aggregate summaries
write CSV/JSON outputs
```

## Main Scripts

### `evaluate_word.py`

Runs SuPreM inference and evaluates it on WORD in the same script.

This script:

```text
loads a SuPreM checkpoint
runs inference on WORD imagesTr
compares predictions to WORD labelsTr
writes Dice/NSD summaries
optionally saves combined predictions
```

It supports:

```text
unet
swinunetr
segresnet
```

This is different from the prediction-directory evaluators because it does not
start from already-saved predictions. It creates predictions during the run.

### `evaluate_word_prediction_dirs.py`

Evaluates already-saved WORD prediction directories against WORD labels.

Use this when inference has already been run and predictions are stored as
NIfTI label maps.

### `evaluate_testing_set_prediction_dirs.py`

Evaluates prediction directories against the testing set annotations.

Target label space:

```text
0 background
1 pancreas
2 kidney
3 liver
```

Predictions can be interpreted as:

```text
target  already in 0-3 testing-set labels
btcv    native BTCV labels, remapped to 0-3
suprem  WORD/SuPreM labels, remapped to 0-3
```

### `evaluate_3d_dataset_prediction_dirs.py`

Evaluates prediction directories against the 3D dataset labels.

It evaluates only labels supported by all three model output spaces:

```text
1 liver
2 spleen
3 left kidney
4 right kidney
```

Other 3D dataset labels are ignored because not all models predict them.

### `evaluate_testing_set_human_agreement.py`

Measures human-human agreement between testing-set annotations:

```text
annotation_1 vs annotation_2
annotation_1 vs annotation_3
annotation_2 vs annotation_3
all three annotations together
```

This script does not use model predictions. It gives a human inter-rater
agreement baseline.

### `check_testing_set_annotation_affines.py`

Diagnostic script for checking annotation geometry.

It reports whether annotations have matching:

```text
shape
voxel spacing
affine matrix
```

This is useful for cases like `UKCHLL007`, where one annotation had labels but
was placed far away in physical space because its NIfTI header was wrong.

## `evaluate_word.py` Walkthrough

`evaluate_word.py` is a complete inference-and-evaluation pipeline.

### `WORD_TO_SUPREM`

Maps each WORD label to one or more SuPreM output channels.

Example:

```python
8: ("pancreas", (10,))
12: ("adrenal", (11, 12))
```

This means:

```text
WORD pancreas label 8 = SuPreM channel 10
WORD adrenal label 12 = SuPreM channels 11 or 12
```

### `parse_args()`

Reads options such as:

```text
--word-root
--checkpoint
--backbone
--output-dir
--device
--roi-size
--spacing
--threshold
--save-predictions
```

### `load_model(args, device)`

Builds the selected architecture and loads the checkpoint.

For `segresnet`, it builds MONAI `SegResNet`.

For `unet` and `swinunetr`, it builds `Universal_model`.

Then it moves the model to the selected device and switches it to evaluation
mode:

```python
model.to(device).eval()
```

### `make_loader(args)`

Finds matching WORD cases:

```text
WORD/imagesTr/*.nii.gz
WORD/labelsTr/*.nii.gz
```

It preprocesses only the CT image:

```text
Load image
Ensure channel-first
Orient to RAS
Resample to 1.5 mm spacing
Normalize CT intensity
Crop foreground/body region
```

The gold label is left untouched. The prediction is later inverted back to the
original label grid.

### `sliding_window_inference(...)`

Runs inference over overlapping 3D patches.

This keeps GPU memory use manageable and returns full-volume logits.

### Logits to Masks

SuPreM outputs 32 independent foreground-structure logits.

The script converts them to binary masks with:

```python
masks = torch.sigmoid(logits).ge(args.threshold)
```

This is not an argmax. Multiple channels can be positive at the same voxel.

### `invert_prediction(batch, transforms, prediction)`

Undo preprocessing so the prediction returns to the original WORD image grid.

It reverses:

```text
crop
spacing
orientation
```

This is needed because the model predicts in preprocessed space, while metrics
must be computed against the original `labelsTr` mask.

### `binary_metrics(prediction, gold, spacing, tolerance_mm)`

Compares one predicted class mask to one gold WORD class mask.

It computes:

```text
Dice / DSC
NSD
TP
FP
FN
```

NSD uses physical voxel spacing from the NIfTI header. If either mask is empty,
NSD is recorded as `NaN` because there is no surface to compare.

### Optional Saved Predictions

If `--save-predictions` is passed, the script saves one combined WORD-style
label map per case.

These saved predictions are mainly for visual inspection. The metrics are
computed from the independent SuPreM masks because a single integer label map
cannot preserve overlapping channels.

## Label-Space Mapping

Different models and datasets use different label spaces, so the evaluators
normalize predictions before computing metrics.

For testing-set evaluation:

```text
BTCV pancreas          -> target pancreas
BTCV left/right kidney -> target kidney
BTCV liver             -> target liver
other BTCV labels      -> background

WORD liver             -> target liver
WORD left/right kidney -> target kidney
WORD pancreas          -> target pancreas
other WORD labels      -> background
```

For 3D dataset evaluation:

```text
BTCV spleen       -> dataset spleen
BTCV right kidney -> dataset right kidney
BTCV left kidney  -> dataset left kidney
BTCV liver        -> dataset liver

WORD liver        -> dataset liver
WORD spleen       -> dataset spleen
WORD left kidney  -> dataset left kidney
WORD right kidney -> dataset right kidney
```

## Common Metric Meaning

### Dice / DSC

Measures overlap between prediction and reference.

```text
1 = perfect overlap
0 = no overlap
```

### NSD

Normalized Surface Dice.

Measures whether predicted and reference boundaries are within a physical
distance tolerance, usually `1 mm`.

### TP / FP / FN

Voxel counts:

```text
TP = predicted class and reference class
FP = predicted class but not reference class
FN = reference class but prediction missed it
```

These counts are used to compute micro Dice.

### Cohen's Kappa

Used in the human-agreement script for two annotators.

It measures agreement corrected for chance:

```text
1 = perfect agreement
0 = chance-level agreement
<0 = worse than chance
```

### Fleiss' Kappa

Used in the human-agreement script for all three annotators together.

It is the multi-rater version of kappa.

## Output Files

Most evaluators write:

```text
per_case_per_class.csv
per_class_summary.csv
overall_summary.json
run_config.txt
```

### `per_case_per_class.csv`

One row per case/class.

Typical columns:

```text
case
label or word_label
class
dsc
nsd
tp
fp
fn
```

### `per_class_summary.csv`

Aggregates each class across all cases.

Typical columns:

```text
mean_case_dsc
mean_case_nsd
micro_dsc
tp
fp
fn
cases_with_dsc
cases_with_nsd
```

### `overall_summary.json`

Compact model-level summary.

Typical fields:

```text
macro_case_dsc
macro_case_nsd
micro_dsc
tp
fp
fn
```

### `run_config.txt`

Records the exact input paths, model specs, annotations, and label settings
used for the run.

## Example Commands

Run `evaluate_word.py` directly:

```bash
python evaluation/evaluate_word.py \
  --word-root /home/s2347484/Seg/SuPreM/WORD/WORD-V0.1.0 \
  --checkpoint /home/s2347484/Seg/SuPreM/pretrained_weights/supervised_suprem_segresnet_2100.pth \
  --backbone segresnet \
  --output-dir results/word_segresnet \
  --save-predictions
```

Evaluate saved testing-set predictions:

```bash
python evaluation/evaluate_testing_set_prediction_dirs.py \
  --cases-root /home/s2347484/Seg/testing_set \
  --annotation-name annotation_1.nii.gz \
  --output-dir results/CURVAS_EVAL \
  --model clip_unet=results/CURVAS_INFERENCE/clip_universal_unet:suprem
```

Run human agreement while excluding the known bad geometry case:

```bash
python evaluation/evaluate_testing_set_human_agreement.py \
  --cases-root /home/s2347484/Seg/testing_set \
  --output-dir results/testing_set_human_agreement \
  --exclude-case UKCHLL007
```

## Revision Note

The evaluation scripts repeat several useful pieces:

```text
argument parsing
NIfTI loading
integer-label validation
case matching
affine/shape validation
binary Dice/NSD calculation
CSV writing
summary aggregation
```

A future cleanup could move these into a shared evaluation utility module. Then
each evaluator would only define:

```text
dataset-specific labels
label-space remapping
which cases/classes to evaluate
which summary metrics to write
```
