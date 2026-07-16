This is a project designed for my IPAB summer internship

Supervisor: Eleonora D'Arnese

## Regenerate all CURVAS predictions with confidence maps

The Slurm array script below runs all three segmentation models over the 20
training, 5 validation, and 65 testing CURVAS cases. For every hard-label mask,
it also creates a spatially aligned confidence map:

```bash
cd /home/s2347484/Seg/SuPreM
sbatch sbatch/infer_all_curvas_three_models_with_confidence.sbatch
```

By default, the results are written under
`SuPreM/results/all_curvas_inference_with_confidence` with this layout:

```text
training|validation|testing/
├── clip_universal_unet/
├── clip_universal_unet_confidence/
├── suprem_segresnet/
├── suprem_segresnet_confidence/
├── swinunetr_5050/
└── swinunetr_5050_confidence/
```

Every confidence-map voxel describes the confidence in the label assigned at
the same voxel of its corresponding hard-label mask. For CLIP Universal U-Net
and SegResNet, this is the assigned WORD label's sigmoid score; their background
confidence is one minus the strongest supported foreground score. For Swin
UNETR, it is the softmax score of the winning BTCV class. These are model
confidence scores and are not guaranteed to be calibrated probabilities. The
random-forest scripts can consume them as three separate continuous features
when all three `--confidence-dir` arguments are supplied. Omitting those
arguments retains the original label-only feature set.

Run five-fold tuning with the prediction and confidence maps using:

```bash
cd /home/s2347484/Seg/SuPreM
sbatch sbatch/random_forest_cross_validation.sbatch
```

The Slurm wrapper reads training inputs from
`results/all_curvas_inference_with_confidence/training` and supplies these
confidence directories automatically:

```text
clip_universal_unet_confidence
suprem_segresnet_confidence
swinunetr_5050_confidence
```

This produces 15 voxel features: 12 one-hot categorical label features plus
one continuous confidence feature from each model. The Python scripts remain
backwards compatible: without `--confidence-dir`, they use only the 12 label
features.

### Why the confidence maps use scaled `uint8`

A confidence value lies between 0 and 1. The default output stores it as an
unsigned 8-bit integer using

```text
stored value = round(confidence × 255)
decoded confidence = stored value / 255
```

For example, a confidence of `0.83` is stored as
`round(0.83 × 255) = 212`. When read back, it becomes
`212 / 255 = 0.83137`, an absolute error of only `0.00137`.

An 8-bit integer has 256 possible values (`0` through `255`), so adjacent
decoded confidence levels are `1 / 255 = 0.00392` apart. Rounding to the nearest
level gives a worst-case error of half that distance:
`0.5 / 255 = 0.00196`, which is below `0.002`.

The storage estimate comes from `uint8` requiring one byte per voxel while
`float32` requires four. For example, a `512 × 512 × 900` map contains
235,929,600 voxels, so one uncompressed confidence map uses approximately
236 MB as `uint8`, compared with 944 MB as `float32`. Across 90 cases and three
models (270 maps), that is approximately 64 GB for `uint8` versus 255 GB for
`float32`. Actual `.nii.gz` sizes vary with compression and image dimensions,
which is why 50–70 GB is a rough estimate rather than an exact total.

Nibabel applies the NIfTI scale automatically, so normal loading returns values
close to the original 0-to-1 confidence range:

```python
import nibabel as nib
import numpy as np

confidence_image = nib.load("case_confidence.nii.gz")
confidence = np.asanyarray(confidence_image.dataobj)
```

## Evaluate downloaded SuPreM models on WORD

`SuPreM/evaluate_word.py` runs inference directly on
`WORD/WORD-V0.1.0/imagesTr` and compares each prediction with the matching
gold label in `labelsTr`. It reports per-case and per-class DSC, normalized
surface Dice (using Google DeepMind's `surface-distance` package), and voxel
TP, FP, and FN.

Install the SuPreM environment and the surface-distance dependency:

```bash
cd SuPreM
pip install -r requirements.txt
```

First run one case as a smoke test:

```bash
python evaluate_word.py \
  --backbone unet \
  --checkpoint pretrained_weights/supervised_suprem_unet_2100.pth \
  --output-dir results/word_unet \
  --limit 1
```

Then remove `--limit 1` to evaluate all 100 training cases. Change both
`--backbone` and `--checkpoint` to evaluate the other downloaded models:

```bash
# Swin UNETR
python evaluate_word.py \
  --backbone swinunetr \
  --checkpoint pretrained_weights/supervised_suprem_swinunetr_2100.pth \
  --output-dir results/word_swinunetr

# SegResNet
python evaluate_word.py \
  --backbone segresnet \
  --checkpoint pretrained_weights/supervised_suprem_segresnet_2100.pth \
  --output-dir results/word_segresnet
```

The default NSD tolerance is 1 mm. Set another explicitly with
`--nsd-tolerance-mm`, since NSD results are not interpretable without the
tolerance. Add `--save-predictions` if combined WORD-numbered NIfTI
predictions are also needed.

### WORD and SuPreM class compatibility

WORD stores one mutually exclusive label map with 16 foreground classes.
SuPreM instead outputs 32 independent binary channels. The following mapping
is used during evaluation:

| WORD ID | WORD class | SuPreM channel | SuPreM class |
|---:|---|---:|---|
| 1 | liver | 6 | liver |
| 2 | spleen | 1 | spleen |
| 3 | left kidney | 3 | left kidney |
| 4 | right kidney | 2 | right kidney |
| 5 | stomach | 7 | stomach |
| 6 | gallbladder | 4 | gall bladder |
| 7 | esophagus | 5 | esophagus |
| 8 | pancreas | 11 | pancreas |
| 9 | duodenum | 14 | duodenum |
| 10 | colon | 18 | colon |
| 11 | intestine | 19 | intestine |
| 12 | adrenal | 12 and 13 | right and left adrenal glands, merged |
| 13 | rectum | 20 | rectum |
| 14 | bladder | 21 | bladder |
| 15 | left head of femur | 23 | left head of femur |
| 16 | right head of femur | 24 | right head of femur |

The remaining SuPreM outputs are not annotated as separate WORD classes and
are excluded from this evaluation:

- aorta, postcava, portal/splenic vein and hepatic vessel;
- right and left lungs;
- prostate and celiac trunk;
- kidney, liver, pancreas, hepatic-vessel, lung and colon tumors;
- kidney cyst.

This mapping is essential because equal numeric IDs do not represent equal
anatomy across the two label systems. For example, WORD label 1 is the liver,
whereas SuPreM channel 1 is the spleen. Comparing the arrays directly would
therefore produce invalid DSC, NSD, TP, FP and FN values. The mapping also
handles differences in class definitions: WORD has one combined adrenal
class, while SuPreM predicts the right and left adrenal glands separately.
The evaluator unions those two SuPreM masks before comparing them with WORD
label 12.

SuPreM channels are independent and may overlap. Metrics are therefore
calculated separately for each mapped binary class. Combined predictions
created with `--save-predictions` are mainly for inspection; the per-class
binary masks are the authoritative inputs to the reported metrics.

Each result directory contains:

- `per_case_per_class.csv`: DSC, NSD, TP, FP, and FN for every case/class.
- `per_class_summary.csv`: macro case metrics, micro DSC, and summed counts.
- `overall_summary.json`: overall macro DSC/NSD, micro DSC, TP, FP, and FN.

### Create three-model agreement maps

Run all three backbones with `--save-predictions`, then compare their saved
WORD-numbered label maps:

```bash
cd /home/s2347484/Seg/SuPreM
python ensemble_agreement.py \
  results/word_unet/predictions \
  results/word_swinunetr/predictions \
  results/word_segresnet/predictions \
  --output-dir results/word_ensemble_agreement
```

Each output is a `uint8` NIfTI file on the same grid as its input predictions.
Its voxel values describe model agreement:

- `1`: all three models predicted the same label.
- `2`: exactly two models predicted the same label.
- `3`: all three models predicted different labels.

Background label `0` participates in the comparison like any foreground
label. The script requires all three directories to contain the same case
filenames, shapes, and affine transforms.

### Run all three models on one image

To run U-Net, Swin UNETR, and SegResNet sequentially on one CT image:

```bash
cd /home/s2347484/Seg/SuPreM
python infer_single_image_three_models.py \
  --image /path/to/ct.nii.gz \
  --output-dir results/single_image
```

The script uses the checkpoints in `pretrained_weights` by default and creates:

```text
results/single_image/
├── unet/ct.nii.gz
├── swinunetr/ct.nii.gz
├── segresnet/ct.nii.gz
└── agreement/ct.nii.gz
```

The three model outputs use WORD label IDs. The agreement output uses labels
`1`, `2`, and `3` for all-model, two-model, and no-model agreement,
respectively. Models are loaded and released one at a time to limit GPU memory
use. Pass `--checkpoint-dir` if the three checkpoint files are elsewhere.

Submit the same single-image workflow to SLURM with:

```bash
cd /home/s2347484/Seg/SuPreM
sbatch infer_single_image_three_models.sbatch \
  /path/to/ct.nii.gz \
  results/single_image
```

The CT path is required. The output directory is optional and defaults to
`results/single_image`. The batch job activates `suprem-h200`, requests one
GPU, stages all three checkpoints to node-local storage, and runs the models
sequentially.

To infer on all 100 images in `WORD/WORD-V0.1.0/imagesTr`, submit the SLURM
array:

```bash
cd /home/s2347484/Seg/SuPreM
sbatch infer_word_images_three_models.sbatch
```

Optionally specify a different output directory:

```bash
sbatch infer_word_images_three_models.sbatch results/word_three_models
```

Each array task processes one image with all three models and writes its
results into the shared `unet`, `swinunetr`, `segresnet`, and `agreement`
subdirectories. Logs include the array job and task IDs. Array tasks load the
three checkpoints directly from `pretrained_weights`; they do not duplicate
the checkpoint files in `/tmp`. Resubmitting the array is safe: complete cases
exit immediately, while partially complete cases reuse existing masks and run
only the missing models. Pass `--overwrite` directly to
`infer_single_image_three_models.py` only when all outputs should be replaced.

### Run Tang et al. Swin UNETR 5050 on WORD

The downloaded `self_supervised_nv_swin_unetr_5050.pt` checkpoint is a
complete BTCV model with background plus 13 abdominal organs. Run inference
on all 100 WORD `imagesTr` scans with:

```bash
cd /home/s2347484/Seg/SuPreM
sbatch infer_word_swinunetr_5050.sbatch
```

Predictions are written to `results/word_swinunetr_5050` by default. Each
output preserves the input filename and uses the model's native BTCV labels:

```text
0 background                 7 stomach
1 spleen                     8 aorta
2 right kidney               9 inferior vena cava
3 left kidney               10 portal and splenic veins
4 gallbladder               11 pancreas
5 esophagus                 12 right adrenal gland
6 liver                     13 left adrenal gland
```

These are BTCV IDs, not WORD IDs. The array runs at most four scans
concurrently and safely skips outputs that already exist when resubmitted.

### Compare the three SuPreM models with Swin 5050

After all four inference outputs are complete, create one four-level agreement
map for every case:

```bash
cd /home/s2347484/Seg/SuPreM
conda activate suprem-h200
python ensemble_agreement/ensemble_agreement_four_models.py
```

The Swin 5050 BTCV labels are first translated to the WORD numbering used by
the three SuPreM outputs. Each agreement map uses:

```text
1 = all four models predicted the same structure
2 = three models predicted the same structure
3 = two models predicted the same structure
4 = all four models predicted different structures
```

A 2-vs-2 draw and a 2-vs-1-vs-1 result both receive agreement label `3`,
because the largest agreeing group contains two models. Swin 5050 structures
that are absent from the saved WORD maps are kept distinct from background.

All outputs are written directly under:

```text
results/word_four_model_agreement/
```

For all 100 cases, use the CPU-only SLURM array:

```bash
cd /home/s2347484/Seg/SuPreM
sbatch sbatch/ensemble_agreement_four_models.sbatch
```

Each array task processes one case, up to eight tasks concurrently. Existing
agreement files are skipped, so the array can safely be resubmitted.

### Count voxels and percentages by label

Calculate the voxel count and percentage of the complete image occupied by
each label:

```bash
cd /home/s2347484/Seg/SuPreM
conda activate suprem-h200
python statistics/label_voxel_statistics.py \
  --image results/single_image/agreement/word_0002.nii.gz
```

The table is printed to the terminal and saved beside the input as
`word_0002_voxel_statistics.csv`. Background label `0` is included by default.
Use `--exclude-background` to omit it from the table; percentages remain
relative to the total number of voxels in the image.

To aggregate every NIfTI label map in a directory:

```bash
python statistics/label_voxel_statistics.py \
  --image results/word_four_model_agreement
```

Directory mode creates `per_case_per_label.csv` and `per_label_summary.csv`.
The summary reports the mean and population standard deviation of voxel counts
and image percentages for each label. If a label is absent from a case, that
case contributes zero to its mean and standard deviation.

### Run on the SLURM cluster

The supplied `SuPreM/evaluate_word.sbatch` requests one GPU on the `saxa`
node in the `Teaching` partition, with 4 CPUs, 32 GB RAM and the partition's
two-day time limit. It processes one sliding-window patch at a time so it fits
the 16 GB H200 MIG profile as well as larger allocations.

Install the two evaluation-specific dependencies once on the login node:

```bash
conda activate suprem-h200
pip install nibabel surface-distance
```

Submit a one-case smoke test:

```bash
cd /home/s2347484/Seg/SuPreM
sbatch evaluate_word.sbatch unet 1
```

After that succeeds, submit the complete evaluation:

```bash
sbatch evaluate_word.sbatch unet
```

The first argument can be `unet`, `swinunetr`, or `segresnet`. Monitor the
job and inspect its logs with:

```bash
squeue -u "$USER"
tail -f slurm_logs/suprem-word-<job-id>.out
```

To evaluate all 100 cases with all three backbones, submit the provided
dependency chain:

```bash
cd /home/s2347484/Seg/SuPreM
bash submit_all_backbones.sh
```

This submits three separate jobs in the following order:

```text
U-Net → Swin UNETR → SegResNet
```

Each job starts only when the preceding job finishes successfully. Separate
allocations are safer than running all models inside one job: each backbone
gets a fresh two-day time limit and releases the GPU before the next job is
scheduled. The output directories are:

```text
results/word_unet/
results/word_swinunetr/
results/word_segresnet/
```

The log names include the backbone and job ID, for example:

```text
slurm_logs/suprem-word-unet-<job-id>.out
slurm_logs/suprem-word-swin-<job-id>.out
slurm_logs/suprem-word-segresnet-<job-id>.out
```

The batch script first copies the selected checkpoint to the node-local
`$SLURM_TMPDIR`. PyTorch checkpoints contain many separate tensor entries,
which can be unusually slow to load directly from the shared cluster
filesystem. Inference evaluates one sliding-window patch per GPU batch to
remain within the smallest GPU memory profile currently observed on `saxa`.
