# Inference Scripts Overview

The inference scripts follow the same high-level pattern:

```text
parse inputs
load CT
preprocess CT
load model
run sliding-window inference
convert logits to masks or labels
invert prediction back to original CT space
save NIfTI with original geometry
print labels/output info
```

## Shared Components

### `parse_args()`

Reads command-line inputs such as:

- input image path
- output path
- checkpoint path
- device, usually `cuda` or `cpu`
- ROI size
- preprocessing spacing
- sliding-window overlap
- sliding-window batch size
- threshold
- overwrite mode

This makes each script runnable from the terminal or from a SLURM job.

### `make_loader(image_path, spacing)`

Builds the MONAI preprocessing pipeline. The common transforms are:

```text
LoadImaged              load NIfTI image
EnsureChannelFirstd     make shape channel-first
Orientationd            orient image to RAS
Spacingd                resample to model inference spacing
ScaleIntensityRanged    CT windowing/normalization
CropForegroundd         crop empty background around body
```

This prepares the CT for inference while keeping enough metadata to undo the
transforms later.

### `load_model(args, device)`

Builds the model architecture and loads checkpoint weights.

The model differs by script:

```text
CLIP Universal U-Net -> Universal_model
SegResNet            -> MONAI SegResNet
Swin UNETR           -> SwinUNETR
```

After loading the weights, the model is moved to the selected device and put in
evaluation mode:

```python
model.to(device).eval()
```

### `sliding_window_inference(...)`

Runs inference over overlapping 3D patches instead of the full CT at once. This
avoids GPU memory overflow.

Important arguments:

```text
roi_size       patch size, usually 96 x 96 x 96
overlap        how much neighboring patches overlap
mode=gaussian  blends patch edges smoothly
```

The output of this step is `logits`, the raw model outputs before activation,
thresholding, or argmax. These logits are not currently saved by the scripts,
but this is where probability/confidence work would begin.

### Logits Post-Processing

Post-processing depends on the model output type.

For CLIP U-Net and SegResNet:

```python
masks = torch.sigmoid(logits).ge(threshold)
```

Each output channel is treated as an independent binary organ prediction.

For Swin UNETR:

```python
prediction = torch.argmax(logits, dim=1)
```

Each voxel gets exactly one native BTCV class.

For CURVAS inference, BTCV outputs are remapped into:

```text
0 background
1 pancreas
2 kidney
3 liver
```

### `invert_prediction(batch, transforms, prediction)`

Returns the prediction from preprocessed space back to the original CT grid.

It reverses:

```text
crop
spacing/resampling
orientation
```

This is important because the model predicts in preprocessed space, but the
saved output needs to align with the original image in Slicer and downstream
evaluation.

### `combine_word_labels(masks)`

Used by the CLIP U-Net and SegResNet scripts.

Those models output independent binary channels, but the saved result is one
combined WORD label map:

```text
0 background
1 liver
2 spleen
3 left kidney
...
16 right femur head
```

Some WORD labels merge multiple model channels. For example, adrenal uses both
left and right adrenal channels.

### `save_prediction(prediction, reference, output_path)`

Saves the final label map as `.nii.gz`.

The function copies spatial metadata from the original CT:

```text
header
affine
qform
sform
```

It also saves the label map as `uint8`, which is appropriate for integer labels
and keeps the file smaller than saving as floating point.

### `main()`

Coordinates the full inference pipeline:

```text
validate files and parameters
check CUDA/CPU availability
load reference image
preprocess image
load model
run inference
post-process logits
invert prediction to original grid
save prediction
print labels present
```

## Example Walkthrough

Example CLIP U-Net command:

```bash
python inference/infer_clip_universal_unet.py \
  --image /home/s2347484/Seg/testing_set/UKCHLL003/image.nii.gz \
  --output results/CURVAS_INFERENCE/clip_universal_unet/UKCHLL003.nii.gz
```

What happens:

1. `parse_args()` reads the image, output, checkpoint, and inference settings.
2. The script loads `image.nii.gz` as the reference CT.
3. `make_loader()` preprocesses the CT:
   - orient to RAS
   - resample to `1.5 x 1.5 x 1.5`
   - normalize CT intensities
   - crop foreground
4. `load_model()` builds CLIP Universal U-Net and loads weights.
5. `sliding_window_inference()` runs the CT through the model patch-by-patch.
6. The model outputs `logits` with 32 channels.
7. `sigmoid + threshold` turns logits into binary masks.
8. `invert_prediction()` maps those masks back to the original CT shape and
   affine.
9. `combine_word_labels()` converts 32 binary masks into one WORD label map.
10. `save_prediction()` writes a NIfTI label map aligned to the original CT.
11. The script prints the saved output path and labels present.

## Revision Note

The scripts are structurally very similar. This consistency is useful, but it
also means there is duplicated code in:

```text
parse_args
make_loader
invert_prediction
save_prediction
checkpoint loading
input validation
```

A future cleanup could move these shared pieces into a common inference utility
module. Then each model-specific script would only need to define:

```text
model architecture
checkpoint loading
logits-to-label conversion
label-space mapping
```
