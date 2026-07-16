# Experiment audit issues

This document tracks issues that could change, bias, invalidate, or make the
CURVAS inference and random-forest experiments difficult to reproduce.

Checkboxes indicate whether an issue has been resolved. Do not mark an item as
complete until its verification step has passed.

## Recommended resolution order

1. Generate and use validity masks for zero-filled confidence regions.
2. Audit overlapping sigmoid labels for both CLIP and SegResNet.
3. Regenerate or repair affected inputs, then rerun confidence CV.
4. Make final random-forest training fail on invalid or missing cases.
5. Add complete spatial and label-value validation.
6. Copy the new confidence-CV winner into final training.
7. Resolve the annotation target and final evaluation methodology.

Do not modify confidence or prediction files while cross-validation is running.
The CV script caches held-out evaluation tables and later reloads training
inputs, so changing files mid-run could make the two stages use different data.

## Critical and high-priority issues

### [ ] 1. Confidence outside the preprocessing crop is zero-filled

**Status:** Code fix implemented; validity masks must still be generated for the
existing dataset and confidence CV must be rerun.

`CropForegroundd` removes exterior air before inference. When outputs are
inverted to the original CT grid, the hard mask is correctly filled with label
0, but the confidence map is also filled with confidence 0. These voxels were
not scored by the model, so zero is not the confidence of the assigned
background label.

Relevant code:

- `inference/infer_clip_universal_unet.py`, confidence inversion near line 320
- `inference/infer_suprem_segresnet.py`, confidence inversion near line 326
- `inference/infer_swinunetr_5050.py`, confidence inversion near line 267

Observed zero-confidence fractions in representative cases:

| Split | Case | Zero-confidence voxels |
|---|---|---:|
| training | UKCHLL001 | 19.371% |
| validation | UKCHLL049 | 25.586% |
| testing | UKCHLL003 | 20.117% |

The zero region was identical across the three models, and every sampled
zero-confidence voxel had hard-label background. A genuine assigned CLIP or
SegResNet confidence cannot be below 0.5 with the current threshold. A maximum
over 14 Swin softmax classes cannot be zero. Therefore these zeros are padding,
not model scores.

**Impact:** The random forest can learn a spurious `all confidences = 0`
indicator for the exterior crop instead of learning only model confidence.

**Implemented resolution:**

- Each inference script can now save a binary validity map by inverting an
  all-ones cropped-grid mask: 1=evaluated and 0=crop exterior.
- If hard-label and confidence outputs already exist, the scripts reconstruct a
  missing validity mask without loading or running the neural network.
- RF fitting samples only common-valid voxels.
- Final RF inference bypasses invalid voxels and assigns background directly.
- CV bypasses invalid voxels in the same way while counting any invalid human
  foreground as false negatives.

**Remaining work:**

- Generate all three validity masks for every case.
- Run the full dataset integrity audit.
- Keep the current confidence CV result only as a diagnostic baseline.
- Rerun confidence CV using the validity-enabled wrapper.

**Verification:**

- Every validity map is binary and retains the corresponding label-map shape
  and affine.
- The three model validity masks match exactly for each case.
- Human foreground outside validity is reported rather than silently excluded.
- The deterministic-background bypass policy is recorded in run metadata.

### [ ] 2. Multi-positive sigmoid labels are resolved by overwrite order

**Status:** Code fixed and synthetic overlap behavior verified; existing CLIP
and SegResNet predictions must still be regenerated.

CLIP and SegResNet have independent sigmoid channels. When several grouped WORD
labels exceed 0.5 at the same voxel, the old loops let the later WORD label
overwrite the earlier label, regardless of which score was higher.

Relevant code:

- `inference/infer_clip_universal_unet.py`, assignment loop near line 207
- `inference/infer_suprem_segresnet.py`, assignment loop near line 214

For example, pancreas at 0.90 can be replaced by a later WORD class at 0.55.
The saved confidence is then also changed to 0.55.

**Impact:** Hard labels and assigned confidences can depend on dictionary order
rather than model evidence.

**Implemented resolution:**

- CLIP and SegResNet now select the supported WORD label with the highest
  sigmoid score and save that same winning score as confidence.
- Grouped labels such as adrenal first take the maximum of their mapped model
  channels.
- Exact equal-score ties retain the first WORD label deterministically.
- The implementation streams the current maximum rather than allocating a
  second 16-channel volume.

**Remaining work:**

- Regenerate all affected CLIP and SegResNet predictions and confidence maps.
- Complete the CLIP overlap audit and extend it to SegResNet to quantify how
  many saved voxels changed.
- Rerun confidence CV and final training after regeneration.

**Verification:** Synthetic cases passed for earlier-label wins, later-label
wins, background, the inclusive 0.5 threshold, grouped adrenal channels, and
deterministic exact ties. Full completion still requires the regenerated real
maps and overlap audit.

### [ ] 3. Weighted consensus uses testing-set-derived weights

**Status:** Confirmed; affects the older weighted-consensus path, not the current
random-forest path.

`ensemble_agreement/consensus_agreement_mask.py` contains default organ weights
that exactly match per-organ Dice scores in
`results/CURVAS_EVAL_65/all_models_per_class_summary.csv`. Those scores were
calculated on the 65 testing cases.

**Impact:** Using those weights to construct a weighted consensus and then
reporting performance on the same 65 testing cases leaks test information and
creates an optimistic estimate.

**Resolution:** Estimate weights and thresholds using training CV or the
validation split only. Freeze them before running a single final test-set
evaluation. Do not use the existing testing-derived defaults for a reported
test result.

**Verification:** The final report records the cases used to estimate every
weight and confirms that none are final testing cases.

## Random-forest implementation issues

### [x] 4. Final training silently skips invalid patients

**Status:** Resolved and fail-fast behavior verified.

`collect_training_samples()` in
`ensemble_agreement/random_forest_stacking_consensus.py` catches loading errors
and shape mismatches and continues even when `--skip-missing` was not requested.
A final run can therefore finish successfully using fewer patients than
intended.

**Resolution:** Loading errors, shape mismatches, and empty samples now raise by
default. They are skipped only when `--skip-missing` is explicitly enabled.
For the official experiment, require exactly the expected 20 training patients
and 5 validation patients.

**Verification:** A synthetic missing-case check confirmed that the default run
raises `FileNotFoundError` rather than fitting a model with that case omitted.

### [x] 5. Hard-mask affine validation is incomplete

**Status:** Resolved and deliberate affine-mismatch failure verified.

Previously, `load_case_predictions()` checked hard-mask shapes but not their affines. A
same-shaped shifted or flipped mask could be flattened and paired voxelwise
without detection. Confidence affines were checked only against the first hard
prediction, and training targets were checked by shape only.

**Resolution:** All hard predictions must now match the first prediction's
shape and affine. Confidence maps, validity masks, and the target annotation
must match that same reference grid. Affines use the evaluation tolerances
`rtol=1e-5` and `atol=1e-5`.

**Verification:** A same-shaped prediction with a deliberately shifted affine
was rejected with a case-, model-, and file-specific error before sampling.

### [x] 6. Unknown prediction labels can silently become background

**Status:** Resolved and unexpected source/target labels were rejected in
focused tests.

Previously, `map_labels()` initialized the remapped output as background and
only replaced known source IDs. An unexpected label such as 255 was consequently
converted to background instead of raising an error.

**Resolution:** Native prediction values are validated against the complete
configured BTCV, SuPreM/WORD, or CURVAS vocabulary before remapping. Both CV
and final training now load targets through the same validator and allow only
CURVAS labels 0, 1, 2, and 3.

**Verification:** Native model label 255 and target label 4 both stopped the
run with case- and file-specific errors.

### [ ] 7. Final-training hyperparameters are fixed manually

**Status:** Waiting for confidence CV.

`sbatch/random_forest_stacking_consensus.sbatch` currently hard-codes:

```text
n_estimators=300
max_depth=5
min_samples_leaf=1
max_features=sqrt
```

These are the previous label-only CV winner. Confidence features can produce a
different winner.

**Resolution:** Do not run final training until the confidence CV has completed.
Either update the wrapper from the new `best_parameters.json`, or add an option
that reads and validates that file automatically.

**Verification:** Final model metadata exactly matches the selected confidence
CV configuration.

### [ ] 8. Only annotation 1 is used as the target

**Status:** Methodological decision required.

Training, validation, and testing cases each contain `annotation_1`,
`annotation_2`, and `annotation_3`, but CV and final training use only
`annotation_1.nii.gz`.

**Impact:** The model learns one annotator rather than a human consensus. This is
valid only if annotator 1 was selected in advance and the experiment is
described as annotator-specific.

**Resolution:** Predefine one of the following before final testing:

- train against a documented human-consensus target;
- repeat training/evaluation for each annotator; or
- retain annotation 1 but clearly report the rater-specific target and evaluate
  sensitivity against annotations 2 and 3.

## Output-integrity and reproducibility issues

### [ ] 9. File existence is treated as successful inference

The inference scripts and Slurm array skip a case when expected paths exist.
They do not validate file size, NIfTI readability, shape, affine, label range,
or confidence encoding before skipping. A truncated output from a killed job can
therefore be treated as complete.

**Resolution:** Write outputs atomically through temporary files, then rename
after a successful save. Before skipping, perform a lightweight integrity check
on every requested output.

### [ ] 10. Evaluation can silently report a partial case set

`evaluation/evaluate_testing_set_prediction_dirs.py` warns about missing
predictions but evaluates the intersection and exits successfully.

**Resolution:** Add strict case matching as the default for official results,
with an explicit option for exploratory partial evaluation. Record expected and
evaluated case IDs in every report.

### [ ] 11. CV overwrite can leave mixed-generation reports

With `--overwrite`, the CV output directory is not cleared or versioned. New
`fold_manifest.json` and `run_config.json` are written near the start, while
metrics and `best_parameters.json` are written only after all fits complete. A
failed rerun can leave new configuration files beside old final results.

**Resolution:** Use a unique output directory per run or write into a temporary
run directory and atomically mark it complete at the end. Include a completion
marker and code/checkpoint hashes.

### [ ] 12. Duplicate Slurm audit submissions can race

`sbatch/audit_clip_label_overlap.sbatch` defaults to one fixed output JSON path.
Submitting it twice creates two jobs that may both write that path.

**Resolution:** Keep only one duplicate submission, or pass a unique output path
to each intentionally different audit. Consider using the Slurm job ID in the
default filename.

### [ ] 13. Audit jobs are unnecessarily restricted to one node

The overlap-audit wrapper requests:

```text
partition=Teaching
nodelist=saxa
gpu=1
cpus=4
memory=128G
```

Pinning to `saxa` prevents the scheduler from using another suitable Teaching
GPU node. A simultaneous RF job already reserves 256 GB on `saxa`, which may
leave insufficient memory or conflict with reservations.

**Resolution:** If the environment and filesystem are available on other
Teaching GPU nodes, remove `--nodelist=saxa`. After measuring peak memory from a
successful audit, reduce the memory request only if the measured margin is safe.

### [ ] 14. Stale documentation excludes UKCHLL007

`evaluation/README.md` describes UKCHLL007 as a known bad-geometry case and
shows it being excluded. A current audit found exact matching CT/annotation
affines for all three UKCHLL007 annotations.

**Resolution:** Remove the stale exclusion instruction after confirming the
dataset correction is intentional and permanent.

## Checks already passed

- [x] All 90 CURVAS cases were discovered: 20 training, 5 validation, 65 testing.
- [x] All 540 expected hard-label and confidence files are present.
- [x] All 540 output headers match their source CT shape and affine.
- [x] All 270 human annotations match their CT grid.
- [x] All 270 confidence headers use scaled uint8 with slope `1/255`.
- [x] Sigmoid is appropriate for the independent CLIP/SegResNet channels.
- [x] Softmax is appropriate for the mutually exclusive 14-class Swin output.
- [x] BTCV, WORD/SuPreM, and CURVAS label mappings are consistent.
- [x] The SegResNet checkpoint loads strictly by matching parameter names.
- [x] CV creates patient-level folds before voxel sampling.
- [x] Confidence feature-key packing round-trips to the expected 15 features.
- [x] Existing unit tests and Python/Slurm syntax checks passed.

## Final experiment gate

Do not report final test-set performance until all of the following are true:

- [ ] Confidence padding policy is fixed and documented.
- [ ] CLIP and SegResNet overlap audits are complete.
- [ ] Any changed base predictions are regenerated or repaired consistently.
- [ ] Confidence CV has been rerun on the final inputs.
- [ ] Final hyperparameters exactly match the new CV winner.
- [ ] Final training uses every intended patient and passes strict grid checks.
- [ ] The human-annotation target strategy is frozen.
- [ ] The test set has not contributed to tuning, weighting, or model selection.
- [ ] Evaluation confirms the complete expected case set.
