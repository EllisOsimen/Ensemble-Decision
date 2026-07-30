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

## Download The Data And Pretrained Models

Run the following commands from the repository root. The downloads are large:
the compressed CURVAS dataset is approximately 29.4 GB in total and the
compressed TotalSegmentator dataset is approximately 23.6 GB. The `--continue`
option allows an interrupted download to be resumed.

### CURVAS

Download all three splits from the
[CURVAS Zenodo record](https://zenodo.org/records/13767408):

```bash
mkdir -p downloads

wget --continue --output-document downloads/training_set.zip \
  "https://zenodo.org/records/13767408/files/training_set.zip?download=1"
wget --continue --output-document downloads/validation_set.zip \
  "https://zenodo.org/records/13767408/files/validation_set.zip?download=1"
wget --continue --output-document downloads/testing_set.zip \
  "https://zenodo.org/records/13767408/files/testing_set.zip?download=1"

unzip downloads/training_set.zip -d .
unzip downloads/validation_set.zip -d .
unzip downloads/testing_set.zip -d .
```

This produces the `training_set/`, `validation_set/`, and `testing_set/`
directories described below.

### TotalSegmentator

Download TotalSegmentator v2.0.1 from the
[TotalSegmentator Zenodo record](https://zenodo.org/records/10047292):

```bash
mkdir -p downloads Totalsegmentator_dataset

wget --continue \
  --output-document downloads/Totalsegmentator_dataset_v201.zip \
  "https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip?download=1"

unzip downloads/Totalsegmentator_dataset_v201.zip \
  -d Totalsegmentator_dataset
```

### Pretrained Models

Download the three required checkpoints from the
[SuPreM Hugging Face repository](https://huggingface.co/MrGiovanni/SuPreM/tree/main), and store them under SuPreM/pretrained_models:

```bash
mkdir -p SuPreM/pretrained_weights

wget --continue \
  --output-document SuPreM/pretrained_weights/supervised_clip_driven_universal_unet_2100.pth \
  "https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_clip_driven_universal_unet_2100.pth?download=true"
wget --continue \
  --output-document SuPreM/pretrained_weights/supervised_suprem_segresnet_2100.pth \
  "https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_segresnet_2100.pth?download=true"
wget --continue \
  --output-document SuPreM/pretrained_weights/self_supervised_nv_swin_unetr_5050.pt \
  "https://huggingface.co/MrGiovanni/SuPreM/resolve/main/self_supervised_nv_swin_unetr_5050.pt?download=true"
```

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
│   ├── sbatch/
│   │   ├── curvas/                     # active CURVAS pipeline
│   │   ├── totalsegmentator/           # active external-evaluation pipeline
│   │   └── legacy/                     # superseded and diagnostic jobs
│   ├── utils/                          # dataset preparation utilities
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
├── Totalsegmentator_dataset/
│   └── s####/
│       ├── ct.nii.gz
│       └── segmentations/
│           ├── pancreas.nii.gz
│           ├── liver.nii.gz
│           ├── kidney_left.nii.gz
│           ├── kidney_right.nii.gz
│           └── ...                     # remaining TotalSegmentator labels
└── external_dataset/                   # generated fixed 50-case subset
    └── s####/
        ├── ct.nii.gz                   # symlink to source CT
        ├── segmentations/              # symlink to source masks
        └── combined_mask.nii.gz        # target labels 0-3
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

The extracted TotalSegmentator v2.0.1 archive uses one `s####/` directory per
CT scan, with the image stored as `ct.nii.gz` and individual anatomical masks
under `segmentations/`. The preparation job described below creates the
lightweight `external_dataset/` view used by inference and evaluation.

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
├── consensus_testing/
│   ├── unweighted/
│   ├── weighted/
│   └── staple/
└── random_forest_config_002_final/
    └── testing_predictions/
```

Hard-label, confidence, and validity maps use one identically named NIfTI file
per case. Confidence values are stored as scaled `uint8`; validity maps are
binary and identify voxels evaluated by every base model. Do not mix case IDs,
NIfTI grids, or affine transforms between these directories.

### Tested software environment

The reported experiments were run with:

- Python 3.10.20
- PyTorch 2.11.0+cu128
- CUDA 12.8 PyTorch build
- cuDNN 9.19.0
- MONAI 1.5.2
- NumPy 2.2.6
- NiBabel 5.4.2
- SciPy 1.15.3
- scikit-learn 1.7.2

Create the environment with:

```bash
conda env create --name suprem-h200 --file environment.yml
conda activate suprem-h200
```

The environment name is defined in `environment.yml`. On a GPU compute node,
verify the core dependencies and GPU availability after installation:

```bash
python -c "import torch, monai, nibabel, numpy, scipy, sklearn, surface_distance; print(f'PyTorch {torch.__version__}, MONAI {monai.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

`CUDA available: False` is expected if this check is run on a login node that
does not expose a GPU.

`environment.yml` is the dependency source of truth for the experiments in
this README. Do not additionally install `SuPreM/requirements.txt`: that file
is retained for the legacy upstream SuPreM code and pins older, incompatible
versions of PyTorch and MONAI.

## Reproduce The Experiment

The Slurm scripts assume a GPU-capable environment named `suprem-h200`; create
the environment above, then adapt the Slurm
resource directives if your cluster differs. Set `CONDA_ENV` when the
environment has another name. Submit all jobs from `SuPreM/` so Slurm can write
to `slurm_logs/`.

```bash
cd /path/to/Ensemble-Decision/SuPreM
mkdir -p slurm_logs
```

### CURVAS

1. Generate base-model masks, assigned-label confidence maps, and validity
   masks for the 20 training, 5 validation, and 65 testing cases:

   ```bash
   curvas_infer_job=$(sbatch --parsable \
     sbatch/curvas/infer_all_curvas_three_models_with_confidence.sbatch)
   ```

2. Generate the unweighted, weighted, and STAPLE consensus masks for all 65
   testing cases:

   ```bash
   curvas_fusion_job=$(sbatch --parsable \
     --dependency=afterok:${curvas_infer_job} \
     sbatch/curvas/fuse_curvas_three_methods.sbatch)
   ```

3. Reproduce the patient-level five-fold random-forest selection, fit the
   selected `config_002` model to all 20 training cases, and apply it to the 65
   testing cases:

   ```bash
   rf_cv_job=$(sbatch --parsable \
     --dependency=afterok:${curvas_infer_job} \
     sbatch/curvas/random_forest_cross_validation.sbatch)

   rf_train_job=$(sbatch --parsable \
     --dependency=afterok:${rf_cv_job} \
     sbatch/curvas/train_final_random_forest_config_002.sbatch)

   curvas_rf_job=$(sbatch --parsable \
     --dependency=afterok:${rf_train_job} \
     sbatch/curvas/infer_curvas_random_forest.sbatch)
   ```

   Five-fold CV operates at patient level; no patient's voxels appear in both
   the fitting and held-out portions of a fold. To regenerate masks using the
   already recorded `config_002` selection, skip `rf_cv_job` and make
   `rf_train_job` depend directly on `curvas_infer_job`.

4. Evaluate the human-only baseline, the three base models, and all four
   consensus methods with the independent expert annotations:

   ```bash
   curvas_human_eval_job=$(sbatch --parsable \
     sbatch/curvas/evaluate_testing_set_human_agreement_all_65.sbatch)

   curvas_base_eval_job=$(sbatch --parsable \
     --dependency=afterok:${curvas_infer_job} \
     sbatch/curvas/evaluate_base_models_human_annotators.sbatch)

   curvas_eval_job=$(sbatch --parsable \
     --dependency=afterok:${curvas_fusion_job}:${curvas_rf_job} \
     sbatch/curvas/evaluate_curvas_consensus_methods.sbatch)
   ```

5. After evaluation finishes, regenerate the final 64-case tables by excluding
   `UKCHLL082` from the canonical 65-case CSVs:

   ```bash
   cd ..
   python SuPreM/statistics/summarise_agreement_excluding_cases.py
   ```

### TotalSegmentator External Evaluation

The external experiment uses the fixed 50 case IDs recorded in
`sbatch/totalsegmentator/totalsegmentator_external_50.txt`. The preparation
job symlinks their CTs and source segmentations into `external_dataset/` and creates
`combined_mask.nii.gz` in the CURVAS label space. It does not modify the
downloaded TotalSegmentator dataset.

1. Prepare the fixed cohort and generate the reference masks:

   ```bash
   totalseg_prepare_job=$(sbatch --parsable \
     sbatch/totalsegmentator/prepare_totalsegmentator_external_50.sbatch)
   ```

2. Run all three base models on the 50 CTs:

   ```bash
   totalseg_infer_job=$(sbatch --parsable \
     --dependency=afterok:${totalseg_prepare_job} \
     sbatch/totalsegmentator/infer_external_dataset_three_models_with_confidence.sbatch)
   ```

3. Generate the three deterministic consensus masks and apply the CURVAS-fitted
   random forest without retraining it on TotalSegmentator:

   ```bash
   totalseg_fusion_job=$(sbatch --parsable \
     --dependency=afterok:${totalseg_infer_job} \
     sbatch/totalsegmentator/fuse_external_dataset_three_methods.sbatch)

   totalseg_rf_job=$(sbatch --parsable \
     --dependency=afterok:${totalseg_infer_job}:${rf_train_job} \
     sbatch/totalsegmentator/infer_external_dataset_random_forest.sbatch)
   ```

4. Evaluate the three base models and all four consensus methods against the
   combined TotalSegmentator reference:

   ```bash
   totalseg_base_eval_job=$(sbatch --parsable \
     --dependency=afterok:${totalseg_infer_job} \
     sbatch/totalsegmentator/evaluate_external_dataset_base_models.sbatch)

   totalseg_eval_job=$(sbatch --parsable \
     --dependency=afterok:${totalseg_fusion_job}:${totalseg_rf_job} \
     sbatch/totalsegmentator/evaluate_external_dataset_consensus_all_methods.sbatch)
   ```

The CURVAS evaluators write the canonical 65-case per-case CSVs and aggregate
files. The summary command does not rerun inference or evaluation: it removes
`UKCHLL082` from the per-case CSVs and recalculates the reported 64-case means
and standard deviations under `SuPreM/results/results_64_testing_set/`.

## Project Provenance

This repository builds on the upstream SuPreM implementation. The CURVAS
experiment-specific workflow is primarily contained in:

- `SuPreM/inference/`
- `SuPreM/ensemble_agreement/`
- `SuPreM/evaluation/`
- `SuPreM/statistics/`
- `SuPreM/sbatch/`

The remaining `SuPreM/` directories retain code from the SuPreM repository
