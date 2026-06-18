This is a project designed for my IPAB summer internship

Supervisor: Eleonora D'Arnese

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
