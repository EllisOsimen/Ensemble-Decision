# Teaching Nodes

Snapshot taken: 2026-07-05 23:18 GB.

This ranking is for CPU/memory evaluation jobs like
`evaluate_testing_set_human_agreement.sbatch`, which request 4 CPUs and 64 GB
RAM and do not request a GPU.

## Current Best Choice

Use `landonia23` right now.

Job `3530098` was submitted to `landonia23` and is running:

```text
JOBID    PARTITION  NAME             ST  NODELIST
3530098  Teaching   human-agreement  R   landonia23
```

## Ranked Nodes

| Rank | Node | State | CPUs | RAM | Notes |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | `landonia23` | idle, now running this job | 12 | 193 GB | Best practical choice after `damnii12` failed at launch. Plenty for 4 CPU / 64 GB evaluation. |
| 2 | `landonia25` | idle | 12 | 193 GB | Same class as `landonia23`; good fallback for this job. |
| 3 | `damnii12` | idle | 40 | 380 GB | Best on paper, but job `3530096` failed instantly on this node with `RaisedSignal:53`; avoid until it looks healthy. |
| 4 | `damnii10` | mixed | 40 | 380 GB | Strong node if enough CPUs/RAM are free. |
| 5 | `damnii07` | mixed | 40 | 380 GB | Strong node, but currently shared with running jobs. |
| 6 | `damnii08` | mixed | 40 | 380 GB | Strong node, but currently shared with running jobs. |
| 7 | `damnii09` | mixed | 40 | 300 GB | Good capacity, a bit less RAM than other `damnii` nodes. |
| 8 | `opencast` | mixed | 32 | 250 GB | Good CPU/RAM fit, but currently shared. |
| 9 | `landonia03` | mixed | 12 | 193 GB | Usable if free enough, smaller than `damnii`/`opencast`. |
| 10 | `landonia05` | mixed | 12 | 193 GB | Usable if free enough, smaller than `damnii`/`opencast`. |
| 11 | `landonia08` | mixed | 12 | 193 GB | Usable if free enough, smaller than `damnii`/`opencast`. |
| 12 | `landonia11` | mixed@ | 12 | 193 GB | Lower priority choice because of the `@` state marker. |
| 13 | `damnii11` | allocated | 40 | 380 GB | Avoid while fully allocated. |
| 14 | `saxa` | draining | 96 | 206 GB shown by `sinfo` | Avoid. Drain reason: `Kill task failed (JobId=3527293 StepId=0)`. |

## Useful Commands

Check your job:

```bash
squeue -u "$USER"
```

Check Teaching nodes:

```bash
sinfo -N -p Teaching -l
```

Submit this evaluation to a specific good node:

```bash
cd /home/s2347484/Seg/SuPreM
sbatch --chdir=/home/s2347484/Seg/SuPreM --nodelist=landonia25 \
  sbatch/evaluate_testing_set_human_agreement.sbatch \
  results/testing_set_human_plus_weighted_agreement_mask \
  /home/s2347484/Seg/testing_set \
  --annotations annotation_1.nii.gz annotation_2.nii.gz annotation_3.nii.gz agreement_mask.nii.gz
```

Avoid hard-coding `saxa` while it is draining.
