# Teaching Nodes

Snapshot source: `sinfo -N -p Teaching -o "%N|%t|%c|%m|%G"` on
2026-07-05.

This ranks the Teaching nodes as if they are all healthy and available. The
ranking is mainly for CPU/memory evaluation jobs in this repo, with GPU
hardware included as useful context. For GPU-heavy model training, prefer newer
GPU hardware more strongly than this table does.

## Normal-Health Ranking

| Rank | Node | CPUs | RAM | GPUs | Why |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `saxa` | 96 | 2,060 GB | H200 resources: `gpu:h200:1`, `gpu:h200_3g.71gb:4`, `gpu:h200_1g.18gb:35` | Best hardware overall by a mile: most CPU, most RAM, and newest GPU class. |
| 2 | `damnii07` | 40 | 380 GB | 8 x RTX 2080 Ti | Strong CPU/RAM node; good default for CPU/memory evaluation if healthy. |
| 3 | `damnii08` | 40 | 380 GB | 8 x RTX 2080 Ti | Same class as `damnii07`; strong CPU/RAM node. |
| 4 | `damnii10` | 40 | 380 GB | 8 x RTX 2080 Ti | Same class as `damnii07`; strong CPU/RAM node. |
| 5 | `damnii11` | 40 | 380 GB | 8 x RTX 2080 Ti | Same class as `damnii07`; strong CPU/RAM node. |
| 6 | `damnii12` | 40 | 380 GB | 8 x RTX 2080 Ti | Same class as `damnii07`; strong CPU/RAM node. |
| 7 | `damnii09` | 40 | 300 GB | 8 x RTX 2080 Ti | Same CPU/GPU class as other `damnii` nodes, but less RAM. |
| 8 | `opencast` | 32 | 250 GB | 2 x RTX 2080 Ti | Solid CPU/RAM fit, but fewer GPUs and less RAM than `damnii`. |
| 9 | `landonia11` | 12 | 193 GB | 8 x RTX A6000 | Best non-`saxa` GPU hardware, but much smaller CPU/RAM than `damnii`. Great GPU choice if healthy. |
| 10 | `landonia03` | 12 | 193 GB | 8 x RTX 2080 Ti | Smaller CPU/RAM node; fine for lightweight evaluation. |
| 11 | `landonia05` | 12 | 193 GB | 8 x RTX 2080 Ti | Same class as `landonia03`. |
| 12 | `landonia08` | 12 | 193 GB | 8 x RTX 2080 Ti | Same class as `landonia03`. |
| 13 | `landonia23` | 12 | 193 GB | 8 x RTX 2080 Ti | Same class as `landonia03`; currently used successfully for the weighted-mask evaluation. |
| 14 | `landonia25` | 12 | 193 GB | 8 x RTX 2080 Ti | Same class as `landonia03`. |

## Quick Choice Guide

For CPU/memory evaluation jobs:

1. Use `saxa` if healthy and not draining.
2. Otherwise use any healthy `damnii` node with enough free memory.
3. Otherwise use `opencast`.
4. Otherwise use an idle `landonia` node.

For GPU-heavy jobs:

1. Use `saxa` if you need H200-class GPU resources.
2. Use `landonia11` if A6000 GPUs are enough and the node is healthy.
3. Use `damnii*` or other `landonia*` nodes for RTX 2080 Ti jobs.

## Current Caveat From This Session

On 2026-07-05, `saxa` was draining with:

```text
Kill task failed (JobId=3527293 StepId=0)
```

Also, a test submission to `damnii12` failed immediately with
`RaisedSignal:53`. The evaluation job was then submitted to `landonia23` and
started successfully as job `3530098`.

Always check current state before pinning a job:

```bash
sinfo -N -p Teaching -l
squeue -u "$USER"
```
