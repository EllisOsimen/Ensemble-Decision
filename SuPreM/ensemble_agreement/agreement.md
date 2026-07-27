# Voting agreement rules

This note describes the two rule-based voting approaches implemented in
`consensus_agreement_mask.py`. Inputs are first converted to the CURVAS target
labels: `0` background, `1` pancreas, `2` kidney, and `3` liver.

STAPLE and random-forest fusion are documented in `README.md`; they are not
direct voting rules.

## Unweighted voting (`legacy` mode)

The reported unweighted method used equal local weights, a `3 x 3 x 3`
neighbourhood, and CLIP U-Net as the final fallback.

| Model votes at a voxel | Current decision | Confidence code |
| --- | --- | --- |
| All three agree, including background | Use the unanimous label | `1` |
| Two agree on a foreground organ | Use the majority organ | `2` |
| Two predict background and one predicts an organ | Keep background, unless that organ component connects to an existing confidence-1/2 region of the same organ; connected voxels are promoted to the organ | `3` |
| All three disagree and local support has a clear winner | Use the label with the strongest equal-weight support in the local neighbourhood | `4` |
| All three disagree and local evidence is insufficient | Use the CLIP U-Net prediction | `5` |

Local support is accepted only when its total support and its lead over the
second-best label are both at least `3`. If CLIP fallback is disabled, the last
case remains label `255` (uncertain) instead.

## Training-derived weighted voting

For each foreground organ, votes are summed using model-specific weights
derived from mean patient-level Dice on the 20 training cases against
`annotation_1.nii.gz`. Background is not scored; it is the default whenever no
foreground candidate is accepted.

| Voting situation | Current decision | Confidence code |
| --- | --- | --- |
| All three agree on a foreground organ | Use that organ as strong unanimous evidence | `1` |
| Any two agree on a foreground organ | Use that organ; the thresholds make every two-model foreground agreement strong | `2` |
| Only one model supports an organ | Accept it only if its weight passes that organ's weak threshold and the candidate component connects to a strong region of the same organ | `3` if accepted; otherwise background `5` |
| Models vote for different organs | Compare the weighted organ scores; a weak winner is accepted only when it passes its threshold and has an unambiguous connection to a strong same-organ region | `3` if accepted; otherwise background `5` |
| All three predict background | Keep background | `5` |

The final weighted testing run used these values:

| Organ | CLIP U-Net | SegResNet | Swin 5050 | Weak threshold | Strong threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pancreas | 0.5754 | 0.7268 | 0.3611 | 0.6511 | 0.8316 |
| Kidney | 0.8688 | 0.9216 | 0.6162 | 0.7425 | 1.2033 |
| Liver | 0.9327 | 0.6068 | 0.8960 | 0.7514 | 1.2178 |

Consequently, a single SegResNet vote can provide weak pancreas evidence;
single CLIP U-Net or SegResNet votes can provide weak kidney evidence; and
single CLIP U-Net or Swin 5050 votes can provide weak liver evidence. Such a
vote still has to connect to a strong region before it is retained. The final
weighted run did not use CLIP fallback, so all unresolved voxels remained
background.
