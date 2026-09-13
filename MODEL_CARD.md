# Model card — sop-video-monitor (development build)

Status on 2026-09-14: every number below is a **development result on a validation split**. No
model has been evaluated on a frozen test split, and no trained weights are published or stored in
this repository. Exact values, intervals and per-video rows live in the linked `reports/`
directories and are recomputed from committed prediction tables by `sop-monitor reproduce-lite`.

## Intended use

Research and portfolio demonstration of a multi-camera SOP-sequence monitor on laboratory assembly
video: per-view temporal action segmentation, SOP checks (order, omission, duration) against
procedure knowledge learned from training annotations, and a queue of suggested deviations for
human review. Outputs are suggestions for a person to review. They are not safety or compliance
decisions, and the system must not stop, reject or grade work on its own.

Out of scope: real factory lines, unseen camera placements, commercial use (the data licences
forbid it), and any claim about error types the datasets do not annotate.

## Components

| component | what it is | where |
|---|---|---|
| frame features | frozen DINOv2 ViT-B/14 (`facebookresearch/dinov2`, `[CLS ; mean patch]`, 1536-d, fp16), weights SHA-256 `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`; not fine-tuned | `src/sop_monitor/features.py` |
| recogniser | MS-TCN++ (Li et al., TPAMI 2020), one network per camera view, causal (left-only padding) or offline; late fusion of the three views' posteriors (mean, geometric, confidence-weighted; nothing tuned); epoch chosen on val F1@10 | `src/sop_monitor/havid_tas.py`, `mstcn.py` |
| SOP knowledge | per-plate precedence graph (strict edges, min support 10), mandatory steps (≥ 90 % of a plate's train recordings), duration windows (1st–99th percentile), all learned from the HA-ViD train split at instruction-sheet step granularity; settings chosen by leave-one-subject-out on train | `sop/ha-vid/sheet/`, `src/sop_monitor/havid_sop.py` |
| deviation review | queue of findings, local review page with synced three-view playback, append-only decisions | `src/sop_monitor/review.py`, `docs/review.md` |
| IndustReal line | procedure-step recognition (PSR) with a causal MS-TCN++ state head and a procedure-prior decoder, used as the metric donor and development testbed | `src/sop_monitor/psr_baseline.py` |

## Data

| dataset | role | licence | split used for the numbers below |
|---|---|---|---|
| HA-ViD (Zheng, Lee, Lu, NeurIPS 2023 D&B) | main | CC BY-NC 4.0 | frozen subject-wise split `splits/ha-vid/`: train 17 subjects / 143 recordings, **val 6 subjects / 18 recordings**, test 7 subjects / 41 recordings (hashed, never read) |
| IndustReal (Schoonbeek et al., WACV 2024) | metric donor, development | Apache-2.0 | participant-disjoint official split, **val** only; test never read |

HA-ViD facts measured on the delivered data (`reports/havid_audit.md`): 203 annotated recordings
× 3 views, h264 1280×720 at 15 fps, frame counts equal to the annotations, 67 native `wrong`
segments of which 16 fall in the test subjects.

## Evaluation (development tier, val)

HA-ViD recogniser, three seeds (`reports/havid_dev_v7_tas_seeds_{lh,rh}`), primitive tasks, mean ±
sample std of the per-seed point estimates:

| run | hand | F1@10 | Edit | MoF |
|---|---|---|---|---|
| causal, mean fusion | left | 30.5 ± 1.4 | 33.6 ± 2.0 | 39.5 ± 1.0 |
| causal, mean fusion | right | 29.9 ± 0.6 | 33.0 ± 0.5 | 34.6 ± 2.3 |
| offline, mean fusion (not an online result) | left | 43.9 ± 2.0 | 41.2 ± 1.1 | 44.2 ± 2.7 |
| offline, mean fusion (not an online result) | right | 42.0 ± 1.6 | 40.1 ± 1.9 | 42.4 ± 2.0 |

HA-ViD SOP checks (`reports/havid_dev_v8_sop_sheet`, seed 0 predictions): on ground-truth step
sequences every synthetic order, omission and duration violation is detected (recall 100 %, Wilson
lower bound 82 %) and 4 / 18 (order), 1 / 18 (omission), 8 / 18 (duration) unperturbed recordings
are flagged; on the recogniser's predicted sequences every recording is flagged. Native `wrong`
segments: 0 of 12 detected (underpowered; ADR 0001 trigger B).

IndustReal PSR (`reports/industreal_dev_v11_psr_epochsel_f1`, three seeds): POS 0.642 ± 0.035,
F1 0.821 ± 0.004, mean delay 23.6 s.

For scale only, not comparable: the HA-ViD paper reports MS-TCN primitive-task F1@10 36.6 / 34.7
(left / right) on its official test subjects with I3D features, one view, offline.

## Known failure modes

- The causal recogniser's most frequent error on every run is a real step predicted as a pause
  (`null`), including the annotated `wrong` steps; it never predicts `wrong`.
- Predicted step sequences are fragmented and miss steps, so SOP checks on them flag every
  recording; smoothing does not recover missed steps (`reports/havid_dev_v6_sop_smoothed`).
- Causality costs ≈ 12–13 F1@10 points against the offline twin; seed noise is 0.3–2.7 points, so
  single-seed differences below ≈ 3 points are not evidence.
- The learned precedence graph encodes the train subjects' orderings; stage-2/3 instruction
  variants and freer stage-1 routes produce order findings that are not errors.

## Licence and provenance

Code: Apache-2.0. HA-ViD derivatives are non-commercial (CC BY-NC 4.0): the cached features, any
trained weights, the learned procedure knowledge in `sop/ha-vid/` and the committed HA-ViD
prediction and step tables under `reports/havid_*`. Raw videos, annotations, feature caches and
weights are not in the repository. Feature caches record the extractor, weights hash, torch version
and GPU in their `meta.json`; every run directory records its configuration and runtime.
