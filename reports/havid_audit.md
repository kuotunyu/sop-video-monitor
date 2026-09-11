# HA-ViD data audit (W1)

Measured on the archives delivered by the authors on 2026-09-12 (Dropbox folder, CC BY-NC 4.0).
Every number below is read from [`havid_audit.json`](havid_audit.json), produced by

```bash
uv run sop-monitor audit-havid \
  --temporal data/external/ha-vid/HAViD_temporalAnnotation.zip \
  --official data/external/ha-vid/ActionSegmentation_data.zip \
  --rgb-dir data/external/ha-vid/HAViD_rgb \
  --out reports/havid_audit.json
```

The download link and password are not recorded anywhere in this repository.

## What arrived

| archive | size | content | on disk |
|---|---|---|---|
| `HAViD_temporalAnnotation.zip` | 2.55 MB | 12 sub-datasets × 203 recordings: `temporal_timestamps` + `collaboration_timestamps` | yes |
| `ActionSegmentation_data.zip` (folder download) | 12.9 GB | official `groundTruth`/`splits`/`mapping.txt` per sub-dataset + 630 I3D feature files (`2048 × T`, float64) | yes |
| `HAViD_rgb.zip` | 29.6 GB | `assembly_dataset_mp4_blurred/s01..s30/*.mp4` (all 3 222 videos, faces blurred) | yes |
| `How to read the file names.txt` | 1 KB | id scheme | yes |
| `HAViD_depth.zip`, `HAViD_skeleton.zip`, `ObjectDetection/`, `ActionRecognition/`, `PretrainedCheckpoints/` | 3.7 GB, 1.4 GB, 110 GB, 0.7 GB+, ? | depth, skeletons, CVAT boxes, clip datasets, checkpoints | not downloaded (outside scope, see README non-goals) |

## Annotation format (measured, not assumed)

- Ids: `S<subject>A<attempt>I<stage><version><camera>`; camera `M0` = side (view 0), `S1` = front
  (view 1), `S2` = top (view 2). Stage digit `0` means stage 1 (`I01`), `2` stage 2 (`I21`–`I26`),
  `3` stage 3 (`I31`, `I32`).
- `temporal_timestamps` rows are `start end label` with **inclusive** 0-based frame indices,
  contiguous from frame 0 with no gaps or overlaps (all 203 × 12 files). The left/right hand and
  primitive-task/atomic-action files of a recording end on the same frame.
- The three views ship **byte-identical** annotation files: the annotation belongs to the
  recording, not to a camera.
- Labels are HR-SAT abbreviations `<verb><manipulated object><target object><tool>` (paper
  Figure 9): verbs `a` approach, `d` disassemble, `g` grasp, `h` hold, `i` insert, `l` slide,
  `m` move, `p` place, `r` rotate, `s` screw. Primitive tasks use only `i/p/r/s/l`.
- Noise labels: `null` = pause, `w` = the paper's **wrong** (error). `w` occurs only at
  primitive-task level; the atomic-action files describe the erroneous motion with ordinary verbs.
- `collaboration_timestamps` rows are `start end status lh_label rh_label` with statuses
  `collaboration`, `parallel`, `single_handed_left`, `single_handed_right`, `pause`; they cover
  every recording exactly.

## Annotation statistics

| | value |
|---|---|
| annotated recordings / subjects | 203 / 30 (17 subjects with 3 recordings, 13 with 13–18) |
| recordings per stage 1 / 2 / 3 | 90 / 67 / 46 |
| frames (per view) | 251 189; per recording min 450, median 1 134, max 2 880 |
| primitive-task classes (lh / rh) | 71 / 71 in the files; `mapping.txt` lists 75 |
| atomic-action classes (lh / rh) | 203 / 203 in the files; `mapping.txt` lists 219 |
| primitive-task segments (lh / rh) | 3 200 / 3 077, median 49 / 53 frames |
| atomic-action segments (lh / rh) | 11 158 / 10 154, median 9 / 10 frames |
| `null` frames, pt (lh / rh) | 87 632 / 93 930 (35 % / 37 %) |
| collaboration frames | collaboration 99 487, parallel 24 163, single-handed L/R 39 907 / 33 609, pause 54 023 |

## The `wrong` label (ADR 0001 trigger B)

| | value |
|---|---|
| `w` segments / frames | **67 / 3 905** (1.6 % of frames) |
| by hand | lh 37, rh 30; 12 segments identical on both hands (two-handed errors) |
| recordings with at least one `w` | 25 of 203 |
| subjects with at least one `w` | 17 of 30 |
| by stage 1 / 2 / 3 | 54 / 8 / 5 |
| segment length | min 4, median 51, max 173 frames |
| inside the official test subjects | **16 segments in 7 recordings** (of 41 test recordings) |

Trigger B in [`../docs/decisions/0001-dataset-and-licences.md`](../docs/decisions/0001-dataset-and-licences.md)
fires when the held-out test subjects contain fewer than 20 `wrong` segments: they contain 16.
Consequence (unchanged from the ADR): HA-ViD stays the main dataset, the native error table is
reported with exact counts and Wilson 95 % intervals and labelled *underpowered*, and the headline
violation metrics come from the synthetic order-violation table, marked as synthetic.

## Official action-segmentation split

- `splits/{train,test}.split1.bundle` are identical across the 12 sub-datasets up to the camera
  suffix: 161 train + 41 test videos per view (123 test videos over three views, as in the paper).
- The split is **by subject**: train S01, S03, S06–S18, S20, S21, S23, S25, S27–S30 (23 subjects);
  test S02, S04, S05, S19, S22, S24, S26 (7 subjects). No subject appears in both.
- One annotated recording, `S02A08I21` (a test subject), is in **neither** bundle; it is kept out
  of both our development and frozen test sets.
- `groundTruth/<video>.txt` equals the temporal file expanded to one label per frame and trimmed
  by **5 frames at each end** (2 436 of 2 436 files); the I3D feature files have exactly that
  length (`T = frames − 10`). Feature files exist for all 609 annotated videos plus 21 videos
  (7 recordings) that have no temporal annotation.
- `mapping.txt` covers every label that occurs in the files.

## Videos

| | value |
|---|---|
| mp4s on disk | 3 222 (all 30 subjects, 108 per subject) |
| annotated videos found | 609 = 203 recordings × 3 views; **every recording has all three views** |
| stream | h264, 1280×720, **15 fps** (all 609; frame count from the container header) |
| video frames − label frames | 0 for all 609 videos: the temporal files cover every frame exactly |
| annotated footage | 13.95 h (4.65 h per view) |

Consequences for the pipeline: frame index *t* of the annotation is frame *t* of the mp4 in
every view (no offset, unlike IndustReal's `11_assy_0_1`); the three views are frame-synchronous
by construction of the dataset (same frame counts, same annotation), which the multi-camera line
relies on; the official benchmark files drop 5 frames at each end, so any comparison with the
paper's numbers must apply the same trim.

## Problems found

None: the audit's `problems` list is empty for annotations, official split and videos.

## Frozen split (`splits/ha-vid/`)

Frozen by `make freeze-havid` from the official bundles (decision in
[`../docs/decisions.md`](../docs/decisions.md)): test = the 7 official test subjects (41
recordings, 123 videos, SHA-256 in `test_sha256.txt`); val = 6 subjects drawn with seed 0 from the
train subjects with three recordings (S01, S08, S10, S12, S18, S30; 18 recordings, 54 videos);
train = the remaining 17 subjects (143 recordings, 429 videos); `S02A08I21` excluded. CI verifies
the test hash.

## What this audit does not show

No model has been run on HA-ViD; there are no HA-ViD metrics. The synthetic order-violation
generator, the multi-view features and the HA-ViD baselines are still to be built (W2–W3).
