# industreal_dev_v1 — first video baseline (development result, validation split)

- Tier: **development result** (see `../README.md`). Real videos, real model outputs, evaluated
  on the IndustReal *validation* split, which was also used to select hyper-parameters. Not a
  headline number; the frozen test split (`splits/industreal/test.csv`, SHA-256 committed) was
  never read.
- Date: 2026-09-11. Author: kuotunyu. Machine: Windows 11, RTX 4090 (24 GB), single GPU.
- Role of the dataset: IndustReal is the project's metric-donor / development dataset
  (ADR 0001). Nothing here is an HA-ViD result and nothing here enters an HA-ViD table.

## 1. Data identity (from `data_audit.json`, all measured on disk)

| item | value |
|---|---|
| dataset | IndustReal (Schoonbeek et al., WACV 2024), Apache-2.0 code + data, local copy under `data/external/industreal/` |
| videos | 86 RGB `.mp4`, all `mpeg4`, 1280×720, 10 fps, frame counts from container headers; 211,931 frames = 5.89 h; 84 videos carry action labels (`15_main_0_1`, `21_assy_1_1` do not) |
| labels | official action-recognition CSVs: train 3,667 rows / 36 videos / 12 participants; val 1,928 / 16 / 5; test 3,678 / 32 / 10; 74 action ids over the whole set (72 seen in train) |
| frame semantics | label frame index = 0-based index into the 10 fps stream (`000027.jpg` → frame 27); every segment ends at or before the probed frame count (last label / frame count between 0.950 and 1.000); segments treated as half-open `[start, end)` because consecutive segments share boundary frames |
| coverage | labelled frames: train 76 %, val 80 %, test 78 % of all frames; the rest is background (`-1`) |
| overlaps | frames covered by two or more segments: train 9.0 %, val 15.7 %, test 10.9 % of labelled frames, dominated by `check_instruction` running concurrently with a manipulation; resolved to the segment with the latest onset (`sop_monitor.industreal.frame_labels`) |
| split | committed `splits/industreal/` regenerated from the local CSVs this round; byte-identical to the committed files, test list SHA-256 `5f119fed…73b1a7` |
| files | `data/manifest.json` lists the zips and CSVs with sizes and SHA-256 (no links, no content) |

HA-ViD status (same audit): `data/external/ha-vid-public/` contains 3 OWL precedence graphs and
7 instruction PDFs — **0 videos, 0 HR-SAT annotation files**. The main dataset is blocked on the
request-form delivery; nothing HA-ViD-related was trained or evaluated.

## 2. Model identity (from `config.json` and the feature cache `meta.json`)

| component | what exactly |
|---|---|
| frame features | `facebookresearch/dinov2` via `torch.hub`, model `dinov2_vits14`, checkpoint `dinov2_vits14_pretrain.pth` (88,283,115 bytes, SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`), frozen |
| preprocessing | every frame (stride 1, i.e. 10 fps), swscale rescale to 392×224 (both multiples of the 14-px patch, ≈16:9), ImageNet mean/std |
| embedding | `[CLS ; mean of patch tokens]` = 768-d, float16, fp16 autocast on CUDA; cached per video under `artifacts/features/industreal/dinov2_vits14_s1/` (not committed, 173 MB) |
| head | multinomial logistic regression on standardised features (train mean/std), full-batch Adam, lr 5e-3, 300 epochs, seed 0, 73 outputs = 72 train action ids + background; torch 2.13.0+cu130 |
| selection (val only) | weight decay ∈ {1e-5, 1e-4, 1e-3} → **1e-3** (val MoF 30.3 / 30.6 / 32.4); causal window ∈ {1, 5, 10, 20, 40} frames → **5** (32.4 / 34.0 / 32.5 / 29.0 / 25.7); centered radius ∈ {0, 2, 5, 10, 20} → **5** (32.4 / 35.0 / 35.7 / 34.5 / 31.7) |
| runs | `frame`: argmax of the per-frame posterior; `causal`: argmax of the mean posterior over the last 5 frames (0.5 s, past only — the honest online baseline); `offline`: argmax over ±5 frames (uses future frames — **not** an online result); `majority`: constant most frequent training label (background) |
| determinism | a second `train-baseline` run on the same GPU reproduced `predictions_val.csv` byte-for-byte and identical `metrics.json` |

## 3. Metric definitions

Implemented in `sop_monitor.metrics.offline` and cross-checked, on this very prediction table,
against an independent transcription of the MS-TCN reference `eval.py`
(`sop_monitor.metrics.reference_mstcn`); `score-predictions` raises if they disagree.

- **MoF**: frame-wise accuracy over all val frames including background, micro-averaged.
- **Edit**: `100 · (1 − Levenshtein / max length)` on the run-length-collapsed label sequences with background segments removed, averaged over videos.
- **F1@k**: segmental F1 at IoU ≥ k/100 (k = 10, 25, 50), background segments removed, TP/FP/FN summed over videos before precision/recall.
- **95 % CI**: percentile bootstrap resampling the **5 val participants** with replacement, 2,000 draws, seed 0. With only 5 resampling units the intervals are coarse; they are reported, not interpreted.

## 4. Result (val, 16 videos, 5 participants, 38,036 frames at 10 fps)

`tables.md` is the authoritative rendering (regenerated and checked by `sop-monitor reproduce-lite`).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority (trivial floor) | 20.0 [15.5, 25.3] | 0.0 | 0.0 | 0.0 | 0.0 |
| frame (no context) | 32.4 [31.7, 33.0] | 11.0 [10.1, 11.8] | 11.9 [11.1, 12.8] | 7.5 [6.8, 8.3] | 4.1 [3.3, 5.0] |
| **causal** (online baseline) | 34.0 [32.7, 35.0] | 23.1 [21.6, 24.9] | 24.1 [22.2, 26.3] | 18.9 [17.1, 20.9] | 10.6 [8.9, 12.6] |
| offline (uses future) | 35.7 [34.3, 36.9] | 27.6 [25.5, 30.0] | 29.2 [27.0, 31.6] | 24.5 [22.0, 27.3] | 15.4 [13.2, 17.8] |

Reading: the per-frame linear probe beats the majority floor by 12 MoF points but is far from
usable; 0.5 s of causal averaging doubles Edit/F1 by removing single-frame flicker, and looking
0.5 s into the future adds a further 2–5 points, which is the price of causality at this stage.
Diagnostics that are not part of the table: the selected head reaches 62.0 MoF on its own
training frames, so the val gap is cross-participant generalisation, not underfitting (1,000
epochs → 63.8 train / 33.0 val). Val contains 65 frames (0.17 %) of two action ids
(`take_small_screw_pin`, `pull_small_screw_pin`) that never occur in train and can never be
predicted.

## 5. Failure cases (causal run; ids resolved with the label CSVs)

- The largest single error mass is **background ↔ `check_instruction`** (1,084 + 503 frames): the
  operator's glance at the instruction sheet looks like idle time from a single global frame
  embedding, and the overlap rule assigns it the foreground when it starts mid-manipulation.
- **Fine-grained pairs on the same object** are swapped: `fit_nut` → `tighten_nut` (342),
  `check_instruction` → `browse_instruction` (357), `align_objects` → `check_instruction` (308),
  `plug_short_pin` → `check_instruction` (258). A 392×224 global embedding cannot resolve which
  small part is in the hand.
- **Background predicted as an action** (`tighten_nut` 502, `take_pin_short` 311,
  `take_objects` 288, `align_objects` 265): the head has no notion of "nothing is happening".
- Worst videos (causal MoF / Edit / F1@50): `20_assy_0_1` 28.0 / 20.6 / 9.9 (below its own
  majority floor of 31.2 — the video is 31 % background), `14_main_0_1` 28.7 / 18.4 / 6.2,
  `26_assy_0_1` 29.6 / 22.8 / 9.1. Best: `26_main_0_1` 39.5, `20_assy_3_6` 37.8. Per-video
  numbers for every run are in `tables.md` / `metrics.json["per_video"]`.

## 6. Cost

| step | wall time | resources |
|---|---|---|
| `audit-industreal` (probe 86 videos, hash 6.2 GB of zips/CSVs) | ≈ 1 min | CPU |
| `extract-features` train + val (52 videos, 116,968 frames) | 563 s (≈ 208 fps, decode-bound; DINOv2-S alone runs ≈ 2,500 img/s) | RTX 4090, ≈ 4 GB VRAM, 173 MB cache |
| `train-baseline` (3 heads, smoothing grid, 4 × 5 bootstraps) | 20 s | RTX 4090 (CPU works, slower) |
| one-off downloads | DINOv2 repo + 88 MB checkpoint from `dl.fbaipublicfiles.com`; torch/torchvision/PyAV wheels came from the local uv cache | — |

## 7. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1   # the checkout path contains non-ASCII characters
uv run sop-monitor audit-industreal --root data/external/industreal --out reports/industreal_dev_v1/data_audit.json --manifest data/manifest.json
uv run sop-monitor extract-features --root data/external/industreal --split train --split val --model dinov2_vits14 --stride 1 --device auto
uv run sop-monitor train-baseline --features artifacts/features/industreal/dinov2_vits14_s1 --labels-dir data/external/industreal/labels --out reports/industreal_dev_v1 --device auto --epochs 300 --n-boot 2000 --seed 0
uv run sop-monitor score-predictions --run reports/industreal_dev_v1      # recompute metrics.json from predictions_val.csv, no data/GPU needed
uv run sop-monitor reproduce-lite                                          # same for every committed run + tables.md check (CI)
```

`make audit`, `make features`, `make baseline` and `make reproduce-lite` wrap the same commands.

## 8. What this is not, and what blocks the next step

- Not an HA-ViD result, not a test-split result, not an online SOP result: the IndustReal online
  metrics (POS, completion F1, detection delay) need the procedure-step-recognition (PSR) labels,
  which are not in the local copy (only the action-recognition labels are). No SOP graph exists
  for IndustReal in this repo, so `check-sop` was not applied to these predictions.
- The official IndustReal action-recognition weights (MViT / SlowFast, `action_recognition_model_weights.zip`)
  are on disk but unused: they are clip-level PySlowFast checkpoints and would need that code base.
- HA-ViD main line: blocked until the request-form delivery (videos + HR-SAT annotations) lands;
  the three precedence graphs are already exported and checked (`sop/ha-vid/*.json`).
