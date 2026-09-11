# industreal_dev_v8_psr_vitb — the v7 protocol on DINOv2 ViT-B/14 frame features

- Tier: **development result**, identical protocol to
  [`../industreal_dev_v7_psr_latency/`](../industreal_dev_v7_psr_latency/README.md): 36 train
  recordings → 16 val recordings, decoders selected on out-of-fold train predictions under 15 s
  and 30 s latency budgets, seeds 0–2. The only change is the frozen frame feature: DINOv2
  **ViT-B/14** (1,536-d `[CLS ; mean patch]`) instead of ViT-S/14 (768-d). The frozen test split is
  untouched. Not HA-ViD.
- Date: 2026-09-12. Author: kuotunyu. Machine: Windows 11, RTX 4090.
- Purpose: v7 concluded that latencies below ≈ 20 s need features that separate installed from
  not-installed sooner. This run asks whether a larger frozen backbone already does that.

## 1. Feature identity (`artifacts/features/industreal/dinov2_vitb14_s1/meta.json`, not committed)

| item | value |
|---|---|
| extractor | `facebookresearch/dinov2` via torch.hub, model `dinov2_vitb14`, checkpoint `dinov2_vitb14_pretrain.pth` (346,378,731 bytes, SHA-256 `0b8b82f85de91b42…`), frozen; fp16 autocast |
| preprocessing | as before: every frame at 10 fps, rescale to 392×224, ImageNet mean/std |
| embedding | `[CLS ; mean patch]` = 1,536-d float16; 52 recordings, 344 MB |
| cost | 52 videos (116,968 frames of train + val) in ≈ 13 min on the RTX 4090, decode-bound as before |

Heads, decoders, grids, priors, budgets and seeds are exactly those of v7 (`config.json`).

## 2. Out-of-fold F1-vs-delay fronts, ViT-B/14 vs ViT-S/14 (seed 0)

| head × decoder | fastest point | ≈ 20 s | ≈ 25 s | ≈ 30 s | slowest point |
|---|---|---|---|---|---|
| mstcn_prior_dwell, ViT-B | **12.3 s / F1 0.490** | 19.4 s / 0.761 | 24.7 s / 0.788 | — | 35.8 s / 0.813 |
| mstcn_prior_dwell, ViT-S (v7) | 17.1 s / 0.552 | 20.0 s / 0.662 | 24.3 s / 0.761 | 28.8 s / 0.780 | 33.3 s / 0.793 |
| linear_prior_dwell, ViT-B | 5.3 s / 0.025 | 20.1 s / 0.687 | 25.6 s / 0.745 | 29.0 s / 0.817 | 35.7 s / 0.845 |
| linear_prior_dwell, ViT-S (v7) | 4.7 s / 0.024 | 20.1 s / 0.568 | 25.4 s / 0.700 | 29.7 s / 0.763 | 36.8 s / 0.813 |

The whole front moved up and to the left: at 20 s the temporal head gains +0.10 F1, the linear head
+0.12; the MS-TCN++ floor drops from 17 s to 12 s.

## 3. Result on val (mean ± std over seeds 0–2; per-seed rows with CIs in `tables.md`)

| run | budget | POS | F1 (system) | mean delay (s) | TP / FP / FN (mean) | ViT-S (v7) |
|---|---|---|---|---|---|---|
| linear_plain_cap15 | 15 s | 0.000 ± 0.000 | 0.353 ± 0.002 | 18.2 ± 0.0 | 104.3 / 411.0 / 4.0 | 0.000 / 0.188 / 18.5 s |
| linear_plain_cap30 | 30 s | 0.127 ± 0.007 | 0.679 ± 0.002 | 32.3 ± 0.0 | 107.0 / 91.3 / 20.0 | 0.076 / 0.604 / 31.1 s |
| linear_prior_dwell_cap15 | 15 s | 0.025 ± 0.002 | 0.428 ± 0.002 | 18.2 ± 0.0 | 104.3 / 324.0 / 6.7 | 0.000 / 0.216 / 18.9 s |
| linear_prior_dwell_cap30 | 30 s | 0.358 ± 0.018 | 0.805 ± 0.009 | 34.6 ± 2.0 | 106.7 / 25.7 / 28.0 | 0.276 / 0.703 / 32.0 s |
| mstcn_plain_cap15 | 15 s | 0.248 ± 0.117 | 0.544 ± 0.037 | 13.0 ± 1.3 | 90.3 / 147.0 / 16.0 | 0.137 / 0.414 / 22.8 s |
| mstcn_plain_cap30 | 30 s | 0.306 ± 0.033 | 0.622 ± 0.020 | 18.6 ± 2.3 | 98.0 / 109.3 / 19.7 | 0.185 / 0.558 / 29.3 s |
| mstcn_prior_dwell_cap15 | 15 s | 0.542 ± 0.127 | 0.676 ± 0.039 | **12.3 ± 1.4** | 90.0 / 72.7 / 21.0 | 0.440 / 0.532 / 22.5 s |
| **mstcn_prior_dwell_cap30** | 30 s | **0.625 ± 0.059** | **0.814 ± 0.020** | 25.1 ± 0.9 | 105.3 / 20.0 / 29.3 | 0.529 / 0.742 / 29.4 s |

Per seed, mstcn_prior_dwell_cap30: seed 0 POS 0.635 [0.585, 0.718] / F1 0.833 [0.764, 0.912] /
26.4 s (106 TP / 10 FP / 32 FN); seed 1 0.547 / 0.787 / 24.4 s; seed 2 0.691 / 0.823 / 24.4 s.
Subsets, seed 0: recordings without error steps (12) POS 0.649 / F1 0.829 / 21.5 s; with error
steps (4) 0.594 / 0.845 / 41.2 s.

Reading:

- **The stronger backbone improves every run on every metric.** For the best configuration
  (MS-TCN++ + prior/dwell, 30 s budget) F1 rises 0.742 → 0.814, POS 0.529 → 0.625, and the
  answer comes 4 s sooner; under the 15 s budget the same head now really runs at 12 s with F1
  0.676 instead of stalling at 22 s with F1 0.532. The linear head reaches F1 0.805 at 30 s.
- **The selection transfers.** Out-of-fold delays of the chosen 30 s settings (26.4 s) match the
  val delays (25.1 s) as in v7; POS and F1 on val are within the seed spread of their out-of-fold
  values (0.796 F1 out-of-fold for seed 0 vs 0.833 on val).
- **Seed spread is larger for the MS-TCN++ runs** (POS ± 0.06 at 30 s, ± 0.13 at 15 s) than for
  the linear ones (± 0.02): the temporal head's 40 fixed epochs on 27–36 recordings are still a
  high-variance fit, and the 15 s decoders amplify it.
- Order of magnitude, still not a comparison: the paper's B3 on the *test* split reports POS 0.797,
  F1 0.883, delay 22.4 s with an object-detection-based state detector. This frozen-feature
  pipeline is now within 0.07 F1 and 3 s of that on a different split.

## 4. SOP checks (`sop_checks_<run>.json`; graphs learned from the train labels, v6 report section 6)

| run (seed 0) | GT flagged | pred flagged | both | only GT | only pred |
|---|---|---|---|---|---|
| mstcn_prior_dwell_cap30_s0 | 3 / 16 | 0 / 16 | 0 | 3 | 0 |
| linear_prior_dwell_cap30_s0 | 3 / 16 | 9 / 16 | 3 | 0 | 6 |

Same picture as with ViT-S: the conservative MS-TCN++ decoder never emits the out-of-order
events, the linear decoder catches the three real deviations but also flags six recordings whose
`S18` (rear rear chassis pin) is emitted before its predecessor. Counts, not rates.

## 5. Failure cases (seed 0, 30 s budget)

- `14_main_2_3` (MS-TCN++ 3 TP / 3 FP / 6 FN; linear 4 / 4 / 5): the 11-step maintenance
  recording remains the hardest for both heads — the learned maintenance prior allows only the
  five components that change in the training maintenance recordings.
- `05_assy_2_2` (6 / 2 / 5 for both): the remove-and-reinstall recording; the second install of
  the rear chassis is still missed.
- `26_assy_1_5` is no longer among the worst three for the MS-TCN++ run (it was in v5–v7).

## 6. Cost

| step | wall time | resources |
|---|---|---|
| `extract-features --model dinov2_vitb14` (52 videos) | ≈ 13 min | RTX 4090, ≈ 4.5 GB VRAM, 344 MB cache; 346 MB checkpoint download |
| `train-psr … --delay-cap 15 --delay-cap 30`, seeds 0–2 | 1,091 s | RTX 4090 (1,536-d inputs make the MS-TCN++ fits ≈ 35 % slower than v7) |

## 7. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1
uv run sop-monitor extract-features --root data/external/industreal --split train --split val --model dinov2_vitb14 --stride 1 --device auto --batch-size 128
# PSR labels as in ../industreal_dev_v5_psr_train/README.md, then:
uv run sop-monitor train-psr --features artifacts/features/industreal/dinov2_vitb14_s1 --psr-dir data/external/industreal/psr --train-split splits/industreal/train.csv --eval-split splits/industreal/val.csv --selection nested --seed 0 --seed 1 --seed 2 --delay-cap 15 --delay-cap 30 --out reports/industreal_dev_v8_psr_vitb --device auto --epochs 300 --mstcn-epochs 40 --n-boot 2000
uv run sop-monitor check-psr-run --run reports/industreal_dev_v8_psr_vitb --run-name mstcn_prior_dwell_cap30_s0 --out sop_checks_mstcn_prior_dwell_cap30_s0.json
uv run sop-monitor reproduce-lite
```

`make psr-latency FEATURES=artifacts/features/industreal/dinov2_vitb14_s1 OUT=reports/industreal_dev_v8_psr_vitb`
wraps the training command. Determinism: not re-run separately; the code path is the one whose
seed-0 re-run reproduced every row in v6, with a different feature cache as input.

## 8. What this says

Frozen frame features were the binding constraint, not the head or the decoder: swapping ViT-S
for ViT-B moved the entire F1-vs-delay front. The next feature step on the spec's list is a
clip-level backbone (VideoMAE-V2) or ViT-L/14; the next protocol step is a
multi-seed MS-TCN++ with epoch selection, since its seed spread now dominates the error bars.
Nothing here is an HA-ViD or test-split result.
