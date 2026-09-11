# industreal_dev_v9_psr_vitl — the v7 protocol on DINOv2 ViT-L/14 frame features

- Tier: **development result**, identical protocol to
  [`../industreal_dev_v7_psr_latency/`](../industreal_dev_v7_psr_latency/README.md) and
  [`../industreal_dev_v8_psr_vitb/`](../industreal_dev_v8_psr_vitb/README.md) (36 train → 16 val
  recordings, out-of-fold decoder selection, 15 s / 30 s budgets, seeds 0–2, MS-TCN++ at a fixed
  40 epochs). The only change against v8: frozen DINOv2 **ViT-L/14** features (2,048-d
  `[CLS ; mean patch]`) instead of ViT-B/14 (1,536-d). Frozen test split untouched. Not HA-ViD.
- Date: 2026-09-12. Author: kuotunyu. Machine: Windows 11, RTX 4090.
- Purpose: v8 showed that the frame features were the binding constraint (ViT-S → ViT-B moved the
  whole F1-vs-delay front). Does the next size step keep paying?

## 1. Feature identity (`artifacts/features/industreal/dinov2_vitl14_s1/meta.json`, not committed)

| item | value |
|---|---|
| extractor | `facebookresearch/dinov2` via torch.hub, model `dinov2_vitl14`, checkpoint `dinov2_vitl14_pretrain.pth` (1,217,586,395 bytes, SHA-256 `d5383ea8f4877b24…`), frozen; fp16 autocast |
| preprocessing / embedding | as before: every frame at 10 fps, 392×224, ImageNet mean/std, `[CLS ; mean patch]` = 2,048-d float16; 52 recordings, 458 MB |
| cost | 116,968 frames in 1,674 s (≈ 70 fps) — the first backbone that is compute-bound rather than decode-bound on this machine, and it shared the GPU with v10 for part of the run |

## 2. Out-of-fold F1-vs-delay fronts, seed 0 (ViT-L vs ViT-B vs ViT-S)

| head × decoder | fastest point | ≈ 20 s | ≈ 25 s | ≈ 30 s | slowest point |
|---|---|---|---|---|---|
| mstcn_prior_dwell, ViT-L | 12.7 s / 0.553 | 19.0 s / 0.771 | 24.3 s / 0.796 | 30.0 s / 0.799 | 32.5 s / 0.805 |
| mstcn_prior_dwell, ViT-B (v8) | 12.3 s / 0.490 | 19.4 s / 0.761 | 24.7 s / 0.788 | — | 35.8 s / 0.813 |
| mstcn_prior_dwell, ViT-S (v7) | 17.1 s / 0.552 | 20.0 s / 0.662 | 24.3 s / 0.761 | 28.8 s / 0.780 | 33.3 s / 0.793 |
| linear_prior_dwell, ViT-L | 5.0 s / 0.032 | 19.8 s / 0.730 | 24.9 s / 0.826 | 29.3 s / 0.849 | 33.7 s / 0.855 |
| linear_prior_dwell, ViT-B (v8) | 5.3 s / 0.025 | 20.1 s / 0.687 | 25.6 s / 0.745 | 29.0 s / 0.817 | 35.7 s / 0.845 |

The linear head's front keeps moving up (+0.04 to +0.08 F1 at 20–25 s); the MS-TCN++ front is
within 0.01 of ViT-B everywhere — the temporal head had already extracted what the larger
backbone adds.

## 3. Result on val (mean ± std over seeds 0–2; per-seed rows with CIs in `tables.md`)

| run | budget | POS | F1 (system) | mean delay (s) | TP / FP / FN (mean) | ViT-B (v8) |
|---|---|---|---|---|---|---|
| linear_plain_cap15 | 15 s | 0.048 ± 0.000 | 0.360 ± 0.000 | 15.1 ± 0.0 | 92.0 / 358.3 / 7.0 | 0.000 / 0.353 / 18.2 s |
| linear_plain_cap30 | 30 s | 0.203 ± 0.010 | 0.695 ± 0.006 | 37.6 ± 0.4 | 103.7 / 71.3 / 28.7 | 0.127 / 0.679 / 32.3 s |
| linear_prior_dwell_cap15 | 15 s | 0.060 ± 0.001 | 0.387 ± 0.001 | 14.9 ± 0.3 | 95.0 / 340.3 / 7.7 | 0.025 / 0.428 / 18.2 s |
| linear_prior_dwell_cap30 | 30 s | 0.365 ± 0.025 | **0.824 ± 0.006** | 34.3 ± 0.4 | 110.7 / 24.7 / 24.7 | 0.358 / 0.805 / 34.6 s |
| mstcn_plain_cap15 | 15 s | 0.122 ± 0.024 | 0.525 ± 0.061 | 17.3 ± 3.3 | 90.3 / 156.0 / 15.0 | 0.248 / 0.544 / 13.0 s |
| mstcn_plain_cap30 | 30 s | 0.221 ± 0.037 | 0.597 ± 0.023 | 28.0 ± 6.7 | 91.7 / 98.7 / 31.7 | 0.306 / 0.622 / 18.6 s |
| mstcn_prior_dwell_cap15 | 15 s | 0.420 ± 0.082 | 0.655 ± 0.069 | 17.6 ± 3.0 | 89.3 / 79.0 / 22.0 | 0.542 / 0.676 / 12.3 s |
| mstcn_prior_dwell_cap30 | 30 s | 0.580 ± 0.045 | 0.795 ± 0.016 | 30.6 ± 2.5 | 101.3 / 25.3 / 26.3 | **0.625 / 0.814 / 25.1 s** |

Per seed, mstcn_prior_dwell_cap30: seed 0 POS 0.517 [0.408, 0.660] / F1 0.796 [0.744, 0.853] /
33.6 s, seed 1 0.616 / 0.775 / 27.4 s, seed 2 0.607 / 0.815 / 30.9 s. Subsets, seed 0: recordings
without error steps (12) POS 0.592 / F1 0.834 / 26.4 s; with error steps (4) 0.292 / 0.680 / 55.0 s.

Reading — diminishing, head-dependent returns:

- **The linear head still benefits** (F1 0.805 → 0.824 at 30 s, FPs 25.7 → 24.7, FNs 28.0 → 24.7);
  with ViT-L the per-frame probe is now the best detector at the 30 s budget.
- **The MS-TCN++ head does not**: at the 30 s budget it is 0.02 F1 and 0.045 POS *below* its ViT-B
  version and 5 s slower, at the 15 s budget it is 0.02 F1 below and 5 s slower. The out-of-fold
  fronts of the two backbones are almost identical for this head, so the val differences are
  within the seed spread; what is not noise is that the selection picked slower operating points.
- **Cost**: 2.2× the extraction time of ViT-B (compute-bound) and a 1.2 GB checkpoint for at best
  +0.02 F1 on one head. ViT-B/14 remains the default feature for this pipeline; ViT-L is worth
  keeping only if the linear head is the deployment target.
- Seed spread of the MS-TCN++ runs stays ± 0.05–0.08 POS: the head's variance, not the features,
  now limits what a backbone comparison can resolve (see v10 / v11 on epoch selection).

## 4. SOP checks (`sop_checks_mstcn_prior_dwell_cap30_s0.json`)

Against the train-learned precedence graphs, the 30 s MS-TCN++ decoder again flags none of the
16 val recordings (13 neither / 3 only-GT): the three real ordering deviations remain invisible to
a decoder tuned for precision.

## 5. Failure cases (seed 0, 30 s budget, MS-TCN++)

`14_main_2_3` (3 TP / 4 FP / 6 FN), `05_assy_2_2` (4 / 4 / 5) and `26_assy_1_5` (5 / 5 / 0) —
the same three recordings as with ViT-B (v8 section 5).

## 6. Cost

| step | wall time | resources |
|---|---|---|
| `extract-features --model dinov2_vitl14` (52 videos) | 1,674 s | RTX 4090 (shared with v10 for part of the run), 458 MB cache, 1.2 GB checkpoint download |
| `train-psr … --delay-cap 15 --delay-cap 30`, seeds 0–2 | 909 s | RTX 4090 (shared with v11 for part of the run) |

## 7. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1
uv run sop-monitor extract-features --root data/external/industreal --split train --split val --model dinov2_vitl14 --stride 1 --device auto --batch-size 64
# PSR labels as in ../industreal_dev_v5_psr_train/README.md, then:
uv run sop-monitor train-psr --features artifacts/features/industreal/dinov2_vitl14_s1 --psr-dir data/external/industreal/psr --train-split splits/industreal/train.csv --eval-split splits/industreal/val.csv --selection nested --seed 0 --seed 1 --seed 2 --delay-cap 15 --delay-cap 30 --out reports/industreal_dev_v9_psr_vitl --device auto --epochs 300 --mstcn-epochs 40 --n-boot 2000
uv run sop-monitor check-psr-run --run reports/industreal_dev_v9_psr_vitl --run-name mstcn_prior_dwell_cap30_s0 --out sop_checks_mstcn_prior_dwell_cap30_s0.json
uv run sop-monitor reproduce-lite
```

`make psr-latency FEATURES=artifacts/features/industreal/dinov2_vitl14_s1 OUT=reports/industreal_dev_v9_psr_vitl`
wraps the training command. Determinism: not re-run separately (same code path as v6–v8).
