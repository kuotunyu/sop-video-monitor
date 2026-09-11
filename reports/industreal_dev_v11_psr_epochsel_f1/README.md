# industreal_dev_v11_psr_epochsel_f1 — MS-TCN++ training length chosen by out-of-fold decoded F1

- Tier: **development result**. Identical to
  [`../industreal_dev_v10_psr_epochsel/`](../industreal_dev_v10_psr_epochsel/README.md)
  (DINOv2 ViT-B/14, 36 train → 16 val recordings, MS-TCN++ + prior/dwell, 30 s budget, epoch grid
  20 / 40 / 60 / 80 on the inner folds, seeds 0–2) except for the criterion that picks the epoch:
  **the out-of-fold decoded F1 of the decoder's best in-budget setting** (the event-level metric
  the run reports) instead of the frame-level held-out BCE. Frozen test split untouched. Not HA-ViD.
- Date: 2026-09-12. Author: kuotunyu. Machine: Windows 11, RTX 4090.
- Purpose: v10 showed that held-out BCE selects a head that decodes worse; this run closes the
  loop by selecting with the metric that is actually reported.

## 1. What the inner folds said (per seed)

| seed | out-of-fold decoded F1 at 20 / 40 / 60 / 80 epochs | held-out BCE at 20 / 40 / 60 / 80 | chosen |
|---|---|---|---|
| 0 | 0.773 / 0.796 / 0.772 / **0.800** | 0.291 / 0.293 / 0.328 / 0.385 | 80 |
| 1 | 0.720 / 0.759 / **0.797** / 0.786 | 0.269 / 0.309 / 0.349 / 0.358 | 60 |
| 2 | 0.764 / **0.807** / 0.804 / 0.793 | 0.261 / 0.296 / 0.294 / 0.350 | 40 |

The two criteria disagree in every seed: BCE is monotonically worse after 20 epochs while the
decoded F1 keeps improving to 40–80. Longer training gives sharper state probabilities, which the
hysteresis decoder rewards even though the frame-level likelihood says the head is overfitting.

## 2. Result on val (mean ± std over seeds 0–2; per-seed rows with CIs in `tables.md`)

| run | epoch choice | POS | F1 (system) | mean delay (s) | TP / FP / FN (mean) |
|---|---|---|---|---|---|
| **this run** (`mstcn_prior_dwell_cap30`) | out-of-fold decoded F1 → 80 / 60 / 40 | **0.642 ± 0.035** | **0.821 ± 0.004** | **23.6 ± 0.7** | 104.7 / 17.7 / 29.0 |
| v8 | fixed 40 | 0.625 ± 0.059 | 0.814 ± 0.020 | 25.1 ± 0.9 | 105.3 / 20.0 / 29.3 |
| v10 | held-out BCE → 20 | 0.545 ± 0.066 | 0.741 ± 0.046 | 21.1 ± 1.3 | 91.7 / 33.7 / 29.3 |

Per seed: 0.620 [0.531, 0.709] / 0.816 [0.776, 0.860] / 23.8 s, 0.614 / 0.823 / 22.7 s,
0.691 / 0.823 / 24.4 s.

Reading:

- **Selecting with the reported metric works.** Against the fixed-40 head the mean is slightly
  better on every axis (POS +0.017, F1 +0.007, delay −1.5 s) and, more importantly, the seed
  spread of F1 collapses from ± 0.020 to ± 0.004 and of POS from ± 0.059 to ± 0.035: the epoch
  that suits each seed's head is found instead of imposed.
- Against v10 (BCE criterion) the gain is large (+0.10 POS, +0.08 F1) at +2.5 s delay, which
  settles the v10 question: the criterion, not the idea, was wrong.
- The out-of-fold decoded F1 of the chosen settings (0.797–0.807) and delays (27–30 s) again
  predict the val numbers (0.816–0.823, 23–24 s) to within the seed spread.
- This is the best development configuration so far: ViT-B/14 frozen features, causal MS-TCN++
  state head with nested epoch selection by decoded F1, prior-and-dwell decoder chosen
  out-of-fold under a 30 s budget. For scale only (different split, different system): the paper's
  B3 reports POS 0.797 / F1 0.883 / delay 22.4 s on the test split.

## 3. SOP checks (`sop_checks_mstcn_prior_dwell_cap30_s0.json`)

13 recordings neither flagged, 2 flagged by the ground truth only, and — for the first time with
an MS-TCN++ decoder — 1 flagged by both: `14_main_2_2`, where the predicted completions place the
rear rear chassis pin (`S18`) before its predecessor exactly as the ground truth does. The
remove-and-reinstall recording (`05_assy_2_2`) and `26_main_0_1` are still missed, and no false
alarm is raised. Counts, not rates.

## 4. Cost

`train-psr … --mstcn-epoch-grid 20 40 60 80 --mstcn-epoch-criterion decoded_f1 --head mstcn
--decoder prior_dwell --delay-cap 30`, seeds 0–2: 1,256 s on the RTX 4090 (shared with v9 for the
first minutes); the four extra grid evaluations per seed run on the CPU.

## 5. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1
# ViT-B/14 features and PSR labels as in ../industreal_dev_v8_psr_vitb/README.md, then:
uv run sop-monitor train-psr --features artifacts/features/industreal/dinov2_vitb14_s1 --psr-dir data/external/industreal/psr --train-split splits/industreal/train.csv --eval-split splits/industreal/val.csv --selection nested --seed 0 --seed 1 --seed 2 --delay-cap 30 --head mstcn --decoder prior_dwell --mstcn-epoch-grid 20 --mstcn-epoch-grid 40 --mstcn-epoch-grid 60 --mstcn-epoch-grid 80 --mstcn-epoch-criterion decoded_f1 --out reports/industreal_dev_v11_psr_epochsel_f1 --device auto --epochs 300 --n-boot 2000
uv run sop-monitor check-psr-run --run reports/industreal_dev_v11_psr_epochsel_f1 --run-name mstcn_prior_dwell_cap30_s0 --out sop_checks_mstcn_prior_dwell_cap30_s0.json
uv run sop-monitor reproduce-lite
```

Determinism: not re-run separately (same training code as v6–v10; the criterion only changes
which snapshot epoch is selected).

## 6. What this settles and what it does not

- Settled: for this pipeline, model selection must use the event-level out-of-fold metric at
  every level — decoder settings (v6), latency budget (v7) and now training length. Frame-level
  proxies (BCE) mislead.
- Not settled: the remaining errors are the same three recordings as in every run since v5
  (`14_main_2_3`, `05_assy_2_2`, `26_assy_1_5`): maintenance procedures the learned prior does
  not cover and remove-and-reinstall sequences. Those need either procedure knowledge beyond
  "which components move" or labels for the maintenance variants; nothing here is an HA-ViD or
  test-split result.
