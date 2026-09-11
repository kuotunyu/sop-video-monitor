# industreal_dev_v7_psr_latency — decoders chosen under a latency budget, with the out-of-fold F1-vs-delay front

- Tier: **development result**, identical protocol to
  [`../industreal_dev_v6_psr_nested/`](../industreal_dev_v6_psr_nested/README.md) (36 train
  recordings → 16 val recordings, decoders selected on out-of-fold train predictions, seeds
  0–2). The single change: instead of "best out-of-fold F1", each decoder is the best-F1 setting
  whose **mean out-of-fold delay stays within a budget** (15 s or 30 s); when no setting fits,
  the fastest one is taken. Both budgets run side by side, so the table has 8 runs × 3 seeds.
  The frozen test split is untouched. Not HA-ViD.
- Date: 2026-09-11. Author: kuotunyu. Machine: Windows 11, RTX 4090.
- Purpose: v6 ended with "every knob that buys precision costs seconds and the selection metric
  ignores delay". This run makes the trade-off explicit and measurable.

## 1. Out-of-fold F1-vs-delay fronts (`config.json["folds"]["train->eval"]["pareto"]`, seed 0)

Every grid setting was scored on the out-of-fold predictions of the 36 training recordings; the
non-dominated points (higher F1, lower mean delay) are recorded per head × decoder. Selected
points of the front:

| head × decoder | fastest point | ≈ 20 s | ≈ 25 s | ≈ 30 s | slowest point |
|---|---|---|---|---|---|
| linear_prior_dwell | 4.7 s / F1 0.024 | 20.1 s / 0.568 | 25.4 s / 0.700 | 29.7 s / 0.763 | 36.8 s / 0.813 |
| mstcn_prior_dwell | 17.1 s / 0.552 | 20.0 s / 0.662 | 24.3 s / 0.761 | 28.8 s / 0.780 | 33.3 s / 0.793 |
| linear_plain | 4.7 s / 0.021 | 20.1 s / 0.454 | 25.2 s / 0.562 | 29.1 s / 0.610 | 40.6 s / 0.660 |
| mstcn_plain | 17.0 s / 0.419 | 20.7 s / 0.524 | 23.1 s / 0.562 | 26.4 s / 0.565 | — |

Two facts stand out. The linear decoders can be made arbitrarily fast but pay for it almost
linearly (F1 0.02 at 5 s, 0.57 at 20 s, 0.76 at 30 s). The MS-TCN++ decoders cannot go below
≈ 17 s at all — the temporal head's own smoothing sets the floor — but hold F1 ≥ 0.55 there and
gain most of their F1 between 17 s and 25 s.

## 2. Result on val (mean ± std over seeds 0–2; per-seed rows with CIs in `tables.md`)

| run | budget | POS | F1 (system) | mean delay (s) | TP / FP / FN (mean) |
|---|---|---|---|---|---|
| linear_plain_cap15 | 15 s | 0.000 ± 0.000 | 0.188 ± 0.022 | 18.5 ± 1.8 | 116.3 / 1202.0 / 1.0 |
| linear_plain_cap30 | 30 s | 0.076 ± 0.003 | 0.604 ± 0.001 | 31.1 ± 0.2 | 101.7 / 117.3 / 24.0 |
| linear_prior_dwell_cap15 | 15 s | 0.000 ± 0.000 | 0.216 ± 0.012 | 18.9 ± 1.2 | 116.7 / 1138.0 / 2.0 |
| linear_prior_dwell_cap30 | 30 s | 0.276 ± 0.006 | 0.703 ± 0.015 | 32.0 ± 1.9 | 98.0 / 53.3 / 28.3 |
| mstcn_plain_cap15 | 15 s (fastest: 17 s) | 0.137 ± 0.051 | 0.414 ± 0.028 | 22.8 ± 2.8 | 67.3 / 187.7 / 13.7 |
| mstcn_plain_cap30 | 30 s | 0.185 ± 0.009 | 0.558 ± 0.023 | 29.3 ± 3.5 | 84.3 / 112.0 / 27.0 |
| mstcn_prior_dwell_cap15 | 15 s (fastest: 17 s) | 0.440 ± 0.050 | 0.532 ± 0.028 | 22.5 ± 2.8 | 67.0 / 109.0 / 20.3 |
| **mstcn_prior_dwell_cap30** | 30 s | **0.529 ± 0.018** | **0.742 ± 0.026** | **29.4 ± 1.7** | 90.3 / 34.7 / 26.3 |

For reference, v6 (same protocol, no budget): linear_prior_dwell 0.366 / 0.759 / 42.5 s,
mstcn_prior_dwell 0.497 / 0.745 / 35.5 s.

Reading:

- **A 30 s budget is free for MS-TCN++ and cheap for the linear head.** mstcn_prior_dwell keeps
  its F1 (0.742 vs 0.745), improves POS (0.529 vs 0.497) and answers 6 s sooner; linear_prior_dwell
  answers 10 s sooner for −0.056 F1. The out-of-fold delay of the chosen settings (28.8 s and
  29.7 s) transferred to val (29.4 s and 32.0 s) to within a few seconds, which is what a
  selection criterion is supposed to do.
- **A 15 s budget is not reachable with these features.** No linear setting within 15 s has
  useful precision (the selected ones emit > 1,100 FPs on val — every install/remove flicker
  becomes an event), and the MS-TCN++ decoders cannot even reach 15 s, so the fallback took the
  fastest setting (17 s out-of-fold, 22–23 s on val) at F1 0.41–0.53. Whether the 17 s floor is
  the head's receptive field or the labels' own completion timing is the open question.
- Seed spread is unchanged from v6 (≤ 0.05 POS, ≤ 0.03 F1 for the MS-TCN++ runs).

## 3. Failure cases (seed 0)

- Under the 15 s budget every recording ends at POS 0 for the linear runs (all 16) — the decoder
  is simply not viable there; `26_assy_1_5` alone collects 136 FPs.
- Under the 30 s budget the failure set is the one already known from v5/v6: `14_main_2_2`,
  `05_assy_2_2`, `14_main_2_3` (missed maintenance removals and the remove-and-reinstall
  recording) for both heads; `26_assy_1_5` for MS-TCN++.

## 3b. SOP checks (`sop_checks_mstcn_prior_dwell_cap30_s0.json`)

Against the train-learned precedence graphs (see the v6 report, section 6) the 30 s MS-TCN++
decoder behaves like its uncapped v6 twin: the three ground-truth deviations are not flagged
and no false alarm is raised (13 neither / 3 only-GT). `reproduce-lite` now recomputes every
committed `sop_checks_*.json` from the completions and the graphs it names.

## 4. Cost

`train-psr … --delay-cap 15 --delay-cap 30`, seeds 0–2: 798 s on the RTX 4090 (the inner and
final fits are shared across budgets; only the val decoding is repeated per budget).

## 5. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1
# archives, labels and features as in ../industreal_dev_v5_psr_train/README.md, then:
uv run sop-monitor train-psr --features artifacts/features/industreal/dinov2_vits14_s1 --psr-dir data/external/industreal/psr --train-split splits/industreal/train.csv --eval-split splits/industreal/val.csv --selection nested --seed 0 --seed 1 --seed 2 --delay-cap 15 --delay-cap 30 --out reports/industreal_dev_v7_psr_latency --device auto --epochs 300 --mstcn-epochs 40 --n-boot 2000
uv run sop-monitor reproduce-lite
```

Determinism: not re-run separately for this directory. The fits and out-of-fold grids are the
v6 code path, whose seed-0 re-run reproduced every completion row; the budget only changes which
grid point `choose_decoder` returns.

## 6. What this settles and what it does not

- Settled: the latency–precision trade-off of this decoder family is now a measured front, and a
  30 s budget is the sensible operating point for it — no F1 lost, several seconds gained.
- Not settled: anything below ≈ 20 s. That needs either features that separate installed from
  not-installed within a second or two of the event (the per-frame DINOv2 embedding does not) or
  a decoder that confirms events from evidence rather than dwell time. Nothing here is an HA-ViD
  or test-split result.
