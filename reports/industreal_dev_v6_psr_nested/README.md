# industreal_dev_v6_psr_nested — decoders selected on out-of-fold train predictions, three seeds

- Tier: **development result**. Same data, features, heads, decoders, metrics and val recordings
  as [`../industreal_dev_v5_psr_train/`](../industreal_dev_v5_psr_train/README.md). Two things
  changed: (1) every decoder is selected on **out-of-fold** predictions of the 36 training
  recordings (deterministic grouped 4-fold by participant; the prior used while scoring a
  recording is learned from that recording's inner-train fold), then the heads are refitted on
  all 36; (2) the whole run is repeated with seeds 0, 1, 2 and reported per seed and as
  mean ± std. The 16 val recordings (5 other participants) are only evaluated; the frozen test
  split is untouched. Not HA-ViD.
- Date: 2026-09-11. Author: kuotunyu. Machine: Windows 11, RTX 4090.
- Purpose: v5 showed that in-sample decoder selection did not transfer across participants and
  was sensitive to tiny label changes; this run tests the fix the v5 report proposed.

## 1. What is new against v5

| item | v5 | v6 |
|---|---|---|
| decoder selection | on the same 36 recordings the head was fitted on | on out-of-fold predictions: 4 inner folds over the 12 training participants, round-robin over the sorted ids — {01, 07, 21}, {02, 11, 22}, {04, 15, 25}, {06, 16, 27}; each inner fold fits the heads on the other 9 participants and predicts these 3; the grid is scored on those predictions with priors learned from the inner-train fold |
| final heads | fitted once on the 36 recordings | same, after selection |
| seeds | 0 (plus scratch seeds in the report) | 0, 1, 2 as committed runs `<head>_<decoder>_s<seed>` |
| runs that emit nothing | would vanish from the table | scored as all-FN (also in `reproduce-lite`) |

Selected decoders are now stable across seeds — identical for both linear runs, and within one
grid step for the MS-TCN++ runs (`tables.md`, "Decoder selected per fold"). They are also more
conservative than the in-sample choices: `ema` 0.9–0.99, `theta_on` 0.7–0.9, dwell 10 (linear)
or 30–60 frames (MS-TCN++).

## 2. Result (val, 16 videos; per-seed rows with participant-bootstrap 95 % CI, then mean ± std over seeds)

| run | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| linear_plain | POS 0.111 / F1 0.593 / 50.2 s / 89-80-42 | 0.111 / 0.593 / 50.1 s / 89-82-42 | 0.106 / 0.594 / 50.1 s / 89-78-42 |
| linear_prior_dwell | 0.370 / 0.759 / 42.8 s / 100-25-35 | 0.357 / 0.759 / 42.4 s / 100-25-35 | 0.372 / 0.759 / 42.4 s / 100-25-35 |
| mstcn_plain | 0.177 / 0.549 / 32.7 s / 82-110-29 | 0.200 / 0.538 / 33.0 s / 79-117-23 | 0.208 / 0.580 / 31.4 s / 86-98-33 |
| mstcn_prior_dwell | 0.472 / 0.737 / 34.1 s / 90-31-32 | 0.473 / 0.735 / 39.2 s / 88-35-28 | 0.547 / 0.763 / 33.2 s / 93-25-31 |

(TP-FP-FN after the delay; CIs per seed are in `tables.md`, e.g. seed 0 mstcn_prior_dwell POS
[0.379, 0.592], F1 [0.696, 0.783].)

| run | POS | F1 (system) | mean delay (s) | TP / FP / FN (mean over seeds) |
|---|---|---|---|---|
| linear_plain | 0.109 ± 0.002 | 0.593 ± 0.000 | 50.1 ± 0.0 | 89.0 / 80.0 / 42.0 |
| linear_prior_dwell | 0.366 ± 0.007 | **0.759 ± 0.000** | 42.5 ± 0.2 | 100.0 / 25.0 / 35.0 |
| mstcn_plain | 0.195 ± 0.013 | 0.556 ± 0.018 | 32.4 ± 0.7 | 82.3 / 108.3 / 28.3 |
| mstcn_prior_dwell | **0.497 ± 0.035** | 0.745 ± 0.013 | 35.5 ± 2.6 | 90.3 / 30.3 / 30.3 |

Subsets, seed 0: linear_prior_dwell — videos without error steps (12) POS 0.441 / F1 0.756 /
31.4 s, with error steps (4) 0.156 / 0.768 / 77.0 s; mstcn_prior_dwell — 0.528 / 0.763 / 25.1 s
and 0.302 / 0.658 / 60.9 s.

Reading:

- **Nested selection did what v5 asked for.** Against v5's in-sample choice (same heads, same
  training data), F1 rises for both prior-and-dwell runs (linear 0.707 → 0.759, MS-TCN++
  0.636 → 0.745) and POS rises (0.304 → 0.366, 0.479 → 0.497); false positives drop to 25 and
  ≈ 30. The seed spread is now small (≤ 0.035 POS, ≤ 0.018 F1), where v5 moved by 0.14 F1 from
  a 3-second label change.
- **The price is delay.** Out-of-fold selection prefers more smoothing and longer dwell, so the
  mean detection delay grows from 27 s to 42 s (linear) and from 30 s to 36 s (MS-TCN++); the
  plain linear decoder even reaches 50 s. Delay and precision are traded through the same knobs,
  and the selection metric (mean per-video F1) does not see delay at all.
- **Linear vs MS-TCN++ is now a wash on F1** (0.759 vs 0.745, both within seed noise), with
  MS-TCN++ ahead on POS (+0.13) and delay (−7 s). More training data plus honest selection made
  the per-frame head competitive; the temporal head's advantage is in ordering and latency, not
  in detection count.
- For scale only: the paper's B3 (object-detection-based state detector + full procedure
  knowledge, *test* split) reports POS 0.797 / F1 0.883 / delay 22.4 s.

## 3. Failure cases (seed 0, `tables.md` per-video rows)

- `05_main_0_1` (linear_prior_dwell 2 TP / 0 FP / 6 FN; mstcn 4 / 1 / 4) and `14_main_2_3`
  (5 / 2 / 5 and 3 / 3 / 6): maintenance recordings whose removals the conservative decoders
  never confirm — the FN side of the precision gain.
- `05_assy_2_2` (6 / 2 / 5 and 5 / 3 / 5): the remove-and-reinstall recording; the second
  install of the same component arrives while the dwell/hysteresis state still says "installed".
- `26_assy_1_5` (mstcn_prior_dwell 3 / 6 / 1): the long error recording keeps producing
  sustained false plateaus, fewer than before (14 → 6 FP) but still the worst FP source.
- Best: `20_assy_3_6` and `24_assy_0_1` (F1 ≥ 0.94 for both prior-and-dwell runs).

## 4. Cost

| step | wall time | resources |
|---|---|---|
| `train-psr --selection nested --seed 0 --seed 1 --seed 2` (per seed: 4 inner MS-TCN++ fits ≈ 60–70 s each on 27 recordings + final fit ≈ 90 s + linear fits + 4 grids on 36 out-of-fold predictions + bootstraps) | 654 s | RTX 4090 |
| everything else | reused from v1 / v5 | — |

## 5. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1
# archives, labels and features as in ../industreal_dev_v5_psr_train/README.md, then:
uv run sop-monitor train-psr --features artifacts/features/industreal/dinov2_vits14_s1 --psr-dir data/external/industreal/psr --train-split splits/industreal/train.csv --eval-split splits/industreal/val.csv --selection nested --seed 0 --seed 1 --seed 2 --out reports/industreal_dev_v6_psr_nested --device auto --epochs 300 --mstcn-epochs 40 --n-boot 2000
uv run sop-monitor reproduce-lite
```

`make psr-nested` wraps the same command.
Determinism: a separate `train-psr --selection nested --seed 0` run on the same GPU reproduced
all 750 seed-0 completion rows of the committed table exactly (the ground truth and every
`<head>_<decoder>_s0` prediction).

## 6. SOP checks on the completion sequences (`sop_checks_<run>.json`)

`sop-monitor learn-sop` learned one precedence graph per recording kind from the 36 train
recordings (an edge only where every co-occurring training recording agrees, ≥ 3 recordings,
transitively reduced): assembly 11 steps / 20 edges (chassis, pins and rear chassis before
brackets and wheels), maintenance 8 steps / 14 edges (removals before re-installs). `check-psr-run`
then ran `sop_graph.check_order` on the ground-truth and the predicted completion sequences of
every val recording. A recording is "flagged" when at least one step precedes an unfinished
predecessor; omissions are listed but not used for flagging (partial procedures always omit
something under a union graph).

| | GT flagged | pred flagged | both | only GT | only pred | neither |
|---|---|---|---|---|---|---|
| `mstcn_prior_dwell_s0` | 3 / 16 | 0 / 16 | 0 | 3 | 0 | 13 |
| `linear_prior_dwell_s0` | 3 / 16 | 6 / 16 | 3 | 0 | 3 | 10 |

The three ground-truth deviations are real: `05_assy_2_2` (rear wheel, bracket and bracket screw
completed before the re-installed rear chassis — the remove-and-reinstall recording) and
`14_main_2_2` / `26_main_0_1` (rear rear chassis pin installed before its predecessor). The
conservative MS-TCN++ decoder suppresses exactly the out-of-order events and flags nothing; the
linear decoder catches all three and raises three false alarms (`14_main_0_1`, `14_main_2_3`,
`26_assy_1_5`). Sixteen recordings and three positives are counts, not rates — this is the first
end-to-end "video → completions → precedence check → deviation" output of the project, not a
violation-detection result.

## 7. What this settles and what it does not

- Settled: decoder selection must be out-of-fold; with it, the val numbers are stable across
  seeds and the v4-vs-v5 discrepancy dissolves into selection noise.
- Not settled: delay. Every knob that buys precision here (smoothing, dwell, thresholds) costs
  seconds, and the selection criterion ignores it; a criterion that trades F1 against delay
  explicitly, or a decoder that confirms events from evidence rather than time, is the next
  methodological step. Nothing here is an HA-ViD or test-split result.
