# industreal_dev_v5_psr_train — the four PSR runs fitted once on the 36 train recordings, evaluated on val

- Tier: **development result**. Same heads, decoders, metrics and val recordings as
  [`../industreal_dev_v4_psr_mstcn/`](../industreal_dev_v4_psr_mstcn/README.md); what changed is
  the protocol: instead of leave-one-participant-out inside val (12–13 training recordings per
  fold), every head is fitted **once on the 36 train-split recordings (12 participants)** and the
  decoder is selected on those same 36 recordings; the 16 val recordings (5 other participants)
  are only evaluated. The frozen test split is still untouched. Not HA-ViD.
- Date: 2026-09-11. Author: kuotunyu. Machine: Windows 11, RTX 4090.

## 1. Data identity (`psr_audit.json`, `data/manifest.json`)

| item | value |
|---|---|
| new archives | `train_p1.zip` 6,371,363,172 B (MD5 `b283383e…1b4bc`), `train_p2.zip` 4,801,035,968 B (`0d07877b…6c0dd`), `train_p3.zip` 4,414,811,027 B (`606509e0…9a662`), `train_p4.zip` 4,837,382,030 B (`40193a8f…0c29f`); all sizes and MD5s match the 4TU listing; only the three `PSR_labels*.csv` per recording and the JPEG name range were extracted |
| recordings with labels | 52 = 36 train + 16 val; the 32 test recordings' archives (23.7 GB) were not downloaded |
| label consistency | for all 52 recordings the raw states reproduce `PSR_labels.csv` / `PSR_labels_with_errors.csv` exactly; all ids within `procedure_info.json`; no label beyond its video |
| frame alignment | 51 recordings: JPEG names `000000..N-1` with `N` = mp4 frame count. **One exception, `11_assy_0_1`**: 2,571 JPEGs named `000030..002600` for a 2,572-frame mp4. Pixel comparison of decoded mp4 frames against the archive's JPEGs (mean abs difference 4.5 for JPEG 000030 vs frame 1, 4.3 for 000060 vs 31, 4.6 for 002600 vs 2571, ≥ 19 for every other pairing) fixes `video frame = JPEG name − 29`; `extract-psr-labels` now records each recording's JPEG range and the loader applies the derived offset (0 everywhere else) |
| ground truth (val) | unchanged: 122 correct completions in 16 recordings, 4 of them with execution errors |

## 2. Model identity (`config.json`)

Identical to v4 apart from the training set: `linear` = 11-way multi-label logistic regression;
`mstcn` = causal MS-TCN++ with sigmoid outputs, 40 epochs, seed 0 (training loss 0.416, 92 s on
the 36 recordings); `plain` = causal EMA + hysteresis; `prior_dwell` = plain + minimum dwell +
procedure prior learned from the 36 training recordings (assembly: base pre-installed, 10 of 11
components active; maintenance: 10 pre-installed, 5 active). Decoders selected on the training
recordings by mean per-video F1:

| run | ema | theta_on | theta_off | min_dwell |
|---|---|---|---|---|
| linear_plain | 0.95 | 0.8 | 0.2 | 0 |
| linear_prior_dwell | 0.9 | 0.8 | 0.2 | 10 |
| mstcn_plain | 0.98 | 0.6 | 0.1 | 0 |
| mstcn_prior_dwell | 0.0 | 0.9 | 0.1 | 30 |

## 3. Result (val, 16 videos, unweighted mean over videos; 95 % CI = participant bootstrap, 2,000 draws)

| run | POS | F1 (system) | mean delay (s) | TP / FP / FN | videos with POS = 0 |
|---|---|---|---|---|---|
| linear_plain | 0.078 [0.014, 0.159] | 0.606 [0.493, 0.718] | 31.3 [25.8, 36.6] | 102 / 117 / 24 | 11 |
| linear_prior_dwell | 0.304 [0.236, 0.354] | **0.707 [0.623, 0.802]** | 27.4 [22.4, 33.9] | 103 / 59 / 27 | 5 |
| mstcn_plain | 0.180 [0.062, 0.331] | 0.516 [0.463, 0.577] | 26.9 [13.4, 51.5] | 79 / 142 / 16 | 10 |
| mstcn_prior_dwell | **0.479 [0.354, 0.646]** | 0.636 [0.564, 0.721] | 29.8 [15.8, 56.6] | 79 / 64 / 29 | 2 |

Same runs under the v4 leave-one-participant-out protocol, for reference: linear_plain 0.040 /
0.479 / 26.8 s, linear_prior_dwell 0.184 / 0.604 / 25.6 s, mstcn_plain 0.317 / 0.551 / 21.5 s,
mstcn_prior_dwell 0.609 / 0.707 / 21.9 s.

Reading — this is a negative result for the hypothesis that motivated the download:

- **More training recordings helped the linear head** (linear_plain F1 0.479 → 0.606,
  linear_prior_dwell 0.604 → 0.707; FPs 246 → 117 without any prior) **but not the MS-TCN++
  head** (mstcn_prior_dwell POS 0.609 → 0.479, F1 0.707 → 0.636). The temporal head fitted on
  12 training participants transfers worse to the 5 val participants than the per-fold heads did,
  and its in-sample decoder selection picked an extreme setting (no smoothing, `theta_on` 0.9,
  dwell 30) that does not transfer either.
- **Decoder selection on training recordings is the unstable part.** The first v5 run, made
  before the frame-offset fix (one of 36 training recordings with labels shifted by 29 frames),
  gave mstcn_prior_dwell POS 0.541 / F1 0.772 / delay 30.7 s with a different selected decoder;
  a 3-second label shift in one training video moved the headline F1 by 0.14 through the
  decoder choice, not through the head. See the seed table below for the same effect across
  seeds.
- The prior still does what it did in v4: FPs 117 → 59 (linear) and 142 → 64 (mstcn), and POS
  rises accordingly, at the cost of ~10 additional FNs.

## 4. Seed variance (same protocol, seeds 1 and 2; scratch runs, not committed)

| run | seed 0 (committed) | seed 1 | seed 2 |
|---|---|---|---|
| linear_plain | POS 0.078 / F1 0.606 / 31.3 s | 0.073 / 0.605 / 30.8 s | 0.078 / 0.603 / 31.3 s |
| linear_prior_dwell | 0.304 / 0.707 / 27.4 s | 0.292 / 0.708 / 27.7 s | 0.306 / 0.707 / 27.5 s |
| mstcn_plain | 0.180 / 0.516 / 26.9 s | 0.202 / 0.492 / 28.7 s | 0.179 / 0.542 / 23.9 s |
| mstcn_prior_dwell | 0.479 / 0.636 / 29.8 s | 0.478 / 0.662 / 32.6 s | 0.528 / 0.688 / 23.1 s |

The linear head is stable across seeds (F1 within 0.003); the MS-TCN++ runs move by up to
0.05 POS / 0.05 F1 / 9 s delay with nothing but the seed changed, which is the size of the gaps
the v4-vs-v5 comparison rests on. Seeds 1 and 2 used 200 bootstrap draws; the point estimates are
unaffected by that.

## 5. Failure cases (`tables.md` per-video rows)

- `mstcn_prior_dwell`: `26_assy_1_5` (2 TP / 14 FP / 1 FN) — the long error recording again;
  with no EMA smoothing the state probabilities cross `theta_on` repeatedly and the 30-frame dwell
  does not filter 14 sustained excursions. `05_assy_2_2` (5 / 11 / 5) — the recording with a
  removal-and-reinstall sequence; both directions get over-emitted.
- `linear_prior_dwell`: `05_main_0_1` (3 / 3 / 5) — maintenance removals are missed after the
  prior restricts the active set to five components learned from the training maintenance
  recordings.
- Best: `20_assy_3_6` (F1 1.000 for mstcn_prior_dwell, 0.87 for mstcn_plain), `20_main_0_1`
  (F1 1.000 for linear_prior_dwell), `14_main_0_1` (0.93).

## 6. Cost

| step | wall time | resources |
|---|---|---|
| download `train_p1..4.zip` (20.4 GB) from 4TU, MD5-verified | 44 min (≈ 6–8 MB/s) | network, 20 GB disk |
| `extract-psr-labels` for six archives (labels + JPEG ranges) | ≈ 2 min | CPU |
| pixel check of the offset recording (3 JPEGs vs 7 decoded frames) | seconds | CPU |
| `train-psr` (linear + MS-TCN++ on 36 recordings, 4 decoder grids on 36 recordings, bootstraps) | 145 s | RTX 4090 |
| `audit-industreal --manifest` (hashes 36 GB of archives) | ≈ 10 min | CPU |

## 7. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1
# archives: https://data.4tu.nl/datasets/b008dd74-020d-4ea4-a8ba-7bb60769d224 -> data/external/industreal/{train_p1..4,val_p1..2}.zip
uv run sop-monitor extract-psr-labels --archive data/external/industreal/val_p1.zip --archive data/external/industreal/val_p2.zip --archive data/external/industreal/train_p1.zip --archive data/external/industreal/train_p2.zip --archive data/external/industreal/train_p3.zip --archive data/external/industreal/train_p4.zip --out data/external/industreal/psr
# features for train and val as in ../industreal_dev_v1/README.md, then:
uv run sop-monitor train-psr --features artifacts/features/industreal/dinov2_vits14_s1 --psr-dir data/external/industreal/psr --train-split splits/industreal/train.csv --eval-split splits/industreal/val.csv --out reports/industreal_dev_v5_psr_train --device auto --epochs 300 --mstcn-epochs 40 --n-boot 2000 --seed 0
uv run sop-monitor reproduce-lite
```

`make extract-psr` (val archives) plus `make psr-train` wrap the same commands.
Determinism: a second `train-psr` run with seed 0 on the same GPU reproduced
`completions_val.csv` byte-for-byte (all four runs).

## 8. What this says about the next step

The bottleneck is no longer the amount of labelled training data but how the decoder is chosen:
selecting it on the training recordings does not transfer across participants and is sensitive to
tiny label changes. The honest fix is a selection protocol with its own held-out participants
(nested inside train) and reporting across several seeds, before any further model work.
