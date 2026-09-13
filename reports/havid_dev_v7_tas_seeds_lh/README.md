# havid_dev_v7_tas_seeds — seed variance of the v3 protocol, both hands (development result)

This directory (`…_lh`) and its twin [`../havid_dev_v7_tas_seeds_rh/`](../havid_dev_v7_tas_seeds_rh/tables.md)
share this README.

- Tier: **development result** on `splits/ha-vid/val.csv` (6 subjects, 18 recordings × 3 views);
  epochs selected on val F1@10; the frozen test subjects were never read.
- Date: 2026-09-14. Author: kuotunyu. Machine: Windows 11, RTX 4090.
- Purpose: every earlier HA-ViD comparison (v1 vs v3, fusion rules, DINOv2 vs I3D) was one seed
  with differences of 1–4 points. This run measures how much of that is seed noise: the v3
  protocol (`train-havid-tas --selection-metric f1@10`) repeated with seeds 1 and 2 next to the
  seed-0 runs of v3.

## 1. Inputs

| seed | left hand | right hand |
|---|---|---|
| 0 | [`../havid_dev_v3_tas_f1sel_lh/`](../havid_dev_v3_tas_f1sel_lh/README.md) | [`../havid_dev_v3_tas_f1sel_rh/`](../havid_dev_v3_tas_f1sel_rh/README.md) |
| 1 | `../havid_dev_v3_tas_f1sel_lh_s1/` | `../havid_dev_v3_tas_f1sel_rh_s1/` |
| 2 | `../havid_dev_v3_tas_f1sel_lh_s2/` | `../havid_dev_v3_tas_f1sel_rh_s2/` |

Each seed directory is a complete `train-havid-tas` run (predictions, metrics with subject-bootstrap
CIs, config) and is recomputed by `reproduce-lite`; so are the two summaries (`seed_summary.json`).
The seed changes the network initialisation and the order of training videos; data, features,
split and hyper-parameters are identical.

## 2. Result (`tables.md` of each directory is authoritative)

Mean ± sample standard deviation over seeds 0–2 (min–max in `tables.md`):

| run | left F1@10 | left Edit | right F1@10 | right Edit |
|---|---|---|---|---|
| fusion_causal (mean) | 30.5 ± 1.4 | 33.6 ± 2.0 | 29.9 ± 0.6 | 33.0 ± 0.5 |
| fusion_geo_causal | 32.2 ± 0.3 | 30.7 ± 2.4 | 31.7 ± 1.7 | 32.9 ± 1.2 |
| fusion_conf_causal | 28.5 ± 1.4 | 34.1 ± 1.5 | 30.8 ± 0.1 | 34.5 ± 0.9 |
| best single causal view | top 30.0 ± 1.9 | top 33.6 ± 2.7 | front 28.5 ± 0.8 | front 33.1 ± 3.5 |
| fusion_offline (mean) | 43.9 ± 2.0 | 41.2 ± 1.1 | 42.0 ± 1.6 | 40.1 ± 1.9 |
| fusion_geo_offline | 43.3 ± 1.6 | 41.1 ± 1.6 | 44.0 ± 1.7 | 41.9 ± 2.2 |

Best epoch per seed, causal networks: left 15/15/45, 15/20/10, 15/15/10; right 20/15/10, 10/20/15,
25/15/10 (side, front, top).

Reading:

- **Seed noise is 0.3–2.7 points** (std) on every metric, i.e. the size of every single-seed
  difference reported so far. Single-seed HA-ViD comparisons below ≈ 3 points are not evidence.
- **The v1 → v3 gain was mostly seed luck.** Seed 0 of v3 was the best of three for the left hand
  (fusion_causal F1@10 32.0, range 29.6–32.0); the three-seed mean is 30.5, +1.4 over v1's single
  seed (29.1) and inside one standard deviation. F1@10 epoch selection is kept because it fixes
  the MoF-selection failure mode (epochs picked for predicting `null`), not because it scores
  higher.
- **Late fusion adds little over the best single causal view:** +0.5 F1@10 (left, mean rule) and
  +1.4 (right). The geometric mean is the only rule ahead on F1@10 for both hands (+2.2 / +3.2 over
  the best view, +1.7 / +1.8 over the mean rule), and on the left hand it is also the most stable
  (± 0.3), but it pays for it in Edit (left 30.7, the lowest causal fusion). The confidence-weighted
  mean has the best causal Edit on both hands (34.1 / 34.5). No rule dominates both metrics.
- **Causal networks stop improving after 10–20 epochs** in 17 of 18 cases, and the causal-vs-offline
  gap (≈ 12–13 F1@10 points for the mean fusion) is far larger than any seed or fusion effect: the
  lever for the online recogniser is the model's access to context, not the fusion rule.
- Seeds 1–2 ran overnight with the machine locked; wall times were 2,075–2,369 s per hand instead of
  1,076–1,108 s for seed 0 (GPU at ≈ 14 %, CPU-bound). `cudnn.deterministic` is set, so the results
  do not depend on the speed.
- All numbers are baselines on a 6-subject validation set, not claims.

## 3. Reproduce

```bash
export PYTHONUTF8=1
make havid-tas-v3          # seed 0 (both hands)
make havid-tas-seeds       # seeds 1 and 2 (both hands), ≈ 18 min per run on an unthrottled RTX 4090
make havid-seeds-summary   # this directory and its _rh twin
uv run sop-monitor reproduce-lite
```

## 4. Not done here

- Three seeds only; the SOP runs (v4–v6) still use the seed-0 predictions.
- No learned fusion, no ASFormer, no class weighting for `w`; no number on the frozen test subjects.
