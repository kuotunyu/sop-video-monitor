# havid_dev_v3_tas_f1sel_rh — epochs chosen by val F1@10, three parameter-free fusion rules, right hand, primitive tasks (development result)

- Tier: **development result** on `splits/ha-vid/val.csv` (6 subjects: S01, S08, S10, S12, S18,
  S30; 18 recordings × 3 views); each network's epoch is selected on val; the frozen test subjects
  (`splits/ha-vid/test.csv`, 7 subjects) were never read. Not comparable with the paper's Table 3.
- Date: 2026-09-13. Author: kuotunyu. Machine: Windows 11, RTX 4090.
- Purpose: two fixes to [`../havid_dev_v1_tas_rh/`](../havid_dev_v1_tas_rh/README.md) —
  epoch selection by a segmental metric (v1's MoF selection favoured `null` and stopped causal
  networks early) and two more late-fusion rules that need no tuning (normalised geometric mean,
  confidence-weighted mean) next to the plain mean.

## 1. Data and features

As v1: HA-ViD primitive-task labels of the right hand (65 classes in train),
train 143 recordings / 17 subjects / 181,015 frames per view, val 18 / 6 / 17,973; frozen DINOv2
ViT-B/14 1536-d features at stride 1 (`artifacts/features/ha-vid/dinov2_vitb14_s1`).

## 2. Model identity (from `config.json`)

| component | what exactly |
|---|---|
| architecture | MS-TCN++ per view as in v1 (986,312 parameters), causal (left-only padding) and offline (symmetric padding) |
| selection (val only) | val **F1@10** every 5 epochs, best epoch kept per network: `view0_causal` 20 (11.4 → 23.5), `view0_offline` 45 (26.6 → 33.3), `view1_causal` 10 (18.6 → 19.5), `view1_offline` 50 (22.2 → 38.0), `view2_causal` 25 (12.6 → 23.3), `view2_offline` 45 (27.3 → 38.0); arrows give val F1@10 at the first (epoch 5) and last (epoch 50) evaluation |
| `fusion_*` | arithmetic mean of the three per-view posteriors |
| `fusion_geo_*` | normalised geometric mean (product of experts: a view assigning ≈ 0 to a class vetoes it) |
| `fusion_conf_*` | per-frame weighted mean, each view weighted by its own maximum posterior |
| `majority` | constant most frequent training label (`null`) |

## 3. Result (val, spec 4.2 metrics, subject bootstrap 2,000 draws, seed 0)

`tables.md` is authoritative.

| run | temporal context | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|
| majority | — | 29.8 [24.6, 34.1] | 8.0 [6.8, 9.3] | 8.9 [5.9, 11.8] | 0.7 [0.0, 2.3] | 0.0 [0.0, 0.0] |
| view0_causal | past only | 29.5 [23.8, 36.9] | 31.9 [26.4, 37.7] | 29.3 [20.9, 40.1] | 22.0 [15.7, 29.5] | 12.5 [6.1, 19.8] |
| view1_causal | past only | 29.8 [22.2, 37.2] | 32.0 [25.7, 37.5] | 28.5 [19.0, 39.5] | 19.5 [12.9, 26.7] | 12.1 [8.1, 17.5] |
| view2_causal | past only | 32.2 [25.3, 42.2] | 31.0 [25.6, 36.0] | 27.7 [21.3, 36.1] | 20.9 [16.7, 27.6] | 11.3 [6.1, 18.9] |
| fusion_causal | past only | 34.3 [26.3, 44.0] | 32.4 [25.6, 39.3] | 30.2 [23.4, 38.4] | 25.1 [18.3, 33.3] | 13.1 [8.3, 19.3] |
| fusion_geo_causal | past only | 35.4 [28.3, 43.3] | 33.0 [26.3, 40.1] | 33.3 [24.1, 43.8] | 26.7 [19.2, 35.5] | 14.7 [9.3, 22.7] |
| fusion_conf_causal | past only | 33.9 [25.2, 44.7] | 33.8 [26.6, 42.1] | 30.8 [23.1, 40.8] | 26.0 [18.7, 34.6] | 14.7 [9.9, 21.5] |
| view0_offline | past and future | 35.4 [28.1, 44.7] | 37.0 [30.2, 42.6] | 37.4 [31.4, 45.5] | 29.3 [21.8, 38.7] | 19.2 [13.0, 27.7] |
| view1_offline | past and future | 35.4 [27.7, 45.7] | 39.6 [32.2, 46.7] | 38.0 [30.4, 48.6] | 33.8 [25.5, 44.3] | 20.5 [13.0, 30.2] |
| view2_offline | past and future | 37.1 [26.2, 51.2] | 43.7 [38.2, 49.1] | 42.6 [33.9, 53.0] | 36.9 [28.0, 47.0] | 24.7 [15.9, 35.0] |
| fusion_offline | past and future | 41.0 [32.5, 53.3] | 38.3 [31.9, 44.4] | 40.2 [33.0, 49.7] | 35.5 [27.4, 46.4] | 22.8 [17.4, 29.9] |
| fusion_geo_offline | past and future | 39.6 [31.3, 51.3] | 40.5 [34.1, 47.3] | 42.1 [34.7, 53.5] | 35.4 [28.0, 46.7] | 23.2 [17.9, 30.3] |
| fusion_conf_offline | past and future | 41.0 [32.2, 53.8] | 39.7 [32.7, 46.4] | 39.8 [32.6, 50.2] | 35.7 [28.3, 46.3] | 23.0 [17.8, 30.5] |

v1 (MoF selection, mean fusion) vs v3 (F1@10 selection), same features and split:

| run | MoF | Edit | F1@10 | F1@50 |
|---|---|---|---|---|
| v1 fusion_causal | 38.6 [32.3, 47.2] | 33.5 [26.8, 39.0] | 30.3 [24.5, 36.9] | 13.6 [9.4, 18.7] |
| v3 fusion_causal | 34.3 [26.3, 44.0] | 32.4 [25.6, 39.3] | 30.2 [23.4, 38.4] | 13.1 [8.3, 19.3] |
| v1 best single causal view (front, `view1_causal`) | 33.2 [25.4, 43.2] | 29.0 [25.1, 33.1] | 26.6 [20.5, 34.3] | 13.1 [8.1, 19.8] |
| v3 best single causal view (side, `view0_causal`) | 29.5 [23.8, 36.9] | 31.9 [26.4, 37.7] | 29.3 [20.9, 40.1] | 12.5 [6.1, 19.8] |
| v1 fusion_offline | 41.8 [33.6, 52.5] | 40.9 [32.2, 48.7] | 41.3 [34.0, 49.9] | 23.4 [17.1, 32.2] |
| v3 fusion_offline | 41.0 [32.5, 53.3] | 38.3 [31.9, 44.4] | 40.2 [33.0, 49.7] | 22.8 [17.4, 29.9] |

Reading:

- For the right hand, F1@10 selection leaves the mean causal fusion where v1 had it (F1@10 30.2
  vs 30.3, Edit 32.4 vs 33.5) and costs 4.3 MoF; every difference is inside the intervals. The
  single causal views did move: all three now sit at F1@10 27.7–29.3 (v1: 24.2–26.6), which is why
  the mean fusion's margin over the best single view shrank to +0.9 F1@10.
- The geometric mean is the best causal rule on F1@10 (33.3 vs 30.2 mean, 30.8 confidence-
  weighted; +4.0 over the best single view) and on F1@25 / F1@50, the confidence-weighted mean
  the best on Edit (33.8); all inside the intervals ([24.1, 43.8] vs [23.4, 38.4] on F1@10), so the
  rules are tied on six subjects, with the geometric mean ahead on both hands' F1@10.
- Causal networks peak early here too (epochs 10–25) and all three score lower at epoch 50 than at
  their best (`view1_causal` 28.5 → 19.5); the offline networks peak at 45–50.
- All numbers are baselines on a 6-subject validation set, not claims.

## 4. Failure cases (`fusion_causal`)

- Most frequent confusions (gt → pred, frames): `sspn4` → `null` 449, `rgw` → `null` 425,
  `sntft` → `null` 385, `w` → `null` 381, `sntn5` → `null` 358: real actions, including the
  `wrong` label, predicted as a pause.
- Worst val recordings by MoF: S18A04I01 13.7, S12A06I01 15.2, S01A04I01 17.1; best: S18A06I01
  63.1, S08A06I01 60.2, S08A05I01 57.5.

## 5. Cost

| step | wall time | resources |
|---|---|---|
| `train-havid-tas` (6 networks × 50 epochs, val every 5, bootstraps) | 1076 s | RTX 4090 |
| feature cache | reused from v1 (2742 s for 483 videos) | — |

## 6. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1
uv run sop-monitor train-havid-tas --features artifacts/features/ha-vid/dinov2_vitb14_s1 --hand rh --level pt --selection-metric f1@10 --out reports/havid_dev_v3_tas_f1sel_rh --device auto --epochs 50 --eval-every 5 --n-boot 2000 --seed 0
uv run sop-monitor score-predictions --run reports/havid_dev_v3_tas_f1sel_rh
```

## 7. Not done here

- No weighted or learned fusion, no ASFormer, no atomic-action level, no online SOP metric.
- No number on the frozen test subjects.
