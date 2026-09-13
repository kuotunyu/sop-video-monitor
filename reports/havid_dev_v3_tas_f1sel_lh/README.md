# havid_dev_v3_tas_f1sel_lh — epochs chosen by val F1@10, three parameter-free fusion rules, left hand, primitive tasks (development result)

- Tier: **development result** on `splits/ha-vid/val.csv` (6 subjects: S01, S08, S10, S12, S18,
  S30; 18 recordings × 3 views); each network's epoch is selected on val; the frozen test subjects
  (`splits/ha-vid/test.csv`, 7 subjects) were never read. Not comparable with the paper's Table 3.
- Date: 2026-09-13. Author: kuotunyu. Machine: Windows 11, RTX 4090.
- Purpose: two fixes to [`../havid_dev_v1_tas_lh/`](../havid_dev_v1_tas_lh/README.md) —
  epoch selection by a segmental metric (v1's MoF selection favoured `null` and stopped causal
  networks early) and two more late-fusion rules that need no tuning (normalised geometric mean,
  confidence-weighted mean) next to the plain mean.

## 1. Data and features

As v1: HA-ViD primitive-task labels of the left hand (63 classes in train),
train 143 recordings / 17 subjects / 181,015 frames per view, val 18 / 6 / 17,973; frozen DINOv2
ViT-B/14 1536-d features at stride 1 (`artifacts/features/ha-vid/dinov2_vitb14_s1`).

## 2. Model identity (from `config.json`)

| component | what exactly |
|---|---|
| architecture | MS-TCN++ per view as in v1 (985,408 parameters), causal (left-only padding) and offline (symmetric padding) |
| selection (val only) | val **F1@10** every 5 epochs, best epoch kept per network: `view0_causal` 15 (16.3 → 24.3), `view0_offline` 50 (20.4 → 42.2), `view1_causal` 15 (20.2 → 25.0), `view1_offline` 20 (20.6 → 36.5), `view2_causal` 15 (13.2 → 27.2), `view2_offline` 50 (22.2 → 40.9); arrows give val F1@10 at the first (epoch 5) and last (epoch 50) evaluation |
| `fusion_*` | arithmetic mean of the three per-view posteriors |
| `fusion_geo_*` | normalised geometric mean (product of experts: a view assigning ≈ 0 to a class vetoes it) |
| `fusion_conf_*` | per-frame weighted mean, each view weighted by its own maximum posterior |
| `majority` | constant most frequent training label (`null`) |

## 3. Result (val, spec 4.2 metrics, subject bootstrap 2,000 draws, seed 0)

`tables.md` is authoritative.

| run | temporal context | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|
| majority | — | 36.6 [27.3, 45.4] | 8.3 [6.7, 10.0] | 9.4 [4.8, 13.9] | 3.6 [0.6, 7.5] | 0.7 [0.0, 2.5] |
| view0_causal | past only | 30.3 [26.6, 34.6] | 32.3 [27.2, 38.4] | 26.2 [19.3, 32.8] | 18.5 [14.1, 23.1] | 9.7 [6.8, 12.4] |
| view1_causal | past only | 31.4 [27.1, 36.6] | 33.4 [26.1, 39.1] | 30.1 [24.3, 36.4] | 22.5 [17.6, 27.8] | 12.4 [7.2, 17.3] |
| view2_causal | past only | 33.1 [29.4, 37.0] | 33.2 [26.9, 38.6] | 27.8 [22.4, 34.5] | 20.3 [13.8, 28.6] | 9.5 [7.4, 12.5] |
| fusion_causal | past only | 39.4 [33.8, 45.2] | 35.4 [30.7, 39.8] | 32.0 [27.7, 37.8] | 22.5 [17.3, 27.9] | 14.1 [10.0, 18.6] |
| fusion_geo_causal | past only | 38.3 [33.7, 43.0] | 33.3 [25.0, 41.4] | 32.2 [25.5, 39.7] | 22.9 [17.0, 29.4] | 13.3 [6.2, 20.0] |
| fusion_conf_causal | past only | 38.5 [32.8, 44.2] | 35.8 [31.0, 41.4] | 30.0 [24.3, 36.3] | 21.5 [17.0, 26.1] | 11.6 [6.6, 15.7] |
| view0_offline | past and future | 38.7 [32.0, 46.6] | 45.8 [40.7, 50.5] | 42.2 [33.9, 52.2] | 35.4 [26.0, 46.5] | 22.1 [14.8, 29.9] |
| view1_offline | past and future | 34.5 [27.9, 43.2] | 38.9 [33.3, 44.1] | 41.2 [33.3, 49.9] | 35.9 [27.1, 44.8] | 20.2 [11.7, 29.0] |
| view2_offline | past and future | 39.8 [33.5, 48.6] | 40.3 [33.8, 45.7] | 40.9 [32.6, 51.2] | 32.6 [21.8, 45.8] | 24.6 [15.4, 35.9] |
| fusion_offline | past and future | 42.1 [35.4, 49.8] | 42.3 [34.7, 48.9] | 43.5 [33.0, 52.4] | 35.8 [25.9, 45.4] | 25.1 [14.4, 35.7] |
| fusion_geo_offline | past and future | 42.2 [34.8, 50.4] | 42.1 [35.3, 48.9] | 43.1 [32.0, 53.6] | 36.2 [24.8, 47.6] | 25.4 [12.8, 37.1] |
| fusion_conf_offline | past and future | 42.2 [36.1, 49.5] | 42.0 [33.6, 49.4] | 43.1 [32.7, 52.0] | 36.0 [26.9, 45.7] | 23.9 [13.5, 34.4] |

v1 (MoF selection, mean fusion) vs v3 (F1@10 selection), same features and split:

| run | MoF | Edit | F1@10 | F1@50 |
|---|---|---|---|---|
| v1 fusion_causal | 42.8 [38.8, 46.9] | 28.3 [20.9, 34.3] | 29.1 [23.2, 34.1] | 12.6 [8.5, 16.4] |
| v3 fusion_causal | 39.4 [33.8, 45.2] | 35.4 [30.7, 39.8] | 32.0 [27.7, 37.8] | 14.1 [10.0, 18.6] |
| v1 best single causal view (top, `view2_causal`) | 41.6 [38.5, 44.8] | 27.1 [21.1, 31.8] | 27.2 [23.2, 30.2] | 12.3 [9.6, 14.3] |
| v3 best single causal view (front, `view1_causal`) | 31.4 [27.1, 36.6] | 33.4 [26.1, 39.1] | 30.1 [24.3, 36.4] | 12.4 [7.2, 17.3] |
| v1 fusion_offline | 45.0 [39.4, 51.0] | 44.1 [37.6, 50.9] | 43.5 [34.0, 53.0] | 25.3 [14.3, 36.6] |
| v3 fusion_offline | 42.1 [35.4, 49.8] | 42.3 [34.7, 48.9] | 43.5 [33.0, 52.4] | 25.1 [14.4, 35.7] |

Reading:

- Selecting the epoch by F1@10 instead of MoF trades frame accuracy for segmentation quality on
  the causal fusion: Edit +7.1 (28.3 → 35.4), F1@10 +2.9, F1@50 +1.5, MoF −3.4. The intervals
  still overlap (v1 Edit [20.9, 34.3] vs v3 [30.7, 39.8]), so on six subjects this is a
  consistent direction, not an established gain. Offline fusion is unchanged (F1@10 43.5 both
  times).
- The three fusion rules are tied on the causal runs: F1@10 32.0 (mean) / 32.2 (geometric) /
  30.0 (confidence-weighted), Edit 35.4 / 33.3 / 35.8; every difference is well inside the
  intervals. No rule beats the best single causal view (front, F1@10 30.1) by more than 2 points,
  although mean fusion adds +6.3 MoF over it.
- With F1@10 selection every causal network is kept at epoch 15 and every one of them scores
  lower at epoch 50 than at its best (e.g. `view2_causal` 27.8 → 27.2, `view1_causal`
  30.1 → 25.0): the causal networks over-fit after ≈ 15 epochs on 143 recordings, while two of
  the three offline networks are still improving at epoch 50.
- All numbers are baselines on a 6-subject validation set, not claims.

## 4. Failure cases (`fusion_causal`)

- Most frequent confusions (gt → pred, frames): `sntsb` → `null` 421, `null` → `sspg3dp` 400,
  `w` → `null` 393, `iusn6` → `null` 377, `rgw` → `null` 335. Unlike v1, one large confusion now
  runs the other way (`null` predicted as a gear-plate screwing step).
- Worst val recordings by MoF: S18A04I01 9.5, S12A06I01 18.7, S10A06I01 25.9; best: S12A04I01
  62.5, S30A05I01 58.6, S18A06I01 54.9.

## 5. Cost

| step | wall time | resources |
|---|---|---|
| `train-havid-tas` (6 networks × 50 epochs, val every 5, bootstraps) | 1108 s | RTX 4090 |
| feature cache | reused from v1 (2742 s for 483 videos) | — |

## 6. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1
uv run sop-monitor train-havid-tas --features artifacts/features/ha-vid/dinov2_vitb14_s1 --hand lh --level pt --selection-metric f1@10 --out reports/havid_dev_v3_tas_f1sel_lh --device auto --epochs 50 --eval-every 5 --n-boot 2000 --seed 0
uv run sop-monitor score-predictions --run reports/havid_dev_v3_tas_f1sel_lh
```

## 7. Not done here

- No weighted or learned fusion, no ASFormer, no atomic-action level, no online SOP metric.
- No number on the frozen test subjects.
