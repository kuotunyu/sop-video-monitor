# havid_dev_v2_i3d_lh — per-view causal/offline MS-TCN++ and late fusion on the authors' I3D features, left hand, primitive tasks (development result)

- Tier: **development result** on `splits/ha-vid/val.csv` (6 subjects: S01, S08, S10, S12, S18,
  S30; 18 recordings × 3 views); the epoch is selected on val; the frozen test subjects
  (`splits/ha-vid/test.csv`, 7 subjects) were never read. Not comparable with the paper's Table 3,
  which uses the official test subjects, a single view and no fusion.
- Date: 2026-09-13. Author: kuotunyu. Machine: Windows 11, RTX 4090.
- Purpose: a control for [`../havid_dev_v1_tas_lh/`](../havid_dev_v1_tas_lh/README.md) — the same
  protocol and code on the features the HA-ViD authors used for their benchmark, to see whether
  the frozen DINOv2 frame features are a reasonable default for the HA-ViD line.

## 1. Data and features

HA-ViD (CC BY-NC 4.0), audited in [`../havid_audit.md`](../havid_audit.md): temporal annotations
are inclusive, contiguous frame ranges at 15 fps; labels are the pt-level HR-SAT codes of
the left hand (63 classes occur in train; the official `mapping.txt` lists 75);
`null` (pause) is an ordinary class as in the paper; `w` (wrong) is one of the classes.
Train: 143 recordings / 17 subjects / 179,585 frames per view. Val: 18 recordings / 6
subjects / 17,793 frames per view. Features: the authors' I3D features from
`ActionSegmentation/data/features` (2048-d, one vector per frame, float64 stored as float32, no
weight digest available), exported by `sop-monitor export-official-features`; they cover
annotation frames 5 … n−6, so the tables have 17,793 val frames instead of 17,973.

## 2. Model identity (from `config.json`)

| component | what exactly |
|---|---|
| architecture | MS-TCN++ (Li et al., TPAMI 2020): prediction-generation stage of 11 dual-dilated layers, 3 refinement stages of 10 dilated residual layers, 64 feature maps, dropout 0.5; 1,018,176 parameters; input = standardised features of one view |
| `view{v}_causal` | one network per view (0 = side `M0`, 1 = front `S1`, 2 = top `S2`); every kernel-3 convolution left-padded, never right-padded → frame *t* sees frames ≤ *t* only |
| `view{v}_offline` | same networks with symmetric padding → see future frames; **not** an online result |
| `fusion_causal` / `fusion_offline` | arithmetic mean of the three per-view posteriors, then argmax; no extra parameters |
| loss / optimiser | cross-entropy on every stage + 0.15 × truncated MSE smoothing (clamp 16); Adam lr 5e-4, one video per step, 50 epochs, seed 0, `cudnn.deterministic` |
| selection (val only) | val MoF every 5 epochs, best epoch kept per network: `view0_causal` 10 (29.6 → 32.4), `view0_offline` 10 (34.9 → 33.2), `view1_causal` 10 (28.0 → 33.3), `view1_offline` 25 (27.9 → 34.6), `view2_causal` 30 (30.8 → 30.5), `view2_offline` 15 (35.7 → 40.1); arrows give val MoF at the first (epoch 5) and last (epoch 50) evaluation |
| `majority` | constant most frequent training label (`null`) |

## 3. Result (val, spec 4.2 metrics, subject bootstrap 2,000 draws, seed 0)

`tables.md` is authoritative.

| run | temporal context | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|
| majority | — | 36.1 [26.7, 44.8] | 8.3 [6.7, 10.0] | 9.4 [4.8, 13.9] | 3.6 [0.6, 7.5] | 0.7 [0.0, 2.5] |
| view0_causal | past only | 36.9 [31.5, 42.4] | 26.6 [22.3, 31.1] | 23.0 [17.7, 29.0] | 19.3 [14.0, 25.2] | 10.5 [8.1, 13.2] |
| view1_causal | past only | 35.8 [29.7, 41.7] | 30.1 [25.7, 34.6] | 26.5 [21.4, 31.9] | 19.8 [14.7, 24.2] | 10.5 [8.0, 13.0] |
| view2_causal | past only | 36.8 [34.2, 39.6] | 22.0 [17.5, 26.8] | 22.6 [16.8, 30.5] | 17.2 [12.0, 23.6] | 9.1 [5.9, 12.8] |
| fusion_causal | past only | 44.1 [40.8, 47.9] | 28.0 [25.9, 31.3] | 28.7 [23.1, 34.9] | 21.6 [16.2, 27.1] | 12.8 [9.9, 15.8] |
| view0_offline | past and future | 38.0 [33.8, 43.4] | 32.7 [27.2, 38.6] | 31.3 [24.2, 39.8] | 26.2 [19.4, 35.2] | 15.3 [9.6, 22.8] |
| view1_offline | past and future | 38.4 [31.9, 46.1] | 35.7 [28.9, 42.4] | 34.8 [27.3, 43.0] | 29.5 [22.1, 37.5] | 19.1 [12.2, 26.5] |
| view2_offline | past and future | 45.2 [41.5, 50.6] | 39.8 [30.8, 49.8] | 43.9 [34.2, 54.1] | 36.0 [25.7, 46.9] | 20.9 [11.6, 30.4] |
| fusion_offline | past and future | 46.9 [43.4, 51.1] | 40.6 [33.0, 48.1] | 41.8 [32.4, 52.3] | 35.1 [25.2, 46.2] | 22.5 [12.7, 31.8] |

DINOv2 (v1) vs I3D (v2), same protocol, val. The frame sets differ slightly: v1 scores all 17,973
val frames, v2 the 17,793 frames the I3D features cover (5 frames fewer at each end of every
recording), so the comparison is close to, but not exactly, paired.

| run | features | MoF | Edit | F1@10 | F1@50 |
|---|---|---|---|---|---|
| fusion_causal | DINOv2 (v1) | 42.8 [38.8, 46.9] | 28.3 [20.9, 34.3] | 29.1 [23.2, 34.1] | 12.6 [8.5, 16.4] |
| fusion_causal | I3D (v2) | 44.1 [40.8, 47.9] | 28.0 [25.9, 31.3] | 28.7 [23.1, 34.9] | 12.8 [9.9, 15.8] |
| fusion_offline | DINOv2 (v1) | 45.0 [39.4, 51.0] | 44.1 [37.6, 50.9] | 43.5 [34.0, 53.0] | 25.3 [14.3, 36.6] |
| fusion_offline | I3D (v2) | 46.9 [43.4, 51.1] | 40.6 [33.0, 48.1] | 41.8 [32.4, 52.3] | 22.5 [12.7, 31.8] |
| best single causal view (by F1@10) | DINOv2 (v1): top, `view2_causal` | 41.6 [38.5, 44.8] | 27.1 [21.1, 31.8] | 27.2 [23.2, 30.2] | 12.3 [9.6, 14.3] |
| best single causal view (by F1@10) | I3D (v2): front, `view1_causal` | 35.8 [29.7, 41.7] | 30.1 [25.7, 34.6] | 26.5 [21.4, 31.9] | 10.5 [8.0, 13.0] |

Reading:

- DINOv2 and I3D are indistinguishable for the causal fusion of the left hand: every difference is
  at most 1.3 points and inside both intervals (I3D +1.3 MoF and +0.2 F1@50, DINOv2 +0.3 Edit and
  +0.4 F1@10).
- For the offline fusion DINOv2 is ahead on Edit (+3.5), F1@10 (+1.7) and F1@50 (+2.8) and I3D on
  MoF (+1.9); all inside the intervals.
- With I3D, fusion is the best causal run on MoF (44.1 vs side 36.9, a gain larger than with
  DINOv2) and on all three F1 thresholds, but the front view alone has the higher Edit (30.1 vs
  28.0). The strongest causal view on F1@10 is the front view (26.5) and the weakest the top view
  (22.6), the reverse of DINOv2 for this hand.
- Causality costs, for fusion, 2.8 MoF, 12.6 Edit and 9.7 F1@50. Among offline runs fusion leads
  on MoF, Edit and F1@50 but the single top view is higher on F1@10 (43.9 vs 41.8) and F1@25
  (36.0 vs 35.1).
- The I3D networks peak early: all six best epochs are between 10 and 30, and every network scores
  lower on val MoF at epoch 50 than at its best epoch (e.g. `view2_offline` 45.2 at epoch 15,
  40.1 at epoch 50).
- All numbers are baselines for the pipeline on a 6-subject validation set, not claims.

## 4. Failure cases (`fusion_causal`, from `metrics.json["confusions"]` and `per_video`)

- Most frequent confusions (gt → pred, frames): `w` → `null` 387, `rgw` → `null` 336,
  `sftg2` → `null` 321, `ibscb` → `null` 263, `sspg3dp` → `null` 222. As with DINOv2, the top
  confusions are real actions predicted as a pause, and the `wrong` label is the most confused one.
- Worst val recordings by MoF: S18A05I01 13.2, S12A06I01 26.2, S08A05I01 29.6; best: S12A04I01
  63.3, S18A06I01 61.1, S12A05I01 60.6.

## 5. Cost

| step | wall time | resources |
|---|---|---|
| `train-havid-tas` (6 networks × 50 epochs, val every 5, bootstraps) | 1131 s (176, 178, 177, 178, 175, 172 s per network in the order of section 2) | RTX 4090, ≈ 2.9 GB VRAM |
| features | official features, no extraction (exported once from the benchmark zip by `make i3d-havid`) | CPU |

## 6. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1
make i3d-havid               # once; exports the official I3D features of the train + val videos
uv run sop-monitor train-havid-tas --features artifacts/features/ha-vid/i3d_official --hand lh --level pt --out reports/havid_dev_v2_i3d_lh --device auto --epochs 50 --eval-every 5 --n-boot 2000 --seed 0
uv run sop-monitor score-predictions --run reports/havid_dev_v2_i3d_lh
```

## 7. Not done here

- Only one hand per run and only primitive tasks; no atomic-action (219-class) level, no ASFormer, no
  hyper-parameter search beyond the epoch, no early or mid-level fusion.
- No online SOP metrics (completion definition for HA-ViD steps is still open), no `wrong`-label
  detection table, no synthetic violations.
- No number on the frozen test subjects.
