# havid_dev_v2_i3d_rh — per-view causal/offline MS-TCN++ and late fusion on the authors' I3D features, right hand, primitive tasks (development result)

- Tier: **development result** on `splits/ha-vid/val.csv` (6 subjects: S01, S08, S10, S12, S18,
  S30; 18 recordings × 3 views); the epoch is selected on val; the frozen test subjects
  (`splits/ha-vid/test.csv`, 7 subjects) were never read. Not comparable with the paper's Table 3,
  which uses the official test subjects, a single view and no fusion.
- Date: 2026-09-13. Author: kuotunyu. Machine: Windows 11, RTX 4090.
- Purpose: a control for [`../havid_dev_v1_tas_rh/`](../havid_dev_v1_tas_rh/README.md) — the same
  protocol and code on the features the HA-ViD authors used for their benchmark, to see whether
  the frozen DINOv2 frame features are a reasonable default for the HA-ViD line.

## 1. Data and features

HA-ViD (CC BY-NC 4.0), audited in [`../havid_audit.md`](../havid_audit.md): temporal annotations
are inclusive, contiguous frame ranges at 15 fps; labels are the pt-level HR-SAT codes of
the right hand (65 classes occur in train; the official `mapping.txt` lists 75);
`null` (pause) is an ordinary class as in the paper; `w` (wrong) is one of the classes.
Train: 143 recordings / 17 subjects / 179,585 frames per view. Val: 18 recordings / 6
subjects / 17,793 frames per view. Features: the authors' I3D features from
`ActionSegmentation/data/features` (2048-d, one vector per frame, float64 stored as float32, no
weight digest available), exported by `sop-monitor export-official-features`; they cover
annotation frames 5 … n−6, so the tables have 17,793 val frames instead of 17,973.

## 2. Model identity (from `config.json`)

| component | what exactly |
|---|---|
| architecture | MS-TCN++ (Li et al., TPAMI 2020): prediction-generation stage of 11 dual-dilated layers, 3 refinement stages of 10 dilated residual layers, 64 feature maps, dropout 0.5; 1,019,080 parameters; input = standardised features of one view |
| `view{v}_causal` | one network per view (0 = side `M0`, 1 = front `S1`, 2 = top `S2`); every kernel-3 convolution left-padded, never right-padded → frame *t* sees frames ≤ *t* only |
| `view{v}_offline` | same networks with symmetric padding → see future frames; **not** an online result |
| `fusion_causal` / `fusion_offline` | arithmetic mean of the three per-view posteriors, then argmax; no extra parameters |
| loss / optimiser | cross-entropy on every stage + 0.15 × truncated MSE smoothing (clamp 16); Adam lr 5e-4, one video per step, 50 epochs, seed 0, `cudnn.deterministic` |
| selection (val only) | val MoF every 5 epochs, best epoch kept per network: `view0_causal` 50 (26.4 → 32.6), `view0_offline` 35 (29.1 → 32.7), `view1_causal` 10 (28.8 → 28.9), `view1_offline` 15 (28.6 → 35.0), `view2_causal` 15 (28.5 → 33.4), `view2_offline` 30 (31.0 → 39.8); arrows give val MoF at the first (epoch 5) and last (epoch 50) evaluation |
| `majority` | constant most frequent training label (`null`) |

## 3. Result (val, spec 4.2 metrics, subject bootstrap 2,000 draws, seed 0)

`tables.md` is authoritative.

| run | temporal context | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|
| majority | — | 29.2 [23.8, 33.5] | 8.0 [6.8, 9.3] | 8.2 [5.1, 11.2] | 0.7 [0.0, 2.3] | 0.0 [0.0, 0.0] |
| view0_causal | past only | 32.6 [25.1, 42.3] | 16.2 [12.0, 20.6] | 15.1 [10.8, 21.1] | 13.2 [9.6, 18.5] | 7.9 [5.5, 11.5] |
| view1_causal | past only | 35.0 [28.6, 41.0] | 30.5 [27.6, 33.3] | 27.2 [21.8, 33.2] | 20.9 [16.9, 25.1] | 11.7 [7.5, 16.4] |
| view2_causal | past only | 37.0 [30.3, 44.4] | 26.3 [20.9, 31.7] | 24.4 [18.5, 30.4] | 20.3 [15.4, 25.1] | 10.0 [7.3, 12.5] |
| fusion_causal | past only | 39.5 [32.9, 48.5] | 21.0 [17.7, 24.2] | 23.0 [18.7, 29.1] | 18.7 [15.4, 23.3] | 10.3 [7.3, 13.3] |
| view0_offline | past and future | 37.9 [29.4, 48.2] | 34.1 [27.6, 42.1] | 34.4 [27.2, 43.4] | 28.5 [21.0, 38.0] | 19.1 [12.9, 26.7] |
| view1_offline | past and future | 36.2 [26.7, 48.3] | 38.1 [29.4, 47.0] | 36.1 [27.3, 47.8] | 29.4 [21.7, 38.8] | 17.7 [10.1, 26.7] |
| view2_offline | past and future | 40.8 [32.8, 51.0] | 41.5 [32.9, 49.9] | 38.4 [30.4, 48.5] | 34.3 [25.1, 44.8] | 23.9 [15.7, 34.1] |
| fusion_offline | past and future | 44.5 [35.6, 56.0] | 39.7 [33.6, 45.3] | 39.6 [30.8, 50.2] | 32.0 [24.6, 40.9] | 23.5 [16.5, 32.1] |

DINOv2 (v1) vs I3D (v2), same protocol, val. The frame sets differ slightly: v1 scores all 17,973
val frames, v2 the 17,793 frames the I3D features cover (5 frames fewer at each end of every
recording), so the comparison is close to, but not exactly, paired.

| run | features | MoF | Edit | F1@10 | F1@50 |
|---|---|---|---|---|---|
| fusion_causal | DINOv2 (v1) | 38.6 [32.3, 47.2] | 33.5 [26.8, 39.0] | 30.3 [24.5, 36.9] | 13.6 [9.4, 18.7] |
| fusion_causal | I3D (v2) | 39.5 [32.9, 48.5] | 21.0 [17.7, 24.2] | 23.0 [18.7, 29.1] | 10.3 [7.3, 13.3] |
| fusion_offline | DINOv2 (v1) | 41.8 [33.6, 52.5] | 40.9 [32.2, 48.7] | 41.3 [34.0, 49.9] | 23.4 [17.1, 32.2] |
| fusion_offline | I3D (v2) | 44.5 [35.6, 56.0] | 39.7 [33.6, 45.3] | 39.6 [30.8, 50.2] | 23.5 [16.5, 32.1] |
| best single causal view (by F1@10) | DINOv2 (v1): front, `view1_causal` | 33.2 [25.4, 43.2] | 29.0 [25.1, 33.1] | 26.6 [20.5, 34.3] | 13.1 [8.1, 19.8] |
| best single causal view (by F1@10) | I3D (v2): front, `view1_causal` | 35.0 [28.6, 41.0] | 30.5 [27.6, 33.3] | 27.2 [21.8, 33.2] | 11.7 [7.5, 16.4] |

Reading:

- For the causal fusion of the right hand DINOv2 is clearly ahead on Edit (33.5 vs 21.0; the
  intervals [26.8, 39.0] and [17.7, 24.2] do not overlap) and ahead on F1@10 (+7.3) and F1@50
  (+3.3) with overlapping intervals; I3D is +0.9 MoF.
- The difference is a fusion effect, not a best-view effect: the best single causal view is the
  front view for both feature sets and the two are within 2 points of each other on every metric
  (I3D +1.8 MoF, +1.5 Edit, +0.6 F1@10; DINOv2 +1.4 F1@50).
- With I3D the side view is weak (causal Edit 16.2, F1@10 15.1), and averaging it in makes the
  causal fusion *worse* than the front view alone on Edit (21.0 vs 30.5, non-overlapping intervals)
  and on F1@10, F1@25 and F1@50; only MoF improves (39.5 vs best single 37.0). Plain posterior
  averaging is therefore not safe when one view is poor; with DINOv2 the same averaging helped.
- For the offline fusion the two feature sets are within 3 points on every metric, all inside the
  intervals (I3D +2.7 MoF and +0.1 F1@50, DINOv2 +1.2 Edit and +1.7 F1@10).
- Causality costs, for the I3D fusion, 5.0 MoF, 18.7 Edit and 13.2 F1@50. Among offline I3D runs
  fusion leads on MoF and F1@10, but the single top view is higher on Edit (41.5 vs 39.7), F1@25
  (34.3 vs 32.0) and F1@50 (23.9 vs 23.5).
- All numbers are baselines for the pipeline on a 6-subject validation set, not claims.

## 4. Failure cases (`fusion_causal`, from `metrics.json["confusions"]` and `per_video`)

- Most frequent confusions (gt → pred, frames): `sntft` → `null` 430, `rgw` → `null` 328,
  `sshc1` → `null` 282, `sspg3dp` → `null` 259, `ibscb` → `null` 251; `w` → `null` follows with
  221 frames. The top confusions are real actions predicted as a pause, as in every other run.
- Worst val recordings by MoF: S12A04I01 12.4, S18A05I01 14.3, S18A04I01 19.4; best: S08A06I01
  66.6, S08A04I01 60.7, S30A06I01 59.4.

## 5. Cost

| step | wall time | resources |
|---|---|---|
| `train-havid-tas` (6 networks × 50 epochs, val every 5, bootstraps) | 1075 s (185, 175, 177, 175, 170, 171 s per network in the order of section 2) | RTX 4090, ≈ 2.9 GB VRAM |
| features | official features, no extraction (exported once from the benchmark zip by `make i3d-havid`) | CPU |

## 6. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1
make i3d-havid               # once; exports the official I3D features of the train + val videos
uv run sop-monitor train-havid-tas --features artifacts/features/ha-vid/i3d_official --hand rh --level pt --out reports/havid_dev_v2_i3d_rh --device auto --epochs 50 --eval-every 5 --n-boot 2000 --seed 0
uv run sop-monitor score-predictions --run reports/havid_dev_v2_i3d_rh
```

## 7. Not done here

- Only one hand per run and only primitive tasks; no atomic-action (219-class) level, no ASFormer, no
  hyper-parameter search beyond the epoch, no early or mid-level fusion.
- No online SOP metrics (completion definition for HA-ViD steps is still open), no `wrong`-label
  detection table, no synthetic violations.
- No number on the frozen test subjects.
