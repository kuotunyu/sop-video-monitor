# havid_dev_v1_tas_rh — per-view causal/offline MS-TCN++ and late fusion, right hand, primitive tasks (development result)

- Tier: **development result** on `splits/ha-vid/val.csv` (6 subjects: S01, S08, S10, S12, S18,
  S30; 18 recordings × 3 views); the epoch is selected on val; the frozen test subjects
  (`splits/ha-vid/test.csv`, 7 subjects) were never read. Not comparable with the paper's Table 3,
  which uses the official test subjects, a single view and no fusion.
- Date: 2026-09-13. Author: kuotunyu. Machine: Windows 11, RTX 4090.
- Purpose: the first HA-ViD numbers of this repository — does a frozen-feature temporal head work
  on HA-ViD primitive tasks at all, which camera view carries the most information, and does
  averaging the three views' posteriors (late fusion) beat the best single view?

## 1. Data and features

HA-ViD (CC BY-NC 4.0), audited in [`../havid_audit.md`](../havid_audit.md): temporal annotations
are inclusive, contiguous frame ranges at 15 fps; labels are the pt-level HR-SAT codes of
the right hand (65 classes occur in train; the official `mapping.txt` lists 75);
`null` (pause) is an ordinary class as in the paper; `w` (wrong) is one of the classes.
Train: 143 recordings / 17 subjects / 181,015 frames per view. Val: 18 recordings / 6
subjects / 17,973 frames per view. Features: frozen DINOv2 ViT-B/14 `[CLS ; mean patch]`
1536-d per frame at stride 1 (`artifacts/features/ha-vid/dinov2_vitb14_s1`, weights SHA-256
`0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`), one cache per camera video.

## 2. Model identity (from `config.json`)

| component | what exactly |
|---|---|
| architecture | MS-TCN++ (Li et al., TPAMI 2020): prediction-generation stage of 11 dual-dilated layers, 3 refinement stages of 10 dilated residual layers, 64 feature maps, dropout 0.5; 986,312 parameters; input = standardised features of one view |
| `view{v}_causal` | one network per view (0 = side `M0`, 1 = front `S1`, 2 = top `S2`); every kernel-3 convolution left-padded, never right-padded → frame *t* sees frames ≤ *t* only |
| `view{v}_offline` | same networks with symmetric padding → see future frames; **not** an online result |
| `fusion_causal` / `fusion_offline` | arithmetic mean of the three per-view posteriors, then argmax; no extra parameters |
| loss / optimiser | cross-entropy on every stage + 0.15 × truncated MSE smoothing (clamp 16); Adam lr 5e-4, one video per step, 50 epochs, seed 0, `cudnn.deterministic` |
| selection (val only) | val MoF every 5 epochs, best epoch kept per network: `view0_causal` 30 (29.6 → 33.2), `view0_offline` 10 (29.8 → 35.9), `view1_causal` 20 (28.1 → 27.1), `view1_offline` 40 (30.7 → 35.4), `view2_causal` 35 (29.9 → 33.4), `view2_offline` 40 (30.7 → 37.2); arrows give val MoF at the first (epoch 5) and last (epoch 50) evaluation |
| `majority` | constant most frequent training label (`null`) |

## 3. Result (val, spec 4.2 metrics, subject bootstrap 2,000 draws, seed 0)

`tables.md` is authoritative.

| run | temporal context | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|
| majority | — | 29.8 [24.6, 34.1] | 8.0 [6.8, 9.3] | 8.9 [5.9, 11.8] | 0.7 [0.0, 2.3] | 0.0 [0.0, 0.0] |
| view0_causal | past only | 35.1 [28.9, 43.3] | 27.7 [22.0, 33.2] | 24.2 [17.6, 31.4] | 18.9 [12.6, 26.0] | 10.7 [6.6, 15.4] |
| view1_causal | past only | 33.2 [25.4, 43.2] | 29.0 [25.1, 33.1] | 26.6 [20.5, 34.3] | 23.8 [17.7, 31.4] | 13.1 [8.1, 19.8] |
| view2_causal | past only | 35.0 [29.2, 44.5] | 27.9 [24.8, 31.0] | 25.5 [21.7, 31.1] | 19.9 [15.4, 26.7] | 9.6 [6.0, 14.2] |
| fusion_causal | past only | 38.6 [32.3, 47.2] | 33.5 [26.8, 39.0] | 30.3 [24.5, 36.9] | 23.8 [17.7, 30.8] | 13.6 [9.4, 18.7] |
| view0_offline | past and future | 36.7 [29.1, 46.8] | 36.9 [28.3, 46.1] | 36.9 [29.2, 47.6] | 30.2 [21.1, 40.8] | 19.7 [12.4, 28.3] |
| view1_offline | past and future | 36.7 [29.6, 45.8] | 38.5 [33.3, 43.7] | 37.4 [31.8, 44.0] | 32.5 [26.1, 39.2] | 22.8 [18.2, 29.1] |
| view2_offline | past and future | 38.5 [32.0, 47.1] | 41.9 [36.6, 48.0] | 40.6 [35.8, 47.3] | 34.0 [27.3, 41.4] | 22.4 [16.6, 30.3] |
| fusion_offline | past and future | 41.8 [33.6, 52.5] | 40.9 [32.2, 48.7] | 41.3 [34.0, 49.9] | 35.0 [26.9, 44.5] | 23.4 [17.1, 32.2] |

Reading:

- Late fusion is the best causal run on MoF, Edit, F1@10 and F1@50 and ties the front view on
  F1@25 (23.8): against the best single causal view on each metric it adds +3.7 F1@10 (30.3 vs
  front 26.6), +4.5 Edit (33.5 vs front 29.0), +3.5 MoF (38.6 vs side 35.1) and +0.5 F1@50 (13.6 vs
  front 13.1); the confidence intervals still overlap, so the gain is not established on six
  subjects.
- Among causal networks the front view is strongest on F1@10 (26.6), Edit (29.0) and F1@50 (13.1)
  and the side view weakest on F1@10 (24.2). The ranking differs from the left hand, where the top
  view led.
- Causality costs, for fusion, 3.2 MoF, 7.4 Edit and 9.8 F1@50 (offline 41.8 / 40.9 / 23.4 vs
  causal 38.6 / 33.5 / 13.6).
- Among offline runs fusion leads on MoF, F1@10, F1@25 and F1@50 but not on Edit, where the single
  top view is higher (41.9 vs 40.9, inside the intervals).
- All numbers are baselines for the pipeline on a 6-subject validation set, not claims.

## 4. Failure cases (`fusion_causal`, from `metrics.json["confusions"]` and `per_video`)

- Most frequent confusions (gt → pred, frames): `sspn4` → `null` 428, `rgw` → `null` 425,
  `w` → `null` 396, `sntft` → `null` 319, `sntsb` → `null` 302. As for the left hand, the top
  confusions are real actions (including the `wrong` label) predicted as a pause.
- Worst val recordings by MoF: S12A04I01 17.1, S18A04I01 18.6, S01A06I01 19.0; best: S18A06I01
  68.1, S08A06I01 67.0, S30A05I01 60.3.

## 5. Cost

| step | wall time | resources |
|---|---|---|
| `train-havid-tas` (6 networks × 50 epochs, val every 5, bootstraps) | 1103 s (176, 188, 179, 184, 173, 184 s per network in the order of section 2) | RTX 4090, ≈ 2.5 GB VRAM |
| feature cache (shared by both hands) | 2742 s for 483 videos (596,964 frames) | RTX 4090, decode-bound |

## 6. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1
make features-havid          # once; DINOv2 ViT-B/14 cache for the train + val videos
uv run sop-monitor train-havid-tas --features artifacts/features/ha-vid/dinov2_vitb14_s1 --hand rh --level pt --out reports/havid_dev_v1_tas_rh --device auto --epochs 50 --eval-every 5 --n-boot 2000 --seed 0
uv run sop-monitor score-predictions --run reports/havid_dev_v1_tas_rh
```

## 7. Not done here

- Only one hand per run and only primitive tasks; no atomic-action (219-class) level, no ASFormer, no
  hyper-parameter search beyond the epoch, no early or mid-level fusion.
- No online SOP metrics (completion definition for HA-ViD steps is still open), no `wrong`-label
  detection table, no synthetic violations.
- No number on the frozen test subjects.
