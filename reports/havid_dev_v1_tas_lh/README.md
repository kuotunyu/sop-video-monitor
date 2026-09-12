# havid_dev_v1_tas_lh — per-view causal/offline MS-TCN++ and late fusion, left hand, primitive tasks (development result)

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
the left hand (63 classes occur in train; the official `mapping.txt` lists 75);
`null` (pause) is an ordinary class as in the paper; `w` (wrong) is one of the classes.
Train: 143 recordings / 17 subjects / 181,015 frames per view. Val: 18 recordings / 6
subjects / 17,973 frames per view. Features: frozen DINOv2 ViT-B/14 `[CLS ; mean patch]`
1536-d per frame at stride 1 (`artifacts/features/ha-vid/dinov2_vitb14_s1`, weights SHA-256
`0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`), one cache per camera video.

## 2. Model identity (from `config.json`)

| component | what exactly |
|---|---|
| architecture | MS-TCN++ (Li et al., TPAMI 2020): prediction-generation stage of 11 dual-dilated layers, 3 refinement stages of 10 dilated residual layers, 64 feature maps, dropout 0.5; 985,408 parameters; input = standardised features of one view |
| `view{v}_causal` | one network per view (0 = side `M0`, 1 = front `S1`, 2 = top `S2`); every kernel-3 convolution left-padded, never right-padded → frame *t* sees frames ≤ *t* only |
| `view{v}_offline` | same networks with symmetric padding → see future frames; **not** an online result |
| `fusion_causal` / `fusion_offline` | arithmetic mean of the three per-view posteriors, then argmax; no extra parameters |
| loss / optimiser | cross-entropy on every stage + 0.15 × truncated MSE smoothing (clamp 16); Adam lr 5e-4, one video per step, 50 epochs, seed 0, `cudnn.deterministic` |
| selection (val only) | val MoF every 5 epochs, best epoch kept per network: `view0_causal` 10 (19.2 → 31.1), `view0_offline` 50 (27.8 → 38.7), `view1_causal` 5 (33.1 → 29.2), `view1_offline` 50 (31.6 → 39.4), `view2_causal` 50 (37.6 → 41.6), `view2_offline` 40 (33.2 → 39.8); arrows give val MoF at the first (epoch 5) and last (epoch 50) evaluation |
| `majority` | constant most frequent training label (`null`) |

## 3. Result (val, spec 4.2 metrics, subject bootstrap 2,000 draws, seed 0)

`tables.md` is authoritative.

| run | temporal context | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|
| majority | — | 36.6 [27.3, 45.4] | 8.3 [6.7, 10.0] | 9.4 [4.8, 13.9] | 3.6 [0.6, 7.5] | 0.7 [0.0, 2.5] |
| view0_causal | past only | 33.5 [27.6, 40.6] | 29.0 [23.3, 34.0] | 25.1 [19.5, 31.1] | 14.8 [10.3, 20.4] | 5.8 [3.1, 9.1] |
| view1_causal | past only | 33.1 [26.1, 40.6] | 18.9 [16.5, 21.6] | 20.2 [17.9, 22.5] | 11.6 [7.3, 14.8] | 5.5 [2.9, 8.5] |
| view2_causal | past only | 41.6 [38.5, 44.8] | 27.1 [21.1, 31.8] | 27.2 [23.2, 30.2] | 20.5 [14.4, 25.7] | 12.3 [9.6, 14.3] |
| fusion_causal | past only | 42.8 [38.8, 46.9] | 28.3 [20.9, 34.3] | 29.1 [23.2, 34.1] | 21.5 [14.0, 28.3] | 12.6 [8.5, 16.4] |
| view0_offline | past and future | 38.7 [32.0, 46.6] | 45.8 [40.7, 50.5] | 42.2 [33.9, 52.2] | 35.4 [26.0, 46.5] | 22.1 [14.8, 29.9] |
| view1_offline | past and future | 39.4 [34.1, 44.8] | 40.7 [34.6, 47.1] | 36.5 [25.9, 49.2] | 31.6 [22.8, 42.7] | 20.3 [12.3, 30.1] |
| view2_offline | past and future | 40.8 [34.2, 49.1] | 41.5 [35.8, 47.6] | 39.9 [31.0, 50.5] | 36.1 [27.4, 46.7] | 23.4 [13.1, 35.9] |
| fusion_offline | past and future | 45.0 [39.4, 51.0] | 44.1 [37.6, 50.9] | 43.5 [34.0, 53.0] | 36.8 [24.6, 48.5] | 25.3 [14.3, 36.6] |

Reading:

- Late fusion is the best causal run on MoF and all three F1 thresholds, but only marginally:
  against the strongest single causal view (top, `view2_causal`) it adds +1.2 MoF (42.8 vs 41.6),
  +1.9 F1@10 (29.1 vs 27.2) and +0.3 F1@50 (12.6 vs 12.3), all well inside overlapping confidence
  intervals. On Edit the side view alone is higher (29.0 vs 28.3, also inside the intervals).
- Among causal networks the top view is strongest on F1@10 (27.2) and MoF (41.6) and the front
  view weakest (F1@10 20.2, Edit 18.9); the side view has the highest causal Edit (29.0).
- Causality is expensive here: for fusion the offline twin is +2.2 MoF, +15.8 Edit and
  +12.7 F1@50 above the causal run.
- Among offline runs fusion leads on MoF and all three F1 thresholds but not on Edit, where the
  single side view is higher (45.8 vs 44.1, inside the intervals).
- The causal side and front networks are *below the majority baseline on MoF* (33.5 and 33.1 vs
  36.6) while far above it on Edit and F1: the constant `null` prediction scores well on MoF
  because `null` covers a third of the frames. MoF-based epoch selection accordingly kept early
  epochs (5 and 10) for these two causal networks, while the side and front offline networks
  were still at their best at the last epoch, 50 (possibly not converged). Epoch selection by a segmental metric is a
  follow-up, not done here.
- All numbers are baselines for the pipeline on a 6-subject validation set, not claims.

## 4. Failure cases (`fusion_causal`, from `metrics.json["confusions"]` and `per_video`)

- Most frequent confusions (gt → pred, frames): `rgw` → `null` 361, `ibscb` → `null` 332,
  `sshc1` → `null` 295, `sshc4dh` → `null` 277, `w` → `null` 262. Every top confusion is a real
  action (including the `wrong` label) predicted as a pause.
- Worst val recordings by MoF: S18A04I01 16.6, S08A05I01 25.1, S10A06I01 25.9; best: S12A05I01
  67.6, S18A06I01 66.0, S12A04I01 62.5.

## 5. Cost

| step | wall time | resources |
|---|---|---|
| `train-havid-tas` (6 networks × 50 epochs, val every 5, bootstraps) | 1077 s (168, 187, 172, 178, 169, 184 s per network in the order of section 2) | RTX 4090, ≈ 2.5 GB VRAM |
| feature cache (shared by both hands) | 2742 s for 483 videos (596,964 frames) | RTX 4090, decode-bound |

## 6. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1
make features-havid          # once; DINOv2 ViT-B/14 cache for the train + val videos
uv run sop-monitor train-havid-tas --features artifacts/features/ha-vid/dinov2_vitb14_s1 --hand lh --level pt --out reports/havid_dev_v1_tas_lh --device auto --epochs 50 --eval-every 5 --n-boot 2000 --seed 0
uv run sop-monitor score-predictions --run reports/havid_dev_v1_tas_lh
```

## 7. Not done here

- Only one hand per run and only primitive tasks; no atomic-action (219-class) level, no ASFormer, no
  hyper-parameter search beyond the epoch, no early or mid-level fusion.
- No online SOP metrics (completion definition for HA-ViD steps is still open), no `wrong`-label
  detection table, no synthetic violations.
- No number on the frozen test subjects.
