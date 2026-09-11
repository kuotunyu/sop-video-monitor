# industreal_dev_v2_mstcn — causal vs non-causal MS-TCN++ on the same features (development result)

- Tier: **development result**, validation split, same data and same feature cache as
  [`../industreal_dev_v1/`](../industreal_dev_v1/README.md); the epoch is selected on val, the
  frozen test split is never read. Not an HA-ViD result.
- Date: 2026-09-11. Author: kuotunyu. Machine: Windows 11, RTX 4090.
- Purpose: the spec's W2/W3 question in miniature — what does a temporal head add over a
  per-frame probe, and what does causality cost? Both heads are trained identically; the only
  difference is the padding of every kernel-3 convolution (left-only vs symmetric).

## 1. Data and features

Identical to `industreal_dev_v1` (see its `data_audit.json`, copied semantics: 10 fps, 0-based
frames, half-open segments, latest onset wins on overlaps): train 36 videos / 12 participants /
78,932 frames; val 16 videos / 5 participants / 38,036 frames; frozen DINOv2 ViT-S/14
`[CLS ; mean patch]` 768-d features at stride 1, weights SHA-256
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.

## 2. Model identity (from `config.json`)

| component | what exactly |
|---|---|
| architecture | MS-TCN++ (Li et al., TPAMI 2020): prediction-generation stage of 11 dual-dilated layers, 3 refinement stages of 10 dilated residual layers, 64 feature maps, dropout 0.5; 940,324 parameters; input = standardised 768-d features |
| `mstcn_causal` | every kernel-3 convolution left-padded by `2·dilation`, never right-padded → frame *t* sees frames ≤ *t* only (numerically verified in `tests/test_mstcn.py`) |
| `mstcn_offline` | same network with symmetric padding → sees future frames; **not** an online result |
| loss / optimiser | cross-entropy on every stage + 0.15 × truncated MSE smoothing (clamp 16) as in MS-TCN; Adam lr 5e-4, one video per step, 50 epochs, seed 0, `cudnn.deterministic` |
| selection (val only) | val MoF evaluated every 5 epochs, best epoch kept: causal **35** (curve 19.9 → 34.8, dips to 31.5 at 40), offline **50** (still improving: 39.1 at the last epoch, so this variant is under-trained rather than converged) |
| `majority` | constant most frequent training label (background) |

## 3. Result (val, spec 4.2 metrics, participant bootstrap 2,000 draws, seed 0)

`tables.md` is authoritative; the linear rows are copied from `industreal_dev_v1/tables.md` for comparison.

| run | temporal context | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|
| majority | — | 20.0 [15.5, 25.3] | 0.0 | 0.0 | 0.0 | 0.0 |
| linear `frame` (v1) | none | 32.4 [31.7, 33.0] | 11.0 [10.1, 11.8] | 11.9 | 7.5 | 4.1 [3.3, 5.0] |
| linear `causal` (v1) | past 0.5 s | 34.0 [32.7, 35.0] | 23.1 [21.6, 24.9] | 24.1 | 18.9 | 10.6 [8.9, 12.6] |
| **`mstcn_causal`** | past only (receptive field ≈ 2·2¹⁰ frames) | 34.8 [31.8, 38.0] | 30.2 [29.9, 30.6] | 28.3 [25.1, 31.5] | 23.6 [19.9, 27.0] | 15.4 [12.7, 18.2] |
| linear `offline` (v1) | ±0.5 s | 35.7 [34.3, 36.9] | 27.6 [25.5, 30.0] | 29.2 | 24.5 | 15.4 [13.2, 17.8] |
| `mstcn_offline` | past and future | 39.1 [36.8, 42.5] | 35.2 [32.1, 37.4] | 37.4 [35.0, 40.2] | 33.1 [30.3, 36.5] | 23.5 [20.5, 27.0] |

Reading:

- The causal temporal head barely moves MoF over the causal linear probe (+0.8, inside the CI)
  but lifts the segment-level metrics clearly (Edit +7, F1@50 +5): it removes fragmentation
  rather than fixing frame-level confusions, which are feature-limited.
- The causal-vs-offline gap on the same architecture is 4.3 MoF / 5.0 Edit / 8.1 F1@50 on val.
  This is the first measured number for the spec's "causal cost" and it is a lower bound on the
  gap, because the offline twin had not converged at 50 epochs.
- All numbers stay far below anything usable for SOP monitoring; they are baselines for the
  pipeline, not claims.

## 4. Failure cases (`mstcn_causal`, from `metrics.json["confusions"]` and `per_video`)

- Same error families as the linear probe: background ↔ `check_instruction`, background
  predicted as manipulations, and fine-grained same-object swaps (`fit_nut` ↔ `tighten_nut`).
- Worst causal videos by MoF: see `tables.md`; the videos that were hardest for the linear probe
  (`20_assy_0_1`, `14_main_0_1`, `26_assy_0_1`) remain the hardest.
- Causal MoF has a wider CI than the linear probe (±3 vs ±1): the temporal head's errors are
  more participant-dependent.

## 5. Cost

| step | wall time | resources |
|---|---|---|
| `train-mstcn` (both variants, 50 epochs each, val every 5, bootstraps) | 105 s (54 s causal, 46 s offline) | RTX 4090, ≈ 2.5 GB VRAM |
| feature cache | reused from v1 (563 s) | — |

## 6. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1
# features as in ../industreal_dev_v1/README.md, then:
uv run sop-monitor train-mstcn --features artifacts/features/industreal/dinov2_vits14_s1 --labels-dir data/external/industreal/labels --out reports/industreal_dev_v2_mstcn --device auto --epochs 50 --eval-every 5 --n-boot 2000 --seed 0
uv run sop-monitor score-predictions --run reports/industreal_dev_v2_mstcn
```

Determinism: a second `train-mstcn` run with the same seed on the same GPU reproduced
`predictions_val.csv` byte-for-byte (`cudnn.deterministic` is set; one video per optimiser step).

## 7. Not done here

- No hyper-parameter search beyond the epoch; no ASFormer; no late fusion (single view).
- No online SOP metrics: the IndustReal procedure-step (PSR) labels are not in the local copy —
  see the root README for the exact blocker.
