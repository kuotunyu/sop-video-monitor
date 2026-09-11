# industreal_dev_v3_psr — first real online metrics (POS / F1 / delay), val, leave-one-participant-out

- Tier: **development result**. The only IndustReal procedure-step (PSR) labels on disk are the
  16 val recordings' (`val_p1.zip` + `val_p2.zip`; train and test archives were not downloaded),
  so the model is trained **leave-one-participant-out inside val**: every scored video comes from
  a head that never saw that participant, but the pool is small (5 participants, 16 videos) and
  decoder settings were chosen on each fold's training videos. Not a headline; not comparable
  with the paper's test-split baselines except in order of magnitude. Not an HA-ViD result.
- Date: 2026-09-11. Author: kuotunyu. Machine: Windows 11, RTX 4090.
- Purpose: the spec 4.3 sanity run — exercise the project's implementation of the IndustReal online
  metrics on real labels and real model output, after aligning it with the reference code.

## 1. Data identity (`psr_audit.json`, `../data_audit.json`, `data/manifest.json`)

| item | value |
|---|---|
| archives | `val_p1.zip` 3,783,132,550 B (MD5 `a6d9464b…95844`), `val_p2.zip` 6,222,333,683 B (MD5 `0db912bd…8b7a4`), both matching the 4TU listing; only `PSR_labels.csv`, `PSR_labels_with_errors.csv`, `PSR_labels_raw.csv` were extracted (`data/external/industreal/psr/<recording>/`) |
| recordings | 16 = all val videos: participants 05, 14, 20, 24, 26; 9 `assy` + 7 `main` (maintenance, with removals) |
| frame alignment | each recording's `rgb/` in the archive holds exactly as many JPEGs as the mp4 has frames (checked for all 16, e.g. `05_assy_0_1` 2,918 = 2,918); label frame indices are 0-based JPEG names, so the 10 fps DINOv2 cache (stride 1) lines up frame-for-frame |
| label consistency | for every recording the completions implied by `PSR_labels_raw.csv` (reference state→step rule) reproduce `PSR_labels.csv` exactly, and with errors included reproduce `PSR_labels_with_errors.csv` exactly; all frames within the video; all ids within `sop/industreal/procedure_info.json` (33 = 11 components × install / incorrect / remove) |
| ground truth | 143 correct completions (34 of them removals); 5 incorrect-install steps in 4 videos (`05_assy_2_2`, `20_assy_3_6`, `24_assy_2_4`, `26_assy_1_5`) — these are excluded from `PSR_labels.csv`, as in the reference evaluation |

## 2. Metric implementation and its behaviour on the real labels (`metrics.json["gt_sanity"]`)

`sop_monitor.metrics.online` was rewritten against the reference `PSR/psr_utils.py`: POS uses the
unrestricted Damerau-Levenshtein with costs insert 1 / delete 1 / substitute 2 / transpose 1,
normalised by the ground-truth length and clipped at 0; system-level TP/FP/FN follow
`determine_performance` (per-step-id occurrence matching, an early prediction is an FP, an
on-time one a TP with delay `pred − gt`); F1 uses the reference's epsilon-smoothed formula. Three
deliberate deviations are listed in the module docstring (fallback when no later candidate
exists, no 100-frame substitute when nothing was detected, "at or after" instead of "strictly
after" when a step id repeats). Scoring the ground truth against controlled perturbations of
itself, averaged over the 16 videos:

| perturbation of the GT | POS | F1 | mean delay (frames) | FP / FN |
|---|---|---|---|---|
| identity | 1.000 | 1.000 | 0.0 | 0 / 0 |
| every completion +30 frames | 1.000 | 1.000 | 30.0 | 0 / 0 |
| last completion dropped | 0.886 | 0.939 | 0.0 | 0 / 16 |
| first two completions' times exchanged | 0.948 | 0.972 | 16.1 | 7 / 0 |

(The transposition case is < 1 − 1/n on average because in several recordings the first two
completions share a frame, so exchanging their times changes nothing.)

## 3. Model identity (`config.json`)

| component | what exactly |
|---|---|
| features | same cache as v1/v2: frozen DINOv2 ViT-S/14 `[CLS ; mean patch]`, 768-d, 10 fps |
| targets | per frame, per component: 1 if `PSR_labels_raw.csv` (forward-filled) says correctly installed, else 0 (errors count as absent) |
| head | 11-way multi-label logistic regression, BCE, full-batch Adam lr 5e-3, weight decay 1e-3, 300 epochs, seed 0; trained on the other four participants' videos of each fold |
| decoder (causal) | EMA of the 11 probabilities, hysteresis: install `3k` on an up-crossing of `theta_on`, removal `3k+2` on a down-crossing of `theta_off`; incorrect installs are never emitted |
| decoder selection | grid `ema ∈ {0, .8, .9, .95, .98, .99} × theta_on ∈ {.6, .7, .8, .9, .95} × theta_off ∈ {.1, .2, .3, .4}` by mean per-video F1 **on the fold's training videos**; chosen settings per fold are in `tables.md` |
| honesty note | the grid was widened once after a first run with `ema ≤ .95, theta_on ≤ .9, theta_off ≥ .2` gave F1 0.494 / POS 0.089 / delay 27.1 s; the widened grid gives 0.479 / 0.040 / 26.8 s — in-sample decoder selection does not transfer across participants, and the numbers below are the widened-grid run |

## 4. Result (val, 16 videos, unweighted mean over videos as in `psr_baseline.py`; participant bootstrap 2,000 draws)

| subset | videos | POS | F1 (system) | mean delay (s) | TP / FP / FN |
|---|---|---|---|---|---|
| all | 16 | 0.040 [0.000, 0.111] | 0.479 [0.401, 0.557] | 26.8 [18.9, 36.6] | 109 / 246 / 13 |
| videos without error steps | 12 | 0.046 [0.000, 0.167] | 0.435 [0.369, 0.548] | 24.3 [17.5, 30.6] | — |
| videos with error steps | 4 | 0.021 [0.000, 0.063] | 0.613 [0.529, 0.759] | 34.2 [20.9, 52.8] | — |

Order of magnitude only, not a comparison: the paper's B3 (object-detection-based assembly-state
detector + procedure knowledge, on the *test* split) reports POS 0.797, F1 0.883, delay 22.4 s.
This floor detects most steps (13 FN out of 122 ground-truth completions) with a delay in the
same range (27 s), but emits more than twice as many completions as there are steps.

## 5. Failure cases (`tables.md` per-video rows, `completions_val.csv`)

- **False positives dominate** (246 FP vs 109 TP): the per-frame state probabilities flicker
  around the hysteresis band, producing install/remove pairs for components that never moved.
  Maintenance videos, where removals are legitimate, are the worst (`26_main_0_1` 25 FP on 8
  steps, `14_main_2_3` 23 FP). Because POS is normalised by the ground-truth length and clipped,
  a video with twice as many predictions as steps scores 0 regardless of order — 13 of 16 videos
  do.
- **The two clean videos show the ceiling of this decoder**: `24_assy_0_1` (7 TP / 2 FP / 0 FN,
  POS 0.556, F1 0.875) and `05_assy_2_2` (12 / 5 / 0, F1 0.828).
- **Delays are long** (median video ≈ 24 s) because the EMA needed to suppress flicker also
  postpones the crossing; `26_assy_1_5` (62 s) is the longest video (4,587 frames) with the
  slowest state changes.
- Nothing in the decoder knows the procedure: no "at most one install per component in an
  assembly", no expected order. That is exactly what the reference's B2/B3 add, and what the SOP
  graph checks in `sop_graph.py` are for once a graph exists for this dataset.

## 6. Cost

| step | wall time | resources |
|---|---|---|
| download `val_p1.zip` + `val_p2.zip` (10.0 GB) from 4TU, MD5-verified | 29 min (≈ 5–8 MB/s) | network, 10 GB disk |
| `extract-psr-labels` (48 CSVs out of 2 archives) | < 1 min | CPU |
| `train-psr` (5 folds × head + 120-point decoder grid + bootstraps) | 70 s | RTX 4090 (CPU would do) |
| `audit-industreal --manifest` (hashes 16 GB of archives) | ≈ 3 min | CPU |

## 7. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1
# archives: https://data.4tu.nl/datasets/b008dd74-020d-4ea4-a8ba-7bb60769d224 -> data/external/industreal/val_p{1,2}.zip
uv run sop-monitor extract-psr-labels --archive data/external/industreal/val_p1.zip --archive data/external/industreal/val_p2.zip --out data/external/industreal/psr
# features as in ../industreal_dev_v1/README.md (val videos), then:
uv run sop-monitor train-psr --features artifacts/features/industreal/dinov2_vits14_s1 --psr-dir data/external/industreal/psr --out reports/industreal_dev_v3_psr --device auto --epochs 300 --n-boot 2000 --seed 0
uv run sop-monitor reproduce-lite     # recomputes metrics.json from completions_val.csv, checks tables.md
```

`make extract-psr` and `make psr-lopo` wrap the same commands. Determinism: a second `train-psr` run
with the same seed on the same GPU reproduced `completions_val.csv` byte-for-byte, and
`reproduce-lite` recomputes `metrics.json` and `tables.md` from that table without features.

## 8. Not done here

- No training on the IndustReal train split (its recording archives, 20 GB, were not downloaded);
  no test-split evaluation (23.7 GB, and the split is frozen for a later formal run).
- No temporal model for the state targets and no procedure-aware decoding — both are the obvious
  next steps and both are cheap on the cached features.
- No HA-ViD: still blocked on the request form.
