# industreal_dev_v4_psr_mstcn — causal MS-TCN++ state head + procedure-prior decoder (val, leave-one-participant-out)

- Tier: **development result**, same data, features, protocol and caveats as
  [`../industreal_dev_v3_psr/`](../industreal_dev_v3_psr/README.md): 16 val recordings, 5
  participants, every scored video comes from a head that never saw its participant, decoder
  settings chosen on each fold's training videos. Small pool, wide CIs, not a headline, not
  comparable with the paper's test-split baselines beyond order of magnitude. Not HA-ViD.
- Date: 2026-09-11. Author: kuotunyu. Machine: Windows 11, RTX 4090.
- Purpose: a 2 × 2 ablation of what v3 was missing — a temporal model for the component states,
  and a decoder that knows the procedure. All four runs share one `completions_val.csv`
  (`run` column) and one ground truth.

## 1. What changed against v3

| axis | `linear` (v3 head) | `mstcn` (new) |
|---|---|---|
| state head | 11-way multi-label logistic regression per frame | causal MS-TCN++ (`sop_monitor.mstcn`, sigmoid outputs, 912,300 parameters): 11 dual-dilated PG layers + 3 × 10 refinement layers, left-only padding so frame *t* sees ≤ *t*; per-stage BCE + truncated-MSE smoothing; Adam 5e-4, one video per step, **fixed 40 epochs** (no epoch selection), seed 0, `cudnn.deterministic` |

| axis | `plain` (v3 decoder) | `prior_dwell` (new) |
|---|---|---|
| smoothing / hysteresis | causal EMA + `theta_on` / `theta_off` | same |
| dwell | none | an event is emitted only after the average has stayed beyond the threshold for `min_dwell ∈ {0, 10, 30, 60}` extra frames (0–6 s at 10 fps) |
| procedure prior | none: every component starts "not installed" and may emit | **learned from the fold's training labels per recording kind** (`assy` / `main`): a component may emit only if it ever changes state in those training recordings, and it starts in their majority initial state. Learned priors were identical across folds: assembly → base pre-installed, 9 of 11 components active; maintenance → 10 components pre-installed, 4–5 active |

Decoder grids are selected on the fold's training videos by mean per-video F1
(`ema ∈ {0, .8, .9, .95, .98, .99} × theta_on ∈ {.6, .7, .8, .9, .95} × theta_off ∈ {.1, .2, .3, .4}`,
plus the dwell grid for `prior_dwell`); the chosen settings per fold and run are in `tables.md`.

## 2. Result (val, 16 videos, unweighted mean over videos; 95 % CI = participant bootstrap, 2,000 draws)

| run | POS | F1 (system) | mean delay (s) | TP / FP / FN | videos with POS = 0 |
|---|---|---|---|---|---|
| `linear_plain` (= v3) | 0.040 [0.000, 0.111] | 0.479 [0.401, 0.557] | 26.8 [18.9, 36.6] | 109 / 246 / 13 | 14 |
| `linear_prior_dwell` | 0.184 [0.043, 0.344] | 0.604 [0.524, 0.696] | 25.6 [18.4, 34.4] | 107 / 137 / 20 | 8 |
| `mstcn_plain` | 0.317 [0.231, 0.414] | 0.551 [0.452, 0.662] | 21.5 [19.4, 23.8] | 83 / 110 / 25 | 7 |
| **`mstcn_prior_dwell`** | **0.609 [0.513, 0.670]** | **0.707 [0.647, 0.756]** | 21.9 [17.8, 27.8] | 86 / 36 / 29 | 0 |

Subsets of the best run: videos without error steps (12) POS 0.624 [0.582, 0.651], F1 0.733
[0.674, 0.808], delay 21.1 s; videos with error steps (4) POS 0.562 [0.344, 0.781], F1 0.629
[0.325, 0.904], delay 24.4 s.

Reading:

- Both axes matter and they compound. The temporal head alone removes more than half of the
  false positives (246 → 110) and turns POS from 0.04 into 0.32; the prior-and-dwell decoder alone
  halves them for the linear head (246 → 137); together they leave 36 FPs and lift POS to 0.61 and
  F1 to 0.71, with no video stuck at POS 0 any more.
- The price is recall: FNs grow from 13 to 29 (of 122 ground-truth completions) because the dwell
  and the prior suppress short or unexpected state changes. Delay does not get worse (≈ 22 s).
- For scale only: the paper's B3 on the *test* split reports POS 0.797 / F1 0.883 / delay 22.4 s
  with an object-detection-based state detector and full procedure knowledge. This run uses a
  frozen global frame embedding and a prior read off 12–13 training recordings.

## 3. Failure cases (`mstcn_prior_dwell`, `tables.md` per-video rows)

- **`26_assy_1_5`** (F1 0.20, 1 TP / 7 FP / 1 FN): the longest recording (4,587 frames, 7.6 min)
  and one of the four with an execution error; the state head oscillates on the late components
  and the dwell cannot suppress 7 sustained false plateaus.
- **`14_main_2_3`** (F1 0.27, 2 / 5 / 6): a maintenance recording with 11 steps (the most among
  `main` videos); the learned `main` prior only allows 4–5 active components, so several true
  removals/installs are structurally impossible to emit — the prior trades recall for precision
  and this video pays for it.
- **`20_main_0_1`** (F1 0.33, 2 / 7 / 1) and **`05_assy_2_2`** (5 FN): early emissions before the
  ground-truth completion frame count as FPs, and dwell of 60 frames on fold 05 delays several true
  events past the end of short plateaus.
- Best videos: `20_assy_3_6` (8 / 0 / 0, F1 1.000, POS 0.875), `24_assy_0_1` (F1 0.941),
  `05_main_0_1` and `14_main_0_1` (F1 0.933).

## 4. Cost

| step | wall time | resources |
|---|---|---|
| `train-psr` (5 folds: linear head + MS-TCN++ 40 epochs ≈ 13 s each, 4 decoder grids of up to 480 settings on 12–13 videos, bootstraps) | 120 s | RTX 4090, < 3 GB VRAM |
| features, archives, labels | reused from v1 / v3 | — |

## 5. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1
# features and PSR labels as in ../industreal_dev_v3_psr/README.md, then:
uv run sop-monitor train-psr --features artifacts/features/industreal/dinov2_vits14_s1 --psr-dir data/external/industreal/psr --out reports/industreal_dev_v4_psr_mstcn --device auto --epochs 300 --mstcn-epochs 40 --n-boot 2000 --seed 0
uv run sop-monitor reproduce-lite     # recomputes every run's metrics from completions_val.csv, checks tables.md
```

`--head` / `--decoder` restrict the run to a subset (`train-psr --head mstcn --decoder prior_dwell`).
Determinism: a second `train-psr` run with the same seed on the same GPU reproduced
`completions_val.csv` byte-for-byte (all four runs).

## 6. Not done here

- No hyper-parameter search for the MS-TCN++ state head (fixed 40 epochs, default widths).
- No use of the procedure *order* (which step is expected next) — the prior only knows which
  components can move and how the recording starts; that is where `sop_graph.py`'s precedence
  checks would plug in once an IndustReal graph exists.
- No train-split training, no test-split evaluation, no HA-ViD (request form pending).
