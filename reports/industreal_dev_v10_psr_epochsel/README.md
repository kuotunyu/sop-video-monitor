# industreal_dev_v10_psr_epochsel — MS-TCN++ training length chosen by held-out BCE (negative result)

- Tier: **development result**. Same data, features (DINOv2 ViT-B/14), protocol, decoder
  selection (out-of-fold, 30 s budget) and seeds as the best run of
  [`../industreal_dev_v8_psr_vitb/`](../industreal_dev_v8_psr_vitb/README.md). One change: the
  MS-TCN++ state head's number of training epochs is no longer fixed at 40 but chosen on the
  inner folds — the inner heads train to 80 epochs, their held-out BCE (last stage) is measured
  at epochs 20 / 40 / 60 / 80, and the epoch with the lowest mean held-out BCE is used both for
  the out-of-fold probabilities that select the decoder and for the final fit. Only the
  `mstcn` × `prior_dwell` run at the 30 s budget was executed. Frozen test split untouched. Not HA-ViD.
- Date: 2026-09-12. Author: kuotunyu. Machine: Windows 11, RTX 4090.
- Purpose: v8 showed the MS-TCN++ seed spread (POS ± 0.06) dominating the error bars with a
  fixed, arbitrary 40 epochs; this run tests whether a validated training length helps.

## 1. What the inner folds said

| seed | mean held-out BCE at 20 / 40 / 60 / 80 epochs | chosen |
|---|---|---|
| 0 | 0.291 / 0.293 / 0.328 / 0.385 | 20 |
| 1 | 0.269 / 0.309 / 0.349 / 0.358 | 20 |
| 2 | 0.261 / 0.296 / 0.294 / 0.350 | 20 |

By held-out BCE the head overfits from 20 epochs on, in every seed. The final heads were
therefore trained for 20 epochs (v8: 40).

## 2. Result on val (mean ± std over seeds 0–2; per-seed rows with CIs in `tables.md`)

| run | epochs | POS | F1 (system) | mean delay (s) | TP / FP / FN (mean) |
|---|---|---|---|---|---|
| mstcn_prior_dwell_cap30, **this run** | 20 (selected) | 0.545 ± 0.066 | 0.741 ± 0.046 | **21.1 ± 1.3** | 91.7 / 33.7 / 29.3 |
| mstcn_prior_dwell_cap30, v8 | 40 (fixed) | **0.625 ± 0.059** | **0.814 ± 0.020** | 25.1 ± 0.9 | 105.3 / 20.0 / 29.3 |

Per seed: 0.598 / 0.805 / 22.3 s, 0.584 / 0.703 / 19.3 s, 0.452 / 0.713 / 21.8 s.

## 3. Reading — the criterion is the problem, not the idea

- Choosing the epoch by held-out **BCE** made the decoded result worse: F1 −0.07, POS −0.08,
  seed spread wider (F1 ± 0.046 vs ± 0.020), for 4 s less delay. The BCE-optimal head is the less
  confident one; the hysteresis decoder wants sharp state probabilities, which the "over-fitted"
  40-epoch head provides. A frame-level proxy and an event-level metric disagree here, exactly the
  kind of mismatch the project's own online-metric work was set up to expose.
- The out-of-fold decoded F1 of the chosen decoders (0.72–0.77) already predicted the drop
  relative to v8's out-of-fold value (0.80): the information to reject 20 epochs was available
  inside the selection loop, but the BCE criterion did not use it.
- Consequence: epoch selection should be driven by the same out-of-fold decoded F1 that selects
  the decoder (snapshots at every grid epoch are already computed, so this costs only CPU grid
  evaluations). That is what `industreal_dev_v11_psr_epochsel_f1` runs.

## 4. Cost

`train-psr … --mstcn-epoch-grid 20 40 60 80 --head mstcn --decoder prior_dwell --delay-cap 30`,
seeds 0–2: 1,862 s on the RTX 4090 while sharing it with the ViT-L/14 extraction (the inner heads
train to 80 epochs instead of 40).

## 5. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1
# ViT-B/14 features and PSR labels as in ../industreal_dev_v8_psr_vitb/README.md, then:
uv run sop-monitor train-psr --features artifacts/features/industreal/dinov2_vitb14_s1 --psr-dir data/external/industreal/psr --train-split splits/industreal/train.csv --eval-split splits/industreal/val.csv --selection nested --seed 0 --seed 1 --seed 2 --delay-cap 30 --head mstcn --decoder prior_dwell --mstcn-epoch-grid 20 --mstcn-epoch-grid 40 --mstcn-epoch-grid 60 --mstcn-epoch-grid 80 --out reports/industreal_dev_v10_psr_epochsel --device auto --epochs 300 --n-boot 2000
uv run sop-monitor reproduce-lite
```

Determinism: not re-run separately (same training code as v6–v8, whose seed-0 re-runs were
byte-identical; the epoch loop only adds evaluation snapshots).
