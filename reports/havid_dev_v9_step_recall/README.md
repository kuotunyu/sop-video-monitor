# havid_dev_v9 — what an output delay buys the causal recogniser (development result)

This README covers the whole v9 study: the 18 look-ahead runs, their seed summaries and the
per-step recall table in this directory.

- Tier: **development result** on `splits/ha-vid/val.csv` (6 subjects, 18 recordings × 3 views);
  epochs selected on val F1@10; the frozen test subjects were not read.
- Date: 2026-09-14. Author: kuotunyu. Machine: Windows 11, RTX 4090.
- Question: the causal recogniser (v7: fusion F1@10 ≈ 30) recognises almost none of the short
  mandatory "insert" steps, while the offline one recognises many of them. How much of that gap
  closes if the online system may answer a few seconds late?

## 1. Method

A causal MS-TCN++ trained with look-ahead L learns to name, at time t, the step of frame t − L
(`train-havid-tas --lookahead-frames L --no-offline`; targets delayed by L frames, outputs advanced
back for scoring, so every metric is frame-aligned and the only cost is an L / 15 s output delay).
Everything else is the v3 protocol: DINOv2 ViT-B/14 features, one network per view, parameter-free
late fusion, epoch by val F1@10. L ∈ {15, 45, 90} frames = 1, 3, 6 s; seeds 0–2; both hands. L = 0
and offline come from the committed v3 seeds (`havid_dev_v7_tas_seeds_*`).

| directory | contents |
|---|---|
| `../havid_dev_v9_tas_la{15,45,90}_{lh,rh}_s{0,1,2}/` | the 18 training runs (predictions, metrics, config) |
| `../havid_dev_v9_tas_lookahead_la{15,45,90}_{lh,rh}/` | mean ± std over seeds (`seed_summary.json`, `tables.md`) |
| this directory | `step_recall.json`, `tables.md`: val frame recall per instruction-sheet step, both hands × three seeds pooled |

Per-step frame recall counts, for every ground-truth frame of a sheet step, whether the predicted
label maps to the same sheet step (`havid-step-recall`). It is the quantity the SOP checks depend
on: a step whose frames are never predicted cannot be observed, so it becomes an omission.

## 2. Results (`tables.md` of each directory is authoritative)

Mean fusion, mean ± std over seeds 0–2:

| setting | delay | left MoF | left Edit | left F1@10 | left F1@50 | right MoF | right Edit | right F1@10 | right F1@50 |
|---|---|---|---|---|---|---|---|---|---|
| causal L = 0 (v7) | 0 s | 39.5 ± 1.0 | 33.6 ± 2.0 | 30.5 ± 1.4 | 13.1 ± 0.9 | 34.6 ± 2.3 | 33.0 ± 0.5 | 29.9 ± 0.6 | 12.3 ± 1.6 |
| causal L = 15 | 1 s | 40.4 ± 1.9 | 31.9 ± 2.3 | 31.9 ± 2.3 | 17.0 ± 1.8 | 37.5 ± 0.9 | 32.3 ± 0.7 | 31.7 ± 1.3 | 14.8 ± 2.0 |
| causal L = 45 | 3 s | **42.2 ± 2.1** | **35.6 ± 2.5** | **36.4 ± 2.0** | 16.6 ± 0.6 | 36.9 ± 1.2 | **34.8 ± 2.8** | **32.5 ± 1.9** | **15.8 ± 0.8** |
| causal L = 90 | 6 s | 37.7 ± 0.6 | 35.2 ± 1.7 | 33.4 ± 1.7 | 11.5 ± 2.3 | 35.0 ± 0.6 | 33.8 ± 1.4 | 30.9 ± 1.5 | 13.5 ± 1.2 |
| offline (v7) | whole recording | 44.2 ± 2.7 | 41.2 ± 1.1 | 43.9 ± 2.0 | 26.6 ± 1.5 | 42.4 ± 2.0 | 40.1 ± 1.9 | 42.0 ± 1.6 | 24.0 ± 1.3 |

Frame recall pooled over steps (this directory):

| setting | all step frames | mandatory step frames |
|---|---|---|
| causal L = 0 | 29.3 % | 32.8 % |
| causal L = 15 | 39.6 % | 45.0 % |
| causal L = 45 | **43.2 %** | **49.7 %** |
| causal L = 90 | 38.1 % | 43.4 % |
| offline | 48.9 % | 55.8 % |

Reading:

- **A 3 s delay closes about three quarters of the causal-offline gap in mandatory-step recall.** Mandatory
  step frame recall rises from 32.8 % to 49.7 % (offline 55.8 %); left-hand F1@10 from 30.5 to 36.4
  (offline 43.9), i.e. under half of the F1@10 gap. The right hand gains less (29.9 → 32.5 F1@10).
- **The effect is not monotone.** L = 90 is worse than L = 45 on every metric in the two tables
  above and on 15 of the 19 mandatory steps (equal on 1, better on 3). A plausible reason, not tested
  here: at a 6 s delay each network must carry the label of a step that may already have ended
  across the following steps, and the v3 training budget does not teach it that. Longer is not
  safer.
- **The gain is on the short insert steps.** Examples (L = 0 → L = 45 → offline): gear `iglft`
  6 → 41 → 37 %, `iplft` 6 → 30 → 72 %, `ipsft` 13 → 35 → 56 %, cylinder `scccb` 25 → 51 → 60 %.
  Some steps stay unrecognised at every delay and offline too: cylinder `ibacb` (0 / 0 / 4 %),
  general `iibn`, `iirn`, `iusn` (≤ 23 % offline). These are a recogniser limit, not a latency limit.
- **The F1@50 gap to offline stays large** (16.6 vs 26.6 left): delay fixes whether a step is seen
  more than where exactly it starts and ends.
- Consequence for the test protocol: the pre-registered rule for L\* (smallest L reaching 90 % of the
  offline mandatory-step recall, else 90) selects L\* = 90, because L = 45 misses the threshold
  (50.19 %) by 0.45 points. L\* stays 90 as written; L = 45 is added to the test as a declared
  amendment (`docs/decisions.md`, `docs/havid_test_protocol.md` §6).
- All numbers are baselines on a 6-subject validation set, not claims.

## 3. Reproduce

```bash
export PYTHONUTF8=1
make havid-tas-lookahead       # 18 runs, 550–1,040 s each on an RTX 4090 shared with a second training job
make havid-lookahead-summary
make havid-step-recall-v9      # this directory
uv run sop-monitor reproduce-lite
```

## 4. Not done here

- L between 45 and 90, other architectures (ASFormer) or longer training for large L.
- Look-ahead for the offline twins (not meaningful) or for the geometric fusion rule as the reported
  row (its numbers are in the seed tables: left F1@10 36.7 ± 0.8 at L = 45).
- No number on the frozen test subjects.
