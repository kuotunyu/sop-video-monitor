# havid_dev_v5_sop_tuned — SOP checks with the graph rule and duration window chosen by leave-one-subject-out on train (development result)

- Tier: **development result** on `splits/ha-vid/val.csv` (6 subjects, 18 recordings). The
  procedure knowledge and every setting are chosen on the train split only (leave-one-subject-out
  over its 17 subjects, `oof.json`); val was looked at once, for this report. The frozen test
  subjects were never read. The synthetic table is **synthetic**; the native table is underpowered.
- Date: 2026-09-13. Author: kuotunyu. Machine: Windows 11 (CPU only, 2 s).
- Purpose: fix the two failures of [`../havid_dev_v4_sop_synthetic/`](../havid_dev_v4_sop_synthetic/README.md)
  without touching val — the strict min-support-3 graph flagged 16 of 18 clean recordings and the
  5–95 % duration window 14 of 18 — by selecting the graph rule (min support × min agreement,
  majority rule now possible) and the duration quantiles out of fold on train.

## 1. What was tuned, and how

`sop-monitor tune-havid-sop` (`oof.json`): for each of the 17 train subjects the knowledge is
learned from the other 16 and applied to that subject's unperturbed sequences. Grid: min support
∈ {3, 5, 10, 20} × min agreement ∈ {1.0, 0.95, 0.9, 0.8} for the order check; quantiles
∈ {[5, 95], [2, 98], [1, 99]} % for the duration check. Selection rule (fixed before val):
order = maximise coverage − recording-level false-alarm rate (coverage = fraction of held-out
recordings on which a synthetic swap is possible at all); duration = the narrowest window with a
step-level false-alarm rate ≤ 5 %, else the widest.

| min support | min agreement | edges (full train) | coverage | recording false alarms | step false alarms | score |
|---|---|---|---|---|---|---|
| 3 | 1.0 (v4) | 78 | 99 % | 93 % | 36.8 % | 0.06 |
| 5 | 1.0 | 53 | 98 % | 83 % | 16.5 % | 0.15 |
| 10 | 1.0 | 38 | 92 % | 71 % | 12.9 % | 0.20 |
| **20** | **1.0** | **26** | **90 %** | **32 %** | **5.6 %** | **0.58** |
| 20 | 0.95 | 33 | 90 % | 40 % | 11.0 % | 0.50 |
| 20 | 0.9 | 27 | 91 % | 43 % | 18.4 % | 0.48 |
| 20 | 0.8 | 34 | 99 % | 55 % | 22.7 % | 0.44 |

(Rows with agreement < 1 at supports 3–10 are in `oof.json`; every one of them has more false
alarms than its strict counterpart.) Duration step false alarms out of fold: [5, 95] % 15.3 %,
[2, 98] % 8.6 %, [1, 99] % 6.6 % — none reaches 5 %, so the widest window is used.

Selected and written to `sop/ha-vid/tuned/`: min support 20, min agreement 1.0, duration
[1, 99] %, mandatory fraction 0.9 (unchanged).

| plate | steps | edges (v4 → v5) | mandatory | bounded |
|---|---|---|---|---|
| cylinder | 31 | 53 → 10 | 4 | 23 |
| gear | 16 | 18 → 15 | 5 | 15 |
| general | 16 | 7 → 1 | 4 | 15 |

## 2. Inputs and checks

As v4: two-hand step sequences of the 18 val recordings (ground truth from the annotations,
predictions from `fusion_causal` of `havid_dev_v3_tas_f1sel_{lh,rh}`), `steps_val.csv` /
`wrong_val.csv`; order / omission / duration checks and the three synthetic perturbations (one per
kind per recording, seed 0). New in this run: the per-step false-alarm rate (flagged steps over all
steps of the unperturbed sequences), which is the rate a per-step alarm would actually show.

## 3. Result (`tables.md` is authoritative)

Synthetic violations on `gt` sequences (18 val recordings, seed 0); Wilson 95 % intervals in percent:

| violation | applicable | detected | recall | clean flagged | false-alarm rate | precision | per-step false alarms |
|---|---|---|---|---|---|---|---|
| order | 17 | 17 | 100.0 [82, 100] | 8 / 18 | 44.4 [25, 66] | 68.0 | 20 / 234 = 8.5 [6, 13] |
| omission | 17 | 17 | 100.0 [82, 100] | 1 / 18 | 5.6 [1, 26] | 94.4 | — |
| duration | 18 | 18 | 100.0 [82, 100] | 11 / 18 | 61.1 [39, 80] | 62.1 | 14 / 234 = 6.0 [4, 10] |

Unperturbed `gt` recordings with any finding: 14 / 18.

Synthetic violations on `pred` sequences (18 val recordings, seed 0):

| violation | applicable | detected | recall | clean flagged | false-alarm rate | precision | per-step false alarms |
|---|---|---|---|---|---|---|---|
| order | 12 | 12 | 100.0 [76, 100] | 14 / 18 | 77.8 [55, 91] | 46.2 | 87 / 354 = 24.6 [20, 29] |
| omission | 10 | 10 | 100.0 [72, 100] | 17 / 18 | 94.4 [74, 99] | 37.0 | — |
| duration | 18 | 18 | 100.0 [82, 100] | 18 / 18 | 100.0 [82, 100] | 50.0 | 204 / 354 = 57.6 [52, 63] |

Unperturbed `pred` recordings with any finding: 18 / 18.

Native `w`: 12 ground-truth segments, 0 detected (recall 0.0 [0, 24] %), 0 predicted — unchanged
from v4 (same recogniser), underpowered.

v4 → v5 on ground truth (recording-level false alarms): order 16 / 18 → 8 / 18, duration
14 / 18 → 11 / 18, omission 1 / 18 → 1 / 18; recall stays 100 % for every kind; order coverage
18 → 17 applicable recordings.

Reading:

- **The strict rule wins; the majority rule loses.** Out of fold, every agreement threshold
  below 1.0 raises the false-alarm rate at every support (at support 20: 32 % → 40–55 %). Edges
  that hold in "most" recordings encode common habits, not constraints; only universal edges with
  enough support survive as SOP knowledge. Support 20 keeps 26 edges (of 78) and halves the
  ground-truth false alarms on val (8 / 18), with 8.5 % of steps flagged.
- **Order is still not a clean recording-level alarm.** 8 of 18 clean recordings carry at least
  one violated universal edge; the val subjects' stage-1 routes are freer than the 26 edges allow.
  The remaining lever is not the rule but the step granularity: several violated edges are between
  screw steps of the same sub-assembly (`sshc*`, `sspg*`), whose order the instruction sheet does
  not fix. Grouping such steps (instruction-sheet granularity) is the next W3 item.
- **Duration** at [1, 99] % is a per-step alarm with 6.0 % [4, 10] flagged steps on real
  sequences, close to the 2 % nominal rate but not at it: durations are heavy-tailed across
  subjects. As a recording-level alarm it still fires on 11 / 18.
- **Omission** is unchanged and remains the only recording-level check that works (1 / 18).
- **Predicted sequences** are still flagged everywhere: the v3 recogniser's fragmented segments
  produce 24.6 % order and 57.6 % duration step false alarms (most duration findings are
  `too_short` fragments). The recogniser, not the checker, is the bottleneck; a segment-level
  smoothing or minimum-duration decoder before the checks is the cheapest next step.
- Deviations are suggestions for human review; nothing here is a safety or compliance guarantee.

## 4. Cost

`tune-havid-sop` 2.4 s, `learn-havid-sop` + `havid-sop` 2 s, CPU only.

## 5. Reproduce

```bash
export PYTHONUTF8=1
make tune-havid-sop                                              # oof.json (deterministic)
make havid-sop-v5 SUPPORT=20 AGREEMENT=1.0 LOWQ=0.01 HIGHQ=0.99   # sop/ha-vid/tuned + this run
uv run sop-monitor reproduce-lite
```

## 6. Not done here

- No instruction-sheet step granularity, no segment smoothing or minimum-duration decoding of
  the predicted sequences before the checks, no online (streaming) checks, no deviation queue.
- No number on the frozen test subjects.
