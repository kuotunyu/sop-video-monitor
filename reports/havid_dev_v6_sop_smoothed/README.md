# havid_dev_v6_sop_smoothed — v5 knowledge with minimum-duration smoothing of the predicted segments (development result, negative)

- Tier: **development result** on `splits/ha-vid/val.csv` (6 subjects, 18 recordings); knowledge
  and settings as in [`../havid_dev_v5_sop_tuned/`](../havid_dev_v5_sop_tuned/README.md); the
  frozen test subjects were never read. Synthetic table = synthetic; native table underpowered.
- Date: 2026-09-13. Author: kuotunyu. Machine: Windows 11 (CPU only, 2 s).
- Purpose: v5 showed that the v3 recogniser's *fragmented* sequences (354 predicted steps against
  234 ground-truth steps on val; 28.5 % of predicted non-`null` segments shorter than 8 frames)
  make every check fire. The cheapest remedy is a parameter-free minimum-duration filter before
  the checks: a predicted segment shorter than the shortest step in the training annotations
  cannot be a step.

## 1. What changed

`havid-sop --min-segment-frames 11` (`havid_sop.smooth_segments`): every predicted per-hand
segment shorter than 11 frames (0.73 s; the 1st percentile of the 1,852 train step lengths,
minimum 7 frames) is absorbed into its predecessor (the first one into its successor) and
same-label neighbours are merged. Ground truth is never smoothed; the knowledge is
`sop/ha-vid/tuned/` (support 20, agreement 1.0, [1, 99] % duration window), unchanged from v5.

## 2. Result (`tables.md` is authoritative; the `gt` tables are identical to v5)

Predicted steps on val: 354 (v5) → 189 (v6); per recording 3–38 → 1–19 (ground truth 7–20).

Synthetic violations on `pred` sequences (18 val recordings, seed 0), v5 → v6:

| violation | recall | clean flagged v5 → v6 | per-step false alarms v5 → v6 |
|---|---|---|---|
| order | 100 % (12 / 12 applicable) | 14 / 18 → 15 / 18 | 87 / 354 = 24.6 % → 63 / 189 = 33.3 % [27, 40] |
| omission | 100 % (11 / 11) | 17 / 18 → 18 / 18 | — |
| duration | 100 % (18 / 18) | 18 / 18 → 17 / 18 | 204 / 354 = 57.6 % → 58 / 189 = 30.7 % [25, 38] |

Unperturbed `pred` recordings with any finding: 18 / 18 (unchanged). Native `w`: 0 of 12, none
predicted (unchanged).

Reading:

- Smoothing removes the fragment count (steps per recording now match the ground truth's range)
  and halves the duration step false alarms (57.6 → 30.7 %, mostly `too_short` fragments), but
  it does not make the sequences *right*: every recording still misses 1–4 mandatory steps
  (omission 18 / 18), up to 4 labels per recording are ones the graph never saw, and a third of
  the surviving steps violate a universal edge (33.3 %, up from 24.6 % because the denominator
  shrank faster than the violations).
- This is a **negative result** for "fix it after the recogniser": the v3 recogniser
  (fusion_causal F1@10 ≈ 30) drops and mislabels steps, and no post-hoc filter recovers them. The
  deployable violation numbers will move only when the recogniser does. Smoothing stays available
  as an option (off by default) because it makes per-step alarm rates interpretable.
- Deviations are suggestions for human review; nothing here is a safety or compliance guarantee.

## 3. Reproduce

```bash
export PYTHONUTF8=1
make havid-sop-v6      # needs reports/havid_dev_v3_tas_f1sel_{lh,rh} and sop/ha-vid/tuned
uv run sop-monitor reproduce-lite
```

## 4. Not done here

- No instruction-sheet step granularity, no better recogniser (the actual bottleneck), no online
  checks, no deviation queue. No number on the frozen test subjects.
