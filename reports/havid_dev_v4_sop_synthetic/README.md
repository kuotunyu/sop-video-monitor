# havid_dev_v4_sop_synthetic — learned SOP checks, synthetic violations and the native `w` table on val (development result)

- Tier: **development result** on `splits/ha-vid/val.csv` (6 subjects, 18 recordings). The
  procedure knowledge is learned from the train split only; the frozen test subjects were never
  read. The synthetic table is **synthetic** (perturbed sequences), as ADR 0001 requires it to be
  marked; the native table is a count with a Wilson interval and is underpowered.
- Date: 2026-09-13. Author: kuotunyu. Machine: Windows 11 (CPU only for this run).
- Purpose: W3's first slice — can order, omission and duration violations be detected from step
  sequences at all (ground truth = checker validity), and how much survives the v3 recogniser's
  errors (predicted sequences = deployable number)?

## 1. Inputs

- Step sequences: both hands' primitive-task segments merged (a same-label segment overlapping on
  the other hand is one two-handed step); `null` and `w` excluded. Ground truth from the temporal
  annotations; predictions from `fusion_causal` of
  [`../havid_dev_v3_tas_f1sel_lh/`](../havid_dev_v3_tas_f1sel_lh/README.md) and
  [`../havid_dev_v3_tas_f1sel_rh/`](../havid_dev_v3_tas_f1sel_rh/README.md). Committed as
  `steps_val.csv` (234 ground-truth steps, 354 predicted steps); `w` segments as `wrong_val.csv`.
- Plate per recording read off the label vocabulary (`havid_sop.plate_of`); val plates: 6
  cylinder, 6 gear, 6 general. Ground-truth sequences have 7–20 steps per recording, predicted
  ones 3–38 (fragmented segments count as separate steps).
- Learned knowledge (`sop/ha-vid/`, from 143 train recordings): precedence edges holding in every
  co-occurring recording (min support 3), mandatory steps (≥ 90 % of a plate's recordings),
  duration bounds (5th–95th percentile, labels seen ≥ 5 times).

| plate | steps | edges | mandatory | bounded |
|---|---|---|---|---|
| cylinder | 31 | 53 | 4 | 23 |
| gear | 16 | 18 | 5 | 15 |
| general | 16 | 7 | 4 | 15 |

## 2. Checks and synthetic violations

- order: `check_order` — a step whose learned predecessors have not all been observed.
- omission: a mandatory step of the plate never observed.
- duration: a step outside its [5th, 95th]-percentile window (seconds at 15 fps).
- Synthetic violations, one per kind per recording (seed 0): move one step before a learned
  predecessor's first occurrence; delete one mandatory step that occurs once; stretch one bounded
  step to 1.5 × its upper bound. Recall = the perturbed step itself is flagged by the matching
  check; false-alarm rate = the unperturbed recording is flagged by that check.

## 3. Result (`tables.md` is authoritative)

Synthetic violations on `gt` sequences (18 val recordings, seed 0); Wilson 95 % intervals in percent:

| violation | applicable | detected | recall | clean flagged | false-alarm rate | precision |
|---|---|---|---|---|---|---|
| order | 18 | 18 | 100.0 [82, 100] | 16 / 18 | 88.9 [67, 97] | 52.9 |
| omission | 17 | 17 | 100.0 [82, 100] | 1 / 18 | 5.6 [1, 26] | 94.4 |
| duration | 18 | 18 | 100.0 [82, 100] | 14 / 18 | 77.8 [55, 91] | 56.2 |

Unperturbed `gt` recordings with any finding: 17 / 18.

Synthetic violations on `pred` sequences (18 val recordings, seed 0):

| violation | applicable | detected | recall | clean flagged | false-alarm rate | precision |
|---|---|---|---|---|---|---|
| order | 15 | 15 | 100.0 [80, 100] | 17 / 18 | 94.4 [74, 99] | 46.9 |
| omission | 10 | 10 | 100.0 [72, 100] | 17 / 18 | 94.4 [74, 99] | 37.0 |
| duration | 18 | 18 | 100.0 [82, 100] | 18 / 18 | 100.0 [82, 100] | 50.0 |

Unperturbed `pred` recordings with any finding: 18 / 18.

Native `w` (wrong) segments on val, both hands — underpowered (fewer than 20 segments):

| ground-truth `w` segments | detected | recall | predicted `w` segments | false predicted | precision |
|---|---|---|---|---|---|
| 12 | 0 | 0.0 [0, 24] | 0 | 0 | — [0, 0] |

Sensitivity of the order check to the graph's minimum support (learned into scratch directories
with `learn-havid-sop --min-support N`, not committed; same val sequences and seed):

| min support | edges cylinder / gear / general | `gt` order recall | `gt` clean flagged | `pred` order recall | `pred` clean flagged |
|---|---|---|---|---|---|
| 3 (committed) | 53 / 18 / 7 | 18 / 18 | 16 / 18 | 15 / 15 | 17 / 18 |
| 10 | 18 / 17 / 3 | 18 / 18 | 12 / 18 | 15 / 15 | 15 / 18 |
| 20 | 10 / 15 / 1 | 17 / 17 | 8 / 18 | 12 / 12 | 14 / 18 |
| 40 | 0 / 0 / 1 | 6 / 6 | 0 / 18 | 0 / 0 | 4 / 18 |

Reading:

- **Order, ground truth:** the check is valid (every synthetic swap is flagged, recall 100 %
  [82, 100]) but the learned graph is too strict: 16 of 18 unperturbed val recordings violate it,
  and the violation counts are bimodal (six recordings with 11–16 violations, the rest with 0–3),
  i.e. whole alternative routes rather than isolated slips. Raising the minimum support trades
  false alarms for coverage (12 / 18 at support 10, 8 / 18 at 20) and at 40 the graph is empty. An
  edge rule of "holds in every co-occurring recording" cannot represent the stage-2/3 instruction
  variants; per-variant graphs or a majority rule are the follow-up.
- **Omission, ground truth:** the only check that works as a recording-level alarm on real
  sequences — recall 100 % [82, 100], one false alarm in 18 (5.6 % [1, 26]), precision 94.4 % —
  because only 4–5 steps per plate are mandatory at the 90 % threshold.
- **Duration, ground truth:** the synthetic stretch (1.5 × the upper bound) is always caught, but
  with 7–20 steps per recording and a per-step window that excludes 10 % of train durations by
  construction, 14 of 18 clean recordings contain at least one out-of-window step. A per-step
  window is not a recording-level alarm; a wider window (1st–99th) or a k-of-n rule is needed.
- **Predicted sequences:** every check fires on 17–18 of 18 unperturbed recordings. The v3
  causal recogniser's sequences are fragmented (up to 38 steps where the ground truth has ≤ 20),
  contain labels the graph never saw (up to 4 per recording) and miss mandatory steps (omission
  false alarms 17 / 18). At F1@10 ≈ 30 the recogniser is the bottleneck, not the checker.
- **Native `w`:** 0 of 12 ground-truth `w` segments are detected and the recogniser never
  predicts `w` at all (its most frequent confusion is `w` → `null`, 381–393 frames per hand). With
  12 segments the recall interval is [0, 24] %: underpowered, as ADR 0001 trigger B foresaw; no
  claim about error detection follows from it.
- Deviations are suggestions for human review; nothing here is a safety or compliance guarantee.

## 4. Cost

`havid-sop` runs in 1 s on CPU; `reproduce-lite` recomputes it from the CSVs and JSON files
without the dataset.

## 5. Reproduce

```bash
export PYTHONUTF8=1
make learn-havid-sop   # rewrites sop/ha-vid/*.json from the train split (idempotent)
make havid-sop         # this run
uv run sop-monitor reproduce-lite
```

## 6. Not done here

- No weighted fusion, no SOP-step granularity from the instruction sheets, no deviation queue or
  review UI, no online (streaming) checks; violations are evaluated on whole-recording sequences.
- No per-variant graphs, no majority-rule edges, no k-of-n duration rule (all named above as
  follow-ups).
- No number on the frozen test subjects.
