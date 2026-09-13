# havid_dev_v8_sop_sheet — SOP checks at instruction-sheet granularity (development result)

- Tier: **development result** on `splits/ha-vid/val.csv` (6 subjects, 18 recordings). Knowledge
  and settings chosen on the train split only (leave-one-subject-out, `oof.json`); val read once
  for this report; the frozen test subjects were never read. Synthetic table = synthetic; native
  table underpowered.
- Date: 2026-09-14. Author: kuotunyu. Machine: Windows 11 (CPU only, seconds).
- Purpose: [`../havid_dev_v5_sop_tuned/`](../havid_dev_v5_sop_tuned/README.md) left 8 of 18 clean
  ground-truth recordings flagged by the order check, many of them between screw steps of the same
  sub-assembly whose order the instruction sheet does not fix. This run changes the *step
  definition*, not the rule.

## 1. Step granularity

`havid_sop.sheet_step`: an HR-SAT primitive-task label is `<verb><manipulated>[<target>[<tool>]]`
with two-character codes (paper Figure 9). The instruction sheets (`I21`–`I26`) repeat steps such
as "Screw M8 screw into cylinder bracket hole" once per hole, never fix which hole comes first and
do not require the tool, so the sheet-level step drops the hole index (`c1`–`c4` → `c`, `g1`–`g3`
→ `g`, `n1`–`n6` → `n`) and the tool code (`dh`, `dp`, `wn`, `ws`). All 73 primitive-task labels of
`mapping.txt` parse; they collapse to 51 sheet steps, e.g. twelve `sshc*` labels → `sshc`, four
`sftg*` → `sftg`, `sntftwn` → `sntft`. The mapping is applied after the two hands' segments are
merged, to ground truth and predictions alike; plates are still read off the original labels.
`run_havid_sop` refuses knowledge learned at a different granularity.

## 2. Tuning (out of fold on train, `oof.json`)

Same grid and selection rule as v5. At sheet granularity the strict rule wins again and the
out-of-fold numbers change sharply:

| granularity | selected support / agreement | edges | coverage | recording false alarms | step false alarms | duration step false alarms at [1, 99] % |
|---|---|---|---|---|---|---|
| primitive task (v5) | 20 / 1.0 | 26 | 90 % | 32 % | 5.6 % | 6.6 % |
| **sheet (v8)** | **10 / 1.0** | **14** | **94 %** | **8 %** | **2.0 %** | **5.0 %** |

(Support 10 and 20 tie at sheet granularity; the rule prefers the smaller support.) Written to
`sop/ha-vid/sheet/`: cylinder 17 steps / 5 edges / 5 mandatory / 10 bounded, gear 11 / 8 / 8 / 10,
general 13 / 1 / 6 / 12. The learned edges read like the sheets: cylinder `ibscb → ibacb → scccb →
lck` (ball seat, ball, cap, slide brackets) and `ickcb → sshc`; gear `sftg → igsft → sntft` and
every gear step `→ rgw` (rotate the handle to check meshing last); general `isbn → sntsb`.

## 3. Result (`tables.md` is authoritative)

Synthetic violations on `gt` sequences (18 val recordings, seed 0); Wilson 95 % intervals in percent:

| violation | applicable | detected | recall | clean flagged | false-alarm rate | precision | per-step false alarms |
|---|---|---|---|---|---|---|---|
| order | 17 | 17 | 100.0 [82, 100] | 4 / 18 | 22.2 [9, 45] | 81.0 | 15 / 234 = 6.4 [4, 10] |
| omission | 17 | 17 | 100.0 [82, 100] | 1 / 18 | 5.6 [1, 26] | 94.4 | — |
| duration | 18 | 18 | 100.0 [82, 100] | 8 / 18 | 44.4 [25, 66] | 69.2 | 10 / 234 = 4.3 [2, 8] |

Unperturbed `gt` recordings with any finding: 10 / 18 (v5: 14 / 18).

Synthetic violations on `pred` sequences (v3 `fusion_causal`, seed 0):

| violation | applicable | detected | recall | clean flagged | false-alarm rate | precision | per-step false alarms |
|---|---|---|---|---|---|---|---|
| order | 12 | 12 | 100.0 [76, 100] | 12 / 18 | 66.7 [44, 84] | 50.0 | 56 / 354 = 15.8 [12, 20] |
| omission | 13 | 13 | 100.0 [77, 100] | 17 / 18 | 94.4 [74, 99] | 43.3 | — |
| duration | 18 | 18 | 100.0 [82, 100] | 18 / 18 | 100.0 [82, 100] | 50.0 | 196 / 354 = 55.4 [50, 60] |

Native `w`: 12 ground-truth segments, 0 detected, none predicted (recogniser unchanged).

Recording level on the unperturbed sequences (counts only; a native `w` marks an erroneous
primitive task, which need not break order, completeness or timing):

| source | group | recordings | order | omission | duration | any |
|---|---|---|---|---|---|---|
| gt | with `w` | 4 | 2 | 1 | 1 | 3 |
| gt | without `w` | 14 | 2 | 0 | 7 | 7 |
| pred | with `w` | 4 | 3 | 4 | 4 | 4 |
| pred | without `w` | 14 | 9 | 13 | 14 | 14 |

v5 → v8 on ground truth, recording-level false alarms: order 8 → 4 of 18, duration 11 → 8,
omission 1 → 1; recall 100 % for every kind in both.

Reading:

- **Granularity was the right lever.** Out of fold the order check's recording-level false-alarm
  rate falls from 32 % to 8 % and its step rate from 5.6 % to 2.0 % at higher coverage; on val the
  clean ground-truth recordings flagged fall from 8 to 4 of 18 with recall unchanged. The duration
  window reaches the 5 % step-level target for the first time (4.3 % [2, 8] on val).
- **The remaining order findings are not all false alarms.** Two of the four flagged ground-truth
  recordings (S10A06I01 with 5 native `w` segments, S12A06I01 with 2) are two of the four val
  recordings that contain a native `wrong` annotation; the order check flags 2 of 4 recordings with
  `w` against 2 of 14 without. S12A06I01 also misses all four mandatory cylinder steps and uses
  three step labels the graph never saw. With four positive recordings this is an observation, not
  a detection rate.
- **Predicted sequences are still flagged everywhere** (any finding 18 / 18; omission 17 / 18;
  duration 55 % of steps): the v3 recogniser remains the bottleneck (v6), and granularity cannot
  fix steps it never predicts.
- Deviations are suggestions for human review; nothing here is a safety or compliance guarantee.

## 4. Reproduce

```bash
export PYTHONUTF8=1
make tune-havid-sop-sheet                                        # oof.json
make havid-sop-v8 SUPPORT=10 AGREEMENT=1.0 LOWQ=0.01 HIGHQ=0.99   # sop/ha-vid/sheet + this run
uv run sop-monitor reproduce-lite
```

## 5. Not done here

- No better recogniser, no online (streaming) checks, no deviation queue. The sheet mapping is a
  rule derived from the label grammar and the sheets' wording, not a step-by-step alignment of
  every instruction variant. No number on the frozen test subjects.
