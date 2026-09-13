# havid_test_v1 — HA-ViD held-out subjects, evaluated once

- Tier: **formal result** on the frozen test split `splits/ha-vid/test.csv` (the official HA-ViD test
  subjects S02, S04, S05, S19, S22, S24, S26: 41 recordings, 123 videos, 49,921 frames per hand).
  Evaluated once under [`docs/havid_test_protocol.md`](../docs/havid_test_protocol.md).
- Date: 2026-09-14, 06:15–06:30. Code: commit `21f9b55` (the test pipeline ran from it). Author:
  kuotunyu. Machine: Windows 11, RTX 4090.
- Rows: causal L = 0 (online, no delay), causal L\* = 90 (online, 6 s output delay; pre-registered),
  offline (not online), and causal L = 45 (3 s delay) — **added after the val results, not
  pre-registered** (protocol §6). Every number is mean ± std over seeds 0–2 unless marked.

## 1. What was fixed before the test, and what happened during it

| step | what | evidence |
|---|---|---|
| protocol | features, recogniser, fusion, seeds, SOP knowledge, selection rules, gate, reporting rules | `docs/havid_test_protocol.md`, commit `012a50a`, written before the v9 sweep finished |
| selections on val | L\* = 90 frames (45 missed the threshold by 0.45 points), m\* = 8 frames | `docs/decisions.md`, commits `da18e02`, `2b6d394` |
| amendment | L = 45 added because L = 90 was dominated on val; L\* unchanged | protocol §6, commit `2b6d394` |
| reproducibility gate | all 18 retrained networks (L = 0 with offline twins, 45, 90) reproduce their committed dev runs frame by frame | `reports/havid_final_gate/` |
| test run | `make havid-test LSTAR=90 MSTAR=8 LEXTRA=45`: DINOv2 features of the 123 test videos, 18 prediction runs, seed summaries, 9 SOP runs, online replay | directories below |
| incident | the last stage (per-step recall) failed on a Makefile quoting bug (a comma inside `$(if …)`); it was run by hand with the protocol's groups; it reads the finished prediction tables and selects nothing. The Makefile is fixed. | this section |

No setting, threshold, knowledge file or code path changed after test predictions existed.

Directories: `havid_test_v1_tas_L{0,45,90}_{lh,rh}_s{0,1,2}/` (predictions and metrics per run),
`havid_test_v1_seeds_L{0,45,90}_{lh,rh}/` (seed summaries), `havid_test_v1_sop_L{0,45,90}_s{0,1,2}/`
(SOP checks), `havid_test_v1_online/` (frame-by-frame replay), `havid_test_v1_step_recall/`.
`reproduce-lite` recomputes all of them from their committed tables.

## 2. Recogniser (primitive tasks, mean fusion of three views)

| row | delay | left MoF | left Edit | left F1@10 | left F1@50 | right MoF | right Edit | right F1@10 | right F1@50 |
|---|---|---|---|---|---|---|---|---|---|
| causal L = 0 | 0 s | 37.1 ± 0.5 | 32.1 ± 0.8 | 26.9 ± 1.8 | 10.4 ± 1.5 | 35.8 ± 1.9 | 32.8 ± 0.9 | 28.8 ± 0.4 | 11.6 ± 1.1 |
| causal L\* = 90 | 6 s | 37.5 ± 2.3 | 32.8 ± 0.7 | 30.5 ± 0.1 | 12.4 ± 0.6 | 37.5 ± 0.5 | 34.8 ± 0.9 | 30.2 ± 1.1 | 12.4 ± 1.6 |
| causal L = 45 (amendment) | 3 s | 40.1 ± 1.5 | 33.6 ± 2.4 | 32.9 ± 1.9 | 14.5 ± 0.9 | 40.1 ± 0.8 | 35.6 ± 1.2 | 34.1 ± 1.2 | 14.1 ± 0.4 |
| offline | whole recording | 43.7 ± 0.8 | 38.5 ± 0.9 | 39.7 ± 1.0 | 21.2 ± 0.3 | 44.3 ± 0.6 | 40.7 ± 1.1 | 42.2 ± 0.2 | 23.0 ± 0.9 |
| majority class | — | 32.2 | 8.3 | 9.1 | 0.3 | 28.3 | 7.9 | 6.8 | 0.0 |

Val → test, F1@10 (left / right): L = 0 30.5 / 29.9 → 26.9 / 28.8; L = 90 33.4 / 30.9 → 30.5 / 30.2;
L = 45 36.4 / 32.5 → 32.9 / 34.1; offline 43.9 / 42.0 → 39.7 / 42.2. The ranking L = 45 > L = 90 >
L = 0 seen on val holds on test for both hands and every metric in the table.

For scale only: the HA-ViD paper's MS-TCN (I3D features, one view at a time, offline, trained on all
official train subjects, averaged over views) reports primitive-task F1@10 36.6 (left) / 34.7 (right)
on these same test subjects. The offline row here (DINOv2, three-view fusion, 17 training subjects,
epoch chosen on 6 val subjects) is a different system; it is not a reproduction of that table.

Per-step frame recall at instruction-sheet granularity (`havid_test_v1_step_recall/tables.md`, both
hands × three seeds pooled):

| row | all step frames | mandatory step frames |
|---|---|---|
| causal L = 0 | 26.5 % | 28.5 % |
| causal L\* = 90 | 34.0 % | 35.5 % |
| causal L = 45 (amendment) | 37.5 % | 40.3 % |
| offline | 42.9 % | 46.4 % |

The insert steps stay the weak point on test: at L = 45, gear `iplft` 6 %, `ipsft` 16 %, general `iibn`
5 %, `iirn` 2 %, `isbn` 3 %, `iusn` 11 %, cylinder `ibacb` 5 %; offline is also low on most of them
(7–26 %), so these are recognition failures more than latency failures.

## 3. SOP checks on test (knowledge `sop/ha-vid/sheet`, learned on train)

**Checker validity, ground-truth sequences** (identical for every row): synthetic violations are
found with 100 % recall (order 37 / 37, omission 41 / 41, duration 41 / 41). Unperturbed recordings
flagged: order 7 / 41 (17.1 %, Wilson [9, 31]), omission 4 / 41 (9.8 % [4, 23]), duration 23 / 41
(56.1 % [41, 70]); any 26 / 41. On val the same checks flagged 4, 1 and 8 of 18. The duration windows
(train [1, 99] % quantiles) generalise worst: more than half of the clean test recordings contain a
step outside its window.

**Deployable numbers, predicted sequences** (unperturbed recordings flagged, pooled over the three
seeds; recall of every synthetic violation stays 100 %):

| row | order | omission | duration | any |
|---|---|---|---|---|
| causal L = 0 | 94 / 121 (78 %) | 116 / 121 (96 %) | 117 / 121 (97 %) | 121 / 121 |
| causal L\* = 90 | 95 / 121 (79 %) | 112 / 121 (93 %) | 119 / 121 (98 %) | 121 / 121 |
| causal L = 45 (amendment) | 61 / 122 (50 %) | 89 / 122 (73 %) | 120 / 122 (98 %) | 122 / 122 |

(121 instead of 123: recording S02A35I32 is predicted as `null` in every frame by both hands in five
of the nine runs, so it has no predicted step sequence; such a recording would be an omission.)

**Native `w` (ADR 0001 trigger B, underpowered):** 16 ground-truth `w` segments in 7 recordings;
detected 0 of 16 in all nine runs (Wilson [0, 19] %); the recogniser predicts `w` twice in one run,
both false.

**Online replay** at the pre-registered m\* = 8 frames (`havid_test_v1_online/tables.md`): the frame
stream and the offline checker agree at m = 1 on 38 / 41 ground-truth recordings (the 3 differences
are adjacent same-label segments, as on val) and on 41 / 41 recordings of every predicted stream.

| stream | recordings flagged, with `w` · without `w` | alarms per recording | first alarm (median s) |
|---|---|---|---|
| ground truth | 5 / 7 · 23 / 34 | 2.3 | 28.2 |
| causal L = 0 | 7 / 7 · 34 / 34 | 11.7 ± 0.9 | 21.3 ± 3.3 |
| causal L\* = 90 | 7 / 7 · 34 / 34 | 9.7 ± 0.5 | 25.6 ± 2.1 |
| causal L = 45 (amendment) | 7 / 7 · 34 / 34 | 9.7 ± 0.2 | 20.4 ± 2.3 |

## 4. Reading

- **The development picture survives the held-out subjects.** Test F1@10 is 0.7–3.6 points below val
  for the causal rows except the right hand at L = 45 (+1.6), and offline moves −4.2 (left) / +0.2
  (right); no row collapses. The six validation subjects did not grossly overfit the protocol, but the
  left hand loses 3–4 points in every row.
- **A 3 s output delay is the best online operating point measured, on test as on val**: +6.0 / +5.3
  F1@10 over L = 0, 40.3 % vs 28.5 % mandatory-step frame recall, order false alarms 50 % instead of
  78 %. The pre-registered L\* = 90 beats L = 0 on every recogniser metric but not on the SOP false
  alarms (order 79 % vs 78 %), and it is worse than L = 45 on every recogniser, step-recall, order and
  omission number (tied on duration flags and alarms per recording); the pre-registered fallback rule
  was the wrong rule, as the amendment recorded before the test.
- **The online SOP monitor is not usable as an alarm on predicted sequences.** Every clean test
  recording is flagged in every run, with about 10 alarms per recording even at the best delay. The
  checks themselves work on ground truth for order and omission (17 % and 10 % false alarms) but not
  for duration (56 %). The bottleneck remains the recogniser, mostly on short insert steps that the
  offline model also misses.
- **Native errors are not detected** (0 of 16), as predicted by ADR 0001 trigger B: the evidence for
  error detection is synthetic only.

## 5. What these numbers may be used for

Allowed wording: "on HA-ViD's fixed three views, held-out subjects", for these rows only, with the
seed spread. Not allowed: any statement about other products, stations, cameras or real factories;
any statement that the system detects real errors; any use of L = 45 as a pre-registered result.

## 6. Reproduce

```bash
export PYTHONUTF8=1
make havid-final-L0 && make havid-final-lstar LSTAR=90 && make havid-final-lstar LSTAR=45
make havid-test LSTAR=90 MSTAR=8 LEXTRA=45      # refuses to run while reports/havid_test_v1_* exist
uv run sop-monitor reproduce-lite
```
