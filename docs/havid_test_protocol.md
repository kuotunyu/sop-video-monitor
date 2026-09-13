# HA-ViD frozen test evaluation: pre-registered protocol

Written 2026-09-14, before the look-ahead sweep (`havid_dev_v9`) finished and before any HA-ViD
test video, feature or annotation row was read. The test split is `splits/ha-vid/test.csv`
(official test subjects S02, S04, S05, S19, S22, S24, S26; 41 recordings, 123 videos; SHA-256 in
`splits/ha-vid/test_sha256.txt`). The evaluation runs **once**. Anything decided after it is
reported as a post-hoc observation, never as a test result.

What had been seen on val when this was written: every committed run up to `havid_dev_v8`, the
v9 runs of seed 0 at look-ahead 15 (both hands) and 45 (left hand), and the per-step frame recall
of the look-ahead-15 seed-0 pair (all step frames 41.5 %, mandatory step frames 47.0 %, against
29.3 % / 32.8 % without look-ahead and 48.9 % / 55.8 % offline, three seeds each).

The IndustReal test split is not evaluated: IndustReal is a metric donor and development dataset
for this project (`docs/decisions/0001-dataset-and-licences.md`), and its test archives were never
downloaded.

## 1. What is fixed before the test is read

| component | fixed value | source |
|---|---|---|
| frame features | DINOv2 ViT-B/14, `[CLS ; mean patch]`, 392 × 224, every frame, weights SHA-256 `0b8b82f8…` | `artifacts/features/ha-vid/dinov2_vitb14_s1/meta.json` |
| recogniser | one MS-TCN++ per view (default `MSTCNSpec`), 50 epochs, epoch chosen on val F1@10, trained on train only | decisions 2026-09-13 |
| fusion | mean of the per-view posteriors (reported row); geometric and confidence-weighted listed next to it | decisions 2026-09-13 |
| seeds | 0, 1, 2; every test number is mean ± std over them | decisions 2026-09-14 |
| SOP granularity and knowledge | instruction-sheet steps, `sop/ha-vid/sheet/` (support 10, agreement 1.0, [1, 99] % duration windows, mandatory ≥ 90 %), learned on train | decisions 2026-09-14 |
| synthetic violations | the three operators of `havid_sop.synthetic_table`, perturbation seed 0 | `havid_dev_v4`–`v8` |
| plate | read off the ground-truth labels, as on val | `havid_sop.plate_of` |

## 2. Selection rules for the two open settings (applied on val only)

**Look-ahead L\*.** Among L ∈ {15, 45, 90} frames, L\* is the smallest L whose mandatory-step frame
recall on val (`havid-step-recall`, causal mean fusion, both hands × three seeds pooled) reaches
90 % of the offline mean fusion's value (same pooling). If no L reaches it, L\* = 90. Rationale:
the SOP checks consume mandatory steps, and a monitor that misses them cannot report omissions or
order; the smallest sufficient delay keeps alarms early.

**Confirmation length m\*.** Among m ∈ {1, 4, 8, 15, 30} frames, on the `havid_dev_v10` online
replay at L\* (three seeds): m\* minimises the mean number of alarms per recording on the predicted
streams, subject to the ground-truth streams flagging (any of order, omission, duration) at most
one more recording than at m = 1. Ties go to the smaller m.

Both choices, with the val numbers that produced them, are logged in `docs/decisions.md` before
the test features are extracted.

## 3. Reproducibility gate before the test

The final networks are retrained with `--checkpoint-dir` (the committed dev runs saved no weights):
causal L = 0 with the offline twins, and causal L\*, for both hands and seeds 0–2. Each retrained
run's `predictions_val.csv` is compared with the committed dev run of the same setting and seed. If
any differs, the val metrics of the retrained runs are reported next to the committed ones and
the test uses the retrained networks; the difference is named in the test report.

## 4. The single test run

1. Extract DINOv2 features for the 123 test videos into `artifacts/features/ha-vid/dinov2_vitb14_s1`
   with the same extractor (no other test data are touched before this step).
2. Predict the test recordings with the saved networks (no training, no epoch or rule selection),
   write `predictions_test.csv` per hand, setting and seed.
3. Score MoF, Edit and F1@{10, 25, 50} with participant bootstrap intervals; summarise seeds.
4. SOP layer on test: synthetic violation table on ground-truth and predicted sequences
   (L = 0 and L\*), native `w` table (16 segments, labelled underpowered by ADR 0001 trigger B),
   recording-level table, online replay at m\* for L = 0 and L\*.
5. Per-step frame recall on test for L = 0, L\* and offline.
6. Reports go to `reports/havid_test_v1_*`; the command refuses to run when that directory exists.

## 5. Reporting rules

- Test and val numbers sit in separate tables; each test table says "held-out subjects, evaluated
  once" and names this protocol.
- The rows are L = 0 (online), L\* (online with an L\* / 15 s output delay) and offline (not online).
- No setting, threshold or knowledge file changes because of the test results. Failures stay in
  the report.
- After the run, `docs/claims_audit.md` and `docs/what_this_does_not_show.md` are updated; only then
  may the wording "on HA-ViD's fixed three views, held-out subjects" be used, and only for these
  rows.
