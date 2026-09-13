# Claims audit

The README's forbidden-claims table (design spec §2, quoted verbatim in `README.md`) lists seven
claims that must not appear in public text. This file records, for each one, whether the repository
currently makes it, where the permitted wording is used instead, and the evidence the wording rests
on. Re-run it before any publication.

Last audit: 2026-09-14, against `README.md`, `MODEL_CARD.md`, `docs/*.md` and every
`reports/*/README.md` (plans under `docs/plans/` are working notes, not public claims).

## How the audit was run

```bash
grep -rniE "real[- ]time|factory|工廠|safety|合規|compliance|commercial|商業|generali[sz]|泛化|robust|VLM|detects? errors|state[- ]of[- ]the[- ]art|SOTA|production[- ]ready" \
  --include="*.md" README.md MODEL_CARD.md docs reports | grep -v docs/plans/
```

Every hit was read in context; the verdicts are below.

| forbidden claim | made? | where the topic appears and how it is worded | evidence |
|---|---|---|---|
| real-factory generalisation | no | `MODEL_CARD.md` "Out of scope: real factory lines"; `docs/what_this_does_not_show.md` "One laboratory, one product"; `reports/industreal_dev_v1` uses "cross-participant generalisation" for a val gap, which is a within-dataset statement | all numbers are on HA-ViD / IndustReal validation subjects (`reports/README.md`) |
| safety or compliance guarantee | no | every SOP report (`havid_dev_v4`–`v8`), `docs/review.md`, the review page and `MODEL_CARD.md` state that deviations are suggestions for human review and not a safety or compliance guarantee | `reports/havid_dev_v8_sop_sheet`: predicted sequences flagged 18 / 18 |
| robustness to unseen error types | no | `docs/what_this_does_not_show.md` "Synthetic violations are not real violations"; SOP reports keep synthetic and native tables apart | HA-ViD native `wrong`: 67 segments, 0 of 12 detected on val (`havid_dev_v8`) and 0 of 16 on the test subjects (`havid_test_v1`) — the README table's "HA-ViD `wrong` 段數未公布" is now measured, and the permitted wording still applies |
| a VLM can detect errors | no | `README.md` lists the VLM verifier as not built; `docs/review.md` reserves `second_opinion` as not implemented | no VLM has been run |
| commercially usable | no | `MODEL_CARD.md` licence section and `docs/what_this_does_not_show.md`: HA-ViD derivatives are CC BY-NC 4.0 | `docs/decisions/0001-dataset-and-licences.md`, `sop/ha-vid/NOTICE.md` |
| real-time on arbitrary hardware | no | the only "real-time" mentions are the README table itself; `docs/what_this_does_not_show.md` states that no streaming throughput is measured | W4 not built; all timings are training / extraction wall times on one RTX 4090 |
| synthetic violations = real violations | no | synthetic and native tables are separate in every SOP report, and the synthetic table's heading says "synthetic" | `reports/havid_dev_v4`–`v8` `tables.md` |

## Wording rules the audit checks for

- HA-ViD results are qualified as "development result on `splits/ha-vid/val.csv`" unless they come
  from `reports/havid_test_v1*`. The phrase "on HA-ViD's fixed three views, held-out subjects" is used
  only for those rows, which were evaluated once under `docs/havid_test_protocol.md` (2026-09-14).
- The L = 45 test row is always labelled as added after the val results and not pre-registered; the
  pre-registered online operating point is L = 90.
- IndustReal numbers never appear in a HA-ViD table (`reports/README.md` keeps separate tables).
- Seed-sensitive comparisons are reported as mean ± std over three seeds
  (`docs/decisions.md`, 2026-09-14).

## Open items before publication

- The HA-ViD test split has been used (`reports/havid_test_v1.md`); it cannot be used again for model
  or setting selection. The IndustReal test split stays unevaluated (development dataset).
- Add the reviewed precision of a deviation queue once a person has reviewed one.
- Re-run this audit after any README or report change.
