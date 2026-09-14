# Claims audit

The design spec's forbidden-claims table (§2, quoted verbatim in the appendix below) lists seven
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
| real-time on arbitrary hardware | no | real-time numbers are stated for one RTX 4090 and replayed files only, with the curve (`reports/stream_bench_v1_dinov2`, `havid_dev_v11_online_head_*`); `docs/what_this_does_not_show.md` lists what is not measured | 24 streams decoded and embedded at 15 fps on one machine; one station end to end at 0.32 × real time |
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

## Appendix: the forbidden-claims table of the design spec (§2, verbatim, zh-TW)

以下每一條在發佈前逐條核對；左欄任何一句出現在 README 即視為 blocker。

| 禁止宣稱 | memo 證據 | README 允許的替代表述 |
|---|---|---|
| 真實工廠泛化 | 所有候選集皆為實驗室裝配台；ATTACH 由 person split 改 view split，平均準確率 57.4 → 30.4；unseen-view TAS 仍是 open problem（memo §4） | 「在 HA-ViD 固定三視角、held-out subjects 上」 |
| 安全或合規保證 | IMPACT 每個 baseline 的 recovery-phase F1 接近零；HoloAssist 最佳模態 F 40.19（memo §4） | 「偏差為建議，一律需人工複核」 |
| 未見錯誤型態的 robustness | IndustReal 僅 38 個 execution errors、14 個只在 val/test（memo §4）；HA-ViD `wrong` 段數未公布（memo §5） | 「僅涵蓋資料集已標註的錯誤型態與明示為 synthetic 的順序違規」 |
| VLM 可偵測錯誤 | zero-shot Qwen2.5-VL-7B 在 MD-VQA 協定 F1 0.0，GRPO 後 53.8／48.0 且需 4×H100；ZeProM 需 4×H100 跑 87.8 分鐘（memo §2） | 「VLM 為非同步第二意見，獨立列表，不進主指標」 |
| 商業可用 | HA-ViD CC BY-NC 4.0；IMPACT data CC BY-NC-SA 4.0（memo §1） | 「權重與快取特徵為 non-commercial 衍生物」 |
| 任意硬體 real-time | 唯一同儕審查的產線系統 I3D + ActionFormer 僅 0.53× real-time，且資料私有（memo §3） | 「單張 RTX 4090 上 N 路 × M fps，附曲線」 |
| 合成違規 = 真實違規 | memo §4 要求 synthetic 表獨立標示 | 兩表分開，表名與圖例含 synthetic |

IndustReal 的數字不會進任何 HA-ViD 表格（設計規格 §3.3）；IndustReal 衍生物依 Apache-2.0 處理。
