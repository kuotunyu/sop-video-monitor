# HA-ViD v3 (F1@10 epoch selection, fusion rules) and W3 SOP layer (havid_dev_v4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> 給人看的一句話（zh-TW）：程式都已寫好、測過並 commit；這份文件只剩「等 v3 訓練結束 → 寫兩份 v3 報告 →
> 跑 W3 的 SOP 檢查（synthetic 違規表 + 原生 `w` 表）→ 寫 v4 報告 → 更新總表／決策紀錄／README → commit」。
> 沒有任何一步需要使用者決定；失敗就停在那一步，把狀況寫進最後的總結。

**Goal:** Commit `reports/havid_dev_v3_tas_f1sel_{lh,rh}` (DINOv2 features, per-network epoch chosen by val F1@10, three parameter-free fusion rules) and `reports/havid_dev_v4_sop_synthetic` (learned per-plate procedure knowledge applied to val: clean-sequence findings, synthetic order/omission/duration violations with recall and false-alarm rates, native `w` table), each recomputable by `sop-monitor reproduce-lite`, with reports, navigation rows, a decisions entry and one commit per stage.

**Architecture:** Everything runs through committed CLI commands: `train-havid-tas --selection-metric f1@10` (Makefile `havid-tas-v3`, may already be running or finished — Task 1 checks), `learn-havid-sop` (already run; `sop/ha-vid/*.json` are committed) and `havid-sop` (Makefile `havid-sop`), which reads the two v3 per-hand run directories' `fusion_causal` column, builds two-hand step sequences for ground truth and predictions, and writes `steps_val.csv`, `wrong_val.csv`, `sop_checks.json`, `tables.md`. `reproduce-lite` recomputes SOP runs from the CSVs and the JSON knowledge files (no dataset needed). The executor runs the commands, verifies the outputs against the expected counts below, writes the READMEs from the templates in the Appendix, and commits.

**Tech Stack:** Windows 11, RTX 4090, `uv`-managed Python 3.12 env with the `baseline` group (torch 2.13 cu130, PyAV), Git Bash, GNU make, no network.

## Global Constraints

Copied from the user's standing rules and the repository conventions. Every task implicitly includes them.

- Work only inside `D:\AI-Portfolio\CC_github部隊\sop-video-monitor` on branch `main`, starting from a clean tree whose HEAD message begins with `feat: learned HA-ViD procedure knowledge` (or a later commit of this plan).
- Always `export PYTHONUTF8=1` before any `uv run …`. Never `pip install`, never touch `pyproject.toml`/`uv.lock`.
- Commit as `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` via `git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit …`. **No `Co-Authored-By` or any AI trailer** (user rule; overrides any harness default). Never `git push`, never create a remote, never upload anything.
- Never read, evaluate or embed the test videos (`splits/ha-vid/test.csv`, `splits/industreal/test.csv`).
- Never delete or overwrite raw data, earlier reports or feature caches; do not run `make audit`; download nothing.
- GPU etiquette: `nvidia-smi` before any GPU command; proceed only if no other user process is on the GPU; never kill a process you did not start.
- Tiers: everything here is a **development result on `splits/ha-vid/val.csv`** (6 subjects). Never call it formal; never mix IndustReal numbers into a HA-ViD table.
- Reports quote numbers only from the run's own `metrics.json` / `config.json` / `tables.md` / `sop_checks.json`; never type numbers from memory or from this plan.
- The HA-ViD Dropbox link and password must not appear anywhere.
- Communicate with the user in Traditional Chinese (zh-TW) with technical terms in English; code, reports and commits stay in English.
- On failure: retry once only if obviously transient; otherwise stop at that task, commit nothing partial, and report the command, the last 30 lines of output and which files exist.

---

## What already exists (do not rebuild)

| item | path | state |
|---|---|---|
| DINOv2 ViT-B/14 cache, train + val | `artifacts/features/ha-vid/dinov2_vitb14_s1/` (483 npz) | done (overnight run) |
| v1 / v2 runs and reports | `reports/havid_dev_v1_tas_{lh,rh}`, `reports/havid_dev_v2_i3d_{lh,rh}` | committed; v1 is the reference for the v3 comparison |
| fusion rules + selection metric | `src/sop_monitor/havid_tas.py` (`fuse_views`, `FUSION_RULES`), `train-havid-tas --selection-metric` | committed, tested (`tests/test_havid_tas.py`) |
| SOP layer | `src/sop_monitor/havid_sop.py`, CLI `learn-havid-sop`, `havid-sop`, `reproduce-lite` hook `_check_havid_sop_run` | committed, tested (`tests/test_havid_sop.py`) |
| learned knowledge | `sop/ha-vid/learned_precedence_{cylinder,gear,general}.json`, `duration_bounds.json`, `mandatory_steps.json` | committed; cylinder 31 steps / 53 edges / 4 mandatory / 23 bounded, gear 16 / 18 / 5 / 15, general 16 / 7 / 4 / 15; train plates gear 50, cylinder 39, general 54 |
| v3 training | `make havid-tas-v3` was started at 11:14 on 2026-09-13 in a background shell of the previous session (log `%LOCALAPPDATA%\Temp\claude\D--AI-Portfolio-CC-github---sop-video-monitor\72649586-31d4-4283-b09d-c4b68ef82dc3\scratchpad\havid_tas_v3.log`) | may be finished, running, or killed by a restart — Task 1 decides |
| report conventions | `reports/README.md` (navigation), `reports/havid_dev_v1_tas_lh/README.md` (structure to mirror), `docs/decisions.md` | Appendix A/B templates |

Expected sizes (measured):

| quantity | value |
|---|---|
| v3 `predictions_val.csv` rows per hand | 17 973 |
| v3 prediction columns | `majority, view0_causal, view0_offline, view1_causal, view1_offline, view2_causal, view2_offline, fusion_causal, fusion_offline, fusion_geo_causal, fusion_geo_offline, fusion_conf_causal, fusion_conf_offline` (13 `pred_*` columns) |
| v3 `config.json` `training.<network>` keys | `best_val_f1@10` (not `best_val_mof`), curve entries keyed `f1@10` |
| v4 `steps_val.csv` | sources `gt` and `pred`, 18 recordings each; `wrong_val.csv` GT `w` segments on val = 12 (S01 5, S10 5, S12 2; none for S08, S18, S30) |
| `reproduce-lite` run list | must contain `havid_dev_v3_tas_f1sel_lh: OK`, `havid_dev_v3_tas_f1sel_rh: OK`, `havid_dev_v4_sop_synthetic: OK` |

---

### Task 0: Sanity (5 min, no GPU)

**Files:** read only.

- [ ] **Step 1: Repository, environment, GPU**

Run:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && git status --short && git log --oneline -1 && export PYTHONUTF8=1 && uv run pytest -q 2>&1 | tail -1 && uv run sop-monitor verify-splits --directory splits/ha-vid | tail -1 && nvidia-smi --query-compute-apps=pid,process_name --format=csv
```
Expected: clean tree; HEAD message begins with `feat: learned HA-ViD procedure knowledge` (or later); `135 passed`; `OK: test list hash matches test_sha256.txt`. The compute-apps list may contain a `python.exe` — that is the v3 training if it is still running (Task 1 Step 1 tells you); any other GPU user means wait, never kill.

- [ ] **Step 2: Keep the machine awake (optional)**

If `mcp__ccd_host__request_keep_awake` is available (ToolSearch `select:mcp__ccd_host__request_keep_awake`), call it with `until: "session_idle"`.

---

### Task 1: v3 runs — verify or (re)run, then report (0–45 min GPU + 25 min writing)

**Files:**
- Create/verify: `reports/havid_dev_v3_tas_f1sel_lh/{predictions_val.csv,metrics.json,config.json,tables.md,README.md}` and `…_rh/…`

**Interfaces:**
- Consumes: the DINOv2 cache; `splits/ha-vid/`; `reports/havid_dev_v1_tas_{lh,rh}/tables.md` for the comparison.
- Produces: the two v3 run directories that Task 2's `havid-sop` reads (`fusion_causal` column of both hands).

- [ ] **Step 1: Decide whether v3 is finished, running, or must be (re)started**

Run:
```bash
L="$LOCALAPPDATA/Temp/claude/D--AI-Portfolio-CC-github---sop-video-monitor/72649586-31d4-4283-b09d-c4b68ef82dc3/scratchpad/havid_tas_v3.log"; [ -f "$L" ] && grep -E "^wall:|exit=|Traceback" "$L"; ls "/d/AI-Portfolio/CC_github部隊/sop-video-monitor/reports" | grep v3; nvidia-smi --query-compute-apps=pid,process_name --format=csv | grep -i python || echo "no python on GPU"
```
Decision:
- Log shows two `wall:` lines and `exit=0`, and both `reports/havid_dev_v3_tas_f1sel_lh` and `_rh` exist with `metrics.json` → go to Step 3.
- A `python.exe` is on the GPU and the log has no `exit=` line → it is still running; wait (check every 10 min; each hand takes ≈ 18 min) until `exit=0`, then Step 3.
- Otherwise (no python on GPU, log missing or without `exit=0`, or a run directory missing `metrics.json`): delete only the *incomplete* v3 directories (`rm -rf reports/havid_dev_v3_tas_f1sel_lh` only if its `metrics.json` is missing; same for `_rh`) and go to Step 2.

- [ ] **Step 2: (Re)start v3 in the background (only if Step 1 said so)**

Run with `run_in_background: true`:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && export PYTHONUTF8=1 && make havid-tas-v3 > /tmp/havid_tas_v3.log 2>&1; echo "exit=$?" >> /tmp/havid_tas_v3.log
```
`make havid-tas-v3` runs `train-havid-tas --features artifacts/features/ha-vid/dinov2_vitb14_s1 --hand lh --selection-metric f1@10 --out reports/havid_dev_v3_tas_f1sel_lh` then the same for `rh` (defaults `--epochs 50 --eval-every 5 --n-boot 2000 --seed 0 --device auto`). Expect ≈ 18 min per hand; wait for `exit=0` (poll the log at most every 10 minutes).

- [ ] **Step 3: Verify both run directories**

Run:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && export PYTHONUTF8=1 && for h in lh rh; do d=reports/havid_dev_v3_tas_f1sel_$h; echo "== $d"; ls $d; tail -n +2 $d/predictions_val.csv | wc -l; head -1 $d/predictions_val.csv | tr ',' '\n' | grep -c pred_; uv run sop-monitor score-predictions --run $d | tail -1; done
```
Expected per hand: `config.json metrics.json predictions_val.csv tables.md` (README.md may be absent yet); `17973`; `13`; `… OK, metrics reproduce`.

- [ ] **Step 4: Print the numbers each README needs**

Save this script once as `/tmp/summarise_v3.py` and run it with `reports/havid_dev_v3_tas_f1sel_lh`, then `…_rh`:
```python
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
c = json.loads((run / "config.json").read_text(encoding="utf-8"))
metric = c["spec"]["mstcn"]["selection_metric"]
print("hand", c["spec"]["hand"], "selection metric", metric, "n_classes", c["classes"]["n_classes"], "majority", c["classes"]["class_id_to_label"][str(c["classes"]["majority_class_id"])])
print("train videos/frames", c["train_videos"], c["train_frames"], "eval videos/frames", c["eval_videos"], c["eval_frames"], "wall_seconds", round(c["wall_seconds"]))
for name, log in c["training"].items():
    curve = log["curve"]
    print(f"{name}: best epoch {log['best_epoch']} val {metric} {log['best_val_' + metric]:.1f}; first {curve[0]['epoch']}:{curve[0][metric]:.1f} last {curve[-1]['epoch']}:{curve[-1][metric]:.1f}; params {log['parameters']}; {log['train_seconds']:.0f}s")
def fmt(e): return f"{e['point']:.1f} [{e['ci95'][0]:.1f}, {e['ci95'][1]:.1f}]"
order = ["majority", "view0_causal", "view1_causal", "view2_causal", "fusion_causal", "fusion_geo_causal", "fusion_conf_causal",
         "view0_offline", "view1_offline", "view2_offline", "fusion_offline", "fusion_geo_offline", "fusion_conf_offline"]
for run_name in order:
    r = m["runs"][run_name]
    print(f"| {run_name} | {fmt(r['mof'])} | {fmt(r['edit'])} | {fmt(r['f1@10'])} | {fmt(r['f1@25'])} | {fmt(r['f1@50'])} |")
labels = c["classes"]["class_id_to_label"]
print("top confusions (gt -> pred, frames) for fusion_causal:")
for conf in m["confusions"][:6]:
    print("  ", labels[str(conf["gt"])], "->", labels[str(conf["pred"])], conf["frames"])
pv = m["per_video"]["fusion_causal"]
print("worst:", sorted(((round(s["mof"], 1), v) for v, s in pv.items()))[:3], "best:", sorted(((round(s["mof"], 1), v) for v, s in pv.items()), reverse=True)[:3])
```
Run: `cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && export PYTHONUTF8=1 && uv run python /tmp/summarise_v3.py reports/havid_dev_v3_tas_f1sel_lh`
Expected: `selection metric f1@10`, `train videos/frames 143 181015`, `eval videos/frames 18 17973`, 13 table rows.

- [ ] **Step 5: Write the two v3 READMEs**

Use Appendix A. The v1 comparison rows come from `reports/havid_dev_v1_tas_<hand>/tables.md` (rows `fusion_causal`, `fusion_offline`, and the best single causal view by F1@10: `view2_causal` for lh, `view1_causal` for rh). In "Reading", state (a) whether F1@10 epoch selection changed the causal numbers vs v1 (fusion_causal F1@10 and Edit, CI overlap), (b) which fusion rule is best on F1@10 and Edit among the causal runs and whether the differences exceed the intervals, (c) the best-epoch pattern (are the causal networks still picking epochs ≤ 10?). Do not speculate beyond the numbers.

- [ ] **Step 6: Verify and commit the v3 runs**

Run:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && export PYTHONUTF8=1 && grep -c "{{" reports/havid_dev_v3_tas_f1sel_lh/README.md reports/havid_dev_v3_tas_f1sel_rh/README.md; uv run sop-monitor reproduce-lite > /tmp/reproduce_v3.log 2>&1 && grep havid /tmp/reproduce_v3.log && git add reports/havid_dev_v3_tas_f1sel_lh reports/havid_dev_v3_tas_f1sel_rh && git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit -q -m "docs: report havid_dev_v3 (epoch by val F1@10, mean/geometric/confidence fusion) on DINOv2 features, val only" && git log --oneline -1
```
Expected: both `grep -c` counts are `0`; the reproduce log lists `havid_dev_v3_tas_f1sel_lh: OK` and `_rh: OK`; the commit line appears (the `&&` chain stops before `git add` if `reproduce-lite` exits 1).

---

### Task 2: W3 SOP run `havid_dev_v4_sop_synthetic` (5 min CPU + 30 min writing)

**Files:**
- Create: `reports/havid_dev_v4_sop_synthetic/{steps_val.csv,wrong_val.csv,sop_checks.json,tables.md,README.md}`

**Interfaces:**
- Consumes: `reports/havid_dev_v3_tas_f1sel_{lh,rh}` (`fusion_causal`), `sop/ha-vid/*.json`, the temporal zip, `splits/ha-vid/val.csv`.
- Produces: the SOP run directory; `reproduce-lite` recomputes it from `steps_val.csv` + `wrong_val.csv` + `sop/ha-vid/*.json` with the seed stored in `sop_checks.json`.

- [ ] **Step 1: Run the SOP evaluation**

Run:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && export PYTHONUTF8=1 && make havid-sop 2>&1 | tee /tmp/havid_sop.log | tail -40
```
(`make havid-sop` = `uv run sop-monitor havid-sop --pred-lh reports/havid_dev_v3_tas_f1sel_lh --pred-rh reports/havid_dev_v3_tas_f1sel_rh --run-name fusion_causal --out reports/havid_dev_v4_sop_synthetic`; defaults `--graphs sop/ha-vid --seed 0`.) Expected: it prints the tables (learned knowledge per plate; synthetic table for `gt` and for `pred`; the native `w` table) and ends with `-> reports/havid_dev_v4_sop_synthetic`.

- [ ] **Step 2: Verify the run directory**

Run:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && export PYTHONUTF8=1 && d=reports/havid_dev_v4_sop_synthetic && ls $d && cut -d, -f3 $d/steps_val.csv | sort | uniq -c && cut -d, -f2 $d/wrong_val.csv | sort | uniq -c && uv run python - <<'EOF'
import json
p = json.load(open("reports/havid_dev_v4_sop_synthetic/sop_checks.json", encoding="utf-8"))
print("graphs", p["graphs"])
for source, t in p["sources"].items():
    print(source, "recordings", t["recordings"], "clean flagged any", t["clean"]["flagged_any"], "by kind", t["clean"]["flagged_by_kind"])
    for kind, e in t["synthetic"].items():
        print(f"  {kind}: applicable {e['applicable']} detected {e['detected']} recall {e['recall']} ci {e['recall_ci95']} false-alarm {e['false_alarm_rate']} precision {e['precision']}")
w = p["wrong"]; print("wrong: gt", w["gt_segments"], "detected", w["detected"], "recall", w["recall"], w["recall_ci95"], "pred", w["pred_segments"], "false", w["false_pred_segments"], "underpowered", w["underpowered"])
print("plates on val", sorted({r["plate"] for r in p["sources"]["gt"]["clean"]["per_recording"].values()}))
EOF
```
Expected: `steps_val.csv` has rows for sources `gt` and `pred` (plus the header line); `wrong_val.csv` has 12 `gt` rows (the header counts once); `graphs` shows cylinder 31/53/4/23, gear 16/18/5/15, general 16/7/4/15; both sources have `recordings 18`; `wrong.gt_segments == 12` and `underpowered True`. Every other number is a result, not an expectation — copy them into the README.

- [ ] **Step 3: Write `reports/havid_dev_v4_sop_synthetic/README.md`**

Use Appendix B, filling every `{{…}}` from Step 2's output or `tables.md`. In "Reading", state for each violation kind, on `gt` and on `pred`: recall with its interval, false-alarm rate on unperturbed sequences, and what that means (e.g. "the order check is valid on ground truth — recall X — but fires on Y of 18 clean recordings, so the learned graph is too strict for the stage-2/3 routes"). State the native `w` result as underpowered with its interval. Do not claim anything about real-factory errors.

- [ ] **Step 4: Verify and commit**

Run:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && export PYTHONUTF8=1 && grep -c "{{" reports/havid_dev_v4_sop_synthetic/README.md; uv run sop-monitor reproduce-lite > /tmp/reproduce_v4.log 2>&1 && grep -E "havid|MISMATCH" /tmp/reproduce_v4.log && git add reports/havid_dev_v4_sop_synthetic && git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit -q -m "docs: report havid_dev_v4: learned SOP checks, synthetic order/omission/duration violations and the native wrong table on val" && git log --oneline -1
```
Expected: `0`; `havid_dev_v4_sop_synthetic: OK` in the log and no `MISMATCH`; commit line.

---

### Task 3: Navigation, decisions, README (20 min, no GPU)

**Files:**
- Modify: `reports/README.md` (HA-ViD table), `docs/decisions.md` (append), `README.md` (module table + CLI table + "還沒有什麼")

- [ ] **Step 1: Add three rows to the HA-ViD table in `reports/README.md`**

After the row that starts with `| \`havid_dev_v2_i3d_rh/\` |` insert (fill the tokens from the runs' `tables.md`):
```markdown
| `havid_dev_v3_tas_f1sel_lh/` | v1 with the epoch chosen by val F1@10 and three parameter-free fusion rules (mean / geometric / confidence-weighted), left hand | fusion_causal F1@10 {{v3 lh}} / Edit {{v3 lh}}; best rule {{rule}} | current HA-ViD TAS protocol |
| `havid_dev_v3_tas_f1sel_rh/` | Same, right hand | fusion_causal F1@10 {{v3 rh}} / Edit {{v3 rh}}; best rule {{rule}} | current HA-ViD TAS protocol |
| `havid_dev_v4_sop_synthetic/` | W3: learned per-plate precedence / mandatory steps / duration bounds applied to val; synthetic violations on ground-truth and on v3 `fusion_causal` sequences; native `w` | order recall gt {{}} / pred {{}}, false alarms {{}} / 18; `w` {{detected}} of 12 (underpowered) | first SOP-layer result |
```
Also update the four existing v1/v2 rows' status column: v1 rows become `superseded by v3 (epoch selection)`, v2 rows stay `control (feature choice)`.

- [ ] **Step 2: Append the decisions entry**

Append to `docs/decisions.md`:
```markdown

## 2026-09-13 — HA-ViD TAS epochs are chosen by val F1@10; fusion stays parameter-free

v1 chose each network's epoch by val MoF, which favoured `null` and stopped two causal networks at
epoch 5–10. v3 (`reports/havid_dev_v3_tas_f1sel_*`) chooses by val F1@10 and adds two
parameter-free fusion rules next to the mean (normalised geometric mean, confidence-weighted
mean), so nothing about the fusion is tuned on val. Result (fusion_causal F1@10, left / right):
v1 29.1 / 30.3 → v3 {{v3 lh}} / {{v3 rh}}; best fusion rule on F1@10: {{rule lh}} / {{rule rh}}
({{inside or outside}} the intervals). Decision: {{one sentence — which selection metric and
which fusion rule the HA-ViD line uses from now on, or "mean fusion stays the default" if the
rules are tied}}.

## 2026-09-13 — the HA-ViD SOP layer learns its procedure knowledge from the train split

The public OWL graphs name steps `PT1`… without HR-SAT codes, so `sop/ha-vid/learned_*.json`
are learned from the 143 train recordings per plate (plate read off the label vocabulary):
precedence edges that hold in every co-occurring recording (min support 3), mandatory steps
(present in ≥ 90 % of a plate's recordings), duration bounds (5th–95th percentile). On val
(`reports/havid_dev_v4_sop_synthetic`): order-violation recall on ground-truth sequences
{{gt order recall}} with {{gt order false alarms}} / 18 clean recordings flagged; on the v3
`fusion_causal` sequences {{pred order recall}} with {{pred false alarms}} / 18 flagged;
omission and duration: {{one clause each}}. Native `w`: {{detected}} of 12 segments detected,
underpowered as ADR 0001 trigger B foresaw. Consequence: the synthetic table is the headline
violation table (marked synthetic); the native table stays a count with a Wilson interval.
```

- [ ] **Step 3: Root `README.md`**

Three exact edits:
1. In the module table, after the `havid_tas.py` row, add: `| \`havid_sop.py\` | HA-ViD SOP 層：雙手 primitive-task 步驟序列、從標籤詞彙判斷 plate、從 train 學 precedence graph／必要步驟／時長界限、synthetic 順序／遺漏／時長違規表、原生 \`w\` 表；\`reproduce-lite\` 可從 CSV 重算 |`
2. In the CLI table row `| 線上 PSR | …`, append `、learn-havid-sop、havid-sop` inside the cell (after `check-psr-run`).
3. In "## 還沒有什麼", replace the bullet `- duration bounds 檔、synthetic 違規生成器與 synthetic 表、偏差偵測的 precision／recall（W3）。` with `- W3 剩餘：weighted／learned fusion、以 SOP step 粒度（說明書）取代 primitive task 的狀態機、偏差佇列；synthetic 表與原生 \`w\` 表已在 \`reports/havid_dev_v4_sop_synthetic\`。`

- [ ] **Step 4: Verify and commit**

Run:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && grep -c "{{" reports/README.md docs/decisions.md README.md; git add reports/README.md docs/decisions.md README.md && git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit -q -m "docs: index havid_dev_v3/v4, log the epoch-selection and SOP-learning decisions" && git log --oneline -4 && git status --short
```
Expected: all counts `0`; three new commits (Task 1, Task 2, this one); clean tree.

---

### Task 4: Wrap up (5 min)

- [ ] **Step 1:** `nvidia-smi --query-compute-apps=pid,process_name --format=csv` shows no python process; `git status --short` is empty.
- [ ] **Step 2: Final message (zh-TW)** in this order: commits with hashes; v3 headline numbers per hand (fusion_causal F1@10 / Edit, best fusion rule) next to v1; the v4 synthetic table (order / omission / duration: recall and false-alarm rate on `gt` and `pred`) and the native `w` line; two or three sentences of interpretation; anything skipped or failed; the follow-ups that remain (weighted or learned fusion; SOP-step granularity from the instruction sheets; W4 streaming layer needs the user's permission to download mediamtx; VLM needs the user's permission to download weights).

---

## Appendix A — v3 run README template

Save as `reports/havid_dev_v3_tas_f1sel_<hand>/README.md`; `<hand>` is `lh`/`rh`, `{{hand}}` is `left`/`right`. Replace every `{{…}}` from the Step 4 script or `tables.md`; keep the rest verbatim.

````markdown
# havid_dev_v3_tas_f1sel_{{lh|rh}} — epochs chosen by val F1@10, three parameter-free fusion rules, {{hand}} hand, primitive tasks (development result)

- Tier: **development result** on `splits/ha-vid/val.csv` (6 subjects: S01, S08, S10, S12, S18,
  S30; 18 recordings × 3 views); each network's epoch is selected on val; the frozen test subjects
  (`splits/ha-vid/test.csv`, 7 subjects) were never read. Not comparable with the paper's Table 3.
- Date: 2026-09-13. Author: kuotunyu. Machine: Windows 11, RTX 4090.
- Purpose: two fixes to [`../havid_dev_v1_tas_{{lh|rh}}/`](../havid_dev_v1_tas_{{lh|rh}}/README.md) —
  epoch selection by a segmental metric (v1's MoF selection favoured `null` and stopped causal
  networks early) and two more late-fusion rules that need no tuning (normalised geometric mean,
  confidence-weighted mean) next to the plain mean.

## 1. Data and features

As v1: HA-ViD primitive-task labels of the {{hand}} hand ({{n_classes minus 1}} classes in train),
train 143 recordings / 17 subjects / 181,015 frames per view, val 18 / 6 / 17,973; frozen DINOv2
ViT-B/14 1536-d features at stride 1 (`artifacts/features/ha-vid/dinov2_vitb14_s1`).

## 2. Model identity (from `config.json`)

| component | what exactly |
|---|---|
| architecture | MS-TCN++ per view as in v1 ({{parameters}} parameters), causal (left-only padding) and offline (symmetric padding) |
| selection (val only) | val **F1@10** every 5 epochs, best epoch kept per network: {{six networks: name best-epoch (first → last val F1@10)}} |
| `fusion_*` | arithmetic mean of the three per-view posteriors |
| `fusion_geo_*` | normalised geometric mean (product of experts: a view assigning ≈ 0 to a class vetoes it) |
| `fusion_conf_*` | per-frame weighted mean, each view weighted by its own maximum posterior |
| `majority` | constant most frequent training label (`{{majority label}}`) |

## 3. Result (val, spec 4.2 metrics, subject bootstrap 2,000 draws, seed 0)

`tables.md` is authoritative.

| run | temporal context | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|
{{the thirteen rows in the script's order; "—" for majority, "past only" for causal rows, "past and future" for offline rows}}

v1 (MoF selection, mean fusion) vs v3 (F1@10 selection), same features and split:

| run | MoF | Edit | F1@10 | F1@50 |
|---|---|---|---|---|
| v1 fusion_causal | {{from v1 tables.md}} | | | |
| v3 fusion_causal | {{from this run}} | | | |
| v1 best single causal view ({{view}}) | {{from v1}} | | | |
| v3 best single causal view ({{view}}) | {{from this run}} | | | |
| v1 fusion_offline | {{from v1}} | | | |
| v3 fusion_offline | {{from this run}} | | | |

Reading:

- {{effect of F1@10 selection on the causal numbers, with CI overlap}}
- {{best fusion rule among causal runs on F1@10 and Edit; differences vs the mean and vs the best single view; CI overlap}}
- {{best-epoch pattern for the causal networks}}
- All numbers are baselines on a 6-subject validation set, not claims.

## 4. Failure cases (`fusion_causal`)

- Most frequent confusions (gt → pred, frames): {{top five}}.
- Worst val recordings by MoF: {{three}}; best: {{three}}.

## 5. Cost

| step | wall time | resources |
|---|---|---|
| `train-havid-tas` (6 networks × 50 epochs, val every 5, bootstraps) | {{wall_seconds}} s | RTX 4090 |
| feature cache | reused from v1 (2742 s for 483 videos) | — |

## 6. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1
uv run sop-monitor train-havid-tas --features artifacts/features/ha-vid/dinov2_vitb14_s1 --hand {{lh|rh}} --level pt --selection-metric f1@10 --out reports/havid_dev_v3_tas_f1sel_{{lh|rh}} --device auto --epochs 50 --eval-every 5 --n-boot 2000 --seed 0
uv run sop-monitor score-predictions --run reports/havid_dev_v3_tas_f1sel_{{lh|rh}}
```

## 7. Not done here

- No weighted or learned fusion, no ASFormer, no atomic-action level, no online SOP metric.
- No number on the frozen test subjects.
````

## Appendix B — v4 SOP run README template

Save as `reports/havid_dev_v4_sop_synthetic/README.md`.

````markdown
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
  `steps_val.csv`; `w` segments as `wrong_val.csv`.
- Plate per recording read off the label vocabulary (`havid_sop.plate_of`); val plates: {{counts}}.
- Learned knowledge (`sop/ha-vid/`, from 143 train recordings): precedence edges holding in every
  co-occurring recording (min support 3), mandatory steps (≥ 90 % of a plate's recordings),
  duration bounds (5th–95th percentile, labels seen ≥ 5 times).

| plate | steps | edges | mandatory | bounded |
|---|---|---|---|---|
| cylinder | {{}} | {{}} | {{}} | {{}} |
| gear | {{}} | {{}} | {{}} | {{}} |
| general | {{}} | {{}} | {{}} | {{}} |

## 2. Checks and synthetic violations

- order: `check_order` — a step whose learned predecessors have not all been observed.
- omission: a mandatory step of the plate never observed.
- duration: a step outside its [5th, 95th]-percentile window (seconds at 15 fps).
- Synthetic violations, one per kind per recording (seed 0): move one step before a learned
  predecessor's first occurrence; delete one mandatory step that occurs once; stretch one bounded
  step to 1.5 × its upper bound. Recall = the perturbed step itself is flagged by the matching
  check; false-alarm rate = the unperturbed recording is flagged by that check.

## 3. Result (`tables.md` is authoritative)

{{paste the two synthetic tables (gt, pred) and the native w table from tables.md}}

Reading:

- {{order, gt vs pred}}
- {{omission, gt vs pred}}
- {{duration, gt vs pred}}
- Native `w`: {{detected}} of 12 ground-truth segments detected by a predicted `w` segment
  ({{recall}} {{interval}}); {{false predicted}} predicted `w` segments overlap no ground truth.
  Underpowered (ADR 0001 trigger B); no claim follows from it.
- Deviations are suggestions for human review; nothing here is a safety or compliance guarantee.

## 4. Cost

`havid-sop` runs in {{seconds, from the shell}} s on CPU; `reproduce-lite` recomputes it from the
CSVs and JSON files without the dataset.

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
- No number on the frozen test subjects.
````
