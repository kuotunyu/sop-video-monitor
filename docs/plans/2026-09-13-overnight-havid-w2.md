# Overnight HA-ViD W2 run (features → per-view MS-TCN++ → fusion → reports) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> 給人看的一句話（zh-TW）：這份文件是今晚無人值守的執行手冊。所有程式都已寫好並測過（HEAD 是訊息以 `docs: overnight runbook` 開頭的那個 commit）；
> 今晚只做四件事：抽 DINOv2 特徵、跑 DINOv2 的 HA-ViD 離線 TAS（左右手）、跑官方 I3D 特徵的對照組、寫報告並 commit。
> 沒有任何一步需要使用者決定；遇到失敗就停在那一步並把狀況寫進最後的總結訊息。

**Goal:** Produce the first HA-ViD development results (`reports/havid_dev_v1_tas_{lh,rh}` on DINOv2 ViT-B/14 features and `reports/havid_dev_v2_i3d_{lh,rh}` on the authors' I3D features), each recomputable by `sop-monitor reproduce-lite`, with reports, navigation-table rows, a decisions-log entry and one commit per stage.

**Architecture:** Everything runs through the already-committed CLI (`sop-monitor extract-features --dataset ha-vid`, `sop-monitor train-havid-tas`) wrapped by Makefile targets (`features-havid`, `havid-tas`, `havid-tas-i3d`). `train-havid-tas` trains a causal and a non-causal MS-TCN++ per camera view on the frozen `splits/ha-vid/train.csv` recordings, selects the epoch on `val.csv`, averages the three views' posteriors (late fusion) and writes `predictions_val.csv`, `metrics.json`, `config.json`, `tables.md`. The executor's job is to run the targets in order, verify each output against the expected counts below, write the run READMEs from the template in the Appendix, and commit.

**Tech Stack:** Windows 11, RTX 4090, `uv`-managed Python 3.12 env with the `baseline` group (torch 2.13 cu130, PyAV), Git Bash for shell commands, `make` (GNU make available in Git Bash), no network needed.

## Global Constraints

Copied from the user's standing rules and the repository conventions. Every task implicitly includes them.

- Work only inside `D:\AI-Portfolio\CC_github部隊\sop-video-monitor` on branch `main`, starting from the commit whose message begins with `docs: overnight runbook` (working tree clean).
- Always `export PYTHONUTF8=1` before any `uv run` (the path contains non-ASCII characters). Use the project env via `uv run …`; never `pip install`, never change `pyproject.toml`/`uv.lock`.
- Commit as `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` using `git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit …`. **No `Co-Authored-By` or any AI trailer in commit messages** (user rule; it overrides any harness default). Never `git push`, never create a remote or a GitHub repo, never upload data, weights or features anywhere.
- Never read, evaluate or embed `splits/ha-vid/test.csv` videos (126 test videos) or `splits/industreal/test.csv`. The Makefile targets already restrict themselves to train + val; do not add `--split test`.
- Never delete or overwrite raw data (`data/external/**`), earlier reports (`reports/industreal_dev_*`), or feature caches; never run `make audit` (it re-hashes 72 GB for no reason tonight).
- Do not download anything (no mediamtx, no VLM weights, no datasets). No paid APIs, no Colab.
- GPU etiquette: before every GPU command run `nvidia-smi` and proceed only if no other user process is on the GPU; never kill a process you did not start.
- Tiers stay separate: everything produced tonight is a **development result on `splits/ha-vid/val.csv`** (6 subjects). Never call it a formal or test result; never put IndustReal numbers into a HA-ViD table.
- Reports quote numbers only from the run's own `metrics.json` / `config.json` / `tables.md` (read them; never type numbers from memory or from this plan). If a number is not in those files, do not write it.
- The HA-ViD Dropbox link and password must not appear in any file, commit message or log.
- Communicate with the user in Traditional Chinese (zh-TW) with technical terms in English; code, reports and commit messages stay in English like the rest of the repository.
- If a step fails: retry once only if the failure is obviously transient (e.g. CUDA out of memory because another process was running). Otherwise stop the pipeline at that task, do not commit partial run directories, and describe the failure precisely in the final message (command, last 30 lines of output, which files exist).

---

## What already exists (do not rebuild)

| item | path | state |
|---|---|---|
| HA-ViD videos | `data/external/ha-vid/HAViD_rgb/assembly_dataset_mp4_blurred/s01..s30/*.mp4` | 3 222 mp4, h264 1280×720 15 fps; audited in `reports/havid_audit.json` |
| temporal annotations | `data/external/ha-vid/HAViD_temporalAnnotation.zip` | read in place by the CLI |
| official benchmark folder | `data/external/ha-vid/ActionSegmentation_data.zip` | mapping/splits/groundTruth + I3D features; read in place |
| frozen split | `splits/ha-vid/{train,val,test}.csv`, `test_sha256.txt`, `manifest.json` | train 143 recordings / 17 subjects (429 videos), val 18 / 6 (54 videos), test 41 / 7 (123 videos, never read) |
| official I3D cache | `artifacts/features/ha-vid/i3d_official/*.npz` (483 files, 4.6 GB) + `meta.json` | exported already; `frames = arange(5, T+5)`, `n_frames` = annotation length |
| code | `src/sop_monitor/havid.py`, `src/sop_monitor/havid_tas.py`, CLI commands in `src/sop_monitor/cli.py`, Makefile targets | tested (129 tests pass), 1-epoch CPU smoke on real I3D data completed |
| report conventions | `reports/README.md` (navigation table), `reports/industreal_dev_v2_mstcn/README.md` (report structure to mirror), `docs/decisions.md` (dated log) | see Appendix A for the template |

Expected sizes that the tasks verify against (all measured on 2026-09-12/13):

| quantity | value |
|---|---|
| train recordings / frames per view (annotation length) | 143 / 181 015 |
| val recordings / frames per view (annotation length) | 18 / 17 973 |
| DINOv2 npz files after Task 1 | 483 (= 429 train + 54 val), dim 1536, dtype float16 |
| `predictions_val.csv` rows, DINOv2 runs (Task 2) | 17 973 (stride 1, untrimmed) |
| `predictions_val.csv` rows, I3D runs (Task 3) | 17 793 (= 17 973 − 18 × 10, the official 5-frame trim at both ends) |
| I3D train frames (config.json `train_frames`) | 179 585 |
| prediction columns per run | `majority, view0_causal, view0_offline, view1_causal, view1_offline, view2_causal, view2_offline, fusion_causal, fusion_offline` |
| classes seen in train, lh_pt (config.json `classes.n_classes`, includes the unused background index 0) | 64 (rh_pt: read it from config.json, not known yet) |

---

### Task 0: Post-reboot sanity check (5 min, no GPU)

**Files:**
- Read only: `splits/ha-vid/test_sha256.txt`, `artifacts/features/ha-vid/i3d_official/meta.json`

**Interfaces:**
- Consumes: the repository at the `docs: overnight runbook…` commit (clean tree).
- Produces: nothing on disk; a go/no-go for Task 1.

- [ ] **Step 1: Confirm the repository state**

Run (Git Bash):
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && git status --short && git log --oneline -1
```
Expected: no output from `git status --short`; the log line's message begins with `docs: overnight runbook`. If the tree is dirty or the commit differs, stop and report (someone changed the repo since the plan was written).

- [ ] **Step 2: Confirm the environment, the split hash and the data**

Run:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && export PYTHONUTF8=1 && uv run python -c "import torch, av; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" && uv run sop-monitor verify-splits --directory splits/ha-vid && ls artifacts/features/ha-vid/i3d_official | wc -l && find data/external/ha-vid/HAViD_rgb -name "*.mp4" | wc -l && df -h /d | tail -1
```
Expected: a line like `2.13.0+cu130 True NVIDIA GeForce RTX 4090`; `OK: test list hash matches test_sha256.txt`; `484` (483 npz + meta.json); `3222`; at least 100 GB free on D:.

- [ ] **Step 3: Confirm the GPU is free**

Run:
```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```
Expected: the second command lists no processes (header only). If another process is using the GPU, wait and re-check every 10 minutes for up to one hour, then stop and report; never kill it.

- [ ] **Step 4: Run the unit tests once (26 s)**

Run:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && export PYTHONUTF8=1 && uv run pytest -q 2>&1 | tail -2
```
Expected: `129 passed` (one warning about a tensor conversion is normal).

- [ ] **Step 5: Keep the machine awake (optional)**

If the tool `mcp__ccd_host__request_keep_awake` is available in the session, load it with ToolSearch (`select:mcp__ccd_host__request_keep_awake`) and call it once. If it is not available, continue; the user set Windows sleep to "never" before leaving.

---

### Task 1: DINOv2 ViT-B/14 frame features for train + val (60–90 min, GPU)

**Files:**
- Create (ignored by git): `artifacts/features/ha-vid/dinov2_vitb14_s1/<video_id>.npz` × 483, `meta.json`, `extraction_log.json`

**Interfaces:**
- Consumes: `splits/ha-vid/{train,val}.csv` (column `video_id`), the mp4s under `data/external/ha-vid/HAViD_rgb/`.
- Produces: the npz cache consumed by Task 2 through `--features artifacts/features/ha-vid/dinov2_vitb14_s1`. Each npz holds `frames` (int64, `arange(n_frames)`), `features` (float16, `(n_frames, 1536)`), `n_frames` (int64).

- [ ] **Step 1: Start the extraction in the background**

Run with `run_in_background: true` and a log file (the command prints one line per video):
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && export PYTHONUTF8=1 && make features-havid > /tmp/features_havid.log 2>&1; echo "exit=$?" >> /tmp/features_havid.log
```
(`make features-havid` expands to `uv run sop-monitor extract-features --dataset ha-vid --split-dir splits/ha-vid --rgb-dir data/external/ha-vid/HAViD_rgb --out artifacts/features/ha-vid --model dinov2_vitb14 --stride 1 --batch-size 128`. The first run also downloads nothing: the DINOv2 weights are already in the torch.hub cache from the IndustReal runs.)

- [ ] **Step 2: Check progress after 10 minutes, then wait for the completion notification**

Run:
```bash
ls "/d/AI-Portfolio/CC_github部隊/sop-video-monitor/artifacts/features/ha-vid/dinov2_vitb14_s1" | wc -l && tail -3 /tmp/features_havid.log
```
Expected after 10 min: roughly 40–80 npz files and per-video lines like `S01A04I01M0: 1605/1605 frames in 9.8s (164 fps)`. If after 20 minutes the count has not increased, read the log and stop if it shows an error. Otherwise wait for the background task to finish (it sends a notification; do not poll more often than every 15 minutes).

- [ ] **Step 3: Verify the cache**

Run:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && export PYTHONUTF8=1 && tail -2 /tmp/features_havid.log && uv run python - <<'EOF'
import json
from pathlib import Path
import numpy as np
from sop_monitor.havid import load_temporal_zip, n_frames_of
from sop_monitor.havid_tas import read_split
cache = Path("artifacts/features/ha-vid/dinov2_vitb14_s1")
meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
print("model", meta["model"], "dim", meta["dim"], "dtype", meta["dtype"], "stride", meta["stride"])
temporal, _ = load_temporal_zip(Path("data/external/ha-vid/HAViD_temporalAnnotation.zip"))
rows = [r for s in ("train", "val") for r in read_split(Path(f"splits/ha-vid/{s}.csv"))]
bad = 0
for row in rows:
    with np.load(cache / f"{row['video_id']}.npz") as p:
        n = int(p["n_frames"]); frames = p["frames"]; feats = p["features"]
    expected = n_frames_of(temporal["view0_lh_pt"][row["recording"]])
    if n != expected or len(frames) != n or feats.shape != (n, 1536) or feats.dtype != np.float16:
        bad += 1; print("BAD", row["video_id"], n, expected, feats.shape, feats.dtype)
print("checked", len(rows), "videos; bad", bad)
log = json.loads((cache / "extraction_log.json").read_text(encoding="utf-8"))
print("extraction seconds", round(log["total_seconds"]), "frames", log["total_sampled"])
EOF
```
Expected: `model dinov2_vitb14 dim 1536 dtype float16 stride 1`, `checked 483 videos; bad 0`, and `frames` equal to 181015 + 17973 = 198988 (if the log shows fewer, some videos were already cached before — that is fine as long as `bad 0`). Write down `extraction seconds` for the reports (it goes into section 5 of both v1 READMEs).

If `bad` is not 0, stop and report the BAD lines; do not start Task 2.

---

### Task 2: `havid_dev_v1_tas_lh` and `havid_dev_v1_tas_rh` on DINOv2 features (30–45 min GPU + 20 min writing)

**Files:**
- Create: `reports/havid_dev_v1_tas_lh/{predictions_val.csv,metrics.json,config.json,tables.md,README.md}` and the same under `reports/havid_dev_v1_tas_rh/`

**Interfaces:**
- Consumes: Task 1's cache; `splits/ha-vid/`; the two archives under `data/external/ha-vid/` (defaults of `train-havid-tas`).
- Produces: two run directories that `sop-monitor reproduce-lite` recomputes; their numbers feed Task 3's comparison table and Task 4's navigation rows.

- [ ] **Step 1: Confirm the GPU is still free, then start both hands in the background**

Run `nvidia-smi --query-compute-apps=pid,process_name --format=csv` (expect header only), then run with `run_in_background: true`:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && export PYTHONUTF8=1 && make havid-tas > /tmp/havid_tas_v1.log 2>&1; echo "exit=$?" >> /tmp/havid_tas_v1.log
```
(`make havid-tas` runs `uv run sop-monitor train-havid-tas --features artifacts/features/ha-vid/dinov2_vitb14_s1 --hand lh --out reports/havid_dev_v1_tas_lh` and then the same with `--hand rh --out reports/havid_dev_v1_tas_rh`; defaults: `--level pt --epochs 50 --eval-every 5 --n-boot 2000 --seed 0 --device auto`.) Each hand trains 6 MS-TCN++ models (3 views × causal/offline); on the 4090 expect 2–4 minutes per model, so 15–25 minutes per hand. If the log has not printed the `wall:` line for the first hand after 60 minutes, check `nvidia-smi`; if the process is alive and the GPU busy, keep waiting; otherwise stop and report.

- [ ] **Step 2: Verify both run directories**

Run:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && export PYTHONUTF8=1 && tail -1 /tmp/havid_tas_v1.log && for h in lh rh; do d=reports/havid_dev_v1_tas_$h; echo "== $d"; ls $d; tail -n +2 $d/predictions_val.csv | wc -l; head -1 $d/predictions_val.csv; uv run sop-monitor score-predictions --run $d; done
```
Expected per hand: the four files `config.json metrics.json predictions_val.csv tables.md`; `17973` rows; the header `video_id,participant,frame,gt,pred_majority,pred_view0_causal,pred_view0_offline,pred_view1_causal,pred_view1_offline,pred_view2_causal,pred_view2_offline,pred_fusion_causal,pred_fusion_offline`; `score-predictions` ends with a line containing `metrics reproduce`. The log's last line must be `exit=0`.

- [ ] **Step 3: Print the numbers each README needs**

Run once per hand (replace `lh` by `rh` the second time) and keep the output; every number in the README comes from here or from `tables.md`:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && export PYTHONUTF8=1 && uv run python - reports/havid_dev_v1_tas_lh <<'EOF'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
c = json.loads((run / "config.json").read_text(encoding="utf-8"))
print("hand", c["spec"]["hand"], "level", c["spec"]["level"], "views", c["spec"]["views"])
print("train videos/frames", c["train_videos"], c["train_frames"], "eval videos/frames", c["eval_videos"], c["eval_frames"])
print("n_classes (incl. unused background 0)", c["classes"]["n_classes"], "majority class id", c["classes"]["majority_class_id"], "=", c["classes"]["class_id_to_label"][str(c["classes"]["majority_class_id"])])
print("features", c["features"].get("model"), c["features"].get("dim"), "weights sha256", (c["features"].get("weights") or {}).get("sha256"))
print("device", c["device"], "wall_seconds", round(c["wall_seconds"]))
for name, log in c["training"].items():
    curve = log["curve"]
    print(f"{name}: best epoch {log['best_epoch']} val MoF {log['best_val_mof']:.1f}; first {curve[0]['epoch']}:{curve[0]['mof']:.1f} last {curve[-1]['epoch']}:{curve[-1]['mof']:.1f}; params {log['parameters']}; {log['train_seconds']:.0f}s")
def fmt(e): return f"{e['point']:.1f} [{e['ci95'][0]:.1f}, {e['ci95'][1]:.1f}]"
for run_name in ["majority", "view0_causal", "view1_causal", "view2_causal", "fusion_causal", "view0_offline", "view1_offline", "view2_offline", "fusion_offline"]:
    r = m["runs"][run_name]
    print(f"| {run_name} | {fmt(r['mof'])} | {fmt(r['edit'])} | {fmt(r['f1@10'])} | {fmt(r['f1@25'])} | {fmt(r['f1@50'])} |")
labels = c["classes"]["class_id_to_label"]
print("top confusions (gt -> pred, frames) for fusion_causal:")
for conf in m["confusions"][:8]:
    print("  ", labels[str(conf["gt"])], "->", labels[str(conf["pred"])], conf["frames"])
pv = m["per_video"]["fusion_causal"]
worst = sorted(pv.items(), key=lambda kv: kv[1]["mof"])[:3]
best = sorted(pv.items(), key=lambda kv: -kv[1]["mof"])[:3]
print("worst videos by fusion_causal MoF:", [(v, round(s["mof"], 1)) for v, s in worst])
print("best videos by fusion_causal MoF:", [(v, round(s["mof"], 1)) for v, s in best])
EOF
```
Expected: it prints without error; `train videos/frames 143 181015`, `eval videos/frames 18 17973`, `features dinov2_vitb14 1536`.

- [ ] **Step 4: Write `reports/havid_dev_v1_tas_lh/README.md` and `reports/havid_dev_v1_tas_rh/README.md`**

Use the template in Appendix A verbatim, replacing every `{{…}}` token with the value the token names (all come from Step 3's output or `tables.md`). Write the "Reading" bullets yourself from the numbers: state whether fusion beats the best single view (compare `fusion_causal` with the best `view*_causal` on F1@10 and Edit, and say whether the difference is inside the overlapping CIs), which view is strongest, and the causal-vs-offline gap for fusion. Do not speculate beyond what the numbers show. Keep the tier sentence and the "Not done here" section exactly as in the template.

- [ ] **Step 5: Verify, then commit the two runs**

Run:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && export PYTHONUTF8=1 && uv run sop-monitor reproduce-lite > /tmp/reproduce_v1.log 2>&1 && tail -4 /tmp/reproduce_v1.log && git add reports/havid_dev_v1_tas_lh reports/havid_dev_v1_tas_rh && git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit -q -m "docs: report the first HA-ViD offline TAS runs on DINOv2 ViT-B/14 (havid_dev_v1, per-view MS-TCN++ + late fusion, val only)" && git log --oneline -1
```
Expected: the chain reaches the commit only if `reproduce-lite` exited 0 (it exits 1 on any mismatch, which stops the `&&` chain before `git add`); `/tmp/reproduce_v1.log` contains `havid_dev_v1_tas_lh: OK` and `havid_dev_v1_tas_rh: OK`; the commit line appears. If the chain stops, read the log, fix nothing by hand in `metrics.json`/`tables.md` (they are generated), and report.

---

### Task 3: `havid_dev_v2_i3d_lh` and `havid_dev_v2_i3d_rh` on the official I3D features (30–45 min GPU + 20 min writing)

**Files:**
- Create: `reports/havid_dev_v2_i3d_lh/{predictions_val.csv,metrics.json,config.json,tables.md,README.md}` and the same under `reports/havid_dev_v2_i3d_rh/`

**Interfaces:**
- Consumes: `artifacts/features/ha-vid/i3d_official/` (already exported), `splits/ha-vid/`, the archives; Task 2's `tables.md` for the comparison table.
- Produces: two run directories; the DINOv2-vs-I3D comparison used in Task 4.

- [ ] **Step 1: Confirm the GPU is free, then start both hands in the background**

Run `nvidia-smi --query-compute-apps=pid,process_name --format=csv` (expect header only), then run with `run_in_background: true`:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && export PYTHONUTF8=1 && make havid-tas-i3d > /tmp/havid_tas_v2.log 2>&1; echo "exit=$?" >> /tmp/havid_tas_v2.log
```
(`make havid-tas-i3d` runs `train-havid-tas --features artifacts/features/ha-vid/i3d_official --hand lh --out reports/havid_dev_v2_i3d_lh` then `--hand rh --out reports/havid_dev_v2_i3d_rh`.) Same waiting rules as Task 2 Step 1.

- [ ] **Step 2: Verify both run directories**

Run:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && export PYTHONUTF8=1 && tail -1 /tmp/havid_tas_v2.log && for h in lh rh; do d=reports/havid_dev_v2_i3d_$h; echo "== $d"; ls $d; tail -n +2 $d/predictions_val.csv | wc -l; uv run sop-monitor score-predictions --run $d; done
```
Expected per hand: the four files; `17793` rows (the I3D features start at annotation frame 5 and end 5 frames early); `metrics reproduce`; log ends with `exit=0`.

- [ ] **Step 3: Print the numbers each README needs**

Run the Step 3 script of Task 2 with `reports/havid_dev_v2_i3d_lh` and then `reports/havid_dev_v2_i3d_rh` as the argument. Expected: `train videos/frames 143 179585`, `eval videos/frames 18 17793`, `features i3d_official 2048`, weights sha256 `None` (the authors' features carry no weight digest).

- [ ] **Step 4: Write the two READMEs**

Use Appendix A again with these substitutions in section 1: the feature sentence becomes "the authors' I3D features from `ActionSegmentation/data/features` (2048-d, one vector per frame, float64 stored as float32, no weight digest available), exported by `sop-monitor export-official-features`; they cover annotation frames 5 … n−6, so the tables have 17 793 val frames instead of 17 973". In section 3 add, after the main table, a second table titled "DINOv2 (v1) vs I3D (v2), same protocol, val" with rows `fusion_causal`, `fusion_offline`, best single causal view, and columns MoF / Edit / F1@10 / F1@50, copying the v1 values from `reports/havid_dev_v1_tas_<hand>/tables.md` and the v2 values from this run's `tables.md`. In "Reading", state which feature set is better per metric and whether the CIs overlap. The cost section's feature line becomes "official features, no extraction".

- [ ] **Step 5: Verify, then commit**

Run:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && export PYTHONUTF8=1 && uv run sop-monitor reproduce-lite > /tmp/reproduce_v2.log 2>&1 && tail -6 /tmp/reproduce_v2.log && git add reports/havid_dev_v2_i3d_lh reports/havid_dev_v2_i3d_rh && git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit -q -m "docs: report the I3D control runs (havid_dev_v2): same protocol as v1 on the authors' features" && git log --oneline -1
```
Expected: the chain reaches the commit only if `reproduce-lite` exited 0; `/tmp/reproduce_v2.log` shows all four HA-ViD runs `OK`; the commit line is printed.

---

### Task 4: Navigation table, decisions log, root README (20 min, no GPU)

**Files:**
- Modify: `reports/README.md` (after the "Which run answers which question" table), `docs/decisions.md` (append), `README.md` (status banner + "還沒有什麼" list + "結果" paragraph)

**Interfaces:**
- Consumes: the four `tables.md` files and READMEs from Tasks 2–3.
- Produces: the committed navigation for the morning review.

- [ ] **Step 1: Add a HA-ViD section to `reports/README.md`**

Insert the following block immediately after the IndustReal run table (after the line that starts with `| \`industreal_dev_v11_psr_epochsel_f1/\` |`) and before the paragraph that starts with "Numbers are means over seeds". Replace each `{{…}}` with the value read from the named file; the verdict column quotes `fusion_causal` F1@10 and Edit points (one decimal) and says in five words or fewer whether fusion helped.

```markdown

## HA-ViD runs (development results on `splits/ha-vid/val.csv`, 6 subjects; test never read)

| run | question | verdict | status |
|---|---|---|---|
| `havid_dev_v1_tas_lh/` | First HA-ViD offline TAS: per-view causal/offline MS-TCN++ + late fusion on DINOv2 ViT-B/14, primitive tasks, left hand | fusion_causal F1@10 {{v1 lh tables.md fusion_causal F1@10 point}} / Edit {{v1 lh Edit point}}; {{fusion vs best view, ≤5 words}} | first HA-ViD baseline |
| `havid_dev_v1_tas_rh/` | Same, right hand | fusion_causal F1@10 {{v1 rh F1@10}} / Edit {{v1 rh Edit}}; {{≤5 words}} | first HA-ViD baseline |
| `havid_dev_v2_i3d_lh/` | Control: same protocol on the authors' I3D features, left hand | fusion_causal F1@10 {{v2 lh F1@10}} / Edit {{v2 lh Edit}}; {{DINOv2 vs I3D, ≤5 words}} | control (feature choice) |
| `havid_dev_v2_i3d_rh/` | Same, right hand | fusion_causal F1@10 {{v2 rh F1@10}} / Edit {{v2 rh Edit}}; {{≤5 words}} | control (feature choice) |

For scale only (official test subjects, I3D, single view, no fusion; not our split): the HA-ViD
paper's Table 3 reports MS-TCN primitive-task F1@10 36.6 (left hand) / 34.7 (right hand) averaged
over the three views.
```

Also change the first paragraph's last sentence from "There is no formal (frozen test split) result and no HA-ViD model result yet; the HA-ViD delivery itself is audited in [`havid_audit.md`](havid_audit.md) (W1)." to "There is no formal (frozen test split) result. The HA-ViD delivery is audited in [`havid_audit.md`](havid_audit.md) (W1) and the first HA-ViD development runs are listed in their own table below."

- [ ] **Step 2: Append the decisions-log entry**

Append to `docs/decisions.md` (after the last entry), filling the tokens from the four `tables.md` files and stating only what they show:

```markdown

## 2026-09-13 — first HA-ViD baselines: per-view MS-TCN++ with late fusion, DINOv2 vs I3D

Protocol: frozen `splits/ha-vid` (train 17 subjects → val 6 subjects), primitive-task labels of
one hand per run, class ids from the official `mapping.txt`, one MS-TCN++ per camera view
(causal and non-causal), epoch selected on val, late fusion = mean of the per-view posteriors.
Fusion vs best single causal view (F1@10, left / right hand): {{v1 lh fusion F1@10}} vs
{{v1 lh best view F1@10}} / {{v1 rh fusion}} vs {{v1 rh best view}}. DINOv2 ViT-B/14 vs the
authors' I3D features, fusion_causal F1@10 (left / right): {{v1 lh}} vs {{v2 lh}} /
{{v1 rh}} vs {{v2 rh}}. Decision: {{one sentence: which feature set is the default for the
HA-ViD line, or "no default yet" if the CIs overlap on both hands}}. The causal-vs-offline gap
for fusion is {{v1 lh offline F1@10 minus causal}} / {{v1 rh}} F1@10 points. All of this is a
development result on val; the frozen test subjects were not read.
```

- [ ] **Step 3: Update the root `README.md`**

Make exactly these three edits:

1. In the status banner (line starting with `> **狀態：`), replace `模型線尚未開始` with `第一批 val 上的離線 TAS 基線已出（`reports/havid_dev_v1_tas_*`、`v2_i3d_*`）`.
2. In the "### 結果" paragraph, replace `所有結果都是 IndustReal validation split 上的**開發結果**；` with `所有結果都是 IndustReal 或 HA-ViD validation split 上的**開發結果**；`.
3. In "## 還沒有什麼", replace the bullet `- HA-ViD 的特徵抽取、任何模型與指標（資料與 split 已就緒，尚未開跑）。` with `- HA-ViD 的線上 PSR 式指標（completion 定義未定）、synthetic 違規表、aa 層與 ASFormer；目前只有 val 上的離線 TAS 基線。`

- [ ] **Step 4: Verify the markdown renders sanely and commit**

Run:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && grep -c "{{" reports/README.md docs/decisions.md README.md reports/havid_dev_v1_tas_lh/README.md reports/havid_dev_v1_tas_rh/README.md reports/havid_dev_v2_i3d_lh/README.md reports/havid_dev_v2_i3d_rh/README.md
```
Expected: every count is `0` (no unreplaced token). Then:
```bash
cd "/d/AI-Portfolio/CC_github部隊/sop-video-monitor" && git add reports/README.md docs/decisions.md README.md && git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit -q -m "docs: index the HA-ViD v1/v2 runs, log the feature decision, update the status" && git log --oneline -4 && git status --short
```
Expected: `git log` shows this commit on top of the Task 3 commit and the Task 2 commit (three commits after the plan commit); `git status --short` prints nothing.

---

### Task 5: Release the GPU and write the morning summary (5 min)

**Files:** none.

- [ ] **Step 1: Confirm nothing is left running**

Run `nvidia-smi --query-compute-apps=pid,process_name --format=csv` — expect header only — and `git status --short` — expect no output.

- [ ] **Step 2: Final message to the user (zh-TW)**

Write one message that stands alone, in this order: (1) what was produced (the four run directories and the three commits, with hashes); (2) the headline numbers per hand as a small table (fusion_causal and best single causal view: F1@10, Edit, MoF, from `tables.md`; DINOv2 and I3D side by side); (3) what the numbers say in three sentences at most (fusion vs single view; DINOv2 vs I3D; causal cost), with CI overlap stated; (4) wall time per stage (feature extraction seconds from Task 1 Step 3; per-hand `wall_seconds` from `config.json`); (5) anything that failed or was skipped, precisely; (6) the two decisions the user still owns: whether to reproduce the paper's official-test numbers once for metric calibration (touches the frozen test subjects), and the step granularity for the SOP state machine (primitive task vs instruction-sheet step).

---

## Appendix A — run README template

Save as `reports/<run>/README.md`. Replace every `{{…}}` token with the value it names (from the Step 3 script output or `tables.md` of that run); keep everything else verbatim. `<run>` is one of `havid_dev_v1_tas_lh`, `havid_dev_v1_tas_rh`, `havid_dev_v2_i3d_lh`, `havid_dev_v2_i3d_rh`; `<hand>` is `left` or `right`; for v2 apply the section-1 / section-3 / section-5 substitutions listed in Task 3 Step 4.

````markdown
# {{run}} — per-view causal/offline MS-TCN++ and late fusion, {{hand}} hand, primitive tasks (development result)

- Tier: **development result** on `splits/ha-vid/val.csv` (6 subjects: S01, S08, S10, S12, S18,
  S30; 18 recordings × 3 views); the epoch is selected on val; the frozen test subjects
  (`splits/ha-vid/test.csv`, 7 subjects) were never read. Not comparable with the paper's Table 3,
  which uses the official test subjects, a single view and no fusion.
- Date: 2026-09-13. Author: kuotunyu. Machine: Windows 11, RTX 4090.
- Purpose: the first HA-ViD numbers of this repository — does a frozen-feature temporal head work
  on HA-ViD primitive tasks at all, which camera view carries the most information, and does
  averaging the three views' posteriors (late fusion) beat the best single view?

## 1. Data and features

HA-ViD (CC BY-NC 4.0), audited in [`../havid_audit.md`](../havid_audit.md): temporal annotations
are inclusive, contiguous frame ranges at 15 fps; labels are the {{level}}-level HR-SAT codes of
the {{hand}} hand ({{n_classes minus 1}} classes occur in train; the official `mapping.txt` lists 75);
`null` (pause) is an ordinary class as in the paper; `w` (wrong) is one of the classes.
Train: 143 recordings / 17 subjects / {{train_frames}} frames per view. Val: 18 recordings / 6
subjects / {{eval_frames}} frames per view. Features: frozen DINOv2 ViT-B/14 `[CLS ; mean patch]`
1536-d per frame at stride 1 (`artifacts/features/ha-vid/dinov2_vitb14_s1`, weights SHA-256
`{{weights sha256}}`), one cache per camera video.

## 2. Model identity (from `config.json`)

| component | what exactly |
|---|---|
| architecture | MS-TCN++ (Li et al., TPAMI 2020): prediction-generation stage of 11 dual-dilated layers, 3 refinement stages of 10 dilated residual layers, 64 feature maps, dropout 0.5; {{parameters}} parameters; input = standardised features of one view |
| `view{v}_causal` | one network per view (0 = side `M0`, 1 = front `S1`, 2 = top `S2`); every kernel-3 convolution left-padded, never right-padded → frame *t* sees frames ≤ *t* only |
| `view{v}_offline` | same networks with symmetric padding → see future frames; **not** an online result |
| `fusion_causal` / `fusion_offline` | arithmetic mean of the three per-view posteriors, then argmax; no extra parameters |
| loss / optimiser | cross-entropy on every stage + 0.15 × truncated MSE smoothing (clamp 16); Adam lr 5e-4, one video per step, 50 epochs, seed 0, `cudnn.deterministic` |
| selection (val only) | val MoF every 5 epochs, best epoch kept per network: {{for each of the six networks: name best-epoch (first→last val MoF)}} |
| `majority` | constant most frequent training label (`{{majority label}}`) |

## 3. Result (val, spec 4.2 metrics, subject bootstrap 2,000 draws, seed 0)

`tables.md` is authoritative.

| run | temporal context | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|
{{the nine rows printed by the Step 3 script, in that order (majority, three causal views, fusion_causal, three offline views, fusion_offline); add the "temporal context" cell: "—" for majority, "past only" for causal rows, "past and future" for offline rows}}

Reading:

- {{fusion vs best single causal view: which is better on F1@10 and Edit, by how much, whether the CIs overlap}}
- {{which view is strongest and which weakest on F1@10 (causal), with the numbers}}
- {{causal-vs-offline gap for fusion: MoF / Edit / F1@50 differences}}
- All numbers are baselines for the pipeline on a 6-subject validation set, not claims.

## 4. Failure cases (`fusion_causal`, from `metrics.json["confusions"]` and `per_video`)

- Most frequent confusions (gt → pred, frames): {{top five from the Step 3 script, as label codes}}.
- Worst val recordings by MoF: {{three worst with values}}; best: {{three best with values}}.

## 5. Cost

| step | wall time | resources |
|---|---|---|
| `train-havid-tas` (6 networks × 50 epochs, val every 5, bootstraps) | {{wall_seconds}} s ({{per-network train seconds, six values}}) | RTX 4090 |
| feature cache (Task 1, shared by both hands) | {{extraction seconds from Task 1 Step 3}} s for 483 videos | RTX 4090 |

## 6. Reproduce

```bash
uv sync --all-extras --group baseline
export PYTHONUTF8=1
make features-havid          # once; DINOv2 ViT-B/14 cache for the train + val videos
uv run sop-monitor train-havid-tas --features artifacts/features/ha-vid/dinov2_vitb14_s1 --hand {{lh|rh}} --level pt --out reports/{{run}} --device auto --epochs 50 --eval-every 5 --n-boot 2000 --seed 0
uv run sop-monitor score-predictions --run reports/{{run}}
```

## 7. Not done here

- Only one hand per run and only primitive tasks; no atomic-action (219-class) level, no ASFormer, no
  hyper-parameter search beyond the epoch, no early or mid-level fusion.
- No online SOP metrics (completion definition for HA-ViD steps is still open), no `wrong`-label
  detection table, no synthetic violations.
- No number on the frozen test subjects.
````

## Appendix B — timeline the user was given

| stage | GPU | estimate |
|---|---|---|
| Task 0 sanity | no | 5 min |
| Task 1 DINOv2 features (483 videos) | yes | 60–90 min |
| Task 2 v1 both hands + READMEs + commit | yes | 30–45 min + 20 min |
| Task 3 v2 both hands + READMEs + commit | yes | 30–45 min + 20 min |
| Task 4 navigation / decisions / README + commit | no | 20 min |
| Task 5 summary | no | 5 min |

Total ≈ 4–5 h; GPU busy ≈ 2–3 h.
