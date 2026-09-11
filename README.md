# sop-video-monitor

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **狀態：第一段流程已可重現，尚無正式研究成果。**
> 目前有：IndustReal 影片與標註的稽核與對齊、凍結的 participant-disjoint split、HA-ViD 三張
> task precedence graph 的匯出與順序／遺漏／時長檢查、離線與線上指標實作（含獨立參考實作交叉核對）、
> 以及一個在 IndustReal **validation split** 上跑通的簡單影片 baseline（frozen DINOv2 frame features +
> linear head，逐影格預測、MoF／Edit／F1@k、participant bootstrap CI、可從已 commit 的預測表重算）。
> 沒有：HA-ViD 影片或 HR-SAT 標註（request form 尚未到貨）、任何 frozen test split 上的正式數字、
> RTSP 串流、causal MS-TCN++／ASFormer、複核 UI、VLM。

sop-video-monitor 的目標是一套多攝影機 SOP 序列監控系統（CV research flagship，開發中）：HA-ViD 的
left／front／top 三路固定視角影片以 RTSP 重播進入系統，經 PyAV 解碼、共享 ring buffer、批次特徵抽取與
每路 causal 分段頭產生即時步驟後驗；SOP 狀態機依資料集附帶的 task precedence graph 檢查順序、時長與
遺漏，把偏差推入佇列，由人工複核 UI 接受或退回，Qwen3-VL-8B 只對被標記的片段提供非同步第二意見。
所有辨識指標在 subject-wise 凍結測試集上附 bootstrap CI；串流層附 backpressure 實測曲線。
以上是設計目標，不是現況；現況見下一節。

## 目前有什麼／還沒有什麼

### 已實作（都有單元測試）

| 模組 | 內容 |
|---|---|
| `metrics/offline.py` | MoF、Edit、F1@{10,25,50}（MS-TCN 參考實作的定義）、每序列充分統計量、以 test subject 為重抽單位的 bootstrap CI |
| `metrics/reference_mstcn.py` | 參考 `eval.py` 控制流程的獨立轉錄，只用來交叉核對；隨機輸入與真實預測表上都與 `offline.py` 一致 |
| `metrics/online.py` | Procedure Order Similarity（Damerau-Levenshtein）、completion F1、mean detection delay（IndustReal 定義；尚未在 IndustReal PSR 標註上做 sanity run，因為本機沒有 PSR 標註） |
| `splits.py`、`industreal.py` | test list SHA-256 契約；IndustReal 標註解析、participant-disjoint split 重建與凍結（`splits/industreal/`，從本機標註重跑結果與已 commit 檔案逐位元一致）；標註 → 逐影格標籤對齊（半開區間、重疊取最新 onset） |
| `sop_graph.py` | HR-SAT OWL 解析、JSON 匯出（`sop/ha-vid/{cylinder,gear,general}_plate.json`）、拓撲排序、順序／遺漏／未知步驟檢查、時長區間檢查 |
| `video.py` | PyAV probe 與循序解碼（含 swscale 縮放），影格索引從 0 起算 |
| `features.py` | 官方 `facebookresearch/dinov2` torch.hub 權重的 frozen frame embedding（CLS ‖ mean patch），fp16 快取，權重 SHA-256 記錄在 `meta.json` |
| `baseline.py` | linear head（full-batch 多項式邏輯迴歸）、causal／centered 平滑、預測表讀寫、從預測表重算指標並交叉核對 |
| `audit.py` | 實測每支影片的 codec／fps／影格數，標註與影格數交叉核對，HA-ViD 公開檔案盤點；產出 `data/manifest.json`（只含檔名、大小、SHA-256） |
| `stream/ring_buffer.py` | `WAIT`／`DROP_OLDEST`／`ADAPTIVE` 三種 backpressure 策略與 `skipped_frames` 計數（in-process 版本） |

CLI（`sop-monitor`）：`freeze-splits`、`check-sop`、`export-sop`、`audit-industreal`、`extract-features`、
`train-baseline`、`score-predictions`、`reproduce-lite`。CI 跑 `ruff`、`pytest` 與 `reproduce-lite`（不裝 torch）。

### 目前的資料與結果

- **IndustReal**（Apache-2.0，指標捐贈／開發資料）：本機有 86 支 1280×720、10 fps 的 RGB 影片與
  三個 action-recognition 標註 CSV（84 支有標註）；稽核結果在
  [`reports/industreal_dev_v1/data_audit.json`](reports/industreal_dev_v1/data_audit.json)。
  本機沒有 procedure-step-recognition（PSR）標註，所以線上指標還不能在 IndustReal 上做 sanity run。
- **HA-ViD**（CC BY-NC 4.0，主力）：本機只有三張 subject-agnostic task precedence graph（OWL）與七份
  組裝說明 PDF，**沒有影片、沒有 HR-SAT 標註**；主力資料線被 request form 阻擋。
- **開發結果**（不是 headline）：[`reports/industreal_dev_v1/`](reports/industreal_dev_v1/) 是
  IndustReal train → val 的第一個影片 baseline，包含真實逐影格預測表、`metrics.json`、`tables.md` 與
  報告。數字在 val 上量測，val 同時也是選 weight decay 與平滑視窗的依據；frozen test split 沒有被讀取。
  IndustReal 的數字不會進任何 HA-ViD 表格（設計規格 §3.3）。
- **正式研究成果**：無。

### 尚不存在

- HA-ViD 影片、HR-SAT parser、HA-ViD subject-wise split、特徵快取、任何 HA-ViD 指標。
- RTSP 重播（mediamtx）、decode thread、batch collector、watchdog、shared-memory ring buffer（W4）。
- causal MS-TCN++、ASFormer、late fusion、VideoMAE-V2 clip 特徵（W2–W3）。
- SOP 狀態機的 hysteresis 雙門檻、duration bounds 檔、synthetic 違規生成器（W3）。
- 複核 UI、deviation queue、VLM verifier（W5）。
- `MODEL_CARD.md`、`docs/claims_audit.md`、`docs/what_this_does_not_show.md`（W6）。
- 任何 frozen test split 上的量測數字。本 README 不列數字；已量測的數字只在 `reports/` 內。

## 非目標與 claim ceiling（設計規格 §2，逐字）

### 非目標

- 不做物件偵測主線，不以 bounding box 為主要輸出；HA-ViD 的 CVAT boxes 只作複核 UI 的輔助視覺化。
- 不接真實攝影機、不用 DeepStream、不做 edge 部署（v2 設計 §8：不採購 Jetson）。
- 不做多節點或分散式訓練；所有數字來自單張 RTX 4090。
- 不追 HA-ViD 官方 leaderboard：官方 test 為 123/609 videos 且未說明 subject 規則（memo §4），與本專案的 subject-wise split 不可比。

### 禁止宣稱

以下每一條在發佈前由 `docs/claims_audit.md` 逐條核對；左欄任何一句出現在 README 即視為 blocker。

| 禁止宣稱 | memo 證據 | README 允許的替代表述 |
|---|---|---|
| 真實工廠泛化 | 所有候選集皆為實驗室裝配台；ATTACH 由 person split 改 view split，平均準確率 57.4 → 30.4；unseen-view TAS 仍是 open problem（memo §4） | 「在 HA-ViD 固定三視角、held-out subjects 上」 |
| 安全或合規保證 | IMPACT 每個 baseline 的 recovery-phase F1 接近零；HoloAssist 最佳模態 F 40.19（memo §4） | 「偏差為建議，一律需人工複核」 |
| 未見錯誤型態的 robustness | IndustReal 僅 38 個 execution errors、14 個只在 val/test（memo §4）；HA-ViD `wrong` 段數未公布（memo §5） | 「僅涵蓋資料集已標註的錯誤型態與明示為 synthetic 的順序違規」 |
| VLM 可偵測錯誤 | zero-shot Qwen2.5-VL-7B 在 MD-VQA 協定 F1 0.0，GRPO 後 53.8／48.0 且需 4×H100；ZeProM 需 4×H100 跑 87.8 分鐘（memo §2） | 「VLM 為非同步第二意見，獨立列表，不進主指標」 |
| 商業可用 | HA-ViD CC BY-NC 4.0；IMPACT data CC BY-NC-SA 4.0（memo §1） | 「權重與快取特徵為 non-commercial 衍生物」 |
| 任意硬體 real-time | 唯一同儕審查的產線系統 I3D + ActionFormer 僅 0.53× real-time，且資料私有（memo §3） | 「單張 RTX 4090 上 N 路 × M fps，附曲線」 |
| 合成違規 = 真實違規 | memo §4 要求 synthetic 表獨立標示 | 兩表分開，表名與圖例含 synthetic |

## 資料集與授權

| 角色 | 資料集 | 授權 | 取得方式 | 本機狀態 | 衍生物處理 |
|---|---|---|---|---|---|
| 主力 | HA-ViD | CC BY-NC 4.0 | request form → Dropbox | 只有公開的 precedence graph 與說明 PDF | 權重與快取特徵為 non-commercial 衍生物，不進 public repo |
| 備援 | IMPACT | code Apache-2.0；data CC BY-NC-SA 4.0（以 repo 為準） | gated Hugging Face + Google Drive | 未取得 | 若成為主力，改標 CC BY-NC-SA 4.0 並註明 share-alike |
| 指標捐贈／開發 | IndustReal | Apache-2.0（code + data） | 4TU 開放下載 | RGB 影片 + action 標註 + 官方 action-recognition 權重（未使用） | 只用於指標實作交叉檢查與工具鏈開發，不混入主表 |

原始影片、標註、快取特徵、訓練權重一律不進 repo；詳見 [`data/README.md`](data/README.md) 與
[`docs/decisions/0001-dataset-and-licences.md`](docs/decisions/0001-dataset-and-licences.md)。
`data/manifest.json` 只記錄本機外部檔案的檔名、大小與 SHA-256。

## 開發

```bash
uv sync --all-extras                    # 基本安裝：指標、split、SOP graph、預測表重算（CI 用）
uv sync --all-extras --group baseline   # 加上 PyAV 與 torch 2.13 (cu130)：解碼、特徵抽取、baseline 訓練
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
make reproduce-lite   # 從已 commit 的 reports/*/predictions_*.csv 重算 metrics.json、檢查 tables.md，再跑單元測試
make reproduce        # audit → features → baseline → reproduce-lite；需要本機 IndustReal 與 baseline group
```

重現 `reports/industreal_dev_v1/` 的完整命令列在該目錄的
[`README.md`](reports/industreal_dev_v1/README.md)；三個層級（engineering smoke／development
result／formal research result）的定義在 [`reports/README.md`](reports/README.md)。

## 授權

程式碼採 Apache-2.0（見 [LICENSE](LICENSE)）。資料集與其衍生物（特徵、權重）依各自授權，不由本 LICENSE 重新授權。
