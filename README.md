# sop-video-monitor

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **狀態：W0 skeleton, no results yet.** 目前只有目錄骨架、純 Python 的指標與 ring buffer 實作及其單元測試。沒有模型、沒有資料、沒有任何量測數字。

sop-video-monitor 是一套多攝影機 SOP 序列監控系統（CV research flagship，開發中）。HA-ViD 的 left／front／top 三路固定視角影片以 RTSP 重播進入系統，經 PyAV 解碼、共享 ring buffer、批次特徵抽取與每路 causal 分段頭產生即時步驟後驗；SOP 狀態機依資料集附帶的 task precedence graph 檢查順序、時長與遺漏，把偏差推入佇列，由人工複核 UI 接受或退回，Qwen3-VL-8B 只對被標記的片段提供非同步第二意見。所有辨識指標在 subject-wise 凍結測試集上附 bootstrap CI；串流層附 backpressure 實測曲線。

## 目前有什麼／還沒有什麼

### 已存在（W0）

- `src/sop_monitor/metrics/offline.py`：MoF、Edit、F1@{10,25,50}（MS-TCN 參考實作的定義）與以 test subject 為重抽單位的 bootstrap CI；純 Python，附 hand-computed 單元測試。
- `src/sop_monitor/metrics/online.py`：Procedure Order Similarity（Damerau-Levenshtein）、completion F1、mean detection delay（IndustReal 定義的轉錄；W3 會在 IndustReal 上做一次實作 sanity check）。
- `src/sop_monitor/stream/ring_buffer.py`：`WAIT`／`DROP_OLDEST`／`ADAPTIVE` 三種 backpressure 策略與 `skipped_frames` 計數器（in-process 版本；shared-memory slot 版本是 W4）。
- `src/sop_monitor/splits.py`：test list 的 SHA-256 hash 契約（排序、`\n` 串接、UTF-8）。
- `src/sop_monitor/sop_graph.py`：precedence graph 與三檢查的型別介面（stub，`NotImplementedError`）。
- `sop-monitor` CLI：`freeze-splits`、`check-sop`、`reproduce-lite` 三個佔位指令，各自標明所屬週次。
- CI：`ruff check`、`ruff format --check`、`pytest`（Python 3.12）。

### 尚不存在

- 任何資料集檔案、`data/manifest.json`、`splits/*.csv`、`sop/graph.json`、特徵快取、模型權重。
- RTSP 重播（mediamtx）、PyAV 解碼、batch collector、watchdog（W4）。
- causal MS-TCN++、ASFormer、late fusion（W2–W3）。
- SOP 狀態機三檢查、duration bounds、synthetic 違規生成器（W3）。
- 複核 UI、deviation queue、VLM verifier（W5）。
- `reports/` 下任何 JSON／表格／圖；`MODEL_CARD.md`、`docs/claims_audit.md`、`docs/what_this_does_not_show.md`（W6）。
- 任何量測數字。本 README 出現的數字全部引用自資料集論文或研究 memo，並非本專案的量測值。

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

| 角色 | 資料集 | 授權 | 取得方式 | 衍生物處理 |
|---|---|---|---|---|
| 主力 | HA-ViD | CC BY-NC 4.0 | request form → Dropbox | 權重與快取特徵為 non-commercial 衍生物，不進 public repo |
| 備援 | IMPACT | code Apache-2.0；data CC BY-NC-SA 4.0（以 repo 為準） | gated Hugging Face + Google Drive | 若成為主力，改標 CC BY-NC-SA 4.0 並註明 share-alike |
| 指標捐贈 | IndustReal | Apache-2.0（code + data） | 4TU 開放下載 | 只用於指標實作交叉檢查，不混入主表 |

原始影片、標註、快取特徵、訓練權重一律不進 repo；詳見 [`data/README.md`](data/README.md) 與 [`docs/decisions/0001-dataset-and-licences.md`](docs/decisions/0001-dataset-and-licences.md)。

## 開發

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
make reproduce-lite   # CI 路徑：從已 commit 的 reports/*.json 重生表格與圖（W2 起）+ 指標單元測試
make reproduce        # 完整路徑：需要資料、特徵快取與 GPU（W1–W6 逐步落地）
```

## 授權

程式碼採 Apache-2.0（見 [LICENSE](LICENSE)）。資料集與其衍生物（特徵、權重）依各自授權，不由本 LICENSE 重新授權。
