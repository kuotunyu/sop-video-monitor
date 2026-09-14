# 模組與 CLI 一覽

從 README 移來的對照表；每個模組都有單元測試。

## 程式（`src/sop_monitor/`）

| 模組 | 內容 |
|---|---|
| `video.py`、`features.py` | PyAV probe／循序解碼；官方 `facebookresearch/dinov2` 權重的 frozen frame embedding（ViT-S/B/L，`[CLS ; mean patch]`），fp16 快取，權重 SHA-256 記錄在 `meta.json` |
| `industreal.py`、`splits.py` | IndustReal action 標註解析、participant-disjoint split 凍結與 test list SHA-256 契約（`splits/industreal/`）、標註 → 逐影格對齊 |
| `industreal_psr.py` | PSR 標註解析與三個 CSV 的交叉核對、官方 state→step 規則轉錄、從 recording 壓縮檔只抽標註、JPEG 名稱位移偵測 |
| `havid.py` | HA-ViD id 與 HR-SAT 標籤解析、temporal／collaboration 標註（inclusive、連續）、官方 split／mapping／groundTruth 交叉核對（頭尾各裁 5 frames）、`wrong` 統計、mp4 probe 與三視角配對、subject-wise split 凍結 |
| `havid_tas.py` | HA-ViD 離線 TAS 線：每個視角各訓 causal／非 causal MS-TCN++、三種不需調參的 late fusion（mean／幾何平均／信心加權）、以 recording 為鍵的預測表；官方 I3D 特徵匯出成同一種 npz 快取；可選 `--checkpoint-dir` 另存選定 epoch 的權重、標準化參數與 val posterior（`mstcn.load_checkpoint` 還原） |
| `havid_predict.py` | 以 checkpoint run 存下的網路評分一個 split（不訓練、不選 epoch），輸出與訓練 run 同格式的目錄；拒絕寫入已存在的目錄，是 frozen test 唯一的評分路徑（`docs/havid_test_protocol.md`） |
| `havid_seeds.py` | HA-ViD TAS 多 seed 彙總：每個 run 指標的 mean ± std 與範圍、各網路每個 seed 選到的 epoch；`reproduce-lite` 從各 run 的 `metrics.json` 重算 |
| `havid_sop.py` | HA-ViD SOP 層：雙手 primitive-task 步驟序列、從標籤詞彙判斷 plate、從 train 學 precedence graph／必要步驟／時長界限、synthetic 順序／遺漏／時長違規表、原生 `w` 表；`reproduce-lite` 可從 CSV 重算 |
| `online.py` | 線上 SOP 監控：逐影格輸入雙手標籤，run-length 最短時長確認、雙手步驟合併；順序／未知步驟在步驟確認時、過長在超過上界的當下、過短在步驟結束時、遺漏在錄影結束時送出，每筆偏差帶偵測影格與延遲；`min_frames=1` 時結果等於離線 `check_sequence`（同一影格開始的步驟視為同時） |
| `online_head.py` | 線上辨識器：每個視角的 causal MS-TCN++ checkpoint 以串流方式接收特徵（causal 保證前綴推論等於整段推論）、逐影格 fusion、依 look-ahead 延遲輸出，雙手標籤齊了就送進線上 SOP 監控；可從快取特徵或解碼影片 + DINOv2 餵入，記錄每個 chunk 的運算延遲與相對 checkpoint run 的一致率 |
| `sop_mermaid.py` | 從 `sop/ha-vid/*/` 的 JSON 生成 `docs/sop_knowledge.md` 的 Mermaid precedence graph（每個 plate 一張，必要步驟標色）；測試會在頁面過期時失敗 |
| `figures.py` | 動畫圖的資料與樣式（不 import Manim）：從已 commit 的 `steps_*.csv` 與 `deviations_*.csv` 讀出步驟條與警報、狀態色與字形對照；場景在 `figures/`，`make figures` 渲染成 `docs/assets/*.gif` |
| `havid_step_recall.py` | 辨識器在說明書步驟層級的診斷：每組 run（例如同一 look-ahead 的雙手 × 三 seed）與一個預測欄位，彙整每個 ground-truth 步驟的 val 影格中被預測成同一步驟的比例，並標出所屬 plate 與是否為必要步驟；`reproduce-lite` 從已 commit 的預測表重算 |
| `havid_online.py` | 把 val 錄影的雙手標籤串流（ground truth 或 causal 辨識器輸出，含 look-ahead 輸出延遲）逐影格重播進線上 SOP 監控，對 min_frames 格點記錄每筆偏差的偵測影格；彙總旗標錄影數、每錄影警報數、首次警報時間、距錄影結束的提前量，並在 min_frames 1 對照離線檢查；`reproduce-lite` 從 CSV 重算 |
| `review.py`、`review_page.py` | 偏差佇列與本機複核介面（W5）：SOP 檢查結果轉成佇列項目、三視角同步播放、接受／退回寫入 append-only JSONL、人工判定的 precision；只綁 127.0.0.1、只供應 val 影片（見 `docs/review.md`） |
| `baseline.py`、`mstcn.py` | 離線 TAS 線：linear head、causal／非 causal MS-TCN++、預測表與從預測表重算指標 |
| `psr_baseline.py` | 線上 PSR 線：linear 與 causal MS-TCN++ state head、EMA／hysteresis／dwell／procedure-prior decoder、out-of-fold 的 decoder／延遲預算／epoch 選擇、多 seed、completion 表與計分 |
| `metrics/offline.py`、`metrics/reference_mstcn.py` | MoF、Edit、F1@k（MS-TCN 定義）與獨立參考轉錄的交叉核對；participant bootstrap |
| `metrics/online.py` | IndustReal PSR 指標（POS、system-level F1、detection delay），逐條對照官方 `psr_utils.py`，三處刻意偏離寫在 docstring |
| `sop_graph.py`、`industreal_sop.py` | HR-SAT OWL 解析與 JSON 匯出（`sop/ha-vid/`）、順序／遺漏／時長檢查；從 train 標註學 IndustReal precedence graph（`sop/industreal/`）並把檢查跑在 completion 序列上 |
| `audit.py` | 影片 probe、標註對影格數的交叉核對、HA-ViD 公開檔盤點、`data/manifest.json` |
| `stream/ring_buffer.py`、`stream/pipeline.py`、`stream/bench.py` | `WAIT`／`DROP_OLDEST`／`ADAPTIVE` backpressure 與 `skipped_frames`；每路攝影機一條解碼執行緒以原始 fps 重播影片、micro-batch 收集器、watchdog；N 路 × 速度的吞吐／延遲／丟幀曲線（`stream-bench`） |

CLI `sop-monitor` 的命令依流程分組：

| 階段 | 命令 |
|---|---|
| 資料 | `audit-industreal`、`audit-havid`、`freeze-splits`（`--dataset industreal | ha-vid`）、`verify-splits`、`extract-psr-labels` |
| 特徵 | `extract-features`（`--dataset industreal | ha-vid`）、`export-official-features` |
| 離線 TAS | `train-baseline`、`train-mstcn`（IndustReal）、`train-havid-tas`（HA-ViD，每視角 + fusion）、`predict-havid-tas`（用已存的 checkpoint 評分，不訓練；test 需 `--confirm-test`） |
| 線上 PSR | `train-psr`、`learn-sop`、`check-psr-run`（IndustReal）、`learn-havid-sop`、`havid-sop`（HA-ViD SOP 層）、`havid-online`（逐影格線上重播）、`havid-online-head`（影格 → 辨識器 → 監控） |
| SOP graph | `export-sop`、`check-sop` |
| 複核 | `build-review-queue`、`review-ui`、`review-summary` |
| 串流 | `stream-bench` |
| 重算 | `score-predictions`、`summarise-havid-seeds`、`havid-step-recall`、`reproduce-lite`、`export-sop-mermaid` |
