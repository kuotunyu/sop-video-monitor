# ADR 0001：資料集選擇與授權處理

- 日期：2026-09-03
- 狀態：accepted（依設計規格 §3；規格本身仍為草稿，數字凡標「提案」者待確認）
- 依據：`docs/superpowers/specs/2026-09-03-sop-video-monitor-design.md` §3，其資料集事實與授權以研究 memo 為準

## 脈絡

專案需要（1）固定多視角、（2）原生的順序／位置錯誤標註、（3）可重建 subject-wise split 的裝配影片資料集，
並且授權允許公開程式碼與 aggregate 指標。沒有單一資料集同時滿足所有需求且授權完全開放，因此以三個角色分工。

## 決定

### 三個角色

| 角色 | 資料集 | 選用理由 | 授權 | 取得方式 |
|---|---|---|---|---|
| 主力 | HA-ViD | 唯一固定三視角（left／front／top）且有原生 HR-SAT `wrong` 標註、附 subject-agnostic task precedence graph | CC BY-NC 4.0 | request form → Dropbox；容量未公布 |
| 備援 | IMPACT | 多視角固定攝影機 + ego；有 partial-order prerequisite graph、compliance 三相標註與 cross-subject split；附預抽特徵 | code Apache-2.0；data CC BY-NC-SA 4.0（以 repo 為準，arXiv 摘要的 CC BY 4.0 不採信） | gated HF + Google Drive；v1.1 標註重驗，視為變動中 |
| 指標捐贈 | IndustReal | POS／completion F1／mean detection delay 的定義來源；有公開 baseline 可做實作 sanity check | Apache-2.0（code + data） | 4TU 開放下載 |

### 排除

Assembly101（容量過大、Drive 連結短期失效）、MECCANO（授權未載明）、ATTACH（research-only 且無錯誤標註）、IKEA ASM（無錯誤標註）。

### 取得計畫

1. Day 0 三線併發：送出 HA-ViD request form、下載 IndustReal、申請 IMPACT gated HF；任何一條不必等另一條。
2. 資料落地於本機（`data/external/` 或 D 槽 datasets 根目錄），不進 repo；下載完成即產生 `data/manifest.json`（只含 metadata）。
3. IMPACT 若先到，W1 的 parser 與 split 工具先以 IMPACT 通過測試（以 adapter 隔離），HA-ViD 到貨後切換資料介面。

### 授權處理

- 原始影片、標註、快取特徵、訓練權重一律不進 public repo；`.gitignore` 之外，CI 加一步掃描被 ignore 目錄不得含原始碼（W1 TODO）。
- 自有程式碼 Apache-2.0；`MODEL_CARD.md`（W6）明示「權重與特徵為 CC BY-NC 4.0 資料的衍生物，僅限非商業用途」。
  若主力切換為 IMPACT，改標 CC BY-NC-SA 4.0 並註明 share-alike 對衍生權重的約束。
- IndustReal 衍生物可依 Apache-2.0 處理，但只用於指標實作的交叉檢查，不混入主表。
- README 與 UI 截圖以介面與圖表為主；資料集影格只放少量並附 attribution。

### 備援觸發（皆為提案）

- 觸發 A（取得延遲）：request 送出後 14 天未收到連結 → 主力改 IMPACT（S2 cross-subject split）；HA-ViD 到貨後降為第二資料集。
- 觸發 B（錯誤標註稀少）：held-out test subjects 內 `wrong` 段數 < 20 → HA-ViD 維持主力，headline 違規指標改由 synthetic 表提供；native 表以精確計數加 Wilson 95% CI 呈報並標「underpowered」。
- 觸發 C（視角不完整）：三視角缺漏或不同步影片 > 5% → 主線退為單視角（front），多視角融合降為 ablation。

## 後果

- README 只能宣稱「在 HA-ViD 固定三視角、held-out subjects 上」，不得宣稱真實工廠泛化或商業可用（規格 §2.2）。
- 權重與特徵第一版不上 Hugging Face；日後若上，走 gated repo 並標 non-commercial（規格 §10 建議）。
- W1 資料稽核（`reports/w1_data_audit.md`）必須實測 `wrong` 段數與三視角配對完整率，才能決定是否觸發 B／C。
