# data/

本目錄除本檔與 `manifest.json`（只含檔名、SHA-256、容量）外全部被 `.gitignore` 排除。
原始影片、標註、影格快取、特徵快取、權重一律不進 repo（設計規格 §3.3、§8.2）。
`data/external/ha-vid-public/` 亦維持 ignore；request form 內容與 Dropbox／Drive 連結永不進 git。

## 資料集來源與授權

| 角色 | 資料集 | 授權 | 取得方式 | 本專案的處理 |
|---|---|---|---|---|
| 主力 | HA-ViD | CC BY-NC 4.0 | 向作者送出 request form（姓名／單位／code of conduct）後由 Dropbox 取得 | 權重與快取特徵視為 non-commercial 衍生物，不公開 |
| 備援 | IMPACT | code Apache-2.0；data CC BY-NC-SA 4.0（以 repo 為準） | gated Hugging Face + Google Drive；v1.1 標註重驗，視為變動中，manifest 記錄版本與 hash | 若成為主力，衍生物改標 CC BY-NC-SA 4.0 並註明 share-alike |
| 指標捐贈 | IndustReal | Apache-2.0（code + data） | 4TU 開放下載 | 只用於線上指標實作的交叉檢查，不混入主表 |

## 本機落點

- 外部下載落在 `data/external/<dataset>/`；大型資料集也可放在 D 槽既有 datasets 根目錄並以 `manifest.json` 指向。
- 下載完成後產生 `data/manifest.json`（W1）；它是唯一被 commit 的資料檔。
- 特徵快取（fp16 memmap）與訓練權重落在 `artifacts/`（ignore）。

決策紀錄見 `docs/decisions/0001-dataset-and-licences.md`。
