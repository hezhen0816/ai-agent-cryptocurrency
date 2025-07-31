# ai-agent-cryptocurrency

> 以 **Python** 打造的加密貨幣量化交易／機器學習代理人（示範專案）

本專案展示一條完整的研究流程：從 **Binance** 下載歷史 OHLCV 資料 → 計算技術指標 → 以簡單均線交叉回測 → 建立監督式機器學習模型。您可以依此框架擴充至其他幣種、時間週期或策略。

---

## 1. 專案流程

```
┌────────────┐   1️⃣ get-date.py
│  Binance   │ ────────────────▶  下載 5 年歷史 K 線
└────────────┘                     ↓
                               2️⃣ 特徵.py
                               計算 TA‑Lib 技術指標
                                   ↓
                               3️⃣ 回測.py
                               vectorbt 回測均線策略
                                   ↓
                               4️⃣ 監督式 ML.py
                               LightGBM 預測未來漲跌
```

---

## 2. 目錄結構

```text
.
├─ data/                    # 原始 K 線、指標、Parquet 等資料
├─ src/
│  ├─ get-date.py           # 歷史資料下載
│  ├─ 特徵.py               # 技術指標特徵工程
│  ├─ 回測.py               # 含手續費／滑價的向量化回測
│  └─ 監督式 ML.py         # LightGBM 時序交叉驗證
├─ requirements.txt         # 依賴套件清單
└─ README.md                # 本文件
```

---

## 3. 安裝與環境設定

> **建議**：先安裝 [Python ≥ 3.11](https://www.python.org/downloads/) 並使用 *virtualenv*／*venv* 隔離環境。

```bash
# 1️⃣ 建立並啟用虛擬環境（Windows 範例）
python -m venv .venv
.venv\Scripts\activate

# 2️⃣ 安裝依賴
pip install -r requirements.txt

# 3️⃣ TA‑Lib 若安裝失敗，可使用隨附 wheel 手動安裝
pip install .\ta_lib‑0.64‑cp312‑cp312‑win_amd64.whl  # 依實際檔名調整
```

---

## 4. 使用流程

| 步驟       | 指令 (在專案根目錄執行)            | 說明                                                              |
| -------- | ------------------------ | --------------------------------------------------------------- |
| ① 下載 K 線 | `python src/get-date.py` | 預設抓取 **ETH/USDT** 五年日線，輸出 `./data_test/binance_ETH_USDT_1d.csv` |
| ② 產生技術指標 | `python src/特徵.py`       | 讀取 CSV → 產生多種指標 → 輸出 Parquet + CSV                              |
| ③ 回測策略   | `python src/回測.py`       | 以 SMA20 / EMA50 黃金／死亡交叉進出場，生成 `backtest.html` 自動開啟              |
| ④ 監督式 ML | `python src/監督式\ ML.py`  | LightGBM + `TimeSeriesSplit`，列印每折 AUC 與平均值                      |

> **備註**：Windows 對中文檔名／空白路徑較敏感，若出現找不到檔案請先檢查路徑是否正確。

---

## 5. 主要依賴套件（摘錄）

- **ccxt**：交易所 API
- **pandas / numpy**：資料處理
- **TA‑Lib**：技術指標
- **vectorbt**：回測框架
- **lightgbm**：梯度提升機器學習模型
- **scikit‑learn**：資料分割與效能評估
- **plotly**：互動式圖表（vectorbt 內部引用）

完整版本請見 `requirements.txt`。

---

## 6. 路線圖 / TODO

-

---

## 7. 免責聲明

> 本專案僅供學術研究與技術交流。任何因使用本程式碼進行實盤交易所產生之盈虧，作者概不負責。加密貨幣投資具有高風險，請務必審慎評估。

