"""下載 K 線並存成 CSV。

流程：
1. 建立一次 Binance 交易所的 ccxt 實例
2. 以 START_DATE 為起點（UTC），迴圈向後抓取 OHLCV
3. 每批最多 1000 根 K 線。抓到資料後，把 cursor 更新為最後一根 K 線時間 +1ms → 不會重疊
4. 若回傳空且 cursor 已經是今天，代表今天尚未生成 K 線 → 結束迴圈
5. 整理成 DataFrame 並寫成 CSV
"""

import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from config import DATA_DIR, SYMBOL, TIMEFRAME, START_DATE

# --- 初始化 -------------------------------------------------------------

exchange = ccxt.binance()  # 只建一次，避免重複連線 / 節省延遲

# --- Helper -------------------------------------------------------------

def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int) -> list:
    """
    從 Binance 抓取 OHLCV。

    Parameters
    ----------
    exchange : ccxt.Exchange
        已初始化的 ccxt 交易所實例
    symbol : str
        例如 "ETH/USDT"
    timeframe : str
        例如 "1d"
    since_ms : int
        從哪個 Unix 毫秒時間戳開始抓取（包含起點）

    Returns
    -------
    list[list]
        [[timestamp, open, high, low, close, volume], ...] 最多 1000 筆
    """
    return exchange.fetch_ohlcv(
        symbol,
        timeframe=timeframe,
        since=since_ms,
        limit=1000,
    )

# --- 主程式 -------------------------------------------------------------

def main() -> None:
    # 1. 設定游標起點與終點
    start = datetime.fromisoformat(START_DATE).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)  # 今天（UTC）

    all_rows: list[list] = []        # 暫存所有 K 線
    cursor = start                   # 讀取游標

    # 2. 迴圈抓取直到抵達 today
    while True:
        since_ms = int(cursor.timestamp() * 1000)
        rows = fetch_ohlcv(exchange, SYMBOL, TIMEFRAME, since_ms)

        # --- 沒抓到資料：可能因為尚未生成今日 K 線 ---------------------
        if not rows:
            if cursor.date() >= now.date():  # 已追到今天
                print("Reached latest available candle — stop.")
                break
            print("Empty fetch (data gap before today), sleep 10 s and retry…")
            time.sleep(10)
            continue

        # --- 抓到資料 ----------------------------------------------------
        all_rows.extend(rows)

        # 更新 cursor → 最後一根時間 +1ms，避免重疊
        last_ms = rows[-1][0]
        cursor = datetime.fromtimestamp(last_ms / 1000, tz=timezone.utc) + timedelta(milliseconds=1)

        time.sleep(0.2)  # 禮貌性延遲，避免觸發 IP rate limit

    # 3. 轉成 DataFrame ---------------------------------------------------
    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")

    # 4. 存檔 -------------------------------------------------------------
    out_path = DATA_DIR / f"{SYMBOL.replace('/', '')}_{TIMEFRAME}.csv"
    df.to_csv(out_path)
    print(f"Saved {len(df):,d} rows to {out_path}")


if __name__ == "__main__":
    main()
