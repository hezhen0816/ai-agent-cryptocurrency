import ccxt, pandas as pd, datetime as dt
import os
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from dateutil.relativedelta import relativedelta
import time
def binance_data(output_dir, symbol, timeframe):
    os.makedirs(output_dir, exist_ok=True)
    exchange = ccxt.binance()
    five_years_ago = datetime.now(ZoneInfo("Asia/Taipei")) - relativedelta(years=5)
    print(five_years_ago)
    since = int(five_years_ago.timestamp() * 1000)
    print(since)
    all_bars   = []
    while True:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not bars:
            break
        all_bars.extend(bars)
        since = bars[-1][0] + 1        # 下一次從上一批最早那根的開盤 + 1ms 開抓
        time.sleep(exchange.rateLimit / 1000) 
    df = pd.DataFrame(all_bars, columns=['Open time','open','high','low','close','volume'])
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')  # 轉換時間戳
    logging.info(f"已取得 {symbol}")
    safe_symbol = symbol.replace("/", "_")
    filename=f"binance_{safe_symbol}_{timeframe}.csv"
    path = os.path.join(output_dir, filename)
    df.set_index('Open time', inplace=True)
    df.to_csv(path, encoding="utf-8-sig")
    logging.info(f"已輸出 {filename}，共 {len(df)} 筆")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    output_dir="./data_test"
    symbol="ETH/USDT"
    timeframe="1d"
    binance_data(output_dir, symbol, timeframe)
    


