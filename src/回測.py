import pandas as pd, vectorbt as vbt
from pathlib import Path
# 1. 載入剛產生的帶指標資料
data_file = Path(__file__).parent / ".." / "data" / "ETH" / "indicators" / "binance_ETH_USDT_1d.Parquet"
df = pd.read_parquet(data_file)

# 2. 取收盤價與指標
price = df['close']
fast  = df['sma20']
slow  = df['ema50']

# 3. 建立進出場訊號
entries = fast > slow          # 上穿
exits   = fast < slow          # 下穿

# 4. 建立投組並回測
pf = vbt.Portfolio.from_signals(price, entries, exits,
                                freq="1D",
                                fees=0.001,     # 假設手續費 0.1%
                                slippage=0.0005 # 假設滑價 0.05%
                               )

print(pf.stats())
fig = pf.plot()
# 寫成 self-contained 的 HTML，並自動開啟
fig.write_html("backtest.html", auto_open=True)