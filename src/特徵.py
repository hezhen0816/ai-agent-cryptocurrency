import pandas as pd
import talib
from pathlib import Path
data_file = Path(__file__).parent / ".." / "data" / "ETH" / "初始資料" / "binance_ETH_USDT_1d.csv"
df = pd.read_csv(data_file, parse_dates=['Open time'], index_col='Open time')
df = df.tz_localize("UTC").tz_convert("Asia/Taipei")   # 時區一致

close  = df['close']           # 保持 Series
high   = df['high']
low    = df['low']
volume = df['volume']

# 移動平均
df['sma20'] = talib.SMA(close, timeperiod=20)
df['ema50'] = talib.EMA(close, timeperiod=50)

# RSI、MACD
df['rsi14'] = talib.RSI(close, timeperiod=14)
macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
df['macd']        = macd
df['macd_signal'] = macd_signal
df['macd_hist']   = macd_hist

# 布林通道
upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
df['bb_upper'] = upper
df['bb_middle'] = middle
df['bb_lower'] = lower

# 成交量指標
df['obv'] = talib.OBV(close, volume)

# 波動度 & 趨勢強度
df['atr14']  = talib.ATR(high, low, close, timeperiod=14)             # 真實波幅
df['adx14']  = talib.ADX(high, low, close, timeperiod=14)             # 趨勢強度
df['cci20']  = talib.CCI(high, low, close, timeperiod=20)             # 商品通道指標
df['trix30'] = talib.TRIX(close, timeperiod=30)                       # TRIX 動量
df['bb_width'] = df['bb_upper'] - df['bb_lower']                      # 布林通道寬度

# 價格動量
df['roc10']  = talib.ROC(close, timeperiod=10)                        # 10 日變化率
df['mom14']  = talib.MOM(close, timeperiod=14)                        # 14 日動量
stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14,
                               slowk_period=3, slowk_matype=0,
                               slowd_period=3, slowd_matype=0)
df['stoch_k'] = stoch_k
df['stoch_d'] = stoch_d

# 多頭 / 空頭動能
df['plus_di']  = talib.PLUS_DI(high, low, close, timeperiod=14)       # +DI
df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)      # –DI
ao = talib.EMA(high + low, timeperiod=5) - talib.EMA(high + low, timeperiod=34)
df['ao'] = ao                                                         # Awesome Oscillator（自行組合）

# 量價關係
df['vwap_20']  = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
df['vol_ma20'] = volume.rolling(20).mean()

# 添加滯後特徵
for lag in [1, 2, 3, 5, 10]:
    df[f'close_lag_{lag}'] = df['close'].shift(lag)
    df[f'rsi14_lag_{lag}'] = df['rsi14'].shift(lag)
    df[f'macd_lag_{lag}'] = df['macd'].shift(lag)

df['rsi_sma'] = df['rsi14'].rolling(5).mean()  # RSI平滑
df['macd_signal_diff'] = df['macd'] - df['macd_signal']  # MACD信號差
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])  # 布林通道位置
df['volume_ratio'] = df['volume'] / df['vol_ma20']  # 成交量比率

# 添加價格型態識別
df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
df['price_acceleration'] = df['close'].pct_change() - df['close'].pct_change().shift(1)

# 添加不同時間週期的特徵
df['sma20_slope'] = df['sma20'].diff(5)  # 移動平均斜率
df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)

# OHLC 相關的衍生特徵
df['hl_range'] = df['high'] - df['low']  # 日內波動範圍
df['oc_range'] = df['close'] - df['open']  # 開收差
df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)  # 上影線
df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']  # 下影線
df['body_ratio'] = abs(df['close'] - df['open']) / df['hl_range']  # 實體比率

# 價格在 OHLC 範圍內的位置
df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
df['open_position'] = (df['open'] - df['low']) / (df['high'] - df['low'])


df.dropna(inplace=True)          # 直接剔除
# 或 df.fillna(method='ffill', inplace=True)  # 向前填值
output_dir = Path(__file__).parent / ".." / "data" / "ETH" / "indicators" / "binance_ETH_USDT_1d.Parquet"
df.to_parquet(output_dir)        # 建議用 Parquet 儲存
output_dir = Path(__file__).parent / ".." / "data" / "ETH" / "indicators" / "binance_ETH_USDT_1d.csv"
df.to_csv(output_dir, encoding="utf-8-sig")  # 也可以 CSV
