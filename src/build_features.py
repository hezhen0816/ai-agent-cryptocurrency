"""將原始 K 線轉成特徵表（修正版）"""
import numpy as np
import pandas as pd
import talib
from config import DATA_DIR, SYMBOL, TIMEFRAME

RAW_PATH  = DATA_DIR / f"{SYMBOL.replace('/', '')}_{TIMEFRAME}.csv"
FEAT_PATH = DATA_DIR / f"{SYMBOL.replace('/', '')}_{TIMEFRAME}_features.parquet"
CSV_PATH  = DATA_DIR / f"{SYMBOL.replace('/', '')}_{TIMEFRAME}_features.csv"

def main() -> None:
    df = pd.read_csv(RAW_PATH, parse_dates=["timestamp"], index_col="timestamp").tz_localize(None)

    close, high, low, volume = (
        df[c].astype(np.float32) for c in ["close", "high", "low", "volume"]
    )

    # ===== 技術指標 =====
    df["sma20"] = talib.SMA(close, 20)
    df["ema50"] = talib.EMA(close, 50)

    df["rsi14"] = talib.RSI(close, 14)
    macd, macd_sig, macd_hist = talib.MACD(close, 12, 26, 9)
    df[["macd", "macd_signal", "macd_hist"]] = np.column_stack([macd, macd_sig, macd_hist])

    bb_u, bb_m, bb_l = talib.BBANDS(close, 20, 2, 2)
    df[["bb_upper", "bb_middle", "bb_lower"]] = np.column_stack([bb_u, bb_m, bb_l])

    df["obv"] = talib.OBV(close, volume)
    df["atr14"] = talib.ATR(high, low, close, 14)
    df["adx14"] = talib.ADX(high, low, close, 14)
    df["cci20"] = talib.CCI(high, low, close, 20)
    df["trix30"] = talib.TRIX(close, 30)
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    df["roc10"] = talib.ROC(close, 10)
    df["mom14"] = talib.MOM(close, 14)
    stoch_k, stoch_d = talib.STOCH(high, low, close, 14, 3, 0, 3, 0)
    df[["stoch_k", "stoch_d"]] = np.column_stack([stoch_k, stoch_d])

    df["plus_di"]  = talib.PLUS_DI(high, low, close, 14)
    df["minus_di"] = talib.MINUS_DI(high, low, close, 14)

    # ➜ Awesome Oscillator：5SMA - 34SMA of HL2
    hl2 = (high + low) / 2
    df["ao"] = talib.SMA(hl2, 5) - talib.SMA(hl2, 34)

    df["vwap_20"]  = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
    df["vol_ma20"] = volume.rolling(20).mean()

    # ===== 滯後與衍生特徵 =====
    for lag in [1, 2, 3, 5, 10]:
        df[f"close_lag_{lag}"] = close.shift(lag)
        df[f"rsi14_lag_{lag}"] = df["rsi14"].shift(lag)
        df[f"macd_lag_{lag}"]  = df["macd"].shift(lag)

    df["rsi_sma"]          = df["rsi14"].rolling(5).mean()
    df["macd_signal_diff"] = df["macd"] - df["macd_signal"]
    df["bb_position"]      = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["volume_ratio"]     = volume / df["vol_ma20"]

    df["higher_high"]       = (high > high.shift(1)).astype(int)
    df["lower_low"]         = (low  < low.shift(1)).astype(int)
    df["price_acceleration"]= close.pct_change() - close.pct_change().shift(1)

    df["sma20_slope"] = df["sma20"].diff(5)
    df["volume_surge"] = (volume > volume.rolling(20).mean() * 2).astype(int)

    df["hl_range"]       = high - low
    df["oc_range"]       = close - df["open"]
    df["upper_shadow"]   = high - df[["open", "close"]].max(axis=1)
    df["lower_shadow"]   = df[["open", "close"]].min(axis=1) - low
    df["body_ratio"]     = (close - df["open"]).abs() / df["hl_range"]
    df["close_position"] = (close - low) / (high - low)
    df["open_position"]  = (df["open"] - low) / (high - low)

    # ===== 清理與輸出 =====
    df = df.dropna().astype(np.float32)
    df.to_parquet(FEAT_PATH)
    df.to_csv(CSV_PATH)
    print(f"Features saved → {FEAT_PATH}  shape={df.shape}")

if __name__ == "__main__":
    main()
