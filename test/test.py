import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import DATA_DIR, SYMBOL, TIMEFRAME

print("[LOG] 啟動回測流程")

# 1. 讀取資料並定義特徵欄位
print("[LOG] 讀取資料中...")
data_file = DATA_DIR / f"{SYMBOL.replace('/', '')}_{TIMEFRAME}_features.parquet"
df = pd.read_parquet(data_file)
print(f"[LOG] 原始資料筆數: {len(df)}")
feat_cols = [
    'sma20','ema50','rsi14','macd','macd_signal','macd_hist','bb_upper',
    'bb_middle','bb_lower','obv','atr14','adx14','cci20','trix30','bb_width',
    'roc10','mom14','stoch_k','stoch_d','plus_di','minus_di','ao','vwap_20','vol_ma20'
]
print(f"[LOG] 特徵欄位數量: {len(feat_cols)}")

# 2. 計算標籤並移除 NaN
df['log_ret_10'] = np.log(df['close'].shift(-10)) - np.log(df['close'])
print(f"[LOG] 計算標籤後，前後 NaN 數: 前 {df['log_ret_10'].isna().sum()} 筆，後 {df['log_ret_10'].isna().sum()} 筆")
drop_cols = ['log_ret_10'] + feat_cols
df_clean = df.dropna(subset=drop_cols)
print(f"[LOG] 清理 NaN 後資料筆數: {len(df_clean)}")

# 3. 定義回測期間
start_date = '2024-08-01'
end_date   = '2025-08-01'
print(f"[LOG] 回測期間設定: {start_date} ~ {end_date}")

# 4. 訓練最終模型（使用回測前所有資料）
train_data = df_clean.loc[:start_date]
print(f"[LOG] 訓練資料筆數: {len(train_data)}")
scaler = StandardScaler().fit(train_data[feat_cols].values)
X_train = scaler.transform(train_data[feat_cols].values)
y_train = (train_data['log_ret_10'] > 0).astype(int).values
print("[LOG] 開始訓練模型...")
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)
print("[LOG] 模型訓練完成")

# 5. 準備回測資料並生成交易訊號
df_bt = df_clean.loc[start_date:end_date].copy()
print(f"[LOG] 回測資料筆數: {len(df_bt)}")
X_test = scaler.transform(df_bt[feat_cols].values)
df_bt['pred_proba'] = clf.predict_proba(X_test)[:, 1]
df_bt['signal']    = (df_bt['pred_proba'] > 0.5).astype(int)
print("[LOG] 前5筆預測機率與訊號:")
print(df_bt[['pred_proba','signal']].head(5))

# 6. 計算日報酬並剔除 NaN
df_bt['ret'] = df_bt['close'].pct_change().shift(-1)
df_bt = df_bt.dropna(subset=['ret'])
print(f"[LOG] 計算並清理報酬後筆數: {len(df_bt)}")

# 7. 策略績效計算
df_bt['strategy_ret']           = df_bt['signal'] * df_bt['ret']
df_bt['cumulative_market']     = (1 + df_bt['ret']).cumprod() - 1
df_bt['cumulative_strategy']   = (1 + df_bt['strategy_ret']).cumprod() - 1

# 8. 進階指標計算
equity = (1 + df_bt['strategy_ret']).cumprod()
rolling_max = equity.cummax()
drawdown = (equity - rolling_max) / rolling_max
max_dd = drawdown.min()
sharpe_ratio = df_bt['strategy_ret'].mean() / df_bt['strategy_ret'].std() * np.sqrt(252)
print(f"[LOG] 最大回撤: {max_dd:.4f}")
print(f"[LOG] 夏普比率: {sharpe_ratio:.4f}")

# 9. 輸出結果
print("[LOG] 回測結果摘要:")
print("  期間：", df_bt.index.min(), "～", df_bt.index.max())
print("  市場累積報酬：{:.4f}".format(df_bt['cumulative_market'].iloc[-1]))
print("  策略累積報酬：{:.4f}".format(df_bt['cumulative_strategy'].iloc[-1]))

# 10. 視覺化：累積報酬與回撤曲線
print("[LOG] 正在繪製圖表...")
plt.figure()
plt.plot(df_bt.index, df_bt['cumulative_market'], label='市場累積報酬')
plt.plot(df_bt.index, df_bt['cumulative_strategy'], label='策略累積報酬')
plt.title('累積報酬曲線')
plt.xlabel('日期')
plt.ylabel('累積報酬')
plt.legend()
plt.show()

plt.figure()
plt.plot(df_bt.index, drawdown, label='回撤')
plt.title('回撤曲線')
plt.xlabel('日期')
plt.ylabel('回撤')
plt.legend()
plt.show()

print("[LOG] 回測流程結束")
