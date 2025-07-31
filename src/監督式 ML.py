import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

data_file = Path(__file__).parent / ".." / "data" / "ETH" / "indicators" / "binance_ETH_USDT_1d.Parquet"
df = pd.read_parquet(data_file)

feat_cols = ['sma20','ema50','rsi14','macd','macd_signal','macd_hist','bb_upper',
             'bb_middle','bb_lower','obv','atr14','adx14','cci20','trix30','bb_width',
             'roc10','mom14','stoch_k','stoch_d','plus_di','minus_di','ao','vwap_20','vol_ma20']

X = df[feat_cols].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

h = 10
df[f'log_ret_{h}'] = np.log(df['close'].shift(-h)) - np.log(df['close'])
y = (df[f'log_ret_{h}'] > 0).astype(int).values

N = len(df)  # 取得總筆數
print(f"總共有 {N} 筆資料，每筆代表一天")

tscv = TimeSeriesSplit(n_splits=10, test_size=100)

for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
    print(f"第 {fold} 折：訓練 {train_idx[0]}–{train_idx[-1]}，測試 {test_idx[0]}–{test_idx[-1]}")

aucs = []
for tr, va in tscv.split(X_scaled):
    X_tr, X_va = X_scaled[tr], X_scaled[va]
    y_tr, y_va = y[tr], y[va]

    clf = lgb.LGBMClassifier()

    clf.fit(X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(200, verbose=False)])

    aucs.append(roc_auc_score(y_va, clf.predict_proba(X_va)[:,1]))

print(f"平均 AUC：{np.mean(aucs):.3f}")
