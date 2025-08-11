import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from config import DATA_DIR, SYMBOL, TIMEFRAME

# 1. 讀資料、定義欄位
data_file = DATA_DIR / f"{SYMBOL.replace('/', '')}_{TIMEFRAME}_features.parquet"
df = pd.read_parquet(data_file)
feat_cols = ['sma20','ema50','rsi14','macd','macd_signal','macd_hist','bb_upper',
             'bb_middle','bb_lower','obv','atr14','adx14','cci20','trix30','bb_width',
             'roc10','mom14','stoch_k','stoch_d','plus_di','minus_di','ao','vwap_20','vol_ma20'
]

# 2. 計算標籤
h = 10
df[f'log_ret_{h}'] = np.log(df['close'].shift(-h)) - np.log(df['close'])

# 3. 清理 NaN，並同時移除特徵與標籤的缺失列
drop_cols = [f'log_ret_{h}'] + feat_cols
df_clean = df.dropna(subset=drop_cols)

# 4. 準備 X, y
X = df_clean[feat_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)    # 只做一次
y = (df_clean[f'log_ret_{h}'] > 0).astype(int).values

# 5. 確認樣本數與形狀
print("樣本數（清理後） =", len(df_clean))
print("X_scaled.shape =", X_scaled.shape)
print("y.shape       =", y.shape)

# 6. 時間序列切分與訓練
tscv = TimeSeriesSplit(n_splits=10, test_size=100)
aucs = []
for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_scaled), 1):
    X_tr, X_va = X_scaled[tr_idx], X_scaled[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    clf = lgb.LGBMClassifier()
    clf.fit(X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(200, verbose=False)])

    aucs.append(roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1]))
    print(f"第 {fold} 折 AUC：{aucs[-1]:.4f}")

print(f"平均 AUC：{np.mean(aucs):.4f}")
