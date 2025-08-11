
#!/usr/bin/env python3
"""End-to-end crypto up/down prediction pipeline.

步驟：
1. 載入 build_features.py 已經產好的技術指標檔
2. 產生『隔日漲跌』標籤
3. 依時間序列切分 train / val / test
4. 使用 LightGBM 訓練二元分類模型
5. 在測試集評估 AUC / Accuracy
6. 以簡易多空策略回測淨值曲線

⚠️ 注意：自 LightGBM v4.0 起，`early_stopping_rounds` 與 `verbose_eval` 參數已移除，需改用 callback 形式。
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from joblib import dump
from config import DATA_DIR, SYMBOL, TIMEFRAME  # 讀入共用設定

# 1. 讀取特徵檔
def load_features():
    file_name = f"{SYMBOL.replace('/', '')}_{TIMEFRAME}_features.parquet"
    return pd.read_parquet(DATA_DIR / file_name)

# 2. 建立標籤：隔日報酬 > 0 ⇒ 1，否則 0
def add_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1d"] = df["close"].pct_change().shift(-1)
    df["up_next_day"] = (df["ret_1d"] > 0).astype(int)
    df = df.dropna(subset=["ret_1d"])
    return df

# 3. 時序切分
def time_split(df: pd.DataFrame):
    train = df.loc[: "2022-12-31"]
    val   = df.loc["2023-01-01":"2023-12-31"]
    test  = df.loc["2024-01-01":]
    return train, val, test

def _xy(split: pd.DataFrame, feature_cols):
    return split[feature_cols], split["up_next_day"]

# 4. LightGBM 訓練 (相容 v4.0+)
def train_lgb(train_df, val_df, feature_cols):
    dtrain = lgb.Dataset(*_xy(train_df, feature_cols))
    dval   = lgb.Dataset(*_xy(val_df, feature_cols), reference=dtrain)

    params = dict(
        objective="binary",
        metric=["binary_logloss", "auc"],
        learning_rate=0.01,
        num_leaves=127,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        seed=42,
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=True),
            lgb.log_evaluation(200),
        ],
    )
    return model

# 5. 模型評估
def evaluate(model, test_df, feature_cols):
    proba = model.predict(test_df[feature_cols])
    pred  = (proba > 0.5).astype(int)
    auc   = roc_auc_score(test_df["up_next_day"], proba)
    acc   = accuracy_score(test_df["up_next_day"], pred)
    print("=== Evaluation on Test Set ===")
    print(f"AUC        : {auc:.4f}")
    print(f"Accuracy   : {acc:.4f}")
    print(classification_report(test_df["up_next_day"], pred, digits=4))
    return auc, acc

# 6. 簡易回測
def backtest(model, df, feature_cols, fee=0.001, threshold=0.55):
    df = df.copy()
    df["proba"] = model.predict(df[feature_cols])
    df["signal"] = (df["proba"] > threshold).astype(int)
    df["pos_shift"] = df["signal"].shift(1).fillna(0)      # 下一根 K 開倉
    df["ret_1d"] = df["close"].pct_change()
    df["strategy_ret"] = df["pos_shift"] * df["ret_1d"] - fee * df["pos_shift"].diff().abs()
    df["cum_equity"] = (1 + df["strategy_ret"].fillna(0)).cumprod()
    return df[["close", "cum_equity", "strategy_ret", "pos_shift", "proba"]]

def main():
    # 載入與標籤化
    df = load_features()
    df = add_label(df)
    feature_cols = [c for c in df.columns if c not in ("ret_1d", "up_next_day")]

    # 切分資料
    train_df, val_df, test_df = time_split(df)

    # 訓練模型
    model = train_lgb(train_df, val_df, feature_cols)

    # 儲存模型
    models_dir = DATA_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "lgbm_updown.joblib"
    dump(model, model_path)
    print(f"Model saved to {model_path}")

    # 測試集評估
    evaluate(model, test_df, feature_cols)

    # 回測
    bt_df = backtest(model, test_df, feature_cols)
    bt_csv = DATA_DIR / "backtest_equity.csv"
    bt_df.to_csv(bt_csv)
    print(f"Backtest equity curve saved to {bt_csv}")

if __name__ == "__main__":
    main()
