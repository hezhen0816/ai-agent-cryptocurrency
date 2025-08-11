#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightGBM 策略回測範例
=====================
本程式示範如何將先前訓練好的 **LightGBM** 方向預測模型
整合進簡易的回測框架，計算策略淨值與績效指標。

步驟：
 1. 讀取特徵資料、模型、Scaler
 2. 依機率門檻產生交易訊號 (signal)
 3. 轉換為持倉 (position)
 4. 計算策略報酬 (含手續費、滑價)
 5. 統計績效指標，並繪製淨值曲線

**注意**：
 - 這裡以現貨做多/做空為例，若交易衍生品請另外考量保證金與資金成本。
 - 示範用最小邏輯 —— 部位僅允許 -1, 0, +1；先進先出，不計槓桿。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Dict

import joblib
import matplotlib.pyplot as plt  # 只用於繪圖，可刪除
import numpy as np
import pandas as pd

# ===== 全局設定（可依需求調整） =====
HORIZON      : int   = 10       # 與訓練時相同
PROBA_UP     : float = 0.55     # 做多門檻
PROBA_DOWN   : float = 0.45     # 做空門檻
FEE_RATE     : float = 0.001    # 手續費 (千分之一)
SLIPPAGE_RATE: float = 0.0005   # 滑價 (千分之 0.5)
RISK_FREE    : float = 0.0      # 年化無風險利率 (計算 Sharpe 時用)

# ===== 資料與模型路徑 =====
from config import DATA_DIR, SYMBOL, TIMEFRAME, MODEL_DIR  # type: ignore

FEAT_PATH  = DATA_DIR / f"{SYMBOL.replace('/', '')}_{TIMEFRAME}_features.parquet"
MODEL_PATH = Path(MODEL_DIR) / "lgbm_eth.pkl"
SCALER_PATH= Path(MODEL_DIR) / "scaler_eth.pkl"

# ===== 工具函式 =====

def setup_logger() -> None:
    """設定 logging 格式"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_data() -> Tuple[pd.DataFrame, object, object]:
    """讀取特徵 DataFrame、模型、Scaler"""
    if not FEAT_PATH.exists():
        raise FileNotFoundError(f"找不到特徵檔：{FEAT_PATH}")
    df = pd.read_parquet(FEAT_PATH)
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("資料列數：%s, 欄位數：%s", f"{len(df):,}", len(df.columns))
    return df, model, scaler


def create_signal(df: pd.DataFrame, model, scaler) -> pd.Series:
    """輸出做多(+1)/做空(-1)/空倉(0) 訊號"""
    exclude = {"open", "high", "low", "close", "volume", f"log_ret_{HORIZON}"}
    X = df[[c for c in df.columns if c not in exclude]].astype(np.float32)

    proba = model.predict_proba(scaler.transform(X))[:, 1]
    signal = np.where(proba > PROBA_UP,  1,
              np.where(proba < PROBA_DOWN, -1, 0)).astype(np.int8)
    return pd.Series(signal, index=df.index, name="signal")


def create_position(signal: pd.Series) -> pd.Series:
    """將訊號轉持倉；遇到 0 則保持原倉位"""
    position = signal.replace(to_replace=0, method="ffill").shift(1).fillna(0)
    position.name = "position"
    return position.astype(np.int8)


def calc_returns(df: pd.DataFrame, position: pd.Series) -> pd.DataFrame:
    """計算策略報酬 (含成本) 與淨值曲線"""
    ret = np.log(df["close"]).diff().fillna(0)

    strategy_ret = position * ret
    cost = position.diff().abs() * (FEE_RATE + SLIPPAGE_RATE)

    net_ret = strategy_ret - cost
    equity_curve = net_ret.cumsum().apply(np.exp)

    out = pd.DataFrame({
        "ret": ret,
        "position": position,
        "strategy_ret": strategy_ret,
        "net_ret": net_ret,
        "equity_curve": equity_curve,
    })
    return out


def performance_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """回傳常用績效指標"""
    cum_ret = df["equity_curve"].iloc[-1] - 1

    # 年化 Sharpe: 假設 1h K 線 → 8760 檔/年
    ann_factor = np.sqrt(24 * 365)
    sharpe = (df["net_ret"].mean() - RISK_FREE/8760) / df["net_ret"].std()
    sharpe *= ann_factor

    drawdown = 1 - df["equity_curve"] / df["equity_curve"].cummax()
    max_dd = drawdown.max()

    return {
        "Cumulative Return": float(cum_ret),
        "Sharpe": float(sharpe),
        "Max Drawdown": float(max_dd),
    }


def plot_equity_curve(df: pd.DataFrame) -> None:
    """繪製策略淨值曲線 (可選)"""
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["equity_curve"], label="Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Strategy Equity Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===== 主程式 =====

def main(plot: bool = True) -> None:
    setup_logger()

    df, model, scaler = load_data()

    signal = create_signal(df, model, scaler)
    position = create_position(signal)
    bt_df = calc_returns(df, position)

    metrics = performance_metrics(bt_df)
    logging.info("===== Backtest Metrics =====")
    for k, v in metrics.items():
        if k == "Sharpe":
            logging.info("%s: %.2f", k, v)
        else:
            logging.info("%s: %.2%%", k, v * 100)

    if plot:
        plot_equity_curve(bt_df)


if __name__ == "__main__":
    main()
