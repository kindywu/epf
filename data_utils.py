"""Shared data loading, feature engineering, and PnL computation.

Used by backtest.py (model training) and rolling_backtest.py (evaluation).
"""

import pandas as pd
import numpy as np
from config import (
    TARGET_DA, TARGET_RT, TARGET_SPREAD, RT_FEATURES,
)


def load_data():
    df = pd.read_excel("data/day_ahead_feature_matrix.xlsx")
    df = df.sort_values(["trade_date", "period"]).reset_index(drop=True)
    return df


def add_features(df):
    """Add engineered features (extra lags, price buckets, interactions)."""
    df = df.copy()

    # Extra lags
    rt_pivot = df.pivot(index="trade_date", columns="period", values=TARGET_RT)
    spread_pivot = df.pivot(index="trade_date", columns="period", values=TARGET_SPREAD)

    for lag_days, label in [(2, "2d"), (3, "3d"), (7, "7d")]:
        rt_lag = np.full(len(df), np.nan)
        sp_lag = np.full(len(df), np.nan)
        for i, row in df.iterrows():
            date_idx = rt_pivot.index.get_loc(row["trade_date"])
            if date_idx >= lag_days:
                lag_date = rt_pivot.index[date_idx - lag_days]
                p = row["period"]
                rt_lag[i] = rt_pivot.loc[lag_date, p]
                sp_lag[i] = spread_pivot.loc[lag_date, p]
        df[f"rt_lag_{label}"] = rt_lag
        df[f"spread_lag_{label}"] = sp_lag

    rt_std = np.full(len(df), np.nan)
    for i, row in df.iterrows():
        date_idx = rt_pivot.index.get_loc(row["trade_date"])
        if date_idx >= 7:
            window = rt_pivot.iloc[max(0, date_idx - 7):date_idx]
            rt_std[i] = window[row["period"]].std()
    df["rt_roll_7d_std"] = rt_std

    # D-1 price buckets
    df["lag_da_floor"] = (df["price_lag_1d"] < 40).astype(int)
    df["lag_da_low"] = ((df["price_lag_1d"] >= 40) & (df["price_lag_1d"] < 100)).astype(int)
    df["lag_da_mid"] = ((df["price_lag_1d"] >= 100) & (df["price_lag_1d"] < 300)).astype(int)
    df["lag_da_high"] = (df["price_lag_1d"] >= 300).astype(int)
    df["lag_spread_sign"] = np.sign(df["spread_lag_1d"])

    # Interactions
    df["ne_high_x_lag_da_floor"] = df["is_ne_high_gen"] * df["lag_da_floor"]
    df["ne_high_x_lag_da_low"] = df["is_ne_high_gen"] * df["lag_da_low"]

    return df


def build_feature_sets():
    """Return feature lists for RT and spread models."""
    EXTRA_LAGS = ["rt_lag_2d", "rt_lag_3d", "rt_lag_7d", "rt_roll_7d_std",
                  "spread_lag_2d", "spread_lag_3d", "spread_lag_7d"]
    PRICE_BUCKETS = ["lag_da_floor", "lag_da_low", "lag_da_mid", "lag_da_high"]
    LAG_SIGN = ["lag_spread_sign"]
    INTERACTIONS = ["ne_high_x_lag_da_floor", "ne_high_x_lag_da_low"]

    rt_features = RT_FEATURES + EXTRA_LAGS + PRICE_BUCKETS + LAG_SIGN + INTERACTIONS
    spread_features = list(dict.fromkeys(rt_features))  # dedup

    return rt_features, spread_features


def compute_pnl(actual_spread, actual_rt, q_da):
    """Savings vs always-DA. Positive = strategy saves money."""
    cost_naive = actual_rt + actual_spread  # Q_da=1 → P_da
    cost_model = actual_rt + q_da * actual_spread
    return (cost_naive - cost_model).sum()
