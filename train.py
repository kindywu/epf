"""Electricity price forecasting — multi-quantile model for day-ahead trading.

Three-line output supports buy/sell decisions under market constraints:
  - P10: sell reference  (10% prob actual is below this → 90% prob it's higher)
  - P50: neutral forecast (median expected price)
  - P90: buy bid price    (90% prob actual is below this → safe to bid here)

Trading rules:
  - Buy: bid at P90, quantity ≤ 120% registered capacity
  - Sell: if P50 > long-term cost + margin, sell from T-2 inventory
"""

import pandas as pd
import numpy as np
import lightgbm as lgb

TOP30 = [
    "day_of_year", "tie_total", "supply_gap", "ne_wind",
    "days_in_solar_term", "price_lag_1d", "hydro_fcst", "tie_新疆",
    "price_lag_7d", "coal_x_thermal_fcst", "dow", "thermal_fcst",
    "price_lag_2d", "solar_term_sin", "tie_湖南", "gen_fcst",
    "wx_temp_2m", "price_lag_3d", "tie_青海", "coal_x_net_load",
    "price_roll_30d_mean", "tie_宁夏", "price_lag_1d_dev_roll7",
    "tie_陕西", "tie_山东", "wx_wind_100m", "load_fcst_x_wx_temp",
    "price_roll_7d_mean", "days_from_holiday", "hydro_ratio",
]

QUANTILES = [0.1, 0.5, 0.9]  # sell-ref, neutral, buy-bid


def load_data():
    df = pd.read_excel("data/day_ahead_feature_matrix.xlsx")
    df = df.sort_values(["trade_date", "period"]).reset_index(drop=True)
    return df


def time_split(df):
    dates = sorted(df["trade_date"].unique())
    n = len(dates)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    masks = {}
    for name, d in [("train", dates[:train_end]),
                     ("val",   dates[train_end:val_end]),
                     ("test",  dates[val_end:])]:
        masks[name] = df["trade_date"].isin(d)
    return masks


def train_models(X_train, y_train, X_val, y_val, quantiles):
    models = {}
    for alpha in quantiles:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=alpha,
            n_estimators=1000, learning_rate=0.03, num_leaves=255,
            min_child_samples=20, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1,
        )
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], eval_metric="quantile",
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        models[alpha] = m
    return models


def evaluate(models, X_test, y_test):
    """Evaluate multi-quantile model with buyer/seller metrics."""
    preds = {alpha: m.predict(X_test) for alpha, m in models.items()}

    for alpha, name, role in [(0.1, "P10", "卖参考"),
                               (0.5, "P50", "中性"),
                               (0.9, "P90", "买入报价")]:
        p = preds[alpha]
        cov = (p >= y_test).mean() * 100
        over = (p - y_test).clip(lower=0).mean()
        under = (y_test - p).clip(lower=0).mean()
        mae = np.abs(y_test - p).mean()
        print(f"  {name} ({role}): MAE={mae:.1f}  Coverage={cov:.1f}%  "
              f"AvgOver={over:.0f}  AvgUnder={under:.0f}")

    # Sell signal quality: P10 > threshold means "price likely high, consider selling"
    print()
    high_mask = y_test > 300
    print(f"  Test 高价时段(>300): {high_mask.sum()} 次 ({high_mask.sum()/len(y_test)*100:.1f}%)")
    for threshold in [250, 300, 350]:
        sell_signal = preds[0.5] > threshold
        if sell_signal.sum() > 0:
            prec = (sell_signal & high_mask).sum() / sell_signal.sum() * 100
        else:
            prec = 0
        rec = (sell_signal & high_mask).sum() / high_mask.sum() * 100
        print(f"  P50>{threshold} → 卖出信号 {sell_signal.sum()} 次, "
              f"精确率={prec:.1f}%, 召回率={rec:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_data()
    masks = time_split(df)
    target = "price_day_ahead"

    X_train = df.loc[masks["train"], TOP30]
    y_train = df.loc[masks["train"], target]
    X_val   = df.loc[masks["val"],   TOP30]
    y_val   = df.loc[masks["val"],   target]
    X_test  = df.loc[masks["test"],  TOP30]
    y_test  = df.loc[masks["test"],  target]

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"Train y mean={y_train.mean():.0f}  Val={y_val.mean():.0f}  Test={y_test.mean():.0f}")

    models = train_models(X_train, y_train, X_val, y_val, QUANTILES)

    print("\n── Test 评估 ──")
    evaluate(models, X_test, y_test)

    # Demo: single day
    print("\n── 单日示例 ──")
    demo_day = pd.Timestamp("2026-04-01")
    demo_mask = masks["test"] & (df["trade_date"] == demo_day)
    demo_X = df.loc[demo_mask, TOP30]
    demo_y = df.loc[demo_mask, target]
    print(f"  {demo_day.date()}  ({len(demo_y)} periods)")
    print(f"  {'Period':<8} {'P10(卖)':>8} {'P50(中)':>8} {'P90(买)':>8} {'实际':>8}")
    for i in range(min(8, len(demo_y))):
        preds = {a: models[a].predict(demo_X.iloc[[i]])[0] for a in QUANTILES}
        print(f"  {i+1:<8} {preds[0.1]:8.0f} {preds[0.5]:8.0f} {preds[0.9]:8.0f} {demo_y.iloc[i]:8.0f}")
