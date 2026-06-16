"""Predict tomorrow's day-ahead prices — multi-quantile model.

Usage:
    python predict.py                          # predict next available day (demo)
    python predict.py --date 2026-06-15        # predict specific date
    python predict.py --train                  # retrain model first, then predict

Output:
    - Console: summary table with trading signals
    - predictions.csv: full 96-period forecast
"""

import argparse
import sys
import joblib
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

MODEL_FILE = "model_quantile.pkl"
QUANTILES = {0.1: "P10(卖参考)", 0.5: "P50(中位)", 0.9: "P90(买入报价)"}


def train_and_save():
    """Train multi-quantile model on ALL historical data and save to disk."""
    print("Loading data...")
    df = pd.read_excel("data/day_ahead_feature_matrix.xlsx")
    df = df.sort_values(["trade_date", "period"]).reset_index(drop=True)

    target = "price_day_ahead"
    X = df[TOP30]
    y = df[target]

    print(f"Training on {len(X)} rows ({df['trade_date'].min().date()} ~ {df['trade_date'].max().date()})...")

    models = {}
    for alpha, name in QUANTILES.items():
        print(f"  Training {name} (q={alpha})...")
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=alpha,
            n_estimators=1000, learning_rate=0.03, num_leaves=255,
            min_child_samples=20, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1,
        )
        m.fit(X, y, callbacks=[lgb.log_evaluation(0)])
        models[alpha] = m

    joblib.dump(models, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")


def load_model():
    """Load pre-trained model."""
    try:
        return joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print(f"Model not found at {MODEL_FILE}. Run with --train first.")
        sys.exit(1)


def predict(models, features_df):
    """Run prediction and return DataFrame with results."""
    X = features_df[TOP30]
    results = features_df[["trade_date", "period", "hour", "minute_slot"]].copy()

    for alpha, name in QUANTILES.items():
        results[name] = models[alpha].predict(X)

    # Add time label HH:MM
    base = pd.Timedelta(minutes=15)
    results["time"] = results.apply(
        lambda r: f"{int(r['hour']):02d}:{int(r['minute_slot'] * 15):02d}", axis=1
    )

    return results


def print_summary(pred_df, long_term_cost=None):
    """Print a trading decision summary."""
    print()
    print("=" * 80)
    print(f"  日 前 预 测  +  交 易 建 议    {pred_df['trade_date'].iloc[0].date()}")
    print("=" * 80)

    # Hourly summary
    hourly = pred_df.groupby("hour").agg(
        **{"P10(卖参考)": ("P10(卖参考)", "mean"),
           "P50(中位)": ("P50(中位)", "mean"),
           "P90(买入报价)": ("P90(买入报价)", "mean")}
    ).round(0).astype(int)

    print(f"\n  {'Hour':<6} {'P10(卖参考)':>12} {'P50(中位)':>12} {'P90(买入报价)':>12}")
    print(f"  {'-'*5}  {'-'*12} {'-'*12} {'-'*12}")
    for h, row in hourly.iterrows():
        print(f"  {int(h):02d}:00  {row['P10(卖参考)']:>10}   {row['P50(中位)']:>10}   {row['P90(买入报价)']:>10}")

    # Trading signals
    print(f"\n  ── 交易建议 ──")
    avg_p50 = pred_df["P50(中位)"].mean()
    avg_p90 = pred_df["P90(买入报价)"].mean()
    min_p90 = pred_df["P90(买入报价)"].min()
    max_p50 = pred_df["P50(中位)"].max()

    print(f"  买入报价: 建议报 P90 (覆盖率 95%), 全天均值 {avg_p90:.0f} 元/MWh")
    print(f"  价格走势: 全天 P50 均值 {avg_p50:.0f} 元/MWh, P50 峰值 {max_p50:.0f}")
    print(f"  最低报价: P90 全天最低 {min_p90:.0f} (此时段竞争最激烈)")

    if long_term_cost:
        sell_periods = pred_df[pred_df["P50(中位)"] > long_term_cost * 1.1]
        print(f"\n  卖出机会: {len(sell_periods)} 个时段 P50 > 长协成本 {long_term_cost}×1.1")
        if len(sell_periods) > 0:
            print(f"    时段: {sell_periods['time'].iloc[0]} ~ {sell_periods['time'].iloc[-1]}")
            print(f"    预期套利: P50 均值 {sell_periods['P50(中位)'].mean():.0f} - 长协 {long_term_cost} "
                  f"= {sell_periods['P50(中位)'].mean()-long_term_cost:.0f} 元/MWh")

    print(f"\n  (详细 96 时段数据已保存到 predictions.csv)")

    # Peak/valley summary
    peak = pred_df.loc[pred_df["P90(买入报价)"].idxmax()]
    valley = pred_df.loc[pred_df["P90(买入报价)"].idxmin()]
    print(f"  价格最高时段: {peak['time']}  P90={peak['P90(买入报价)']:.0f}")
    print(f"  价格最低时段: {valley['time']}  P90={valley['P90(买入报价)']:.0f}")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day-ahead electricity price prediction")
    parser.add_argument("--train", action="store_true", help="Retrain model before predicting")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD). Default: last available")
    parser.add_argument("--lt-cost", type=float, default=None,
                        help="Your long-term contract cost for sell signals")
    args = parser.parse_args()

    # Train or load
    if args.train:
        train_and_save()
    models = load_model()

    # Load features for target date
    df = pd.read_excel("data/day_ahead_feature_matrix.xlsx")
    df = df.sort_values(["trade_date", "period"]).reset_index(drop=True)

    if args.date:
        target_date = pd.Timestamp(args.date)
        target_data = df[df["trade_date"] == target_date]
        if len(target_data) == 0:
            print(f"No data for {args.date}. Available: {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}")
            sys.exit(1)
    else:
        # Default: use last complete day as demo
        last_date = df["trade_date"].max()
        target_data = df[df["trade_date"] == last_date]
        target_date = last_date
        print(f"[Demo mode] Predicting for last available date: {target_date.date()}")

    predictions = predict(models, target_data)
    print_summary(predictions, long_term_cost=args.lt_cost)

    # Save full results
    out_cols = ["trade_date", "period", "time"] + list(QUANTILES.values())
    predictions[out_cols].to_csv("predictions.csv", index=False, float_format="%.1f")
    print(f"\nFull 96-period predictions saved to predictions.csv")
