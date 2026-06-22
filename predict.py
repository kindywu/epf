"""Predict day-ahead & real-time prices — hybrid spread-direction strategy.

Hybrid strategy:
  - 新能源大发时段: classifier → 77% direction accuracy
  - 其他时段: dual-head spread = pred_da − pred_rt → 54% direction accuracy

Usage:
    python predict.py                          # predict last available date (demo)
    python predict.py --date 2026-06-20        # predict specific date
    python predict.py --backtest               # run full backtest (trains + evaluates)

Output:
    - Console: trading summary with DA/RT/spread/direction
    - predictions.csv: 96-period forecast
"""

import argparse
import sys
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb

from config import (
    DA_FEATURES, RT_FEATURES, RT_LAG_FEATURES,
    TARGET_DA, TARGET_RT, TARGET_SPREAD,
    MODEL_FILE_DA, MODEL_FILE_RT,
)

MODEL_FILE_CLF = "model_clf_ne.pkl"
CLF_THRESHOLD = 20


def load_models():
    """Load all three models."""
    models = {}
    for name, path in [("da", MODEL_FILE_DA), ("rt", MODEL_FILE_RT), ("clf", MODEL_FILE_CLF)]:
        try:
            models[name] = joblib.load(path)
        except FileNotFoundError:
            print(f"Model not found: {path}. Run backtest first: python backtest.py")
            sys.exit(1)
    return models["da"], models["rt"], models["clf"]


def predict_hybrid(da_model, rt_model, clf_model, features_df, rt_feats, spread_feats):
    """Run hybrid prediction.

    Uses classifier for 新能源大发 periods, dual-head for others.
    """
    X_da = features_df[DA_FEATURES]
    X_rt = features_df[rt_feats]
    X_clf = features_df[spread_feats]

    results = features_df[["trade_date", "period", "hour", "minute_slot"]].copy()

    # DA P50
    results["DA_P50"] = da_model.predict(X_da)

    # RT P50
    results["RT_P50"] = rt_model.predict(X_rt)

    # Spread from dual-head
    results["Spread_Dual"] = results["DA_P50"] - results["RT_P50"]

    # Classifier probability
    results["Clf_Prob"] = clf_model.predict_proba(X_clf)[:, 1]

    # Hybrid direction
    is_ne = features_df["is_ne_high_gen"].values.astype(bool)

    # In 新能源大发: classifier — clf_prob > 0.5 → spread > 20 → DA >> RT → 少报
    # In other: dual-head spread sign
    results["Pred_Sign"] = np.where(
        is_ne,
        np.where(results["Clf_Prob"] > 0.5, 1, -1),  # 1 = DA>RT (少报), -1 = DA<RT (多报)
        np.sign(results["Spread_Dual"]),
    )

    # Confidence: clf_prob for NE, |spread| magnitude for others
    results["Confidence"] = np.where(
        is_ne,
        np.maximum(results["Clf_Prob"], 1 - results["Clf_Prob"]),
        np.abs(results["Spread_Dual"]).clip(upper=100) / 100,
    )

    # Direction label
    results["Direction"] = np.where(
        results["Pred_Sign"] > 0.5, "📕 日前高(少报)",
        np.where(results["Pred_Sign"] < -0.5, "📗 日前低(多报)", "📙 价差小")
    )

    # Strategy Q_da: 1 = 多报 (buy DA), 0 = 少报 (buy RT)
    results["Q_da"] = (results["Pred_Sign"] < 0).astype(int)

    # Time label
    results["time"] = results.apply(
        lambda r: f"{int(r['hour']):02d}:{int(r['minute_slot'] * 15):02d}", axis=1
    )

    return results


def print_summary(pred_df):
    """Print trading summary."""
    trade_date = pred_df["trade_date"].iloc[0].date()
    n_buy = (pred_df["Q_da"] == 1).sum()
    n_sell = (pred_df["Q_da"] == 0).sum()
    n_ne_gen = pred_df["is_ne_high_gen"].sum() if "is_ne_high_gen" in pred_df.columns else 0

    print()
    print("=" * 80)
    print(f"  混 合 策 略 预 测    {trade_date}")
    print("=" * 80)

    print(f"\n  日前均价: {pred_df['DA_P50'].mean():.0f}  实时预期: {pred_df['RT_P50'].mean():.0f}  "
          f"价差: {(pred_df['DA_P50'] - pred_df['RT_P50']).mean():+.0f}")
    print(f"  多报(买日前): {n_buy} 时段  少报(等实时): {n_sell} 时段")
    if n_ne_gen > 0:
        print(f"  新能源大发段: {n_ne_gen} 时段 (使用分类器, 准确率 ~77%)")

    # Hourly summary
    hourly = pred_df.groupby("hour").agg(
        DA_P50=("DA_P50", "mean"),
        RT_P50=("RT_P50", "mean"),
        Spread=("Spread_Dual", "mean"),
        Direction=("Direction", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ""),
        Confidence=("Confidence", "mean"),
    ).round(0)

    print(f"\n  {'Hour':<6} {'DA':>8} {'RT':>8} {'Spread':>8} {'Conf':>6}  Direction")
    print(f"  {'-'*5}  {'-'*8} {'-'*8} {'-'*8} {'-'*6}  {'-'*20}")
    for h, row in hourly.iterrows():
        print(f"  {int(h):02d}:00  {row['DA_P50']:8.0f} {row['RT_P50']:8.0f} "
              f"{row['Spread']:8.0f} {row['Confidence']:5.0%}  {row['Direction']}")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid electricity price prediction")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD). Default: last available")
    parser.add_argument("--backtest", action="store_true", help="Run full backtest instead")
    args = parser.parse_args()

    if args.backtest:
        import subprocess
        subprocess.run([sys.executable, "backtest.py"])
        sys.exit(0)

    # Load models
    da_model, rt_model, clf_model = load_models()

    # Load data and build features
    from backtest import add_features, build_feature_sets
    df = pd.read_excel("data/day_ahead_feature_matrix.xlsx")
    df = df.sort_values(["trade_date", "period"]).reset_index(drop=True)
    df = add_features(df)
    rt_feats, spread_feats = build_feature_sets()

    # Select target date
    if args.date:
        target_date = pd.Timestamp(args.date)
        target_data = df[df["trade_date"] == target_date]
        if len(target_data) == 0:
            print(f"No data for {args.date}. "
                  f"Available: {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}")
            sys.exit(1)
    else:
        last_date = df["trade_date"].max()
        target_data = df[df["trade_date"] == last_date]
        target_date = last_date
        print(f"[Demo mode] Last available date: {target_date.date()}")

    predictions = predict_hybrid(da_model, rt_model, clf_model, target_data, rt_feats, spread_feats)
    print_summary(predictions)

    # Save
    out_cols = ["trade_date", "period", "time", "DA_P50", "RT_P50",
                "Spread_Dual", "Clf_Prob", "Direction", "Confidence", "Q_da"]
    predictions[out_cols].to_csv("predictions.csv", index=False, float_format="%.2f")
    print(f"\nFull 96-period predictions saved to predictions.csv")
