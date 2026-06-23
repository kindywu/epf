"""Rolling-window backtest — the official evaluation pipeline.

Fixes the Train/Test distribution shift problem (policy change 2026-01-01)
by retraining daily on a sliding window instead of a single fixed split.

Usage:
    # Standalone: compare window sizes
    python rolling_backtest.py

    # From backtest.py: evaluate after model training
    from rolling_backtest import evaluate
    evaluate(df, window_days=365)
"""

import sys
import pandas as pd
import numpy as np
import lightgbm as lgb

from data_utils import load_data, add_features, build_feature_sets, compute_pnl
from config import (
    DA_FEATURES, TARGET_DA, TARGET_RT, TARGET_SPREAD, REGIMES,
)

LOG_FILE = open("data/rolling_backtest.log", "w", buffering=1)


def log(msg=""):
    print(msg, flush=True)
    LOG_FILE.write(msg + "\n")
    LOG_FILE.flush()


# ── Model params (same as backtest.py) ──
LGB_DA = dict(
    n_estimators=1000, learning_rate=0.03, num_leaves=255,
    min_child_samples=20, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
)
LGB_RT = dict(
    n_estimators=2000, learning_rate=0.015, num_leaves=127,
    min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1,
)


def train_model(X_tr, y_tr, X_val, y_val, params, alpha=0.5, sample_weight=None):
    p = {**params, "objective": "quantile", "alpha": alpha}
    m = lgb.LGBMRegressor(**p)
    kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)], **kw)
    return m


def evaluate(df, window_days=365):
    """Run rolling-window backtest and print full PnL report.

    Returns dict with all metrics for programmatic use.
    """
    all_dates = sorted(df["trade_date"].unique())
    policy_break = pd.Timestamp("2026-01-01")
    test_dates = [d for d in all_dates if d >= policy_break]

    log(f"滚动窗口回测 (window={window_days}d)")
    log(f"测试期: {test_dates[0].date()} ~ {test_dates[-1].date()}  ({len(test_dates)} 天)")
    log()

    rt_feats, _ = build_feature_sets()

    # ── Run rolling window ──
    records = []  # (test_date, pred_spread[96], actual_spread[96], actual_rt[96], q_da[96], ...)
    daily_pnl = []
    daily_dir_acc = []

    for i, test_date in enumerate(test_dates):
        test_idx = all_dates.index(test_date)
        train_start = max(0, test_idx - window_days)
        train_end = test_idx - 1
        train_dates = all_dates[train_start:train_end + 1]
        val_dates = train_dates[-min(7, len(train_dates) // 5):]

        tr_mask = df["trade_date"].isin(train_dates)
        val_mask = df["trade_date"].isin(val_dates)
        test_mask = df["trade_date"] == test_date

        if tr_mask.sum() < 500:
            continue

        # ── Weights ──
        spread_w = np.clip(np.abs(df.loc[tr_mask, TARGET_SPREAD].values), 0, 200)

        # ── DA model ──
        da_model = train_model(
            df.loc[tr_mask, DA_FEATURES], df.loc[tr_mask, TARGET_DA],
            df.loc[val_mask, DA_FEATURES], df.loc[val_mask, TARGET_DA],
            LGB_DA, 0.5,
        )

        # ── RT model ──
        pred_da_tr = da_model.predict(df.loc[tr_mask, DA_FEATURES])
        X_tr_rt = df.loc[tr_mask, rt_feats].copy()
        X_tr_rt["pred_da"] = pred_da_tr
        pred_da_val = da_model.predict(df.loc[val_mask, DA_FEATURES])
        X_val_rt = df.loc[val_mask, rt_feats].copy()
        X_val_rt["pred_da"] = pred_da_val

        rt_model = train_model(
            X_tr_rt, df.loc[tr_mask, TARGET_RT],
            X_val_rt, df.loc[val_mask, TARGET_RT],
            LGB_RT, 0.5, sample_weight=spread_w,
        )

        # ── Predict ──
        test_pred_da = da_model.predict(df.loc[test_mask, DA_FEATURES])
        X_test_rt = df.loc[test_mask, rt_feats].copy()
        X_test_rt["pred_da"] = test_pred_da
        test_pred_rt = rt_model.predict(X_test_rt)
        test_pred_spread = test_pred_da - test_pred_rt

        q_da = (test_pred_spread < 0).astype(float)
        actual_spread = df.loc[test_mask, TARGET_SPREAD].values
        actual_rt = df.loc[test_mask, TARGET_RT].values
        actual_da = df.loc[test_mask, TARGET_DA].values

        pnl = compute_pnl(actual_spread, actual_rt, q_da)
        daily_pnl.append(pnl)

        nonzero = np.abs(actual_spread) > 1e-6
        if nonzero.any():
            daily_dir_acc.append(
                (np.sign(test_pred_spread[nonzero]) == np.sign(actual_spread[nonzero])).mean()
            )

        # Store per-period records for breakdowns
        for p in range(len(q_da)):
            records.append({
                "trade_date": test_date,
                "period": p + 1,
                "pred_da": test_pred_da[p],
                "pred_rt": test_pred_rt[p],
                "pred_spread": test_pred_spread[p],
                "actual_da": actual_da[p],
                "actual_rt": actual_rt[p],
                "actual_spread": actual_spread[p],
                "q_da": q_da[p],
                "is_ne_high_gen": df.loc[test_mask, "is_ne_high_gen"].values[p],
                "is_load_peak": df.loc[test_mask, "is_load_peak"].values[p],
                "is_grid_other": df.loc[test_mask, "is_grid_other"].values[p],
            })

        if (i + 1) % 20 == 0:
            cum = sum(daily_pnl)
            n = len(daily_pnl)
            log(f"  [{i+1:>3}/{len(test_dates)}] {test_date.date()}  "
                f"cum_PnL={cum:>8.0f}  avg={cum/(n*96):.2f}/MWh  "
                f"dir={np.mean(daily_dir_acc[-20:])*100:.1f}%")

    rec_df = pd.DataFrame(records)
    n_days = len(daily_pnl)
    n_periods = n_days * 96
    total_pnl = sum(daily_pnl)
    pnl_per_mwh = total_pnl / n_periods

    # ── Oracle ──
    oracle_q_da = (rec_df["actual_spread"].values < 0).astype(float)
    oracle_pnl = compute_pnl(rec_df["actual_spread"].values, rec_df["actual_rt"].values, oracle_q_da)
    oracle_per_mwh = oracle_pnl / n_periods

    # ══════════════════════════════════════════════════════════════════
    # Report
    # ══════════════════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log(f"  滚动窗口回测结果 (window={window_days}d)")
    log(f"{'='*70}")
    log(f"  测试天数: {n_days}  测试时段: {n_periods:,}")
    log(f"  平均 DA: {rec_df['actual_da'].mean():.0f}  平均 RT: {rec_df['actual_rt'].mean():.0f}  "
        f"平均 |spread|: {np.abs(rec_df['actual_spread']).mean():.0f}")

    log(f"\n  {'策略':<28} {'总PnL':>10} {'元/MWh':>8} {'元/天':>8} {'方向Acc':>8} {'捕获率':>8}")
    log(f"  {'-'*28} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    log(f"  {'Always DA (baseline)':<28} {'0':>10} {'0.00':>8} {'0':>8} {'—':>8} {'0%':>8}")

    dir_acc = np.mean(daily_dir_acc) * 100 if daily_dir_acc else 0
    log(f"  {'Dual-Head (滚动)':<28} {total_pnl:>10.0f} {pnl_per_mwh:>8.2f} "
        f"{total_pnl/n_days:>8.0f} {dir_acc:>7.1f}% {pnl_per_mwh/oracle_per_mwh*100:>7.1f}%")
    log(f"  {'Oracle (理论上限)':<28} {oracle_pnl:>10.0f} {oracle_per_mwh:>8.2f} "
        f"{oracle_pnl/n_days:>8.0f} {'—':>8} {'100%':>8}")

    # ── PnL by |spread| magnitude ──
    log(f"\n  {'─'*60}")
    log(f"  PnL 按 |Spread| 分段")
    log(f"  {'|Spread|':<12} {'n':>6} {'PnL':>10} {'PnL/period':>10} {'方向Acc':>8} {'占总PnL':>10}")
    log(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    abs_spread = np.abs(rec_df["actual_spread"].values)
    nonzero = abs_spread > 1e-6
    for lo, hi in [(0, 20), (20, 50), (50, 100), (100, 200), (200, 999)]:
        mask = (abs_spread > lo) & (abs_spread <= hi)
        n = mask.sum()
        if n == 0:
            continue
        pnl_mag = compute_pnl(
            rec_df.loc[mask, "actual_spread"].values,
            rec_df.loc[mask, "actual_rt"].values,
            rec_df.loc[mask, "q_da"].values,
        )
        pred_sign = np.sign(rec_df.loc[mask, "pred_spread"].values)
        actual_sign = np.sign(rec_df.loc[mask, "actual_spread"].values)
        nm = nonzero[mask]
        dir_mag = (pred_sign[nm] == actual_sign[nm]).mean() * 100 if nm.sum() > 0 else 0
        pct = pnl_mag / oracle_pnl * 100 if oracle_pnl != 0 else 0
        log(f"  {f'{lo}-{hi}':<12} {n:>6} {pnl_mag:>10.0f} {pnl_mag/n:>10.2f} {dir_mag:>7.1f}% {pct:>9.1f}%")

    # ── Per-regime ──
    log(f"\n  {'─'*60}")
    log(f"  PnL 按 Regime 分段")
    log(f"  {'Regime':<14} {'n':>6} {'PnL/period':>10}")
    log(f"  {'-'*14} {'-'*6} {'-'*10}")
    for rname, rcol in REGIMES.items():
        rmask = rec_df[rcol].values.astype(bool)
        n_r = rmask.sum()
        if n_r == 0:
            continue
        r_pnl = compute_pnl(
            rec_df.loc[rmask, "actual_spread"].values,
            rec_df.loc[rmask, "actual_rt"].values,
            rec_df.loc[rmask, "q_da"].values,
        )
        log(f"  {rname:<14} {n_r:>6} {r_pnl/n_r:>10.2f}")

    # ── Daily PnL distribution ──
    daily_pnl_arr = np.array(daily_pnl)
    log(f"\n  {'─'*60}")
    log(f"  日 PnL 分布")
    log(f"  均值: {daily_pnl_arr.mean():.0f} 元/天")
    log(f"  标准差: {daily_pnl_arr.std():.0f} 元/天")
    log(f"  正收益日: {(daily_pnl_arr > 0).mean()*100:.0f}%")
    log(f"  最佳日: {daily_pnl_arr.max():.0f} 元")
    log(f"  最差日: {daily_pnl_arr.min():.0f} 元")

    # ── Monetary value ──
    log(f"\n  {'─'*60}")
    log(f"  价值换算 (基于 {pnl_per_mwh:.2f} 元/MWh)")
    log(f"  每 MWh: {pnl_per_mwh:.2f} 元")
    log(f"  每 MW/天: {pnl_per_mwh*24:.0f} 元")
    log(f"  每 MW/年: {pnl_per_mwh*24*365:.0f} 元")
    for mw, lbl in [(100, "100MW"), (300, "300MW"), (1000, "1000MW (1GW)")]:
        day_profit = pnl_per_mwh * 24 * mw
        yr_profit = day_profit * 365
        log(f"  {lbl}: {day_profit:,.0f} 元/天 = {yr_profit/1e4:,.0f} 万/年")

    log(f"\n  Oracle 上限: {oracle_per_mwh:.2f} 元/MWh")
    for mw in [100, 300, 1000]:
        log(f"  {mw}MW: {oracle_per_mwh*24*mw*365/1e4:,.0f} 万/年")

    return {
        "window_days": window_days,
        "n_days": n_days,
        "total_pnl": total_pnl,
        "pnl_per_mwh": pnl_per_mwh,
        "pnl_per_day": total_pnl / n_days,
        "dir_acc": dir_acc,
        "oracle_per_mwh": oracle_per_mwh,
        "capture_rate": pnl_per_mwh / oracle_pnl * n_periods if oracle_pnl else 0,
        "daily_pnl": daily_pnl,
    }


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_data()
    df = add_features(df)

    all_dates = sorted(df["trade_date"].unique())
    policy_break = pd.Timestamp("2026-01-01")
    test_dates = [d for d in all_dates if d >= policy_break]

    log(f"滚动窗口回测: {len(test_dates)} 天 ({test_dates[0].date()} ~ {test_dates[-1].date()})")
    log(f"测试期全部在新政策下 (2026-01-01 后)")
    log()

    configs = [
        (90, "90d 窗口"),
        (180, "180d 窗口"),
        (365, "365d 窗口"),
        (545, "545d 窗口"),
    ]

    results = []
    for window, label in configs:
        log(f"\n{'='*60}")
        log(f"  {label}")
        log(f"{'='*60}")
        r = evaluate(df, window_days=window)
        r["label"] = label
        results.append(r)

    # ── Comparison ──
    log(f"\n{'='*75}")
    log("  窗口大小对比")
    log(f"{'='*75}")
    log(f"{'配置':<20} {'天数':>5} {'总PnL':>10} {'元/MWh':>8} {'元/天':>8} {'方向Acc':>8} {'捕获率':>8}")
    log(f"{'-'*20} {'-'*5} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        log(f"{r['label']:<20} {r['n_days']:>5} {r['total_pnl']:>10.0f} "
            f"{r['pnl_per_mwh']:>8.2f} {r['pnl_per_day']:>8.0f} "
            f"{r['dir_acc']:>7.1f}% {r['capture_rate']:>7.1f}%")

    LOG_FILE.close()
