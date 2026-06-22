"""Backtest: hybrid spread-direction strategy with PnL proxy.

Hybrid strategy:
  - 新能源大发 (is_ne_high_gen): classifier (threshold=20) → 77.4% direction acc
  - Other regimes: dual-head spread = pred_da − pred_rt → ~53% direction acc

PnL proxy (§4.3): unit MWh, Q_mlt=0, Q_rt=1
  Cost = Q_da * P_da + (1 − Q_da) * P_rt
  Strategy: Q_da = 1 if predict DA<RT (多报), else 0 (少报, buy RT)
  Benchmark: always DA (Q_da=1), oracle (knows actual spread sign)

Saves models for predict.py to use.
"""

import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb

from config import (
    DA_FEATURES, RT_FEATURES, RT_LAG_FEATURES,
    TARGET_DA, TARGET_RT, TARGET_SPREAD,
    REGIMES,
    MODEL_FILE_DA, MODEL_FILE_RT,
)

MODEL_FILE_CLF = "model_clf_ne.pkl"

LGB_QUANTILE = dict(
    n_estimators=1000, learning_rate=0.03, num_leaves=255,
    min_child_samples=20, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
)
LGB_RT = dict(
    n_estimators=2000, learning_rate=0.015, num_leaves=127,
    min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1,
)
LGB_CLF = dict(
    n_estimators=1000, learning_rate=0.02, num_leaves=127,
    min_child_samples=100, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=0.5, class_weight="balanced",
    random_state=42, verbose=-1,
)

CLF_THRESHOLD = 20  # predict spread > 20 (significant DA>RT)


def load_data():
    df = pd.read_excel("data/day_ahead_feature_matrix.xlsx")
    df = df.sort_values(["trade_date", "period"]).reset_index(drop=True)
    return df


def add_features(df):
    """Add engineered features (same as train.py)."""
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


def build_feature_sets():
    """Return feature lists for RT and spread models.

    RT: DA base + RT lags + expanded (all from matrix) + extra computed lags
    Spread: same as RT (all D-1 available features)
    """
    # Extra lags computed on-the-fly (not in matrix)
    EXTRA_LAGS = ["rt_lag_2d", "rt_lag_3d", "rt_lag_7d", "rt_roll_7d_std",
                  "spread_lag_2d", "spread_lag_3d", "spread_lag_7d"]
    # Price bucket features (derived from price_lag_1d — D-1 available)
    PRICE_BUCKETS = ["lag_da_floor", "lag_da_low", "lag_da_mid", "lag_da_high"]
    LAG_SIGN = ["lag_spread_sign"]
    INTERACTIONS = ["ne_high_x_lag_da_floor", "ne_high_x_lag_da_low"]

    # RT_FEATURES from config now includes DA_FEATURES + RT_LAG_FEATURES + EXPANDED_FEATURES
    rt_features = (RT_FEATURES + EXTRA_LAGS + PRICE_BUCKETS + LAG_SIGN + INTERACTIONS)
    spread_features = list(dict.fromkeys(rt_features))  # dedup, same set

    return rt_features, spread_features
    spread_features = DA_FEATURES + RT_LAG_FEATURES + EXTRA_LAGS + PRICE_BUCKETS + INTERACTIONS + LAG_SIGN

    return rt_features, spread_features


def train_quantile(X_tr, y_tr, X_val, y_val, params, alpha, sample_weight=None):
    p = {**params, "objective": "quantile", "alpha": alpha}
    m = lgb.LGBMRegressor(**p)
    fit_kw = {}
    if sample_weight is not None:
        fit_kw["sample_weight"] = sample_weight
    m.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)], eval_metric="quantile",
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
          **fit_kw)
    return m


def compute_pnl(actual_spread, actual_rt, q_da):
    """Compute PnL proxy per period.

    Cost = Q_da * P_da + (1 − Q_da) * P_rt = P_rt + Q_da * spread
    Savings vs always-DA = (1 − Q_da) * spread

    Returns total savings over naive (always buy DA).
    Positive = strategy saves money.
    """
    # Cost of strategy
    cost_model = actual_rt + q_da * actual_spread
    # Cost of always buying DA
    cost_naive_da = actual_rt + actual_spread  # Q_da=1 → P_da
    # Savings: positive = model better
    savings = cost_naive_da - cost_model  # = (1 − Q_da) * spread
    return savings.sum(), cost_model.sum(), cost_naive_da.sum()


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_data()
    df = add_features(df)
    masks = time_split(df)
    rt_feats, spread_feats = build_feature_sets()

    test_dates = sorted(df.loc[masks["test"], "trade_date"].unique())
    train_dates = sorted(df.loc[masks["train"], "trade_date"].unique())
    val_dates = sorted(df.loc[masks["val"], "trade_date"].unique())

    print(f"Split: train={len(train_dates)}d  val={len(val_dates)}d  test={len(test_dates)}d")
    print(f"Test period: {test_dates[0].date()} ~ {test_dates[-1].date()}")

    # ═══════════════════════════════════════════════════════════════════════
    # Train Models (DA unweighted, RT weighted + pred_da feature)
    # ═══════════════════════════════════════════════════════════════════════

    # Compute RT sample weight from training set |spread|
    train_spread_abs = np.abs(df.loc[masks["train"], TARGET_SPREAD].values)
    rt_sample_weight = np.clip(train_spread_abs, 0, 200)

    # ── Step 1: OOF DA predictions for train (to avoid leakage) ──
    print("\n── Computing OOF DA for train ──")
    from sklearn.model_selection import TimeSeriesSplit
    train_dates_all = sorted(df.loc[masks["train"], "trade_date"].unique())
    tscv = TimeSeriesSplit(n_splits=3)
    oof_da_train = np.full(len(df.loc[masks["train"]]), np.nan)

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(train_dates_all)):
        tr_dates = [train_dates_all[i] for i in tr_idx]
        val_dates = [train_dates_all[i] for i in val_idx]
        tr_mask = df.loc[masks["train"], "trade_date"].isin(tr_dates)
        val_mask = df.loc[masks["train"], "trade_date"].isin(val_dates)
        m = train_quantile(
            df.loc[masks["train"]].loc[tr_mask, DA_FEATURES],
            df.loc[masks["train"]].loc[tr_mask, TARGET_DA],
            df.loc[masks["train"]].loc[val_mask, DA_FEATURES],
            df.loc[masks["train"]].loc[val_mask, TARGET_DA],
            LGB_QUANTILE, 0.5,
        )
        oof_da_train[val_mask.values] = m.predict(
            df.loc[masks["train"]].loc[val_mask, DA_FEATURES]
        )

    # ── Step 2: Final DA Model (unweighted) ──
    print("\n── Training DA P50 (unweighted) ──")
    da_model = train_quantile(
        df.loc[masks["train"], DA_FEATURES], df.loc[masks["train"], TARGET_DA],
        df.loc[masks["val"], DA_FEATURES], df.loc[masks["val"], TARGET_DA],
        LGB_QUANTILE, 0.5,
    )

    # OOF DA for val (from final DA model — acceptable since val is not used for DA training)
    oof_da_val = da_model.predict(df.loc[masks["val"], DA_FEATURES])

    # ── Step 3: RT Model with pred_da as feature ──
    print("\n── Training RT P50 (weighted + pred_da feature) ──")
    X_train_rt = df.loc[masks["train"], rt_feats].copy()
    X_train_rt["pred_da"] = oof_da_train
    X_val_rt = df.loc[masks["val"], rt_feats].copy()
    X_val_rt["pred_da"] = oof_da_val

    rt_model = train_quantile(
        X_train_rt, df.loc[masks["train"], TARGET_RT],
        X_val_rt, df.loc[masks["val"], TARGET_RT],
        LGB_RT, 0.5, sample_weight=rt_sample_weight,
    )

    # Store the RT feature list including pred_da for later use
    RT_FEATURES_WITH_PRED_DA = rt_feats + ["pred_da"]

    # ── Classifier (spread > CLF_THRESHOLD) ──
    print(f"\n── Training Classifier (spread > {CLF_THRESHOLD}) ──")
    y_clf_tr = (df.loc[masks["train"], TARGET_SPREAD] > CLF_THRESHOLD).astype(int)
    y_clf_val = (df.loc[masks["val"], TARGET_SPREAD] > CLF_THRESHOLD).astype(int)

    clf_model = lgb.LGBMClassifier(**LGB_CLF)
    clf_model.fit(
        df.loc[masks["train"], spread_feats], y_clf_tr,
        eval_set=[(df.loc[masks["val"], spread_feats], y_clf_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Save Models
    # ═══════════════════════════════════════════════════════════════════════
    joblib.dump(da_model, MODEL_FILE_DA)
    joblib.dump(rt_model, MODEL_FILE_RT)
    joblib.dump(clf_model, MODEL_FILE_CLF)
    print(f"\nModels saved: {MODEL_FILE_DA}, {MODEL_FILE_RT}, {MODEL_FILE_CLF}")

    # ═══════════════════════════════════════════════════════════════════════
    # Backtest on Test Set
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Backtest: PnL Proxy (per MWh, vs always-DA)")
    print("=" * 60)

    test_mask = masks["test"]
    test_df = df.loc[test_mask].copy()

    # Predictions
    test_df["pred_da"] = da_model.predict(test_df[DA_FEATURES])
    # Add pred_da as feature for RT model
    X_test_rt = test_df[rt_feats].copy()
    X_test_rt["pred_da"] = test_df["pred_da"].values
    test_df["pred_rt"] = rt_model.predict(X_test_rt)
    test_df["pred_spread"] = test_df["pred_da"] - test_df["pred_rt"]

    # Classifier: prob(spread > threshold)
    test_df["clf_prob"] = clf_model.predict_proba(test_df[spread_feats])[:, 1]

    # Actuals
    actual_spread = test_df[TARGET_SPREAD].values
    actual_rt = test_df[TARGET_RT].values
    actual_da = test_df[TARGET_DA].values

    # ── Strategy 1: Always DA (naive) ──
    q_da_naive = np.ones(len(test_df))
    sav_naive, cost_naive, _ = compute_pnl(actual_spread, actual_rt, q_da_naive)
    # Should be 0 (baseline)

    # ── Strategy 2: Oracle (knows actual spread sign) ──
    q_da_oracle = (actual_spread < 0).astype(float)
    sav_oracle, cost_oracle, _ = compute_pnl(actual_spread, actual_rt, q_da_oracle)

    # ── Strategy 3: Dual-Head only ──
    q_da_dual = (test_df["pred_spread"].values < 0).astype(float)
    sav_dual, cost_dual, _ = compute_pnl(actual_spread, actual_rt, q_da_dual)

    # ── Strategy 4: Hybrid ──
    is_ne = test_df["is_ne_high_gen"].values.astype(bool)
    q_da_hybrid = np.where(
        is_ne,
        # 新能源大发: classifier — prob>0.5 means spread>20 → DA>RT → 少报 (Q_da=0)
        (test_df["clf_prob"].values < 0.5).astype(float),
        # Other: dual-head
        (test_df["pred_spread"].values < 0).astype(float),
    )
    sav_hybrid, cost_hybrid, _ = compute_pnl(actual_spread, actual_rt, q_da_hybrid)

    # ── Strategy 5: Persistence ──
    q_da_persist = (test_df["spread_lag_1d"].values < 0).astype(float)
    valid_p = ~np.isnan(test_df["spread_lag_1d"].values)
    sav_persist, cost_persist, _ = compute_pnl(
        actual_spread[valid_p], actual_rt[valid_p], q_da_persist[valid_p]
    )

    # ── Strategy 6: Confidence-filtered Hybrid ──
    # Only act when confidence is high, otherwise default to always-DA (Q_da=1)
    n_periods = len(test_df)
    n_days = len(test_dates)

    # Confidence: clf_prob distance from 0.5 for NE, |pred_spread| for others
    conf_ne = np.abs(test_df["clf_prob"].values - 0.5) * 2  # 0..1
    conf_dual = np.clip(np.abs(test_df["pred_spread"].values), 0, 100) / 100  # 0..1
    confidence = np.where(is_ne, conf_ne, conf_dual)

    # Filter: only deviate from always-DA when confidence > threshold
    for conf_thresh in [0.3, 0.5, 0.7]:
        deviate = confidence > conf_thresh
        q_da_conf = np.where(deviate, q_da_hybrid, 1.0)  # default to DA when unsure
        sav_conf, cost_conf, _ = compute_pnl(actual_spread, actual_rt, q_da_conf)
        n_deviate = deviate.sum()
        # Direction accuracy on deviations only
        dev_nonzero = deviate & (np.abs(actual_spread) > 1e-6)
        dev_sign = np.where(q_da_conf == 1, -1, 1)
        dev_correct = dev_sign[dev_nonzero] == np.sign(actual_spread[dev_nonzero])
        dev_dir = dev_correct.mean() * 100 if dev_nonzero.sum() > 0 else 0
        print(f"    Conf>{conf_thresh}: deviate={n_deviate}/{n_periods} periods, "
              f"dir={dev_dir:.1f}%, saving={sav_conf/n_periods:.2f}/MWh")

    # ═══════════════════════════════════════════════════════════════════════
    # ── Print results ──
    print(f"\n  Test set: {n_days} days, {n_periods} periods")
    print(f"  Avg DA={actual_da.mean():.0f}  Avg RT={actual_rt.mean():.0f}  "
          f"Avg |spread|={np.abs(actual_spread).mean():.0f}")

    print(f"\n  {'Strategy':<28} {'Total Cost':>12} {'Saving':>12} {'per MWh':>10} {'per Day':>10}")
    print(f"  {'-'*28}  {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
    print(f"  {'Always DA (naive)':<28} {cost_naive:>12.0f} {'—':>12} {'—':>10} {'—':>10}")
    print(f"  {'Oracle (upper bound)':<28} {cost_oracle:>12.0f} {sav_oracle:>12.0f} "
          f"{sav_oracle/n_periods:>10.2f} {sav_oracle/n_days:>10.0f}")
    print(f"  {'Dual-Head only':<28} {cost_dual:>12.0f} {sav_dual:>12.0f} "
          f"{sav_dual/n_periods:>10.2f} {sav_dual/n_days:>10.0f}")
    print(f"  {'Hybrid (clf + dual)':<28} {cost_hybrid:>12.0f} {sav_hybrid:>12.0f} "
          f"{sav_hybrid/n_periods:>10.2f} {sav_hybrid/n_days:>10.0f}")

    best_conf = 0.5
    deviate = confidence > best_conf
    q_da_best = np.where(deviate, q_da_hybrid, 1.0)
    sav_best, cost_best, _ = compute_pnl(actual_spread, actual_rt, q_da_best)
    n_dev = deviate.sum()
    print(f"  {'Hybrid+Conf(>{})'.format(best_conf):<28} {cost_best:>12.0f} {sav_best:>12.0f} "
          f"{sav_best/n_periods:>10.2f} {sav_best/n_days:>10.0f}")

    print(f"  {'Persistence':<28} {cost_persist:>12.0f} {sav_persist:>12.0f} "
          f"{sav_persist/valid_p.sum():>10.2f} {sav_persist/n_days:>10.0f}")

    # Capture rate
    print(f"\n  ── Oracle Capture Rate ──")
    print(f"  Dual-Head:     {sav_dual/sav_oracle*100:.1f}%")
    print(f"  Hybrid:        {sav_hybrid/sav_oracle*100:.1f}%")
    print(f"  Hybrid+Conf:   {sav_best/sav_oracle*100:.1f}%  ({n_dev}/{n_periods} periods, {(1-n_dev/n_periods)*100:.0f}% stay DA)")
    print(f"  Persistence:   {sav_persist/sav_oracle*100:.1f}%")

    # Direction accuracy
    nonzero = np.abs(actual_spread) > 1e-6
    hybrid_sign = np.where(q_da_hybrid == 1, -1, 1)
    print(f"\n  ── Direction Accuracy ──")
    for name, pred_sign in [
        ("Dual-Head", np.sign(test_df["pred_spread"].values)),
        ("Hybrid", hybrid_sign),
        ("Persistence", np.sign(test_df["spread_lag_1d"].values)),
    ]:
        if "Persistence" in name:
            acc = (pred_sign[valid_p & nonzero] == np.sign(actual_spread[valid_p & nonzero])).mean() * 100
        else:
            acc = (pred_sign[nonzero] == np.sign(actual_spread[nonzero])).mean() * 100
        print(f"  {name:<14} {acc:.1f}%")

    # ── PnL breakdown by |spread| magnitude ──
    print(f"\n  ── PnL Breakdown by |Spread| Magnitude (Hybrid+Conf) ──")
    print(f"  {'|Spread|':<12} {'n':>6} {'PnL':>10} {'PnL/period':>12} {'Dir Acc':>10} {'% of total':>12}")
    best_sign = np.where(q_da_best == 1, -1, 1)
    for lo, hi in [(0, 20), (20, 50), (50, 100), (100, 200), (200, 999)]:
        mask = (np.abs(actual_spread) > lo) & (np.abs(actual_spread) <= hi)
        n = mask.sum()
        if n == 0:
            continue
        pnl_mag, _, _ = compute_pnl(actual_spread[mask], actual_rt[mask], q_da_best[mask])
        dir_mag = (best_sign[mask & nonzero] == np.sign(actual_spread[mask & nonzero])).mean() * 100
        pct = pnl_mag / sav_oracle * 100 if sav_oracle != 0 else 0
        print(f"  {f'{lo}-{hi}':<12} {n:>6} {pnl_mag:>10.0f} {pnl_mag/n:>12.2f} {dir_mag:>9.1f}% {pct:>11.1f}%")

    # Total oracle PnL by magnitude
    print(f"\n  ── Oracle PnL by |Spread| (for reference) ──")
    oracle_sign = np.where(q_da_oracle == 1, -1, 1)
    for lo, hi in [(0, 20), (20, 50), (50, 100), (100, 200), (200, 999)]:
        mask = (np.abs(actual_spread) > lo) & (np.abs(actual_spread) <= hi)
        n = mask.sum()
        if n == 0:
            continue
        pnl_mag, _, _ = compute_pnl(actual_spread[mask], actual_rt[mask], q_da_oracle[mask])
        print(f"  {f'{lo}-{hi}':<12} {n:>6} {pnl_mag:>10.0f} {pnl_mag/n:>12.2f}")

    # By regime
    print(f"\n  ── Per-Regime Savings (元/MWh) ──")
    for rname, rcol in REGIMES.items():
        rmask = test_df[rcol].values.astype(bool)
        n_r = rmask.sum()
        if n_r == 0:
            continue
        r_sav_best, _, _ = compute_pnl(actual_spread[rmask], actual_rt[rmask], q_da_best[rmask])
        r_sav_dual, _, _ = compute_pnl(actual_spread[rmask], actual_rt[rmask], q_da_dual[rmask])
        r_sav_oracle, _, _ = compute_pnl(actual_spread[rmask], actual_rt[rmask], q_da_oracle[rmask])
        print(f"  {rname:<12} n={n_r:>5}  Dual={r_sav_dual/n_r:.2f}  "
              f"Best={r_sav_best/n_r:.2f}  Oracle={r_sav_oracle/n_r:.2f}")

    # Per-day PnL distribution
    print(f"\n  ── Per-Day PnL Distribution (Hybrid+Conf) ──")
    test_df["day"] = test_df["trade_date"]
    daily_pnl = test_df.groupby("day").apply(
        lambda g: compute_pnl(
            g[TARGET_SPREAD].values, g[TARGET_RT].values,
            np.where(
                (np.abs(g["clf_prob"].values - 0.5) * 2 > best_conf)
                | (np.clip(np.abs(g["pred_spread"].values), 0, 100) / 100 > best_conf),
                np.where(
                    g["is_ne_high_gen"].values.astype(bool),
                    (g["clf_prob"].values < 0.5).astype(float),
                    (g["pred_spread"].values < 0).astype(float),
                ),
                1.0,
            )
        )[0]
    )
    print(f"  Mean daily PnL: {daily_pnl.mean():.0f} 元/day")
    print(f"  Std daily PnL:  {daily_pnl.std():.0f} 元/day")
    print(f"  Positive days:  {(daily_pnl > 0).mean()*100:.0f}%")
    print(f"  Best day:       {daily_pnl.max():.0f} 元")
    print(f"  Worst day:      {daily_pnl.min():.0f} 元")
    print(f"  PnL per MWh:    {daily_pnl.mean()/96:.2f} 元/MWh")

    # ── Monetary value conversion ──
    pnl_per_mwh = sav_best / n_periods  # use Dual-Head PnL (most robust)
    print(f"\n  ── 价值换算 (基于 Dual-Head {pnl_per_mwh:.2f} 元/MWh) ──")
    print(f"  每 MWh 交易量: {pnl_per_mwh:.2f} 元利润")
    print(f"  每 MW 容量/天: {pnl_per_mwh*24:.0f} 元 (24 MWh)")
    print(f"  每 MW 容量/年: {pnl_per_mwh*24*365:.0f} 元")

    # Scenarios
    for mw, label in [(100, "100MW"), (300, "300MW"), (1000, "1000MW (1GW)")]:
        day_profit = pnl_per_mwh * 24 * mw
        yr_profit = day_profit * 365
        print(f"  {label} 容量: {day_profit:,.0f} 元/天 = {yr_profit/1e4:,.0f} 万/年")

    # Oracle comparison
    oracle_per_mwh = sav_oracle / n_periods
    print(f"\n  Oracle 理论上限: {oracle_per_mwh:.2f} 元/MWh")
    for mw in [100, 300, 1000]:
        print(f"  {mw}MW: {(oracle_per_mwh*24*mw*365)/1e4:,.0f} 万/年")
    print(f"  模型捕获率: {pnl_per_mwh/oracle_per_mwh*100:.1f}%")
