"""Electricity price forecasting — optimized spread/direction prediction.

Key findings from EDA:
  - DA price level is highly predictive of spread direction (but is a label)
  - Floor regime (DA<40): DA>RT only 4.0%  → almost always DA<RT
  - DA 40-100: DA>RT = 69.6% → strong signal
  - We CAN use D-1 lagged prices + DA prediction as proxies

Approaches:
  Plan A: spread = pred_da − pred_rt (dual-head quantile)
  Plan B: direct spread regression (L1)
  Plan C: direction classifier
  Two-stage: use pred_da as feature for spread/direction
"""

import pandas as pd
import numpy as np
import lightgbm as lgb

from config import (
    DA_FEATURES, RT_FEATURES, RT_LAG_FEATURES,
    TARGET_DA, TARGET_RT, TARGET_SPREAD,
    QUANTILES, REGIMES,
)

LGB_QUANTILE = dict(
    n_estimators=1000, learning_rate=0.03, num_leaves=255,
    min_child_samples=20, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, verbose=-1,
)

LGB_RT = dict(
    n_estimators=2000, learning_rate=0.015, num_leaves=127,
    min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=0.5,
    random_state=42, verbose=-1,
)


def load_data():
    df = pd.read_excel("data/day_ahead_feature_matrix.xlsx")
    df = df.sort_values(["trade_date", "period"]).reset_index(drop=True)
    return df


def add_features(df):
    """Add extra lag features + engineered features."""
    df = df.copy()

    # ── Extra RT/spread lags ──
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

    # RT volatility
    rt_std = np.full(len(df), np.nan)
    for i, row in df.iterrows():
        date_idx = rt_pivot.index.get_loc(row["trade_date"])
        if date_idx >= 7:
            window = rt_pivot.iloc[max(0, date_idx - 7):date_idx]
            rt_std[i] = window[row["period"]].std()
    df["rt_roll_7d_std"] = rt_std

    # ── D-1 price bucket features (D-1 available!) ──
    # price_lag_1d is the D-1 DA price — a proxy for today's price regime
    df["lag_da_floor"] = (df["price_lag_1d"] < 40).astype(int)
    df["lag_da_low"] = ((df["price_lag_1d"] >= 40) & (df["price_lag_1d"] < 100)).astype(int)
    df["lag_da_mid"] = ((df["price_lag_1d"] >= 100) & (df["price_lag_1d"] < 300)).astype(int)
    df["lag_da_high"] = (df["price_lag_1d"] >= 300).astype(int)

    # ── D-1 spread direction ──
    df["lag_spread_sign"] = np.sign(df["spread_lag_1d"])

    # ── Regime × price interactions ──
    df["ne_high_x_lag_da_floor"] = df["is_ne_high_gen"] * df["lag_da_floor"]
    df["ne_high_x_lag_da_low"] = df["is_ne_high_gen"] * df["lag_da_low"]
    df["load_peak_x_lag_da_high"] = df["is_load_peak"] * df["lag_da_high"]

    # ── Spread momentum ──
    df["spread_change_1d"] = df["spread_da_rt"] - df["spread_lag_1d"]

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


def train_quantile(X_tr, y_tr, X_val, y_val, params, alpha):
    p = {**params, "objective": "quantile", "alpha": alpha}
    m = lgb.LGBMRegressor(**p)
    m.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)], eval_metric="quantile",
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    return m


def similar_day_baseline(df, test_dates, train_dates, target_col, n_similar=5):
    preds, actuals = [], []
    train_df = df[df["trade_date"].isin(train_dates)]
    for test_date in test_dates:
        test_day = df[df["trade_date"] == test_date]
        if len(test_day) != 96:
            continue
        same_dow = train_df[train_df["trade_date"].dt.dayofweek == test_date.dayofweek]
        if same_dow.empty:
            same_dow = train_df
        candidates = same_dow["trade_date"].unique()
        if len(candidates) == 0:
            continue
        test_nl = test_day["net_load"].values
        distances = []
        for t in candidates:
            t_day = train_df[train_df["trade_date"] == t]
            if len(t_day) != 96:
                continue
            dist = np.sqrt(np.mean((test_nl - t_day["net_load"].values) ** 2))
            distances.append((t, dist))
        distances.sort(key=lambda x: x[1])
        similar = train_df[train_df["trade_date"].isin([d[0] for d in distances[:n_similar]])]
        avg = similar.groupby("period")[target_col].mean().values
        if len(avg) == 96:
            preds.extend(avg)
            actuals.extend(test_day[target_col].values)
    return np.array(preds), np.array(actuals)


def spread_metrics(actual, pred, regime_masks=None):
    mae = np.abs(actual - pred).mean()
    nonzero = np.abs(actual) > 1e-6
    nz = nonzero.sum()
    dir_acc = (np.sign(pred[nonzero]) == np.sign(actual[nonzero])).mean() * 100 if nz > 0 else 0
    result = {"spread_mae": mae, "dir_acc": dir_acc}
    if regime_masks:
        for name, mask in regime_masks.items():
            if mask.sum() == 0:
                continue
            result[f"{name}_mae"] = np.abs(actual[mask] - pred[mask]).mean()
            r_nz = nonzero & mask
            result[f"{name}_dir"] = (np.sign(pred[r_nz]) == np.sign(actual[r_nz])).mean() * 100 if r_nz.sum() > 0 else 0
    return result


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_data()
    df = add_features(df)
    masks = time_split(df)

    # Build extended feature sets
    EXTRA_LAGS = ["rt_lag_2d", "rt_lag_3d", "rt_lag_7d", "rt_roll_7d_std",
                  "spread_lag_2d", "spread_lag_3d", "spread_lag_7d"]
    PRICE_BUCKETS = ["lag_da_floor", "lag_da_low", "lag_da_mid", "lag_da_high"]
    INTERACTIONS = ["ne_high_x_lag_da_floor", "ne_high_x_lag_da_low", "load_peak_x_lag_da_high"]
    LAG_SIGN = ["lag_spread_sign"]

    RT_FEATURES_V2 = RT_FEATURES + EXTRA_LAGS + [c for c in PRICE_BUCKETS + LAG_SIGN if c not in RT_FEATURES]
    SPREAD_FEATURES = DA_FEATURES + RT_LAG_FEATURES + EXTRA_LAGS + PRICE_BUCKETS + INTERACTIONS + LAG_SIGN

    # Remove spread_change_1d from features (it uses D-day spread → leakage!)
    # spread_change_1d was computed above but is NOT D-1 available (uses D-day spread)

    test_dates_arr = sorted(df.loc[masks["test"], "trade_date"].unique())
    train_dates_arr = sorted(df.loc[masks["train"], "trade_date"].unique())
    val_dates_arr = sorted(df.loc[masks["val"], "trade_date"].unique())

    print(f"Split: train={len(train_dates_arr)}d  val={len(val_dates_arr)}d  "
          f"test={len(test_dates_arr)}d")
    print(f"Features — DA: {len(DA_FEATURES)}  RT: {len(RT_FEATURES_V2)}  "
          f"Spread: {len(SPREAD_FEATURES)}")

    regime_masks = {name: df.loc[masks["test"], col].values.astype(bool)
                    for name, col in REGIMES.items()}

    # ═══════════════════════════════════════════════════════════════════════
    # Plan A: Dual-Head
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  方 案 A : 双 头 回 归")
    print("=" * 60)

    X_tr_da = df.loc[masks["train"], DA_FEATURES]
    y_tr_da = df.loc[masks["train"], TARGET_DA]
    X_val_da = df.loc[masks["val"], DA_FEATURES]
    y_val_da = df.loc[masks["val"], TARGET_DA]
    X_te_da = df.loc[masks["test"], DA_FEATURES]
    y_te_da = df.loc[masks["test"], TARGET_DA]

    print(f"DA y — Train={y_tr_da.mean():.0f}  Val={y_val_da.mean():.0f}  "
          f"Test={y_te_da.mean():.0f}")
    da_p50 = train_quantile(X_tr_da, y_tr_da, X_val_da, y_val_da, LGB_QUANTILE, 0.5)
    pred_da = da_p50.predict(X_te_da)
    da_mae = np.abs(y_te_da.values - pred_da).mean()
    print(f"  DA P50 MAE = {da_mae:.1f}")

    X_tr_rt = df.loc[masks["train"], RT_FEATURES_V2]
    y_tr_rt = df.loc[masks["train"], TARGET_RT]
    X_val_rt = df.loc[masks["val"], RT_FEATURES_V2]
    y_val_rt = df.loc[masks["val"], TARGET_RT]
    X_te_rt = df.loc[masks["test"], RT_FEATURES_V2]
    y_te_rt = df.loc[masks["test"], TARGET_RT]

    print(f"RT y — Train={y_tr_rt.mean():.0f}  Val={y_val_rt.mean():.0f}  "
          f"Test={y_te_rt.mean():.0f}")
    rt_p50 = train_quantile(X_tr_rt, y_tr_rt, X_val_rt, y_val_rt, LGB_RT, 0.5)
    pred_rt = rt_p50.predict(X_te_rt)
    rt_mae = np.abs(y_te_rt.values - pred_rt).mean()
    print(f"  RT P50 MAE = {rt_mae:.1f}")

    actual_spread = df.loc[masks["test"], TARGET_SPREAD].values
    pred_spread_a = pred_da - pred_rt
    result_a = spread_metrics(actual_spread, pred_spread_a, regime_masks)
    print(f"  Spread MAE={result_a['spread_mae']:.1f}  Dir={result_a['dir_acc']:.1f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # Plan B: Direct Spread (L1)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  方 案 B : 直 接 Spread (L1)")
    print("=" * 60)

    X_tr_sp = df.loc[masks["train"], SPREAD_FEATURES]
    y_tr_sp = df.loc[masks["train"], TARGET_SPREAD]
    X_val_sp = df.loc[masks["val"], SPREAD_FEATURES]
    y_val_sp = df.loc[masks["val"], TARGET_SPREAD]
    X_te_sp = df.loc[masks["test"], SPREAD_FEATURES]

    print(f"Spread y — Train={y_tr_sp.mean():.1f}  Val={y_val_sp.mean():.1f}  "
          f"Test={actual_spread.mean():.1f}  σ={actual_spread.std():.1f}")

    sp_model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=3000, learning_rate=0.005,
        num_leaves=63, min_child_samples=100, subsample=0.7,
        colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
        random_state=42, verbose=-1,
    )
    sp_model.fit(X_tr_sp, y_tr_sp,
                 eval_set=[(X_val_sp, y_val_sp)],
                 callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    pred_spread_b = sp_model.predict(X_te_sp)
    result_b = spread_metrics(actual_spread, pred_spread_b, regime_masks)
    print(f"  Spread MAE={result_b['spread_mae']:.1f}  Dir={result_b['dir_acc']:.1f}%  "
          f"Pred mean={pred_spread_b.mean():.1f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Plan C: Direction Classifier
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  方 案 C : 方 向 分 类 器")
    print("=" * 60)

    # Binary: DA > RT or not (exclude near-zero for cleaner signal)
    for threshold in [0, 10, 20]:
        y_cls_tr = (df.loc[masks["train"], TARGET_SPREAD] > threshold).astype(int)
        y_cls_val = (df.loc[masks["val"], TARGET_SPREAD] > threshold).astype(int)
        y_cls_te = (df.loc[masks["test"], TARGET_SPREAD] > threshold).astype(int)

        n_pos_tr = y_cls_tr.sum()
        n_neg_tr = len(y_cls_tr) - n_pos_tr
        print(f"\n  Threshold={threshold}: "
              f"DA>RT={n_pos_tr/len(y_cls_tr)*100:.1f}% in train")

        clf = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.02, num_leaves=127,
            min_child_samples=100, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=0.5,
            class_weight="balanced", random_state=42, verbose=-1,
        )
        clf.fit(X_tr_sp, y_cls_tr,
                eval_set=[(X_val_sp, y_cls_val)],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

        pred_prob = clf.predict_proba(X_te_sp)[:, 1]
        pred_cls = (pred_prob > 0.5).astype(int)

        # Evaluate direction accuracy
        te_actual_sign = (actual_spread > threshold).astype(int)
        te_nonzero = np.abs(actual_spread) > 1e-6
        dir_acc_clf = (pred_cls[te_nonzero] == te_actual_sign[te_nonzero]).mean() * 100
        print(f"  Classifier Dir Acc (|spread|>0) = {dir_acc_clf:.1f}%")

        # By regime
        for name, mask in regime_masks.items():
            m = mask & te_nonzero
            if m.sum() > 0:
                acc = (pred_cls[m] == te_actual_sign[m]).mean() * 100
                print(f"    {name}: Dir={acc:.1f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # Two-Stage: use pred_da as feature for spread
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  两 阶 段 : pred_da 作 为 spread 特 征")
    print("=" * 60)

    # Add pred_da as a feature for train/val (use cross-val style: pred from previous fold)
    # Simpler: add DA lag bucket features (already done) which are D-1 proxies
    # But let's also try: train spread model with DA price features INCLUDING price_lag_*
    # which are already in SPREAD_FEATURES

    # What if we use the DA model's prediction as a feature for spread?
    # For train: we need out-of-fold DA predictions to avoid leakage
    # Simple approach: use price_lag_1d as a proxy (already in features)

    # Let's try: add pred_da_bucket as categorical info
    # For test set we have pred_da, for train we need OOF
    # Quick hack: use a simple time-series split for OOF DA preds on train

    from sklearn.model_selection import TimeSeriesSplit

    print("  Computing OOF DA predictions for train set...")
    oof_da = np.full(len(df), np.nan)
    tscv = TimeSeriesSplit(n_splits=3)
    train_idx = np.where(masks["train"].values)[0]
    train_dates_for_cv = sorted(df.loc[masks["train"], "trade_date"].unique())

    # Split train dates into 3 folds chronologically
    fold_size = len(train_dates_for_cv) // 3
    for fold in range(3):
        fold_start = fold * fold_size
        fold_val_dates = train_dates_for_cv[fold_start:fold_start + fold_size]
        fold_train_dates = [d for d in train_dates_for_cv if d not in fold_val_dates]

        fold_train_mask = df["trade_date"].isin(fold_train_dates)
        fold_val_mask = df["trade_date"].isin(fold_val_dates)

        m = train_quantile(
            df.loc[fold_train_mask, DA_FEATURES], df.loc[fold_train_mask, TARGET_DA],
            df.loc[fold_val_mask, DA_FEATURES], df.loc[fold_val_mask, TARGET_DA],
            LGB_QUANTILE, 0.5,
        )
        oof_da[fold_val_mask.values] = m.predict(df.loc[fold_val_mask, DA_FEATURES])

    # For val and test, use the main DA model
    val_mask_arr = masks["val"].values
    test_mask_arr = masks["test"].values
    oof_da[val_mask_arr] = da_p50.predict(df.loc[val_mask_arr, DA_FEATURES])
    oof_da[test_mask_arr] = pred_da

    # Add OOF pred_da and derived features
    df["pred_da_oof"] = oof_da
    df["pred_da_floor"] = (oof_da < 40).astype(int)
    df["pred_da_low"] = ((oof_da >= 40) & (oof_da < 100)).astype(int)
    df["pred_da_high"] = (oof_da >= 300).astype(int)
    df["pred_da_x_ne_high"] = df["is_ne_high_gen"] * df["pred_da_floor"]

    TWO_STAGE_FEATURES = SPREAD_FEATURES + ["pred_da_oof", "pred_da_floor",
                                              "pred_da_low", "pred_da_high",
                                              "pred_da_x_ne_high"]

    X_tr_2s = df.loc[masks["train"], TWO_STAGE_FEATURES]
    X_val_2s = df.loc[masks["val"], TWO_STAGE_FEATURES]
    X_te_2s = df.loc[masks["test"], TWO_STAGE_FEATURES]

    print(f"  Two-stage features: {len(TWO_STAGE_FEATURES)}")

    ts_model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=3000, learning_rate=0.005,
        num_leaves=63, min_child_samples=100, subsample=0.7,
        colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
        random_state=42, verbose=-1,
    )
    ts_model.fit(X_tr_2s, y_tr_sp,
                 eval_set=[(X_val_2s, y_val_sp)],
                 callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    pred_spread_2s = ts_model.predict(X_te_2s)
    result_2s = spread_metrics(actual_spread, pred_spread_2s, regime_masks)
    print(f"  Spread MAE={result_2s['spread_mae']:.1f}  Dir={result_2s['dir_acc']:.1f}%  "
          f"Pred mean={pred_spread_2s.mean():.1f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Baselines
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  基 线")
    print("=" * 60)

    persist_spread = df.loc[masks["test"], "spread_lag_1d"].values
    persist_valid = ~np.isnan(persist_spread)
    persist_masks = {k: v[persist_valid] for k, v in regime_masks.items()}
    result_p = spread_metrics(actual_spread[persist_valid], persist_spread[persist_valid], persist_masks)
    print(f"  Persistence:  MAE={result_p['spread_mae']:.1f}  Dir={result_p['dir_acc']:.1f}%")

    sim_preds, sim_actuals = similar_day_baseline(df, test_dates_arr, train_dates_arr, TARGET_SPREAD)
    if len(sim_preds) > 0:
        result_sim = spread_metrics(sim_actuals, sim_preds)
        print(f"  Similar-Day:  MAE={result_sim['spread_mae']:.1f}  Dir={result_sim['dir_acc']:.1f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # Final Comparison
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  最 终 对 比")
    print("=" * 60)

    print(f"\n  {'Approach':<28} {'Spread MAE':>12} {'Dir Acc':>10}")
    print(f"  {'-'*28}  {'-'*12} {'-'*10}")
    print(f"  {'方案 A (双头 quantile)':<28} {result_a['spread_mae']:>12.1f} {result_a['dir_acc']:>9.1f}%")
    print(f"  {'方案 B (直接Spread L1)':<28} {result_b['spread_mae']:>12.1f} {result_b['dir_acc']:>9.1f}%")
    print(f"  {'两阶段 (pred_da特征)':<28} {result_2s['spread_mae']:>12.1f} {result_2s['dir_acc']:>9.1f}%")
    print(f"  {'持久性基线':<28} {result_p['spread_mae']:>12.1f} {result_p['dir_acc']:>9.1f}%")
    if len(sim_preds) > 0:
        print(f"  {'相似日基线':<28} {result_sim['spread_mae']:>12.1f} {result_sim['dir_acc']:>9.1f}%")

    # Best
    all_results = {"方案A": result_a, "方案B": result_b, "两阶段": result_2s,
                   "持久性": result_p}
    if len(sim_preds) > 0:
        all_results["相似日"] = result_sim

    best_mae = min(all_results.items(), key=lambda x: x[1]["spread_mae"])
    best_dir = max(all_results.items(), key=lambda x: x[1]["dir_acc"])
    print(f"\n  Best MAE:  {best_mae[0]} ({best_mae[1]['spread_mae']:.1f})")
    print(f"  Best Dir:  {best_dir[0]} ({best_dir[1]['dir_acc']:.1f}%)")

    # By regime
    print(f"\n  ── 分时段方向准确率 ──")
    print(f"  {'时段':<12} {'方案A':>8} {'方案B':>8} {'两阶段':>8} {'持久性':>8}")
    for rname in REGIMES:
        a = result_a.get(f"{rname}_dir", 0)
        b = result_b.get(f"{rname}_dir", 0)
        t = result_2s.get(f"{rname}_dir", 0)
        p = result_p.get(f"{rname}_dir", 0)
        best = max(a, b, t, p)
        markers = []
        if a == best: markers.append("A")
        if b == best: markers.append("B")
        if t == best: markers.append("2S")
        mark_str = " ✓" + "/".join(markers) if best > min(a, b, t, p) else ""
        print(f"  {rname:<12} {a:>7.1f}% {b:>7.1f}% {t:>7.1f}% {p:>7.1f}%{mark_str}")

    # ── Feature importance (two-stage) ──
    print(f"\n  ── 两阶段模型 Top-20 特征重要性 ──")
    importances = pd.Series(ts_model.feature_importances_, index=TWO_STAGE_FEATURES)
    for i, (feat, imp) in enumerate(importances.sort_values(ascending=False).head(20).items()):
        marker = " ← new" if feat.startswith("pred_da_") else ""
        print(f"  {i+1:2d}. {feat:<30s} {imp:.4f}{marker}")
