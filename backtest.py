"""Train models for predict.py and run rolling-window backtest.

Trains 3 models (DA P50, RT P50, classifier) and saves them for predict.py.
Evaluation uses rolling-window methodology (not fixed split) to avoid the
Train/Test distribution shift from the 2026-01-01 policy change.
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
from data_utils import load_data, add_features, build_feature_sets, compute_pnl
from rolling_backtest import evaluate

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
    # Rolling-window backtest (official evaluation)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Rolling-Window Backtest (window=365d)")
    print("=" * 60)
    evaluate(df, window_days=365)
