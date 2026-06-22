"""Experiment: sample weighting by |spread| magnitude.

Hypothesis: weighting training samples by |spread| forces the model to
prioritize large-spread periods, where direction errors are most costly.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from backtest import load_data, add_features, build_feature_sets, time_split, compute_pnl
from config import DA_FEATURES, TARGET_DA, TARGET_RT, TARGET_SPREAD, REGIMES

LGB_BASE = dict(
    n_estimators=1000, learning_rate=0.03, num_leaves=255,
    min_child_samples=20, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
)

LGB_RT = dict(
    n_estimators=2000, learning_rate=0.015, num_leaves=127,
    min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1,
)

def train_p50(X_tr, y_tr, X_val, y_val, params, sample_weight=None):
    p = {**params, "objective": "quantile", "alpha": 0.5}
    m = lgb.LGBMRegressor(**p)
    kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="quantile",
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)], **kw)
    return m

# ═══════════════════════════════════════════════════════════════════════════
df = load_data()
df = add_features(df)
masks = time_split(df)
rt_feats, sp_feats = build_feature_sets()

# Weight schemes to test
train_spread = np.abs(df.loc[masks["train"], TARGET_SPREAD].values)

schemes = {
    "unweighted": None,
    "|spread|": train_spread,
    "|spread|²": train_spread ** 2,
    "clip(|spread|,0,200)": np.clip(train_spread, 0, 200),
    "1+|spread|/50": 1 + train_spread / 50,
}

results = {}
for name, sw in schemes.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # DA model
    print("── DA ──")
    da_m = train_p50(
        df.loc[masks["train"], DA_FEATURES], df.loc[masks["train"], TARGET_DA],
        df.loc[masks["val"], DA_FEATURES], df.loc[masks["val"], TARGET_DA],
        LGB_BASE, sample_weight=sw,
    )

    # RT model
    print("── RT ──")
    rt_m = train_p50(
        df.loc[masks["train"], rt_feats], df.loc[masks["train"], TARGET_RT],
        df.loc[masks["val"], rt_feats], df.loc[masks["val"], TARGET_RT],
        LGB_RT, sample_weight=sw,
    )

    # Predict
    pred_da = da_m.predict(df.loc[masks["test"], DA_FEATURES])
    pred_rt = rt_m.predict(df.loc[masks["test"], rt_feats])
    pred_spread = pred_da - pred_rt
    actual_spread = df.loc[masks["test"], TARGET_SPREAD].values
    actual_rt = df.loc[masks["test"], TARGET_RT].values

    # Metrics
    nonzero = np.abs(actual_spread) > 1e-6
    dir_acc = (np.sign(pred_spread[nonzero]) == np.sign(actual_spread[nonzero])).mean() * 100
    spread_mae = np.abs(actual_spread - pred_spread).mean()

    q_da = (pred_spread < 0).astype(float)
    sav, _, _ = compute_pnl(actual_spread, actual_rt, q_da)

    n_periods = len(pred_da)
    print(f"\n  DA MAE={np.abs(df.loc[masks['test'],TARGET_DA].values-pred_da).mean():.1f}  "
          f"RT MAE={np.abs(df.loc[masks['test'],TARGET_RT].values-pred_rt).mean():.1f}")
    print(f"  Spread MAE={spread_mae:.1f}  Dir={dir_acc:.1f}%  "
          f"PnL={sav/n_periods:.2f}/MWh  ({sav:.0f} total)")

    # By |spread| magnitude
    print(f"  PnL by |spread|:")
    for lo, hi in [(0,20),(20,50),(50,100),(100,200),(200,999)]:
        m = (np.abs(actual_spread)>lo) & (np.abs(actual_spread)<=hi)
        if m.sum()>0:
            s,_,_ = compute_pnl(actual_spread[m], actual_rt[m], q_da[m])
            print(f"    {lo:>3}-{hi:<4}: {s/m.sum():6.2f}/MWh  n={m.sum()}")

    results[name] = {"dir": dir_acc, "spread_mae": spread_mae, "pnl_per_mwh": sav/n_periods}

# Summary
print(f"\n{'='*60}")
print(f"  Summary")
print(f"{'='*60}")
print(f"  {'Scheme':<25} {'Dir':>7} {'Spread MAE':>12} {'PnL/MWh':>10}")
print(f"  {'-'*25} {'-'*7} {'-'*12} {'-'*10}")
for name, r in results.items():
    print(f"  {name:<25} {r['dir']:>6.1f}% {r['spread_mae']:>12.1f} {r['pnl_per_mwh']:>9.2f}")
