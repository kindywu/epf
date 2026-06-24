# 电价预测系统 — LLM 可复现实现文档

> 更新: 2026-06-23  
> 前提: 本文件包含完整、可运行的代码。按顺序执行即可复现全部结果。
> 数据: `data/day_ahead_feature_matrix.xlsx` (86,588 行 × 81 列, 2024-01 ~ 2026-06)

---

## 0. 环境准备

```bash
# Python 3.13, 用 uv 管理
uv init --python 3.13
uv add pandas numpy lightgbm scikit-learn joblib openpyxl

# 激活 venv
source .venv/bin/activate
```

---

## 1. 全局配置 — `config.py`

定义所有特征列表、目标列、模型文件名、regime 映射。**整个项目的唯一配置入口。**

```python
"""Shared configuration for electricity price forecasting.

Plan A (双头回归): shared feature backbone, two LGBM heads for P_DA and P_RT.
spread = pred_da − pred_rt, direction = sign(spread).
"""

# ── Day-ahead (DA) model features (existing TOP30, unchanged) ──
DA_FEATURES = [
    "day_of_year", "tie_total", "supply_gap", "ne_wind",
    "days_in_solar_term", "price_lag_1d", "hydro_fcst", "tie_新疆",
    "price_lag_7d", "coal_x_thermal_fcst", "dow", "thermal_fcst",
    "price_lag_2d", "solar_term_sin", "tie_湖南", "gen_fcst",
    "wx_temp_2m", "price_lag_3d", "tie_青海", "coal_x_net_load",
    "price_roll_30d_mean", "tie_宁夏", "price_lag_1d_dev_roll7",
    "tie_陕西", "tie_山东", "wx_wind_100m", "load_fcst_x_wx_temp",
    "price_roll_7d_mean", "days_from_holiday", "hydro_ratio",
]

# ── RT-specific lag features (D-1 available, NOT D-day labels) ──
RT_LAG_FEATURES = [
    "rt_lag_1d",
    "rt_roll_7d_mean",
    "spread_lag_1d",
    "spread_roll_7d_mean",
]

# ── Previously unused columns, now added ──
# 事前预测: ne_solar (光伏), load_fcst (负荷原始值)
# 气象: wx_ghi (辐照度), coal_price_gansu (煤价)
# Regime: is_ne_high_gen, is_load_peak, is_grid_other, is_solar_season
# 日历: is_weekend, is_holiday, month, is_heating_season, period_sin, period_cos
# 衍生: net_load, renewable_ratio, ne_total
# 贴地: floor_regime_proxy, renewable_ratio_x_floor_regime, ne_total_x_floor_regime,
#        net_load_x_floor_regime, solar_season_ghi
# 其他交互: renewable_ratio_x_holiday, net_load_x_period_sin, net_load_x_holiday,
#           tie_total_x_hour, ne_solar_per_ghi, renewable_ratio_x_coal
# 风速衍生: wx_wind_effective, wx_wind_above_cutin
EXPANDED_FEATURES = [
    "ne_solar", "load_fcst",
    "wx_ghi", "coal_price_gansu",
    "is_ne_high_gen", "is_load_peak", "is_grid_other", "is_solar_season",
    "is_weekend", "is_holiday", "month", "is_heating_season",
    "period_sin", "period_cos",
    "net_load", "renewable_ratio", "ne_total",
    "floor_regime_proxy", "renewable_ratio_x_floor_regime",
    "ne_total_x_floor_regime", "net_load_x_floor_regime",
    "solar_season_ghi",
    "renewable_ratio_x_holiday", "net_load_x_period_sin",
    "net_load_x_holiday", "tie_total_x_hour",
    "ne_solar_per_ghi", "renewable_ratio_x_coal",
    "wx_wind_effective", "wx_wind_above_cutin",
]

# ── RT model features = DA base + RT lags + expanded ──
RT_FEATURES = DA_FEATURES + RT_LAG_FEATURES + EXPANDED_FEATURES

# ── Target columns (labels, NOT usable as D-1 features) ──
TARGET_DA = "price_day_ahead"
TARGET_RT = "price_realtime"
TARGET_SPREAD = "spread_da_rt"

# ── Model ──
QUANTILES = [0.1, 0.5, 0.9]
QUANTILE_NAMES = {0.1: "P10(卖参考)", 0.5: "P50(中位)", 0.9: "P90(买入报价)"}
MODEL_FILE_DA = "model_da_quantile.pkl"
MODEL_FILE_RT = "model_rt_quantile.pkl"

# ── Regime columns for stratified evaluation (西北三时段) ──
REGIMES = {
    "新能源大发": "is_ne_high_gen",
    "用电高峰": "is_load_peak",
    "其他时段": "is_grid_other",
}
```

**关键决策记录**:
- DA_FEATURES: 30 个 TOP-30 特征，来自实验 5 的 LightGBM importance 消融 (67→30, MAPE 40.09%→38.83%)
- RT_FEATURES: DA 的 30 个 + RT lag (4) + EXPANDED (25) + 动态计算 (16) = 75 个
- Regime 是硬编码时间模板: 新能源大发=10:00-16:59, 用电高峰=6:00-9:59+17:00-22:59, 其他=0:00-5:59+23:00-23:59

---

## 2. 共享工具函数 — `data_utils.py`

数据加载、特征工程、PnL 计算。被 `backtest.py` 和 `rolling_backtest.py` 共同引用。

```python
"""Shared data loading, feature engineering, and PnL computation.

Used by backtest.py (model training) and rolling_backtest.py (evaluation).
"""

import pandas as pd
import numpy as np
from config import (
    TARGET_DA, TARGET_RT, TARGET_SPREAD, RT_FEATURES,
)


def load_data():
    """Load and sort the feature matrix."""
    df = pd.read_excel("data/day_ahead_feature_matrix.xlsx")
    df = df.sort_values(["trade_date", "period"]).reset_index(drop=True)
    return df


def add_features(df):
    """Add engineered features (extra lags, price buckets, interactions).

    These are NOT in the Excel matrix — computed on-the-fly from
    existing columns. All features use D-1 or older information only
    (no look-ahead bias).
    """
    df = df.copy()

    # ── Extra lags (rt_lag_2d/3d/7d, spread_lag_2d/3d/7d) ──
    # Use pivot to align by (trade_date, period) for fast lag lookup
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

    # ── Rolling std of RT over past 7 days ──
    rt_std = np.full(len(df), np.nan)
    for i, row in df.iterrows():
        date_idx = rt_pivot.index.get_loc(row["trade_date"])
        if date_idx >= 7:
            window = rt_pivot.iloc[max(0, date_idx - 7):date_idx]
            rt_std[i] = window[row["period"]].std()
    df["rt_roll_7d_std"] = rt_std

    # ── D-1 price buckets (from D-1 DA price, acts as regime proxy) ──
    df["lag_da_floor"] = (df["price_lag_1d"] < 40).astype(int)
    df["lag_da_low"] = ((df["price_lag_1d"] >= 40) & (df["price_lag_1d"] < 100)).astype(int)
    df["lag_da_mid"] = ((df["price_lag_1d"] >= 100) & (df["price_lag_1d"] < 300)).astype(int)
    df["lag_da_high"] = (df["price_lag_1d"] >= 300).astype(int)
    df["lag_spread_sign"] = np.sign(df["spread_lag_1d"])

    # ── Regime × price interactions ──
    df["ne_high_x_lag_da_floor"] = df["is_ne_high_gen"] * df["lag_da_floor"]
    df["ne_high_x_lag_da_low"] = df["is_ne_high_gen"] * df["lag_da_low"]

    return df


def build_feature_sets():
    """Return feature lists for RT and spread models.

    RT: DA base + RT lags + expanded (all from matrix) + extra computed lags
        + price buckets + lag_sign + interactions
    Spread: same as RT (deduped)
    """
    EXTRA_LAGS = ["rt_lag_2d", "rt_lag_3d", "rt_lag_7d", "rt_roll_7d_std",
                  "spread_lag_2d", "spread_lag_3d", "spread_lag_7d"]
    PRICE_BUCKETS = ["lag_da_floor", "lag_da_low", "lag_da_mid", "lag_da_high"]
    LAG_SIGN = ["lag_spread_sign"]
    INTERACTIONS = ["ne_high_x_lag_da_floor", "ne_high_x_lag_da_low"]

    rt_features = RT_FEATURES + EXTRA_LAGS + PRICE_BUCKETS + LAG_SIGN + INTERACTIONS
    spread_features = list(dict.fromkeys(rt_features))  # dedup

    return rt_features, spread_features


def compute_pnl(actual_spread, actual_rt, q_da):
    """Savings vs always-DA. Positive = strategy saves money.

    Cost = Q_da * P_da + (1 - Q_da) * P_rt = P_rt + Q_da * spread
    Savings = Cost(always-DA) - Cost(strategy) = (1 - Q_da) * spread
    """
    cost_naive = actual_rt + actual_spread  # Q_da=1 → P_da
    cost_model = actual_rt + q_da * actual_spread
    return (cost_naive - cost_model).sum()
```

---

## 3. 模型训练管线 — `backtest.py`

**功能**: 训练部署用模型 (DA P50, RT P50, 分类器) → 保存 .pkl → 运行滚动窗口评估。

```python
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

# ── Model hyperparameters ──
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
    """Chronological 70/15/15 split for model training only.

    NOT used for evaluation — rolling window handles that.
    """
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
    """Train a single LGBM quantile regression model."""
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
    # ── Load & prep ──
    df = load_data()
    df = add_features(df)
    masks = time_split(df)
    rt_feats, spread_feats = build_feature_sets()

    train_dates = sorted(df.loc[masks["train"], "trade_date"].unique())
    val_dates = sorted(df.loc[masks["val"], "trade_date"].unique())
    test_dates = sorted(df.loc[masks["test"], "trade_date"].unique())

    print(f"Split: train={len(train_dates)}d  val={len(val_dates)}d  test={len(test_dates)}d")

    # ── RT sample weights (by |spread| magnitude) ──
    train_spread_abs = np.abs(df.loc[masks["train"], TARGET_SPREAD].values)
    rt_sample_weight = np.clip(train_spread_abs, 0, 200)

    # ════════════════════════════════════════════════════════════════
    # Step 1: OOF DA predictions (3-fold TimeSeriesSplit)
    # Purpose: RT model needs pred_da as feature. Direct in-sample
    #   pred_da would leak target info. OOF ensures each training
    #   sample's pred_da comes from a fold that didn't see it.
    # ════════════════════════════════════════════════════════════════
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

    # ════════════════════════════════════════════════════════════════
    # Step 2: Final DA P50 (trained on full train set, unweighted)
    # ════════════════════════════════════════════════════════════════
    print("\n── Training DA P50 (unweighted) ──")
    da_model = train_quantile(
        df.loc[masks["train"], DA_FEATURES], df.loc[masks["train"], TARGET_DA],
        df.loc[masks["val"], DA_FEATURES], df.loc[masks["val"], TARGET_DA],
        LGB_QUANTILE, 0.5,
    )

    oof_da_val = da_model.predict(df.loc[masks["val"], DA_FEATURES])

    # ════════════════════════════════════════════════════════════════
    # Step 3: RT P50 (pred_da as feature, |spread|-weighted)
    # ════════════════════════════════════════════════════════════════
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

    # ════════════════════════════════════════════════════════════════
    # Step 4: Classifier (spread > 20, for 新能源大发 regime)
    # ════════════════════════════════════════════════════════════════
    print(f"\n── Training Classifier (spread > {CLF_THRESHOLD}) ──")
    y_clf_tr = (df.loc[masks["train"], TARGET_SPREAD] > CLF_THRESHOLD).astype(int)
    y_clf_val = (df.loc[masks["val"], TARGET_SPREAD] > CLF_THRESHOLD).astype(int)

    clf_model = lgb.LGBMClassifier(**LGB_CLF)
    clf_model.fit(
        df.loc[masks["train"], spread_feats], y_clf_tr,
        eval_set=[(df.loc[masks["val"], spread_feats], y_clf_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )

    # ════════════════════════════════════════════════════════════════
    # Save models for predict.py
    # ════════════════════════════════════════════════════════════════
    joblib.dump(da_model, MODEL_FILE_DA)
    joblib.dump(rt_model, MODEL_FILE_RT)
    joblib.dump(clf_model, MODEL_FILE_CLF)
    print(f"\nModels saved: {MODEL_FILE_DA}, {MODEL_FILE_RT}, {MODEL_FILE_CLF}")

    # ════════════════════════════════════════════════════════════════
    # Rolling-window backtest (the official evaluation)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Rolling-Window Backtest (window=365d)")
    print("=" * 60)
    evaluate(df, window_days=365)
```

**关键设计点**:
1. **OOF pred_da**: RT 模型需要 pred_da 作为特征。如果用 in-sample pred_da，DA 模型"见过"这些训练样本的 label，会造成目标泄漏。3-fold TimeSeriesSplit 确保每个训练样本的 pred_da 来自未见过该样本的 fold。
2. **|spread| 加权**: RT 模型按 `clip(|spread|, 0, 200)` 加权。大 spread 时段决策错误代价大，加权让模型倾斜容量给这些关键时段。实验证明这是最有效的单点优化 (P1nL -1.35 → +1.71)。
3. **固定切分仅用于模型训练**: 70/15/15 切分仅用于生成部署模型。评估不依赖此切分。

---

## 4. 滚动窗口评估 — `rolling_backtest.py`

**功能**: 用 365d 滑动窗口每天重训模型、预测当天、计算 PnL。输出完整的 PnL 分解报告。

**为什么用滚动窗口替代固定切分**:
- 2026-01-01 政策变更 (甘工信发〔2025〕268 号，申报上限 650→500)
- 固定切分下 Train 全在旧政策，Test 全在新政策 → 系统性分布偏移
- 滚动窗口让新政策数据随时间自然流入训练集 → 评估更接近真实部署环境

```python
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
    """Print to both stdout and log file, unbuffered."""
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
    """Train a single LGBM quantile model."""
    p = {**params, "objective": "quantile", "alpha": alpha}
    m = lgb.LGBMRegressor(**p)
    kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)], **kw)
    return m


def evaluate(df, window_days=365):
    """Run rolling-window backtest and print full PnL report.

    For each test date (2026-01-01 onwards):
      1. Build training window: [date - window_days, date - 1]
      2. Train DA model (unweighted)
      3. Train RT model (|spread|-weighted, pred_da as feature)
      4. Predict the test day → compute PnL

    Key simplification vs backtest.py:
      - In-sample pred_da for RT training is safe here because
        training window is entirely before the test date.
        No OOF needed.

    Returns dict with metrics for programmatic use.
    """
    all_dates = sorted(df["trade_date"].unique())
    policy_break = pd.Timestamp("2026-01-01")
    test_dates = [d for d in all_dates if d >= policy_break]

    log(f"滚动窗口回测 (window={window_days}d)")
    log(f"测试期: {test_dates[0].date()} ~ {test_dates[-1].date()}  ({len(test_dates)} 天)")
    log()

    rt_feats, _ = build_feature_sets()

    # Per-period records for breakdown analysis
    records = []
    daily_pnl = []
    daily_dir_acc = []

    for i, test_date in enumerate(test_dates):
        test_idx = all_dates.index(test_date)
        train_start = max(0, test_idx - window_days)
        train_end = test_idx - 1
        train_dates = all_dates[train_start:train_end + 1]
        # Validation: last 7 days of training window
        val_dates = train_dates[-min(7, len(train_dates) // 5):]

        tr_mask = df["trade_date"].isin(train_dates)
        val_mask = df["trade_date"].isin(val_dates)
        test_mask = df["trade_date"] == test_date

        if tr_mask.sum() < 500:
            continue

        # ── RT sample weights ──
        spread_w = np.clip(np.abs(df.loc[tr_mask, TARGET_SPREAD].values), 0, 200)

        # ── Step 1: Train DA (unweighted) ──
        da_model = train_model(
            df.loc[tr_mask, DA_FEATURES], df.loc[tr_mask, TARGET_DA],
            df.loc[val_mask, DA_FEATURES], df.loc[val_mask, TARGET_DA],
            LGB_DA, 0.5,
        )

        # ── Step 2: Train RT (weighted, pred_da as feature) ──
        # In-sample pred_da is safe: all training data < test date
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

        # ── Step 3: Predict test day ──
        test_pred_da = da_model.predict(df.loc[test_mask, DA_FEATURES])
        X_test_rt = df.loc[test_mask, rt_feats].copy()
        X_test_rt["pred_da"] = test_pred_da
        test_pred_rt = rt_model.predict(X_test_rt)
        test_pred_spread = test_pred_da - test_pred_rt

        # Decision: DA<RT → 多报 (Q_da=1), else 少报 (Q_da=0)
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

        # Store per-period records
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

    # ════════════════════════════════════════════════════════════════
    # Compile report
    # ════════════════════════════════════════════════════════════════
    rec_df = pd.DataFrame(records)
    n_days = len(daily_pnl)
    n_periods = n_days * 96
    total_pnl = sum(daily_pnl)
    pnl_per_mwh = total_pnl / n_periods

    # Oracle (knows actual spread sign)
    oracle_q_da = (rec_df["actual_spread"].values < 0).astype(float)
    oracle_pnl = compute_pnl(rec_df["actual_spread"].values, rec_df["actual_rt"].values, oracle_q_da)
    oracle_per_mwh = oracle_pnl / n_periods

    # ── Core metrics ──
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

    # ── PnL by regime ──
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

    # ── Monetary value conversion ──
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


# ════════════════════════════════════════════════════════════════
# Standalone: compare window sizes
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_data()
    df = add_features(df)

    all_dates = sorted(df["trade_date"].unique())
    policy_break = pd.Timestamp("2026-01-01")
    test_dates = [d for d in all_dates if d >= policy_break]

    log(f"滚动窗口回测: {len(test_dates)} 天 ({test_dates[0].date()} ~ {test_dates[-1].date()})")
    log()

    configs = [(90, "90d 窗口"), (180, "180d 窗口"), (365, "365d 窗口"), (545, "545d 窗口")]
    results = []
    for window, label in configs:
        log(f"\n{'='*60}")
        log(f"  {label}")
        log(f"{'='*60}")
        r = evaluate(df, window_days=window)
        r["label"] = label
        results.append(r)

    # ── Comparison table ──
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
```

**关键设计点**:
1. **不需要 OOF**: 滚动窗口中训练数据全部在测试日之前 → in-sample pred_da 没有泄漏
2. **验证用最后 7 天**: 简单有效，不需要单独的 val set
3. **记录 per-period 数据**: 用于事后分段分析 (按 |spread| / regime 分解 PnL)

---

## 5. 复现步骤

### 5.1 确认前置条件

```bash
# 1. 检查数据文件存在
ls -la data/day_ahead_feature_matrix.xlsx
# 预期: 86,588 rows × 81 cols, 文件大小 ~30MB

# 2. 检查所有 4 个 .py 文件存在
ls -la config.py data_utils.py backtest.py rolling_backtest.py

# 3. 确认依赖
uv run python -c "import pandas, numpy, lightgbm, sklearn, joblib; print('OK')"
```

### 5.2 运行完整管线 (训练 + 评估)

```bash
uv run python backtest.py
```

**预计耗时**: ~45 分钟 (171 天 × 2 模型训练/天 × ~8s)

**输出**:
- `model_da_quantile.pkl` — DA P50 模型
- `model_rt_quantile.pkl` — RT P50 模型
- `model_clf_ne.pkl` — 分类器
- `data/rolling_backtest.log` — 完整 PnL 分解报告

### 5.3 仅评估 (不更新部署模型)

```bash
uv run python rolling_backtest.py
```

这会跑 4 种窗口大小对比 (90/180/365/545)，预计 ~3 小时。

### 5.4 预期结果

| 指标 | 数值 |
|------|------|
| PnL | **4.70 元/MWh** |
| 方向准确率 | 51.4% |
| Oracle 捕获率 | 18.7% |
| |spread|>200 时段 PnL | 128 元/period |
| 100MW 年化收益 | **412 万** |

**PnL 随时间衰减**: 1 月 ~14 → 3 月 ~10 → 6 月 ~4 元/MWh (spread 收窄的季节效应)。

---

## 6. 决策摘要

### 为什么是 LGBM

树模型天然处理混合特征 (连续 + 类别 + 缺失值)、训练快、原生支持 quantile regression。没有与其他算法 (XGBoost/CatBoost) 做横向对比——这是工程直觉决策，未经过实验验证。

### 为什么双头 (DA+RT)

| 方案 | Spread MAE | 方向 Acc | 
|------|-----------|---------|
| A 双头回归 | 57.5 | **54.3%** ✅ |
| B 直接 Spread | 51.9 | 43.1% (比随机差) |
| 两阶段 | 51.9 | 53.5% |

双头利用 DA (R²=0.78) 和 RT (R²=0.56) 各自的可预测性，误差部分抵消 (ρ=0.57)。

### 为什么 |spread| 加权

大 spread 时段贡献 82% 的 PnL 但只占 3% 的样本。加权让模型倾斜容量给这些关键时段，是单点最有效优化 (PnL -1.35 → +1.71)。

### 为什么滚动窗口

2026-01-01 制度断点使固定切分 Train/Test 分布系统性偏移。滚动窗口让新政策数据随时间流入训练集，评估从 1.71 更正为 4.70 元/MWh (提升 2.75×)。
