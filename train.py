import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_data():
    df = pd.read_excel("data/day_ahead_feature_matrix.xlsx")
    df = df.sort_values(["trade_date", "period"]).reset_index(drop=True)
    return df


def prepare(df):
    target = "price_day_ahead"
    drop_cols = ["trade_date", "timestamp", "is_complete", "n_sources_ok"]
    features = [c for c in df.columns if c not in drop_cols and c != target]
    return features, target


def time_split(df):
    dates = sorted(df["trade_date"].unique())
    n = len(dates)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]
    mask_train = df["trade_date"].isin(train_dates)
    mask_val = df["trade_date"].isin(val_dates)
    mask_test = df["trade_date"].isin(test_dates)
    return mask_train, mask_val, mask_test


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100


def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100


def evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test,
             log_transform=False):
    results = {}
    for name, X, y in [("Train", X_train, y_train),
                        ("Val",   X_val,   y_val),
                        ("Test",  X_test,  y_test)]:
        pred_raw = model.predict(X)
        if log_transform:
            pred = np.expm1(pred_raw)
        else:
            pred = pred_raw

        results[name] = {
            "MAE":  mean_absolute_error(y, pred),
            "RMSE": np.sqrt(mean_squared_error(y, pred)),
            "MAPE": mape(y, pred),
            "sMAPE": smape(y, pred),
        }
    return results


def print_results(results):
    print(f"{'':>6} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'sMAPE':>8}")
    for split, m in results.items():
        print(f"{split:>6} {m['MAE']:8.2f} {m['RMSE']:8.2f} {m['MAPE']:7.2f}% {m['sMAPE']:7.2f}%")


def print_feature_importance(model, features, top_n=15):
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    for i, (name, v) in enumerate(imp.head(top_n).items()):
        print(f"  {i+1:2d}. {name:<35s} {v:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 2: log-transform + hyperparameter tuning
# ═══════════════════════════════════════════════════════════════════════════

df = load_data()
features, target = prepare(df)
mask_train, mask_val, mask_test = time_split(df)

X_train = df.loc[mask_train, features]
y_train = df.loc[mask_train, target]
X_val   = df.loc[mask_val,   features]
y_val   = df.loc[mask_val,   target]
X_test  = df.loc[mask_test,  features]
y_test  = df.loc[mask_test,  target]

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
print(f"Train y mean: {y_train.mean():.1f}  Val y mean: {y_val.mean():.1f}  Test y mean: {y_test.mean():.1f}")

# ── Baseline (from experiment 1) ────────────────────────────────────────
print("\n" + "=" * 60)
print("EXP 1 - Baseline")
print("=" * 60)
model1 = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=127,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
)
model1.fit(X_train, y_train,
           eval_set=[(X_val, y_val)], eval_metric="mae",
           callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
results1 = evaluate(model1, X_train, y_train, X_val, y_val, X_test, y_test)
print_results(results1)

# ── Log-transform target ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXP 2 - Log-transform target")
print("=" * 60)
y_train_log = np.log1p(y_train)
y_val_log   = np.log1p(y_val)

model2 = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=127,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
)
model2.fit(X_train, y_train_log,
           eval_set=[(X_val, y_val_log)], eval_metric="mae",
           callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
results2 = evaluate(model2, X_train, y_train, X_val, y_val, X_test, y_test, log_transform=True)  # fixed: raw y
print_results(results2)

# ── Tuned + log-transform ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXP 3 - Tuned + log-transform")
print("=" * 60)
model3 = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.03, num_leaves=255,
    min_child_samples=20, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, verbose=-1,
)
model3.fit(X_train, y_train_log,
           eval_set=[(X_val, y_val_log)], eval_metric="mae",
           callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
results3 = evaluate(model3, X_train, y_train, X_val, y_val, X_test, y_test, log_transform=True)
print_results(results3)

# ── Tuned + log + more features attention ────────────────────────────────
print("\n" + "=" * 60)
print("EXP 4 - Tuned + log + deeper trees")
print("=" * 60)
model4 = lgb.LGBMRegressor(
    n_estimators=1500, learning_rate=0.02, num_leaves=511,
    min_child_samples=10, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=0.5,
    random_state=42, verbose=-1, n_jobs=-1,
)
model4.fit(X_train, y_train_log,
           eval_set=[(X_val, y_val_log)], eval_metric="mae",
           callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
results4 = evaluate(model4, X_train, y_train, X_val, y_val, X_test, y_test, log_transform=True)
print_results(results4)

print("\n── EXP 4 Feature Importance ──")
print_feature_importance(model4, features)
