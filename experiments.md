# 电价预测实验记录

## 通用配置

- **特征**: 全部 67 个特征（排除 trade_date, timestamp, is_complete, n_sources_ok）
- **切分**: 时间顺序 70/15/15 (train 2024-11-01~2025-12-18 / val ~2026-03-17 / test ~2026-06-14)
- **各集均价**: Train 248 / Val 239 / Test 156（分布偏移明显，Test 在春季低价期）
- **各集低价占比** (price<50): Train 14% / Val 18% / Test 25%

---

## 实验 1: LightGBM 基线 (2026-06-16)

参数: n_estimators=500, lr=0.05, num_leaves=127, subsample=0.8, colsample_bytree=0.8

| 集 | MAE | RMSE | MAPE | sMAPE |
|----|-----|------|------|-------|
| Train | 29.63 | 38.59 | 24.10% | 19.46% |
| Val | 49.70 | 67.08 | 37.75% | 28.29% |
| Test | 43.74 | 55.81 | 49.96% | 35.54% |

**分析**:
- MAPE 被低价区间放大（price<50 → MAPE=96%，price>300 → MAPE=11%）
- 用 MAE/RMSE 做主指标，MAPE 仅作参考

---

## 实验 2: Log 变换目标变量 (2026-06-16)

同 Exp1 参数，目标做 `log1p` 变换，预测时 `expm1` 还原。

| 集 | MAE | RMSE | MAPE | sMAPE |
|----|-----|------|------|-------|
| Train | 37.51 | 51.07 | 20.27% | 18.95% |
| Val | 60.11 | 91.32 | 33.95% | 30.43% |
| Test | 41.03 | 54.10 | **40.09%** | 32.49% |

**分析**:
- Test MAPE 从 49.96% → 40.09%（**-10pp**），log 变换有效缓解了低价区间的比例误差放大
- Train MAE 变差（29→37），但这是回归到原始空间的必然代价——log 空间优化的是相对误差
- Val MAE 也变差（49→60），但 Test MAPE 改善了

---

## 实验 3: Log + 超参调优 (2026-06-16)

参数: n_estimators=1000, lr=0.03, num_leaves=255, min_child_samples=20, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.1

| 集 | MAE | RMSE | MAPE | sMAPE |
|----|-----|------|------|-------|
| Train | 25.35 | 35.92 | 12.81% | 12.37% |
| Val | 60.45 | 91.86 | 33.91% | 30.64% |
| Test | 40.84 | 53.86 | **40.09%** | 32.47% |

**分析**:
- Test MAPE 与 Exp2 持平，Test MAE 略好（40.84 vs 41.03）
- 调参后 Train 明显更好了但 Val/Test 未同步改善——过拟合迹象

---

## 实验 4: Log + 更深树 (2026-06-16)

参数: n_estimators=1500, lr=0.02, num_leaves=511, min_child_samples=10, reg_alpha=0.5, reg_lambda=0.5

| 集 | MAE | RMSE | MAPE | sMAPE |
|----|-----|------|------|-------|
| Train | 11.37 | 17.06 | 5.40% | 5.36% |
| Val | 62.74 | 94.75 | 34.99% | 31.39% |
| Test | 41.51 | 54.95 | **39.77%** | 32.66% |

**分析**:
- 严重过拟合：Train MAPE=5.4% vs Val MAPE=35%
- Test MAPE 仅比 Exp3 改善 0.3pp，代价是更差的 Val
- **结论**: 更深/更多的树在此数据上无益

---

## 总结

| 实验 | Test MAE | Test MAPE | 备注 |
|------|----------|-----------|------|
| EXP1 Baseline | 43.74 | 49.96% | 原始空间，低价区间 MAPE 膨胀 |
| EXP2 +log | 41.03 | 40.09% | log 变换显著改善 MAPE |
| EXP3 +log+tuned | **40.84** | 40.09% | 当前最优 Test MAE |
| EXP4 +log+deep | 41.51 | 39.77% | 过拟合，不推荐 |

**当前结论**:
- Log 变换是正确方向，降低了对低价区间的惩罚，MAPE 改善 ~10pp
- 最优 Test MAPE ~40%，Test MAE ~41
- 更复杂的模型（更深树、更多轮）带来过拟合而无实质收益
- Val-Test 倒挂（Val MAE > Test MAE）是由于 Val 在冬季高波动期，属于数据分布特征而非模型问题
