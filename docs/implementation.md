# 电价预测系统实现文档

> 更新日期: 2026-06-23  
> 覆盖版本: 滚动窗口评估管线

---

## 1. 项目概述

### 1.1 目标

帮助甘肃电力市场买方（化工等高耗能连续生产用户）在日前-实时现货市场上做最优申报决策。核心问题是**日度二元决策**：对每个 15 分钟时段，判断日前价 (DA) 是否低于实时价 (RT)，若 DA<RT 则在日前多报（锁定低价），否则少报（等实时市场）。

### 1.2 文件结构

```
├── config.py              # 特征定义、模型参数、regime 定义
├── data_utils.py          # 共享函数：数据加载、特征工程、PnL 计算
├── backtest.py            # 模型训练 + 保存 + 滚动窗口评估
├── rolling_backtest.py    # 滚动窗口评估核心逻辑 + 窗口大小对比
├── predict.py             # 单日预测 CLI（加载已训练模型）
├── app.py                 # Streamlit UI
├── experiment_weights.py  # 样本权重消融实验
├── fetch_weather.py       # 天气数据抓取
│
├── data/
│   ├── day_ahead_feature_matrix.xlsx  # 核心特征矩阵 (86,588 × 81)
│   ├── weather/                        # 原始天气数据 (Parquet)
│   └── rolling_backtest.log           # 回测输出日志
│
├── model_da_quantile.pkl  # DA P50 模型 (部署用)
├── model_rt_quantile.pkl  # RT P50 模型 (部署用)
├── model_clf_ne.pkl       # 新能源大发分类器 (部署用)
│
├── docs/
│   ├── analysis_report.md      # 分析报告 (模型性能 + PnL)
│   ├── data_quality_report.md  # 数据质量检查报告
│   ├── story.md / story2.md    # 案例叙事
│   └── talk.md                 # 战略分析
│
└── experiments.md         # 实验记录
```

### 1.3 依赖

```bash
uv add pandas numpy lightgbm scikit-learn joblib openpyxl streamlit
```

---

## 2. 数据管线

### 2.1 数据源

特征矩阵 `data/day_ahead_feature_matrix.xlsx` 包含：

| 类别 | 来源 | 粒度 | 覆盖 |
|------|------|------|------|
| 价格 (DA/RT/Spread) | 电网结算系统 | 15min | 2024-01-01 ~ 2026-06-20 |
| 事前预测 (新能源/负荷/水电) | 调度机构 | 15min | 同上 |
| 联络线 | 调度机构 | 15min | 同上 |
| 天气 (温度/风速/GHI) | Open-Meteo API | 1h | 同上 (backfill 完成) |
| 煤价 | 公开数据 | 月度 | 2024-05 起 |
| 日历/节气/假期 | 计算 | — | 全覆盖 |

**关键数据质量问题**: 见 `docs/data_quality_report.md`。

### 2.2 制度断点

**甘工信发〔2025〕268 号** (2025-12-31 签发, 2026-01-01 生效):

| 变更项 | 旧 | 新 |
|--------|----|----|
| 申报上限 | 650 元/MWh | 500 元/MWh |
| 新能源 | 不参与 | 报量不报价 |
| 火电补偿 | 申报价 | min(307.8, 申报价) |

这意味着 2024-01-01 ~ 2025-12-31 与 2026-01-01 ~ 2026-06-20 的定价机制不同，是**结构性断点**。

### 2.3 数据加载与特征工程

`data_utils.py` 提供三个核心函数：

```python
def load_data():
    """加载并排序特征矩阵。"""
    df = pd.read_excel("data/day_ahead_feature_matrix.xlsx")
    return df.sort_values(["trade_date", "period"]).reset_index(drop=True)

def add_features(df):
    """动态计算特征：扩展滞后、价格桶、交互项。"""
    # 扩展滞后 (rt_lag_2d/3d/7d, spread_lag_2d/3d/7d)
    # 滚动统计 (rt_roll_7d_std)
    # D-1 价格桶 (lag_da_floor/low/mid/high)
    # 交互项 (ne_high × lag_da_*, lag_spread_sign)
    ...

def build_feature_sets():
    """返回 RT 和 Spread 模型的特征列表。"""
    # RT: DA_FEATURES + RT_LAG_FEATURES + EXPANDED_FEATURES
    #      + EXTRA_LAGS + PRICE_BUCKETS + LAG_SIGN + INTERACTIONS
    # Spread: 与 RT 相同 (去重)
    ...
```

---

## 3. 特征体系

### 3.1 DA 模型特征 (30 个 TOP-30)

从 `config.py` 的 `DA_FEATURES`:

| 类别 | 特征 |
|------|------|
| 事前预测 (7) | `ne_wind`, `hydro_fcst`, `gen_fcst`, `thermal_fcst`, `tie_*` (6 条联络线中 6 条) |
| 历史价格 (8) | `price_lag_1d/2d/3d/7d`, `price_roll_7d/30d_mean`, `price_lag_1d_dev_roll7` |
| 日历 (5) | `day_of_year`, `dow`, `days_in_solar_term`, `solar_term_sin`, `days_from_holiday` |
| 气象 (2) | `wx_temp_2m`, `wx_wind_100m` |
| 衍生/交互 (8) | `supply_gap`, `coal_x_thermal_fcst`, `coal_x_net_load`, `load_fcst_x_wx_temp`, `hydro_ratio` 等 |

**选型依据**: 实验 5 消融 (67 → 30 特征, Test MAPE 40.09% → 38.83%)。

### 3.2 RT 模型特征 (75 个)

DA 的 30 个 + `RT_LAG_FEATURES` (4) + `EXPANDED_FEATURES` (25) + 动态计算 (16):

- RT/spread 历史滞后 + 滚动均值
- 扩展事前预测 (ne_solar, load_fcst)
- Regime 标记 (is_ne_high_gen, is_load_peak, is_grid_other)
- 日历扩展 (is_weekend, is_holiday, month, period_sin/cos)
- 贴地交互 (floor_regime_proxy 等 5 个)
- 价格桶 + 交互 (lag_da_floor 等, ne_high_x_lag_da_*)
- pred_da (DA 模型预测值, 作为 RT 特征)

### 3.3 特征消融证据

| 操作 | 效果 | 来源 |
|------|------|------|
| 67 → TOP-30 (DA) | MAPE -1.26pp | experiments.md Exp 5 |
| 天气特征移除 | DA R² -0.001 (可忽略) | analysis_report.md §4.4 |
| 样本加权 (RT) | PnL -1.35 → +1.71 | experiment_weights.py |

---

## 4. 模型架构

### 4.1 三个模型

| 模型 | 算法 | 目标 | 特征 | 加权 | 用途 |
|------|------|------|------|------|------|
| DA P50 | LGBM Quantile (α=0.5) | `price_day_ahead` | DA_FEATURES (30) | 无 | 预测日前价 |
| RT P50 | LGBM Quantile (α=0.5) | `price_realtime` | RT_FEATURES + pred_da (75+1) | clip(\|spread\|, 0, 200) | 预测实时价 |
| 分类器 | LGBMClassifier | spread > 20 | RT_FEATURES | class_weight=balanced | 新能源大发 regime |

### 4.2 参数配置

```python
# DA 模型 (config.py → backtest.py LGB_QUANTILE)
LGBMRegressor(
    objective="quantile", alpha=0.5,
    n_estimators=1000, learning_rate=0.03, num_leaves=255,
    min_child_samples=20, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42,
)

# RT 模型 (LGB_RT)
LGBMRegressor(
    objective="quantile", alpha=0.5,
    n_estimators=2000, learning_rate=0.015, num_leaves=127,
    min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=0.5, random_state=42,
)
```

### 4.3 为什么双头 (DA + RT) 而非单头 (Spread 直接回归)

| 方案 | Spread MAE | 方向准确率 | 选择 |
|------|-----------|-----------|------|
| A 双头回归 (pred_da - pred_rt) | 57.5 | **54.3%** | ✅ 采用 |
| B 直接 Spread 回归 | 51.9 | 43.1% | ✗ 方向比随机差 |
| 两阶段 (pred_da 特征) | 51.9 | 53.5% | ✗ 无增益 |

双头利用 DA 和 RT 各自的可预测性 (D-1 R²: DA=0.78, RT=0.56)，且预测误差部分抵消 (ρ=0.57)。

---

## 5. 训练管线 (`backtest.py`)

### 5.1 流程

```
1. load_data() → add_features()
2. 固定切分 (70/15/15) → train_masks/val_masks
3. Step 1: 3-fold OOF pred_da (避免目标泄漏)
4. Step 2: 训练最终 DA 模型 (不加权)
5. Step 3: 训练 RT 模型 (pred_da 特征, |spread| 加权)
6. Step 4: 训练分类器 (spread > 20)
7. 保存三个模型 → .pkl 文件
8. 调用 evaluate() → 滚动窗口评估 (见 §6)
```

### 5.2 OOF pred_da 机制

```python
# 3-fold TimeSeriesSplit on train dates
tscv = TimeSeriesSplit(n_splits=3)
for fold, (tr_idx, val_idx) in enumerate(tscv.split(train_dates)):
    # 训练 DA → 预测 fold 验证集 → 收集 OOF pred_da
oof_da_train = [fold1_pred, fold2_pred, fold3_pred]
```

**目的**: RT 模型训练时需要 pred_da 作为特征。直接使用 DA 模型的 in-sample 预测会导致目标泄漏 (DA 模型见过这些样本的 label)。OOF 确保 RT 训练时的 pred_da 来自未见过的 fold。

### 5.3 样本权重

```python
# RT 模型: 按 |spread| 大小加权
rt_sample_weight = np.clip(|spread_da_rt|, 0, 200)
```

**理由**: 大 spread 时段的决策错误代价更大 (PnL 贡献主要来自 |spread|>200 的时段)。加权让模型容量倾斜给这些关键时段，实验证明这将 PnL 从 -1.35 翻转为 +1.71。

---

## 6. 评估管线 (`rolling_backtest.py`)

### 6.1 为什么用滚动窗口替代固定切分

**固定切分的问题**:

```
Train (631d)                Val     Test (136d)
├─────────────────────────┼──────┼──────────────┤
2024-01                 2025-09  2026-02    2026-06
  全部旧政策                  ←跨越断点→  全部新政策
```

- Train 全在旧政策 (申报上限 650)，Test 全在新政策 (上限 500)
- 分布系统性偏移: Test 均价 165 仅为 Train (238) 的 69%
- 评估混淆了"预测能力"和"分布外泛化"

**滚动窗口的解决**:

```
每天都用过去 365 天重新训练，预测明天:

第 1 天: 用 2025-01-01~2025-12-31 训练 → 预测 2026-01-01
第 30 天: 用 2025-02-01~2026-01-30 训练 → 预测 2026-01-31
...
第 171 天: 用 2025-07-01~2026-06-19 训练 → 预测 2026-06-20
```

新政策数据随时间自然流入训练窗口，模型自动适应。

### 6.2 实现细节

```python
def evaluate(df, window_days=365):
    """滚动窗口回测 + 完整 PnL 报告。"""
    test_dates = [d for d in all_dates if d >= pd.Timestamp("2026-01-01")]

    for test_date in test_dates:
        # 1. 构建训练窗口: [test_date - window_days, test_date - 1]
        train_dates = all_dates[test_idx - window_days : test_idx]

        # 2. 训练 DA (不加权)
        da_model = train_model(X_tr_da, y_tr_da, ...)

        # 3. 训练 RT (|spread| 加权) — 注意: 不需要 OOF!
        #    因为训练数据全部在 test_date 之前，不存在目标泄漏
        pred_da_tr = da_model.predict(X_tr_da)         # in-sample, 安全
        X_tr_rt["pred_da"] = pred_da_tr
        rt_model = train_model(X_tr_rt, y_tr_rt, ..., sample_weight=spread_w)

        # 4. 预测明天
        pred_da = da_model.predict(X_test_da)
        pred_rt = rt_model.predict(X_test_rt + pred_da)
        pred_spread = pred_da - pred_rt

        # 5. 决策 + 计算 PnL
        q_da = (pred_spread < 0).astype(float)
        pnl = (1 - q_da) * actual_spread  # 节省 = 少报 × spread

    # 输出完整报告: PnL 按 |spread| 分段、按 regime、日分布、价值换算
```

**关键简化**: 滚动窗口中 pred_da 不需要 OOF —— 训练数据全部在测试日之前，无泄漏。

### 6.3 窗口大小选择

| 窗口 | PnL/MWh | 方向 Acc | 分析 |
|------|---------|---------|------|
| 90d | 3.71 | 52.0% | 太短，不稳定 |
| 180d | 4.18 | 50.9% | 仅跨 2 季 |
| **365d** | **4.70** | **51.4%** | **最优 — 完整年周期** |
| 545d | 4.25 | 51.5% | 旧政策拖累 |
| 365d+衰减 | 4.26 | 51.4% | 衰减无增益 |

365 天 = 覆盖完整季节周期 (四季 + 供暖/非供暖)，同时旧政策数据占比随窗口滑动自然降低。

### 6.4 评估输出

报告包含:
1. **核心指标**: 总 PnL, 元/MWh, 元/天, 方向准确率, Oracle 捕获率
2. **PnL 按 |Spread| 分段**: 识别哪些价差幅度贡献了利润/亏损
3. **PnL 按 Regime**: 新能源大发/用电高峰/其他时段的分段表现
4. **日 PnL 分布**: 均值/标准差/最差日/最佳日
5. **价值换算**: 各容量下的年化收益估算

---

## 7. 预测服务 (`predict.py`)

```python
# 加载部署模型 (由 backtest.py 训练并保存)
da_model = joblib.load("model_da_quantile.pkl")
rt_model = joblib.load("model_rt_quantile.pkl")
clf_model = joblib.load("model_clf_ne.pkl")

# 加载特征矩阵, 构建目标日期特征
df = load_data()
df = add_features(df)
target_date = pd.Timestamp("2026-06-21")

# 预测
pred_da = da_model.predict(X_da[target_date])
pred_rt = rt_model.predict(X_rt[target_date] + pred_da)
pred_spread = pred_da - pred_rt

# Hybrid 策略
if is_ne_high_gen:
    direction = clf_model.predict_proba(...)[:, 1] > 0.5
else:
    direction = pred_spread < 0  # DA<RT → 多报

# 输出: predictions.csv
```

---

## 8. 运行指南

### 8.1 训练 + 评估

```bash
# 完整管线: 训练模型 + 滚动窗口回测
uv run python backtest.py

# 输出:
#   - model_da_quantile.pkl, model_rt_quantile.pkl, model_clf_ne.pkl
#   - data/rolling_backtest.log (完整 PnL 报告)
```

### 8.2 单日预测

```bash
# 预测指定日期
uv run python predict.py --date 2026-06-21

# 输出: predictions.csv (96 periods × 决策)
```

### 8.3 窗口大小对比实验

```bash
# 对比多种窗口大小 (仅评估, 不更新部署模型)
uv run python rolling_backtest.py
```

### 8.4 Streamlit 界面

```bash
uv run streamlit run app.py
```

---

## 9. 当前结果摘要 (2026-06-23)

| 指标 | 数值 |
|------|------|
| 评估方式 | 365d 滚动窗口 |
| 测试期 | 2026-01-01 ~ 2026-06-20 (171 天) |
| PnL | **4.70 元/MWh** (452 元/天) |
| 方向准确率 | 51.4% |
| Oracle 捕获率 | 18.7% |
| 正收益日 | 53% |
| 100MW 年化 | **412 万** |
| 核心 PnL 来源 | \|spread\|>200 的时段 (3% 样本, 82% 利润) |

---

## 10. 待改进项

| # | 方向 | 预期收益 | 难度 |
|---|------|---------|------|
| 1 | 中等 spread (50-100) 的方向判断 — 当前在此区间亏钱 | 中 | 中 |
| 2 | 新能源大发 regime 的分类器效果有限 (P1nL 仅 0.75/MWh) | 低-中 | 低 |
| 3 | add_features() 中的 iterrows 性能优化 | — | 低 |
| 4 | 新政策数据积累 (等 2027 年后可独立评估新政策) | 高 (评估可信度) | — |
