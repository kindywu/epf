"""Day-ahead electricity price prediction — Streamlit UI.

Cost optimization across two markets:
  - Day-ahead: bid today, cleared at marginal price, delivers tomorrow
  - Real-time:  same-day, covers shortfall at real-time price

Optimal bid for a buyer = expected real-time price (the cutoff).
  - If day-ahead clearing < real-time → buy day-ahead (bid high to ensure dispatch)
  - If day-ahead clearing > real-time → buy real-time (bid low, skip day-ahead)
  - Shortfall is settled at real-time price — not catastrophic, just cost optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from datetime import date, timedelta

TOP30 = [
    "day_of_year", "tie_total", "supply_gap", "ne_wind",
    "days_in_solar_term", "price_lag_1d", "hydro_fcst", "tie_新疆",
    "price_lag_7d", "coal_x_thermal_fcst", "dow", "thermal_fcst",
    "price_lag_2d", "solar_term_sin", "tie_湖南", "gen_fcst",
    "wx_temp_2m", "price_lag_3d", "tie_青海", "coal_x_net_load",
    "price_roll_30d_mean", "tie_宁夏", "price_lag_1d_dev_roll7",
    "tie_陕西", "tie_山东", "wx_wind_100m", "load_fcst_x_wx_temp",
    "price_roll_7d_mean", "days_from_holiday", "hydro_ratio",
]

MODEL_FILE = "model_quantile.pkl"
DATA_FILE = "data/day_ahead_feature_matrix.xlsx"


@st.cache_resource
def load_historical():
    df = pd.read_excel(DATA_FILE)
    return df.sort_values(["trade_date", "period"]).reset_index(drop=True)


@st.cache_resource
def load_models():
    try:
        return joblib.load(MODEL_FILE)
    except FileNotFoundError:
        return None


def build_calendar_features(target_date):
    rows = []
    for period in range(1, 97):
        slot = (period - 1) % 4
        hour = (period - 1) // 4
        ts = pd.Timestamp(target_date) + pd.Timedelta(minutes=15 * (period - 1))
        rows.append({
            "period": period, "hour": hour, "minute_slot": slot,
            "timestamp": ts, "trade_date": target_date,
            "day_of_year": ts.day_of_year, "dow": ts.dayofweek,
            "is_weekend": int(ts.dayofweek >= 5), "month": ts.month,
            "week_of_year": ts.isocalendar().week,
            "is_holiday": 0, "is_adjusted_workday": 0,
            "days_to_holiday": 7, "days_from_holiday": 7, "holiday_tier": 0,
            "is_heating_season": int(ts.month in [11, 12, 1, 2, 3]),
            "days_in_solar_term": ts.day_of_year % 15 + 1,
            "solar_term_sin": np.sin(2 * np.pi * ts.day_of_year / 365.25),
            "solar_term_cos": np.cos(2 * np.pi * ts.day_of_year / 365.25),
            "period_sin": np.sin(2 * np.pi * period / 96),
            "period_cos": np.cos(2 * np.pi * period / 96),
            "is_solar_season": int(ts.month in [4, 5, 6, 7, 8, 9]),
            "is_daytime_peak": int(8 <= hour <= 18),
        })
    return pd.DataFrame(rows)


def compute_lag_features(df_hist, df_tomorrow):
    hist_prices = df_hist["price_day_ahead"].values
    n = len(hist_prices)
    last_7d = hist_prices[-96 * 7:]
    last_30d = hist_prices[-96 * 30:]
    roll_7d_mean = np.mean(last_7d)
    roll_30d_mean = np.mean(last_30d)
    roll_7d_std = np.std(last_7d)

    for i, row in df_tomorrow.iterrows():
        p = row["period"] - 1
        idx_1d = n - 96 + p
        df_tomorrow.at[i, "price_lag_1d"] = hist_prices[idx_1d] if idx_1d >= 0 else np.nan
        df_tomorrow.at[i, "price_lag_2d"] = hist_prices[idx_1d - 96] if idx_1d >= 96 else np.nan
        df_tomorrow.at[i, "price_lag_3d"] = hist_prices[idx_1d - 192] if idx_1d >= 192 else np.nan
        idx_7d = n - 96 * 7 + p
        df_tomorrow.at[i, "price_lag_7d"] = hist_prices[idx_7d] if idx_7d >= 0 else np.nan
        df_tomorrow.at[i, "price_lag_1d_dev_roll7"] = (
            df_tomorrow.at[i, "price_lag_1d"] - roll_7d_mean
        )
        df_tomorrow.at[i, "price_roll_7d_mean"] = roll_7d_mean
        df_tomorrow.at[i, "price_roll_30d_mean"] = roll_30d_mean
        df_tomorrow.at[i, "price_roll_7d_std"] = roll_7d_std


def compute_derived(df):
    df["ne_total"] = df["ne_wind"] + df.get("ne_solar", 0)
    df["tie_total"] = (df.get("tie_宁夏", 0) + df.get("tie_山东", 0)
                       + df.get("tie_新疆", 0) + df.get("tie_湖南", 0)
                       + df.get("tie_陕西", 0) + df.get("tie_青海", 0))
    df["thermal_fcst"] = (df["gen_fcst"] - df["hydro_fcst"]
                          - df["ne_wind"] - df.get("ne_solar", 0))
    df["net_load"] = df["load_fcst"] - df["ne_wind"] - df.get("ne_solar", 0)
    df["supply_gap"] = df["gen_fcst"] - df["load_fcst"]
    df["renewable_ratio"] = (df["ne_wind"] + df.get("ne_solar", 0)) / df["gen_fcst"].clip(lower=1)
    df["hydro_ratio"] = df["hydro_fcst"] / df["gen_fcst"].clip(lower=1)
    df["wx_wind_effective"] = df.get("wx_wind_100m", 0)
    df["wx_wind_above_cutin"] = (df.get("wx_wind_100m", 0) > 3).astype(int)
    df["coal_x_net_load"] = df.get("coal_price_gansu", 500) * df["net_load"]
    df["coal_x_thermal_fcst"] = df.get("coal_price_gansu", 500) * df["thermal_fcst"]
    df["renewable_ratio_x_coal"] = df["renewable_ratio"] * df.get("coal_price_gansu", 500)
    df["load_fcst_x_wx_temp"] = df["load_fcst"] * df.get("wx_temp_2m", 0)
    return df


def auto_fill_external(df_hist, target_date):
    """Use last available day's external features as placeholder."""
    last_date = df_hist["trade_date"].max()
    last_day = df_hist[df_hist["trade_date"] == last_date]
    if len(last_day) != 96:
        target_dow = target_date.weekday()
        similar = df_hist[df_hist["trade_date"].dt.dayofweek == target_dow]
        if len(similar) >= 96:
            last_day = similar[similar["trade_date"] == similar["trade_date"].max()]
        else:
            return None

    externals = ["load_fcst", "gen_fcst", "hydro_fcst", "ne_wind", "ne_solar",
                 "tie_宁夏", "tie_山东", "tie_新疆", "tie_湖南", "tie_陕西", "tie_青海",
                 "wx_temp_2m", "wx_wind_100m", "wx_ghi", "coal_price_gansu"]
    values = {}
    for col in externals:
        values[col] = last_day[col].values if col in last_day.columns else np.full(96, np.nan)
    return values


# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="日前电价预测", page_icon="⚡", layout="wide")
st.title("⚡ 甘肃电力现货 — 日前电价预测 & 交易建议")

df_hist = load_historical()
models = load_models()
last_hist_date = df_hist["trade_date"].max().date()

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📅 预测日期")
    target_date = st.date_input(
        "选择日期", value=last_hist_date + timedelta(days=1),
        min_value=last_hist_date,
    )

    st.header("⚖️ 实时价预期")
    st.caption("缺口按实时价结算。实时价 < 日前预期 → 等实时更划算。")
    rt_mode = st.radio("实时价预期方式", ["固定值", "分时段"], horizontal=True)
    if rt_mode == "固定值":
        rt_price = st.number_input("预期实时价 (元/MWh)", value=180, step=10)
        rt_hourly = [rt_price] * 24
    else:
        st.caption("输入每小时实时价预期")
        rt_hourly = []
        for h in range(0, 24, 3):
            cols = st.columns(3)
            for j in range(3):
                hour = h + j
                with cols[j]:
                    v = st.number_input(f"{hour:02d}:00", value=180, step=10, key=f"rt_{hour}")
                    rt_hourly.append(v)

    st.header("💰 长协参数")
    lt_cost = st.number_input("长协成本 (元/MWh)", value=200, step=10)
    lt_qty = st.number_input("可用长协库存 (MWh/时段)", value=0, step=10,
                             help="T-2以上锁的、未到期的库存，可以日前卖出套利")

    st.header("📊 外部数据")
    st.caption("一键生成填充模拟值，未来对接正式服务。")
    if st.button("🔄 一键生成所有外部数据", width="stretch"):
        st.rerun()

    st.divider()
    if st.button("🛠 重新训练模型", width="stretch"):
        with st.spinner("训练中..."):
            X = df_hist[TOP30]
            y = df_hist["price_day_ahead"]
            new_models = {}
            for alpha in [0.1, 0.5, 0.9]:
                m = lgb.LGBMRegressor(
                    objective="quantile", alpha=alpha,
                    n_estimators=1000, learning_rate=0.03, num_leaves=255,
                    min_child_samples=20, subsample=0.7, colsample_bytree=0.7,
                    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
                )
                m.fit(X, y)
                new_models[alpha] = m
            joblib.dump(new_models, MODEL_FILE)
            st.cache_resource.clear()
        st.success(f"模型已训练 ({len(df_hist)} 行)")
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────
if not models:
    st.warning("⚠️ 模型未训练。请在侧边栏点击「重新训练模型」。")
    st.stop()

auto_data = auto_fill_external(df_hist, target_date)
if auto_data:
    st.info(f"📌 外部数据使用 {df_hist['trade_date'].max().date()} 的模拟值。未来对接调度/天气/交易系统。")

with st.spinner("构建特征 & 预测..."):
    df_tomorrow = build_calendar_features(target_date)
    for col, vals in auto_data.items():
        df_tomorrow[col] = vals
    compute_lag_features(df_hist, df_tomorrow)
    compute_derived(df_tomorrow)
    preds = {a: models[a].predict(df_tomorrow[TOP30]) for a in [0.1, 0.5, 0.9]}

# Build result table with trading logic
rt_by_period = np.repeat(rt_hourly, 4)[:96]

df_result = pd.DataFrame({
    "period": range(1, 97),
    "time": [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)],
    "日前P10": preds[0.1].round(0),
    "日前P50": preds[0.5].round(0),
    "日前P90": preds[0.9].round(0),
    "预期实时价": rt_by_period,
})

# Trading logic
df_result["日前vs实时"] = df_result["日前P50"] - df_result["预期实时价"]
df_result["建议"] = np.where(
    df_result["日前P50"] < df_result["预期实时价"],
    "📗 日前买",  # day-ahead cheaper → buy DA
    "📕 等实时",  # real-time cheaper → wait for RT
)
# Edge: if spread is small (<10), neutral
df_result.loc[df_result["日前vs实时"].abs() < 10, "建议"] = "📙 价差小"
# If very cheap in DA (<100), always buy
df_result.loc[df_result["日前P50"] < 100, "建议"] = "📗 地板价(日前买)"

# Sell signal: day-ahead high + have inventory
df_result["卖出信号"] = (
    (df_result["日前P50"] > lt_cost * 1.1) & (lt_qty > 0)
)

# ── Top Metrics ──────────────────────────────────────────────────────────
st.header(f"📈 {target_date} 预测结果")

da_buy = (df_result["建议"] == "📗 日前买").sum()
da_sell = df_result["卖出信号"].sum()
rt_wait = (df_result["建议"] == "📕 等实时").sum()
neutral = (df_result["建议"] == "📙 价差小").sum()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📗 日前买入", f"{da_buy} 时段")
col2.metric("📕 等实时", f"{rt_wait} 时段")
col3.metric("📙 价差小/地板", f"{neutral} 时段")
col4.metric("日前均价(P50)", f"{preds[0.5].mean():.0f} 元")
col5.metric("卖出信号", f"{da_sell} 时段", delta=f"库存{lt_qty}MWh" if lt_qty > 0 else "无库存")

# ── Chart ─────────────────────────────────────────────────────────────────
st.subheader("日前价预测 vs 实时价预期")

chart_df = df_result.set_index("period")[["日前P10", "日前P50", "日前P90", "预期实时价"]]
# Add a horizontal band for real-time price
st.line_chart(chart_df, height=350, color=["#ffcccc", "#cc3333", "#ff6666", "#3366cc"])

# ── Hourly decision matrix ────────────────────────────────────────────────
st.subheader("分时交易建议")

hourly = df_result.copy()
hourly["hour"] = hourly["period"].apply(lambda p: (p - 1) // 4)
hourly_agg = hourly.groupby("hour").agg(
    日前P50=("日前P50", "mean"),
    预期实时=("预期实时价", "mean"),
    价差=("日前vs实时", "mean"),
    DA买入=("建议", lambda x: (x == "📗 日前买").sum()),
    RT等待=("建议", lambda x: (x == "📕 等实时").sum()),
    卖出=("卖出信号", "sum"),
).round(0)
hourly_agg.index = [f"{h:02d}:00" for h in hourly_agg.index]
hourly_agg["建议动作"] = hourly_agg.apply(
    lambda r: "📗 日前买" if r["DA买入"] >= 3
    else ("📕 等实时" if r["RT等待"] >= 3 else "📙 混合"),
    axis=1,
)

st.dataframe(
    hourly_agg[["日前P50", "预期实时", "价差", "DA买入", "RT等待", "卖出", "建议动作"]],
    width="stretch", height=420,
    column_config={
        "日前P50": "日前P50", "预期实时": "预期实时", "价差": "价差",
        "DA买入": "日前买", "RT等待": "等实时", "卖出": "卖出信号",
    },
)

# ── Strategy Summary ──────────────────────────────────────────────────────
st.subheader("📋 交易策略")

# Cost comparison
avg_da = preds[0.5].mean()
avg_rt = np.mean(rt_by_period)
if avg_da < avg_rt:
    st.success(
        f"日前均价 {avg_da:.0f} < 实时预期 {avg_rt:.0f} → "
        f"**倾向于日前多买**，预计每 MWh 省 {avg_rt - avg_da:.0f} 元。"
        f"建议报价略高于预期实时价以确保成交。"
    )
else:
    st.warning(
        f"日前均价 {avg_da:.0f} > 实时预期 {avg_rt:.0f} → "
        f"**倾向于等实时**，预计每 MWh 省 {avg_da - avg_rt:.0f} 元。"
        f"日前报低价，买不到的量走实时。"
    )

# Sell advice
if lt_qty > 0 and da_sell > 0:
    sell_df = df_result[df_result["卖出信号"]]
    profit = sell_df["日前P50"].mean() - lt_cost
    st.success(
        f"💰 卖出信号: {da_sell} 个时段 P50 > 长协成本 {lt_cost}×1.1。"
        f"卖出库存 {lt_qty} MWh/时段，预期套利 {profit:.0f} 元/MWh。"
    )
elif lt_qty == 0:
    st.info("💡 当前无长协库存。如有 T-2 以上锁定的低成本库存，可在日前高价时卖出套利。")

# ── Download ──────────────────────────────────────────────────────────────
st.divider()
csv = df_result.to_csv(index=False)
st.download_button(
    "⬇️ 下载 96 时段完整预测 (CSV)", csv,
    f"prediction_{target_date}.csv", "text/csv", width="stretch",
)

with st.expander("查看完整 96 时段数据"):
    st.dataframe(
        df_result[["period", "time", "日前P10", "日前P50", "日前P90",
                    "预期实时价", "日前vs实时", "建议", "卖出信号"]],
        width="stretch", height=400,
    )
