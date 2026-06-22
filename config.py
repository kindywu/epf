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
