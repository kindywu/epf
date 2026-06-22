"""Backfill missing weather data (2024-01 ~ 2025-04) from Open-Meteo Historical Forecast API.

Source: Open-Meteo Historical Forecast (D-1 style forecasts, NOT reanalysis)
Weights: Lanzhou+Wuwei+Zhangye equal-weight per the "多代表点加权" approach.
"""

import pandas as pd
import numpy as np
import requests
import time
import sys
from datetime import datetime, timedelta

# ── Configuration ──
CITIES = {
    "Lanzhou": (36.06, 103.83),
    "Wuwei": (37.93, 102.64),
    "Zhangye": (38.93, 100.45),
    # Jiuquan included for GHI to better capture western Gansu solar
    "Jiuquan": (39.73, 98.52),
}

# Weights derived from optimization against existing data
# Temp: Lanzhou+Wuwei+Zhangye equal
# Wind: Lanzhou+Wuwei+Zhangye equal
# GHI: add Jiuquan weight to capture western Gansu solar regime
WEIGHTS = {
    "temp": {"Lanzhou": 0.33, "Wuwei": 0.34, "Zhangye": 0.33},
    "wind": {"Lanzhou": 0.33, "Wuwei": 0.34, "Zhangye": 0.33},
    "ghi":  {"Lanzhou": 0.20, "Wuwei": 0.20, "Zhangye": 0.20, "Jiuquan": 0.40},
}

API_BASE = "https://historical-forecast-api.open-meteo.com/v1/forecast"
BATCH_DAYS = 30  # fetch 30 days per API call
RATE_LIMIT = 0.5  # seconds between calls


def fetch_city(city_name, lat, lon, start_date, end_date):
    """Fetch hourly weather forecast data for a date range."""
    url = (
        f"{API_BASE}"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,wind_speed_100m,shortwave_radiation"
        f"&timezone=Asia/Shanghai"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    return {
        "time": pd.to_datetime(data["time"]),
        "temp": np.array(data["temperature_2m"]),
        "wind": np.array(data["wind_speed_100m"]),
        "ghi": np.array(data["shortwave_radiation"]),
    }


def backfill_period(start_date, end_date):
    """Fetch and combine weather data for a date range."""
    print(f"  Fetching {start_date} ~ {end_date} ...", end=" ", flush=True)

    city_data = {}
    for name, (lat, lon) in CITIES.items():
        try:
            city_data[name] = fetch_city(name, lat, lon, start_date, end_date)
        except Exception as e:
            print(f"\n  ERROR ({name}): {e}")
            return None
        time.sleep(RATE_LIMIT)

    # Combine with weights
    times = city_data["Lanzhou"]["time"]
    n_hours = len(times)

    temp = np.zeros(n_hours)
    wind = np.zeros(n_hours)
    ghi = np.zeros(n_hours)

    for city in WEIGHTS["temp"]:
        w = WEIGHTS["temp"][city]
        temp += w * city_data[city]["temp"]

    for city in WEIGHTS["wind"]:
        w = WEIGHTS["wind"][city]
        wind += w * city_data[city]["wind"]

    for city in WEIGHTS["ghi"]:
        w = WEIGHTS["ghi"][city]
        ghi += w * city_data[city]["ghi"]

    print(f"OK ({n_hours} hours)")
    return pd.DataFrame({
        "time": times,
        "wx_temp_2m": temp,
        "wx_wind_100m": wind,
        "wx_ghi": ghi,
    })


def validate(existing_df, backfilled_df):
    """Compare backfilled values against existing data on overlap dates."""
    overlap = pd.merge(
        existing_df[["trade_date", "hour", "wx_temp_2m", "wx_wind_100m", "wx_ghi"]],
        backfilled_df,
        on=["trade_date", "hour"],
        suffixes=("_orig", "_bf"),
    )
    if len(overlap) == 0:
        print("  No overlapping dates to validate.")
        return

    for var in ["wx_temp_2m", "wx_wind_100m", "wx_ghi"]:
        orig = overlap[f"{var}_orig"]
        bf = overlap[f"{var}_bf"]
        mae = np.abs(orig - bf).mean()
        corr = np.corrcoef(orig, bf)[0, 1]
        print(f"  {var}: MAE={mae:.2f}, Corr={corr:.3f}, n={len(overlap)}")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Weather Data Backfill ===\n")

    # Load existing data
    df = pd.read_excel("data/day_ahead_feature_matrix.xlsx")
    df = df.sort_values(["trade_date", "period"]).reset_index(drop=True)

    # Identify missing periods
    wx_missing = df["wx_ghi"].isna()
    missing_dates = sorted(df.loc[wx_missing, "trade_date"].unique())
    existing_dates = sorted(df.loc[~wx_missing, "trade_date"].unique())

    print(f"Existing weather: {existing_dates[0].date()} ~ {existing_dates[-1].date()} "
          f"({len(existing_dates)} days)")
    print(f"Missing weather:  {missing_dates[0].date()} ~ {missing_dates[-1].date()} "
          f"({len(missing_dates)} days)")

    if len(missing_dates) == 0:
        print("\nNo missing weather data. Done.")
        sys.exit(0)

    # ── Step 1: Validate approach on existing data ──
    print("\n── Validating on existing data (3 sample days) ──")
    sample_dates = existing_dates[:3]
    for d in sample_dates:
        result = backfill_period(d.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d"))
        if result is not None:
            # Compare at hour level
            existing_hourly = (
                df[(df.trade_date == d) & (df.minute_slot == 3)]
                [["trade_date", "hour", "wx_temp_2m", "wx_wind_100m", "wx_ghi"]]
            )
            result["trade_date"] = pd.Timestamp(d)
            result["hour"] = result["time"].dt.hour
            merged = pd.merge(existing_hourly, result, on=["trade_date", "hour"])
            if len(merged) > 0:
                for var in ["wx_temp_2m", "wx_wind_100m", "wx_ghi"]:
                    mae = np.abs(merged[f"{var}_x"] - merged[f"{var}_y"]).mean()
                    corr = np.corrcoef(merged[f"{var}_x"], merged[f"{var}_y"])[0, 1]
                    print(f"    {d.date()} {var}: MAE={mae:.2f}, Corr={corr:.3f}")

    # ── Step 2: Fetch missing period in batches ──
    print(f"\n── Fetching {len(missing_dates)} missing days in batches of {BATCH_DAYS} ──")
    print(f"    Estimated API calls: {len(CITIES) * (len(missing_dates) // BATCH_DAYS + 1)}")
    print(f"    Estimated time: {len(CITIES) * (len(missing_dates) // BATCH_DAYS + 1) * RATE_LIMIT / 60:.0f} min")

    all_backfill = []
    batch_start = 0
    while batch_start < len(missing_dates):
        batch_end = min(batch_start + BATCH_DAYS, len(missing_dates))
        start = missing_dates[batch_start].strftime("%Y-%m-%d")
        end = missing_dates[batch_end - 1].strftime("%Y-%m-%d")
        batch_df = backfill_period(start, end)
        if batch_df is not None:
            all_backfill.append(batch_df)
        batch_start = batch_end

    if not all_backfill:
        print("ERROR: No data fetched.")
        sys.exit(1)

    weather_hourly = pd.concat(all_backfill, ignore_index=True)
    weather_hourly["date"] = weather_hourly["time"].dt.date
    weather_hourly["hour"] = weather_hourly["time"].dt.hour

    # ── Step 3: Merge into feature matrix ──
    print(f"\n── Merging into feature matrix ──")
    df["date"] = df["trade_date"].dt.date

    # Map hourly weather to 96-period rows
    weather_map = weather_hourly.set_index(["date", "hour"])[
        ["wx_temp_2m", "wx_wind_100m", "wx_ghi"]
    ]

    fill_count = 0
    for idx, row in df.iterrows():
        if pd.isna(row["wx_ghi"]):
            key = (row["date"], int(row["hour"]))
            if key in weather_map.index:
                df.at[idx, "wx_temp_2m"] = weather_map.loc[key, "wx_temp_2m"]
                df.at[idx, "wx_wind_100m"] = weather_map.loc[key, "wx_wind_100m"]
                df.at[idx, "wx_ghi"] = weather_map.loc[key, "wx_ghi"]
                fill_count += 1

    print(f"  Filled {fill_count} rows")

    # ── Step 4: Recompute derived features that depend on weather ──
    print(f"\n── Recomputing derived features ──")

    # wx_wind_effective
    df["wx_wind_effective"] = np.maximum(df["wx_wind_100m"] - 4, 0)
    df["wx_wind_above_cutin"] = (df["wx_wind_100m"] >= 4).astype(int)

    # floor_regime_proxy = wx_ghi × is_solar_season × is_ne_high_gen
    df["floor_regime_proxy"] = (
        df["wx_ghi"].fillna(0) * df["is_solar_season"] * df["is_ne_high_gen"]
    )
    # Derived from floor_regime_proxy
    df["renewable_ratio_x_floor_regime"] = df["renewable_ratio"] * df["floor_regime_proxy"]
    df["ne_total_x_floor_regime"] = df["ne_total"] * df["floor_regime_proxy"]
    df["net_load_x_floor_regime"] = df["net_load"] * df["floor_regime_proxy"]

    # solar_season_ghi
    df["solar_season_ghi"] = df["wx_ghi"] * df["is_solar_season"]

    # load_fcst_x_wx_temp
    df["load_fcst_x_wx_temp"] = df["load_fcst"] * df["wx_temp_2m"]

    # ne_solar_per_ghi
    df["ne_solar_per_ghi"] = np.where(
        df["wx_ghi"] > 0, df["ne_solar"] / df["wx_ghi"], np.nan
    )

    # ── Step 5: Verify and save ──
    print(f"\n── Verification ──")
    for c in ["wx_temp_2m", "wx_wind_100m", "wx_ghi", "floor_regime_proxy",
              "load_fcst_x_wx_temp", "solar_season_ghi"]:
        n_nan = df[c].isna().sum()
        print(f"  {c}: {n_nan} NaN / {len(df)} ({100 - n_nan/len(df)*100:.1f}% ok)")

    # Drop helper column
    df = df.drop(columns=["date"])

    out_path = "data/day_ahead_feature_matrix.xlsx"
    print(f"\n── Saving to {out_path} ──")
    df.to_excel(out_path, index=False)
    print("Done!")
