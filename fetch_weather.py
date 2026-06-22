"""Fetch weather forecast data from Open-Meteo Historical Forecast API.

Saves raw hourly data per city to data/weather/ as Parquet files.
Then validates against existing feature matrix data.

Usage:
    python fetch_weather.py              # fetch all missing dates
    python fetch_weather.py --validate   # only validate against existing
    python fetch_weather.py --merge      # apply backfill to feature matrix
"""

import argparse
import sys
import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path

# ── Config ──
CITIES = {
    # name: (lat, lon)
    "Lanzhou": (36.06, 103.83),
    "Wuwei": (37.93, 102.64),
    "Zhangye": (38.93, 100.45),
    "Jiuquan": (39.73, 98.52),
}

API_BASE = "https://historical-forecast-api.open-meteo.com/v1/forecast"
WEATHER_DIR = Path("data/weather")
WEATHER_DIR.mkdir(parents=True, exist_ok=True)

# ── Unit mapping (Open-Meteo → our columns) ──
# All units match: °C, m/s, W/m² — verified
OPEN_METEO_FIELDS = {
    "temperature_2m": "wx_temp_2m",
    "wind_speed_100m": "wx_wind_100m",
    "shortwave_radiation": "wx_ghi",
}

# Hourly fields requested from API
HOURLY_PARAMS = "temperature_2m,wind_speed_100m,shortwave_radiation"


def fetch_city(city, lat, lon, start_date, end_date):
    """Fetch hourly weather for one city, save as Parquet."""
    fname = WEATHER_DIR / f"{city}.parquet"

    url = (
        f"{API_BASE}"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly={HOURLY_PARAMS}"
        f"&timezone=Asia/Shanghai"
    )
    print(f"  Fetching {city} ({lat}, {lon}): {start_date} ~ {end_date} ...", end=" ", flush=True)

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame({
        "time": pd.to_datetime(data["time"]),
        "wx_temp_2m": data["temperature_2m"],
        "wx_wind_100m": data["wind_speed_100m"],
        "wx_ghi": data["shortwave_radiation"],
    })

    # Merge with existing if file exists
    if fname.exists():
        existing = pd.read_parquet(fname)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)

    df.to_parquet(fname, index=False)
    print(f"{len(df)} hours total")
    return df


def fetch_all(start_date, end_date):
    """Fetch all cities for a date range."""
    for city, (lat, lon) in CITIES.items():
        fetch_city(city, lat, lon, start_date, end_date)
        time.sleep(0.5)  # rate limit


def load_weather():
    """Load and combine all city weather data, returning hourly DataFrame per city."""
    data = {}
    for city in CITIES:
        fname = WEATHER_DIR / f"{city}.parquet"
        if fname.exists():
            data[city] = pd.read_parquet(fname)
            print(f"  {city}: {len(data[city])} hours "
                  f"({data[city]['time'].min().date()} ~ {data[city]['time'].max().date()})")
        else:
            print(f"  {city}: NOT FOUND")
    return data


def validate_against_existing(city_data):
    """Compare API data against existing feature matrix values."""
    print("\n" + "=" * 60)
    print("  VALIDATION: API vs existing feature matrix")
    print("=" * 60)

    df = pd.read_excel("data/day_ahead_feature_matrix.xlsx")
    df = df.sort_values(["trade_date", "period"]).reset_index(drop=True)

    # Get hourly existing data (one row per hour, use minute_slot==3 = :00 boundary)
    existing = df[df["minute_slot"] == 3][
        ["trade_date", "hour", "wx_temp_2m", "wx_wind_100m", "wx_ghi"]
    ].dropna(subset=["wx_ghi"])
    existing["time"] = existing.apply(
        lambda r: pd.Timestamp(r["trade_date"]) + pd.Timedelta(hours=int(r["hour"])),
        axis=1,
    )

    print(f"  Existing data: {len(existing)} hourly rows "
          f"({existing['trade_date'].min().date()} ~ {existing['trade_date'].max().date()})")

    # Try different weight combinations
    WEIGHT_COMBOS = {
        "Lanzhou only": {"Lanzhou": 1.0},
        "Wuwei only": {"Wuwei": 1.0},
        "Zhangye only": {"Zhangye": 1.0},
        "L+W+Z equal": {"Lanzhou": 0.33, "Wuwei": 0.34, "Zhangye": 0.33},
        "L+W+Z+J equal": {"Lanzhou": 0.25, "Wuwei": 0.25, "Zhangye": 0.25, "Jiuquan": 0.25},
    }

    print(f"\n  {'Weight Combo':<20} {'Temp':>18} {'Wind':>18} {'GHI':>18}")
    print(f"  {'':20} {'MAE':>6} {'Corr':>6} {'Bias':>6}  "
          f"{'MAE':>6} {'Corr':>6} {'Bias':>6}  {'MAE':>6} {'Corr':>6} {'Bias':>6}")

    best = {}
    for combo_name, weights in WEIGHT_COMBOS.items():
        # Build combined hourly DataFrame
        combined = None
        for city, w in weights.items():
            if city not in city_data:
                continue
            cdf = city_data[city].copy()
            cdf = cdf.rename(columns={
                "wx_temp_2m": f"t_{city}",
                "wx_wind_100m": f"w_{city}",
                "wx_ghi": f"g_{city}",
            })
            if combined is None:
                combined = cdf
            else:
                combined = pd.merge(combined, cdf, on="time", how="inner")

        # Weighted average
        combined["temp"] = sum(w * combined[f"t_{c}"] for c, w in weights.items())
        combined["wind"] = sum(w * combined[f"w_{c}"] for c, w in weights.items())
        combined["ghi"] = sum(w * combined[f"g_{c}"] for c, w in weights.items())

        # Merge with existing on time
        merged = pd.merge(existing, combined[["time", "temp", "wind", "ghi"]], on="time", how="inner")

        if len(merged) < 10:
            print(f"  {combo_name:<20}  (only {len(merged)} matching hours)")
            continue

        stats = {}
        for var, api_col, unit in [("Temp", "temp", "°C"), ("Wind", "wind", "m/s"), ("GHI", "ghi", "W/m²")]:
            orig = merged[f"wx_{var.lower().replace('temp','temp_2m').replace('wind','wind_100m')}"]
            # fix column names
            if var == "Temp":
                orig_col = "wx_temp_2m"
            elif var == "Wind":
                orig_col = "wx_wind_100m"
            else:
                orig_col = "wx_ghi"

            orig = merged[orig_col]
            api = merged[api_col]
            mae = np.abs(orig - api).mean()
            corr = np.corrcoef(orig, api)[0, 1] if len(orig) > 1 else 0
            bias = (api - orig).mean()
            stats[var] = (mae, corr, bias)

        print(f"  {combo_name:<20}  "
              f"{stats['Temp'][0]:5.1f} {stats['Temp'][1]:5.3f} {stats['Temp'][2]:+5.1f}  "
              f"{stats['Wind'][0]:5.1f} {stats['Wind'][1]:5.3f} {stats['Wind'][2]:+5.1f}  "
              f"{stats['GHI'][0]:5.0f} {stats['GHI'][1]:5.3f} {stats['GHI'][2]:+5.0f}")

        # Track best by MAE
        if "Temp" not in best or stats["Temp"][0] < best["Temp"][1]:
            best["Temp"] = (combo_name, stats["Temp"][0])
        if "Wind" not in best or stats["Wind"][0] < best["Wind"][1]:
            best["Wind"] = (combo_name, stats["Wind"][0])
        if "GHI" not in best or stats["GHI"][0] < best["GHI"][1]:
            best["GHI"] = (combo_name, stats["GHI"][0])

    print(f"\n  Best per variable:")
    for var in ["Temp", "Wind", "GHI"]:
        print(f"    {var}: {best[var][0]} (MAE={best[var][1]:.1f})")


def apply_backfill(city_data, weights):
    """Apply weighted weather data to fill NaN in feature matrix."""
    print("\n" + "=" * 60)
    print("  APPLYING BACKFILL")
    print("=" * 60)

    # Build combined hourly DataFrame
    combined = None
    for city, w in weights.items():
        cdf = city_data[city].copy()
        cdf = cdf.rename(columns={
            "wx_temp_2m": f"t_{city}",
            "wx_wind_100m": f"w_{city}",
            "wx_ghi": f"g_{city}",
        })
        if combined is None:
            combined = cdf
        else:
            combined = pd.merge(combined, cdf, on="time", how="inner")

    combined["wx_temp_2m"] = sum(w * combined[f"t_{c}"] for c, w in weights.items())
    combined["wx_wind_100m"] = sum(w * combined[f"w_{c}"] for c, w in weights.items())
    combined["wx_ghi"] = sum(w * combined[f"g_{c}"] for c, w in weights.items())
    combined["date"] = combined["time"].dt.date
    combined["hour"] = combined["time"].dt.hour

    weather_lookup = combined.set_index(["date", "hour"])[
        ["wx_temp_2m", "wx_wind_100m", "wx_ghi"]
    ]

    df = pd.read_excel("data/day_ahead_feature_matrix.xlsx")
    df = df.sort_values(["trade_date", "period"]).reset_index(drop=True)

    df["date"] = df["trade_date"].dt.date
    fill_count = 0
    for idx, row in df.iterrows():
        if pd.isna(row["wx_ghi"]):
            key = (row["date"], int(row["hour"]))
            if key in weather_lookup.index:
                df.at[idx, "wx_temp_2m"] = weather_lookup.loc[key, "wx_temp_2m"]
                df.at[idx, "wx_wind_100m"] = weather_lookup.loc[key, "wx_wind_100m"]
                df.at[idx, "wx_ghi"] = weather_lookup.loc[key, "wx_ghi"]
                fill_count += 1

    print(f"  Filled {fill_count} NaN rows")

    # Recompute derived features
    df["wx_wind_effective"] = np.maximum(df["wx_wind_100m"] - 4, 0)
    df["wx_wind_above_cutin"] = (df["wx_wind_100m"] >= 4).astype(int)
    df["floor_regime_proxy"] = (
        df["wx_ghi"] * df["is_solar_season"] * df["is_ne_high_gen"]
    )
    df["renewable_ratio_x_floor_regime"] = df["renewable_ratio"] * df["floor_regime_proxy"]
    df["ne_total_x_floor_regime"] = df["ne_total"] * df["floor_regime_proxy"]
    df["net_load_x_floor_regime"] = df["net_load"] * df["floor_regime_proxy"]
    df["solar_season_ghi"] = df["wx_ghi"] * df["is_solar_season"]
    df["load_fcst_x_wx_temp"] = df["load_fcst"] * df["wx_temp_2m"]
    df["ne_solar_per_ghi"] = np.where(df["wx_ghi"] > 0, df["ne_solar"] / df["wx_ghi"], np.nan)

    # Verify
    print(f"\n  Verification:")
    for c in ["wx_temp_2m", "wx_wind_100m", "wx_ghi", "floor_regime_proxy"]:
        n_nan = df[c].isna().sum()
        print(f"    {c}: {n_nan} NaN / {len(df)}")

    df = df.drop(columns=["date"])
    out_path = "data/day_ahead_feature_matrix.xlsx"
    df.to_excel(out_path, index=False)
    print(f"\n  Saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather data pipeline")
    parser.add_argument("--validate", action="store_true", help="Validate against existing")
    parser.add_argument("--merge", action="store_true", help="Apply backfill")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2025-04-30", help="End date")
    args = parser.parse_args()

    if args.validate:
        data = load_weather()
        if data:
            validate_against_existing(data)
        sys.exit(0)

    if args.merge:
        data = load_weather()
        if not data:
            print("No weather data. Run fetch first.")
            sys.exit(1)
        # Use best weights from validation (hard-coded after running --validate)
        weights = {"Lanzhou": 0.33, "Wuwei": 0.34, "Zhangye": 0.33}
        apply_backfill(data, weights)
        sys.exit(0)

    # Default: fetch
    print("=== Fetching Weather Data ===\n")
    print(f"Cities: {list(CITIES.keys())}")
    print(f"Date range: {args.start} ~ {args.end}")
    print(f"Output: {WEATHER_DIR}/\n")

    fetch_all(args.start, args.end)

    print("\n=== Fetch Complete ===")
    data = load_weather()
    if data:
        validate_against_existing(data)
