from pathlib import Path
import numpy as np
import pandas as pd

DATE_FORMAT = "%d-%m-%Y"
NUMERIC_COLS = [
    "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max",
    "apparent_temperature_min", "precipitation_sum", "rain_sum", "weather_code",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant"
]

def load_raw_data(path="data/india_2000_2024_daily_weather.csv"):
    df = pd.read_csv(path)
    required = {"city", "date", *NUMERIC_COLS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df

def season_from_month(month):
    # Broad meteorological seasons for India; monsoon is June-September.
    if month in (12,  1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Summer"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-monsoon"

def iqr_outlier_summary(df, columns=None):
    columns = columns or [c for c in NUMERIC_COLS if c in df.columns]
    rows = []
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({"column": col, "lower_bound": low, "upper_bound": high,
                     "outlier_count": int(((s < low) | (s > high)).sum())})
    return pd.DataFrame(rows)

def clean_data(df):
    original = df.copy()
    report = {
        "original_rows": int(len(original)),
        "original_missing_cells": int(original.isna().sum().sum()),
        "original_duplicates": int(original.duplicated().sum())
    }
    out = df.copy()
    out["city"] = out["city"].astype("string").str.strip().str.title()
    out["date"] = pd.to_datetime(out["date"], format=DATE_FORMAT, errors="coerce")
    for col in NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Remove rows without identifiers; interpolate/fill numeric gaps within each city.
    out = out.dropna(subset=["city", "date"])
    out = out.sort_values(["city", "date"]).drop_duplicates(subset=["city", "date"], keep="first")
    # Physical validity checks. Negative rainfall/wind and invalid wind bearings are impossible.
    for col in ["precipitation_sum", "rain_sum", "wind_speed_10m_max", "wind_gusts_10m_max"]:
        out.loc[out[col] < 0, col] = np.nan
    out.loc[~out["wind_direction_10m_dominant"].between(0, 360), "wind_direction_10m_dominant"] = np.nan
    out.loc[~out["weather_code"].between(0, 99), "weather_code"] = np.nan
    # Temperature max should not be below min. Treat inconsistent pairs as missing.
    bad_temp = out["temperature_2m_max"] < out["temperature_2m_min"]
    out.loc[bad_temp, ["temperature_2m_max", "temperature_2m_min"]] = np.nan
    # Rainfall should not exceed total precipitation by a material amount; retain the source
    # values but flag the inconsistency in a separate column.
    out["rain_precipitation_inconsistent"] = (
        out["rain_sum"].notna() & out["precipitation_sum"].notna() &
        (out["rain_sum"] > out["precipitation_sum"] + 0.1)
    )

    # Fill numeric missing values using within-city time interpolation, then city median.
    for col in NUMERIC_COLS:
        out[col] = out.groupby("city", observed=True)[col].transform(
            lambda s: s.interpolate(limit_direction="both")
        )
        out[col] = out[col].fillna(out.groupby("city", observed=True)[col].transform("median"))
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())

    # Keep genuine extremes. IQR is used for reporting, not automatic deletion/clipping.
    out["year"] = out["date"].dt.year.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    out["day"] = out["date"].dt.day.astype(int)
    out["day_of_year"] = out["date"].dt.dayofyear.astype(int)
    out["season"] = out["month"].map(season_from_month)
    out["temperature_difference"] = out["temperature_2m_max"] - out["temperature_2m_min"]
    out["average_temperature"] = (out["temperature_2m_max"] + out["temperature_2m_min"]) / 2
    out["rainfall_category"] = pd.cut(
        out["rain_sum"], bins=[-0.001, 0.1, 2.5, 7.5, 50, np.inf],
        labels=["No rain", "Light", "Moderate", "Heavy", "Very heavy"]
    ).astype("string")
    out = out.sort_values(["city", "date"]).reset_index(drop=True)

    report.update({
        "cleaned_rows": int(len(out)),
        "cleaned_missing_cells": int(out.isna().sum().sum()),
        "cleaned_duplicates": int(out.duplicated(subset=["city", "date"]).sum()),
        "removed_rows": int(report["original_rows"] - len(out))
    })
    return out, report

def run_cleaning(input_path="data/india_2000_2024_daily_weather.csv",
                 output_path="data/cleaned_weather.csv"):
    raw = load_raw_data(input_path)
    cleaned, report = clean_data(raw)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    return raw, cleaned, report
