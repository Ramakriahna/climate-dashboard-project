import pandas as pd

def yearly_temperature(df):
    return df.groupby(["city", "year"], as_index=False).agg(
        avg_max_temp=("temperature_2m_max", "mean"),
        avg_min_temp=("temperature_2m_min", "mean"),
        avg_temp=("average_temperature", "mean")
    )

def annual_rainfall(df):
    return df.groupby(["city", "year"], as_index=False).agg(
        annual_rainfall=("rain_sum", "sum"),
        annual_precipitation=("precipitation_sum", "sum")
    )

def monthly_patterns(df):
    return df.groupby(["city", "month"], as_index=False).agg(
        avg_temperature=("average_temperature", "mean"),
        avg_max_temp=("temperature_2m_max", "mean"),
        avg_min_temp=("temperature_2m_min", "mean"),
        avg_rainfall=("rain_sum", "mean"),
        avg_wind_speed=("wind_speed_10m_max", "mean")
    )

def seasonal_patterns(df):
    return df.groupby(["city", "season"], as_index=False).agg(
        avg_temperature=("average_temperature", "mean"),
        avg_rainfall=("rain_sum", "mean"),
        total_rainfall=("rain_sum", "sum"),
        avg_wind_speed=("wind_speed_10m_max", "mean")
    )

def city_summary(df):
    return df.groupby("city", as_index=False).agg(
        avg_temperature=("average_temperature", "mean"),
        avg_max_temperature=("temperature_2m_max", "mean"),
        avg_min_temperature=("temperature_2m_min", "mean"),
        total_rainfall=("rain_sum", "sum"),
        avg_wind_speed=("wind_speed_10m_max", "mean"),
        max_recorded_temperature=("temperature_2m_max", "max")
    )
