import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid", palette="crest")

def temperature_trend(df, cities=None):
    data = df if not cities else df[df["city"].isin(cities)]
    yearly = data.groupby(["year", "city"], as_index=False)["average_temperature"].mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=yearly, x="year", y="average_temperature", hue="city", ax=ax)
    ax.set(title="Annual Average Temperature by City", xlabel="Year", ylabel="Temperature (°C)")
    fig.tight_layout()
    return fig

def monthly_temperature(df, cities=None):
    data = df if not cities else df[df["city"].isin(cities)]
    monthly = data.groupby("month", as_index=False)["average_temperature"].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=monthly, x="month", y="average_temperature", marker="o", ax=ax)
    ax.set(title="Average Temperature by Month", xlabel="Month", ylabel="Temperature (°C)")
    ax.set_xticks(range(1, 13))
    fig.tight_layout()
    return fig

def rainfall_distribution(df):
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df["rain_sum"], bins=45, log_scale=(False, True), ax=ax)
    ax.set(title="Rainfall Distribution (log count scale)", xlabel="Rainfall (mm)", ylabel="Count")
    fig.tight_layout()
    return fig

def city_temperature(df):
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=df, x="city", y="average_temperature", ax=ax)
    ax.set(title="City-wise Temperature Distribution", xlabel="City", ylabel="Temperature (°C)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig

def city_rainfall(df):
    summary = df.groupby("city", as_index=False)["rain_sum"].mean().sort_values("rain_sum", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=summary, x="city", y="rain_sum", ax=ax)
    ax.set(title="Average Daily Rainfall by City", xlabel="City", ylabel="Rainfall (mm/day)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig

def monthly_heatmap(df):
    pivot = df.pivot_table(index="city", columns="month", values="average_temperature", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(pivot, cmap="coolwarm", annot=True, fmt=".1f", ax=ax)
    ax.set(title="City vs Month: Average Temperature (°C)", xlabel="Month", ylabel="City")
    fig.tight_layout()
    return fig

def wind_distribution(df):
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df["wind_speed_10m_max"], bins=35, kde=True, ax=ax)
    ax.set(title="Maximum Wind Speed Distribution", xlabel="Wind speed (km/h)", ylabel="Records")
    fig.tight_layout()
    return fig

def correlation_heatmap(df):
    cols = ["temperature_2m_max", "temperature_2m_min", "apparent_temperature_max",
            "apparent_temperature_min", "precipitation_sum", "rain_sum",
            "wind_speed_10m_max", "wind_gusts_10m_max"]
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df[cols].corr(numeric_only=True), annot=True, cmap="vlag", center=0, fmt=".2f", ax=ax)
    ax.set(title="Correlation Between Weather Variables")
    fig.tight_layout()
    return fig

def seasonal_chart(df):
    data = df.groupby("season", as_index=False).agg(
        temperature=("average_temperature", "mean"), rainfall=("rain_sum", "mean"))
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=data, x="season", y="temperature", ax=ax)
    ax.set(title="Seasonal Average Temperature", xlabel="Season", ylabel="Temperature (°C)")
    fig.tight_layout()
    return fig
