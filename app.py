from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data_cleaning import load_raw_data, clean_data, iqr_outlier_summary
from eda import city_summary
from visualization import (temperature_trend, monthly_temperature, rainfall_distribution,
    city_temperature, city_rainfall, monthly_heatmap, wind_distribution,
    correlation_heatmap, seasonal_chart)
from model_training import train_models
from prediction import predict_horizon

st.set_page_config(page_title="India Climate Intelligence", page_icon="🌦️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#f4f9ff 0%,#eef6f7 100%);}
[data-testid="stMetric"] {background:#ffffff;padding:16px;border-radius:14px;
 box-shadow:0 3px 14px rgba(30,70,100,.08);border:1px solid #e2edf5;}
h1,h2,h3 {color:#123b5d;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    raw = load_raw_data()
    clean, report = clean_data(raw)
    return raw, clean, report

raw, data, cleaning_report = get_data()
if not Path("data/cleaned_weather.csv").exists():
    data.to_csv("data/cleaned_weather.csv", index=False, date_format="%Y-%m-%d")

st.sidebar.title("🌦️ Climate Intelligence")
page = st.sidebar.radio("Navigate", ["Home", "Data Cleaning", "Climate Analysis",
                                      "Weather Prediction", "City Comparison"])
st.sidebar.caption("Historical data: 2000–2024")
cities = sorted(data["city"].dropna().unique().tolist())

if page == "Home":
    st.title("India Climate Analysis & Weather Prediction")
    st.write("Explore 25 years of historical weather observations across 10 Indian cities.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Weather records", f"{len(data):,}")
    c2.metric("Cities", len(cities))
    c3.metric("Average temperature", f"{data['average_temperature'].mean():.1f} °C")
    c4.metric("Maximum recorded", f"{data['temperature_2m_max'].max():.1f} °C")
    c5,c6,c7 = st.columns(3)
    c5.metric("Total rainfall", f"{data['rain_sum'].sum():,.0f} mm")
    c6.metric("Start date", data["date"].min().strftime("%d %b %Y"))
    c7.metric("End date", data["date"].max().strftime("%d %b %Y"))
    st.subheader("Historical temperature trend")
    st.pyplot(temperature_trend(data), use_container_width=True)
    st.info("Predictions are machine-learning estimates from historical patterns, not official forecasts.")

elif page == "Data Cleaning":
    st.title("Data Cleaning & Quality")
    st.subheader("Original dataset preview")
    st.dataframe(raw.head(20), use_container_width=True)
    a,b,c = st.columns(3)
    a.metric("Original rows", f"{cleaning_report['original_rows']:,}")
    b.metric("Missing cells", f"{cleaning_report['original_missing_cells']:,}")
    c.metric("Duplicate rows", f"{cleaning_report['original_duplicates']:,}")
    st.subheader("Missing values by column")
    st.dataframe(raw.isna().sum().rename("missing_count").to_frame(), use_container_width=True)
    st.subheader("IQR outlier report")
    st.caption("IQR flags unusual observations for review. Genuine extreme weather is retained.")
    st.dataframe(iqr_outlier_summary(data), use_container_width=True)
    st.subheader("Cleaning summary")
    st.json(cleaning_report)
    st.subheader("Cleaned dataset preview")
    st.dataframe(data.head(20), use_container_width=True)
    st.download_button("⬇️ Download cleaned CSV", data.to_csv(index=False).encode("utf-8"),
                       "cleaned_weather.csv", "text/csv")

elif page == "Climate Analysis":
    st.title("Historical Climate Analysis")
    selected = st.multiselect("Filter cities", cities, default=cities[:4])
    if not selected:
        st.warning("Select at least one city.")
        st.stop()
    filtered = data[data["city"].isin(selected)]
    st.pyplot(temperature_trend(data, selected), use_container_width=True)
    left,right = st.columns(2)
    with left: st.pyplot(monthly_temperature(filtered), use_container_width=True)
    with right: st.pyplot(rainfall_distribution(filtered), use_container_width=True)
    left,right = st.columns(2)
    with left: st.pyplot(city_temperature(filtered), use_container_width=True)
    with right: st.pyplot(city_rainfall(filtered), use_container_width=True)
    st.pyplot(monthly_heatmap(filtered), use_container_width=True)
    left,right = st.columns(2)
    with left: st.pyplot(seasonal_chart(filtered), use_container_width=True)
    with right: st.pyplot(wind_distribution(filtered), use_container_width=True)
    st.pyplot(correlation_heatmap(filtered), use_container_width=True)
    st.subheader("Yearly temperature and rainfall tables")
    yearly = filtered.groupby("year", as_index=False).agg(
        avg_max_temp=("temperature_2m_max","mean"),
        avg_min_temp=("temperature_2m_min","mean"),
        rainfall=("rain_sum","sum"))
    st.dataframe(yearly, use_container_width=True)

elif page == "Weather Prediction":
    st.title("Machine Learning Weather Estimates")
    st.warning("These are historical-pattern model estimates, not official live forecasts. Sudden storms and extreme events cannot be reliably inferred from this dataset alone.")
    city = st.selectbox("Select city", cities)
    horizon = st.slider("Prediction horizon (days)", 1, 30, 7)
    model_file = Path("models/climate_models.joblib")
    if not model_file.exists():
        st.info("Models have not been trained yet. Start training below; this may take a few minutes.")
        if st.button("Train and evaluate models", type="primary"):
            with st.spinner("Training chronological models..."):
                metrics, _, cutoff = train_models(data)
            st.success(f"Training complete. Test period starts after {pd.Timestamp(cutoff).date()}.")
            st.dataframe(metrics.sort_values(["target","MAE"]), use_container_width=True)
            st.rerun()
    else:
        if st.button("Retrain models"):
            with st.spinner("Training models..."):
                metrics, _, cutoff = train_models(data)
            st.success("Models retrained.")
            st.rerun()
        try:
            forecast = predict_horizon(data, city, horizon)
            x1,x2,x3 = st.columns(3)
            x1.metric("Average predicted max", f"{forecast['predicted_max_temp'].mean():.1f} °C")
            x2.metric("Average predicted min", f"{forecast['predicted_min_temp'].mean():.1f} °C")
            x3.metric("Total predicted rainfall", f"{forecast['predicted_rainfall'].sum():.1f} mm")
            st.dataframe(forecast, use_container_width=True)
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(forecast["date"], forecast["predicted_max_temp"], marker="o", label="Predicted max")
            ax.plot(forecast["date"], forecast["predicted_min_temp"], marker="o", label="Predicted min")
            ax.set(title=f"{city}: estimated temperature", xlabel="Date", ylabel="Temperature (°C)")
            ax.legend(); fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            fig2, ax2 = plt.subplots(figsize=(10,3))
            ax2.bar(forecast["date"].dt.strftime("%d %b"), forecast["predicted_rainfall"])
            ax2.set(title="Estimated daily rainfall", xlabel="Date", ylabel="Rainfall (mm)")
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            metrics_path = Path("models/model_metrics.csv")
            if metrics_path.exists():
                st.subheader("Model evaluation (chronological holdout)")
                st.dataframe(pd.read_csv(metrics_path).sort_values(["target","MAE"]), use_container_width=True)
        except Exception as exc:
            st.error(f"Prediction could not be generated: {exc}")

elif page == "City Comparison":
    st.title("Compare Indian Cities")
    selected = st.multiselect("Choose two or more cities", cities, default=cities[:3])
    if len(selected) < 2:
        st.info("Choose at least two cities to compare.")
        st.stop()
    subset = data[data["city"].isin(selected)]
    summary = city_summary(subset)
    st.dataframe(summary, use_container_width=True)
    left,right = st.columns(2)
    with left: st.pyplot(city_temperature(subset), use_container_width=True)
    with right: st.pyplot(city_rainfall(subset), use_container_width=True)
    st.pyplot(temperature_trend(subset, selected), use_container_width=True)
    seasonal = subset.groupby(["city","season"], as_index=False).agg(
        avg_temp=("average_temperature","mean"), avg_rain=("rain_sum","mean"))
    st.subheader("Seasonal comparison")
    st.dataframe(seasonal, use_container_width=True)
