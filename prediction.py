from pathlib import Path
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("models/climate_models.joblib")

def load_bundle(path=MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError("Trained models not found. Run: python model_training.py")
    return joblib.load(path)

def _feature_row(city, date, history):
    doy = date.dayofyear
    row = {"city": city, "year": date.year, "month": date.month, "day": date.day,
           "day_of_year": doy, "sin_day": np.sin(2*np.pi*doy/365.25),
           "cos_day": np.cos(2*np.pi*doy/365.25)}
    for col, short in [("temperature_2m_max", "max"), ("temperature_2m_min", "min"), ("rain_sum", "rain")]:
        values = history[col].astype(float).tolist()
        row[f"lag1_{short}"] = values[-1]
        row[f"lag7_{short}"] = values[-7] if len(values) >= 7 else float(np.mean(values))
        row[f"lag30_{short}"] = values[-30] if len(values) >= 30 else float(np.mean(values))
        row[f"roll7_{short}"] = float(np.mean(values[-7:]))
    return row

def predict_horizon(df, city, horizon=7, model_path=MODEL_PATH):
    if not 1 <= int(horizon) <= 30:
        raise ValueError("Prediction horizon must be between 1 and 30 days.")
    bundle = load_bundle(model_path)
    models = bundle["models"]
    history = df[df["city"] == city].sort_values("date").copy()
    if history.empty:
        raise ValueError(f"No historical records found for {city}.")
    last_date = pd.to_datetime(history["date"]).max()
    hist = history[["temperature_2m_max", "temperature_2m_min", "rain_sum"]].tail(30).copy()
    rows = []
    for step in range(1, int(horizon)+1):
        date = last_date + pd.Timedelta(days=step)
        features = pd.DataFrame([_feature_row(city, date, hist)])
        tmax = float(models["temperature_2m_max"].predict(features)[0])
        tmin = float(models["temperature_2m_min"].predict(features)[0])
        rain = max(0.0, float(models["rain_sum"].predict(features)[0]))
        if tmin > tmax:
            tmin, tmax = tmax, tmin
        rows.append({"date": date, "city": city, "predicted_max_temp": tmax,
                     "predicted_min_temp": tmin, "predicted_rainfall": rain})
        hist.loc[len(hist)] = [tmax, tmin, rain]
        hist = hist.tail(30).reset_index(drop=True)
    result = pd.DataFrame(rows)
    result["expected_weather"] = np.where(result["predicted_rainfall"] >= 7.5, "Rainy",
                                  np.where(result["predicted_rainfall"] >= 0.1, "Light rain", "Mostly dry"))
    return result
