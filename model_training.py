from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGETS = ["temperature_2m_max", "temperature_2m_min", "rain_sum"]
FEATURES = ["city", "year", "month", "day", "day_of_year", "sin_day", "cos_day",
            "lag1_max", "lag7_max", "lag30_max", "roll7_max",
            "lag1_min", "lag7_min", "lag30_min", "roll7_min",
            "lag1_rain", "lag7_rain", "lag30_rain", "roll7_rain"]

def make_features(df):
    x = df.sort_values(["city", "date"]).copy()
    doy = x["date"].dt.dayofyear
    x["year"], x["month"], x["day"] = x["date"].dt.year, x["date"].dt.month, x["date"].dt.day
    x["day_of_year"] = doy
    x["sin_day"] = np.sin(2 * np.pi * doy / 365.25)
    x["cos_day"] = np.cos(2 * np.pi * doy / 365.25)
    for col, short in [("temperature_2m_max", "max"), ("temperature_2m_min", "min"), ("rain_sum", "rain")]:
        grouped = x.groupby("city", observed=True)[col]
        x[f"lag1_{short}"] = grouped.shift(1)
        x[f"lag7_{short}"] = grouped.shift(7)
        x[f"lag30_{short}"] = grouped.shift(30)
        x[f"roll7_{short}"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    return x

def train_models(df, model_dir="models"):
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    data = make_features(df).dropna(subset=["lag30_max", "lag30_min", "lag30_rain"]).copy()
    cutoff = data["date"].quantile(0.80)
    train, test = data[data["date"] <= cutoff], data[data["date"] > cutoff]
    if train.empty or test.empty:
        raise ValueError("Not enough chronological data to create train/test sets.")
    categorical = ["city"]
    numeric = [c for c in FEATURES if c not in categorical]
    preprocessor = ColumnTransformer([
        ("city", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric)
    ])
    candidates = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=3,
                                                random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=2,
                                                        learning_rate=0.05, random_state=42)
    }
    bundle, metric_rows = {}, []
    for target in TARGETS:
        best_model, best_mae, best_name = None, float("inf"), None
        for name, estimator in candidates.items():
            pipeline = Pipeline([("prep", preprocessor), ("model", estimator)])
            pipeline.fit(train[FEATURES], train[target])
            pred = pipeline.predict(test[FEATURES])
            mae = mean_absolute_error(test[target], pred)
            rmse = float(np.sqrt(mean_squared_error(test[target], pred)))
            r2 = r2_score(test[target], pred)
            metric_rows.append({"target": target, "model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
            if mae < best_mae:
                best_model, best_mae, best_name = pipeline, mae, name
        bundle[target] = best_model
        bundle[f"{target}_best_model"] = best_name
    metrics = pd.DataFrame(metric_rows)
    joblib.dump({"models": bundle, "features": FEATURES, "metrics": metrics}, Path(model_dir)/"climate_models.joblib")
    metrics.to_csv(Path(model_dir)/"model_metrics.csv", index=False)
    return metrics, bundle, cutoff

if __name__ == "__main__":
    from data_cleaning import run_cleaning
    _, clean, _ = run_cleaning()
    metrics, _, cutoff = train_models(clean)
    print(metrics.sort_values(["target", "MAE"]).to_string(index=False))
    print(f"Chronological test begins after: {cutoff}")
