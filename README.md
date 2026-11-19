# AI-Based Climate Analysis and Weather Prediction Dashboard

## Dataset
Place the provided CSV at `data/india_2000_2024_daily_weather.csv`.
The included copy contains 91,320 daily rows, 10 Indian cities, and dates from 2000-01-01 to 2024-12-31.

## Project modules
- `data_cleaning.py`: validates columns and values, parses dates, removes duplicate city/date rows, fills missing values, creates calendar/season/temperature/rain features, and exports the cleaned CSV.
- `eda.py`: reusable yearly, monthly, seasonal, and city summary aggregations.
- `visualization.py`: Matplotlib/Seaborn chart functions.
- `model_training.py`: builds lag/calendar features, uses a chronological 80/20 date split, compares Ridge, Random Forest, and Gradient Boosting, and saves the best model per target.
- `prediction.py`: generates recursive 1–30 day estimates from the latest available observation.
- `app.py`: Streamlit dashboard with Home, Data Cleaning, Climate Analysis, Weather Prediction, and City Comparison pages.

## Setup in VS Code (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run
From the project root:
```powershell
python -c "from data_cleaning import run_cleaning; raw, clean, report = run_cleaning(); print(report); print(clean.head())"
python model_training.py
streamlit run app.py
```
The dashboard can also clean data automatically on first launch. The first model training may take several minutes.

## Important modelling notes
- Model test data is the latest 20% of dates; no random shuffle is used.
- Models estimate daily maximum temperature, minimum temperature, and rainfall.
- The selected model for each target is chosen using holdout MAE.
- Prediction is recursive: each predicted day becomes input history for the next day. Uncertainty therefore grows with longer horizons.
- The dataset does not include atmospheric pressure, satellite/radar data, real-time observations, or numerical weather prediction inputs. Treat outputs as experimental statistical estimates, not operational forecasts.
- IQR results are reported, not automatically clipped or deleted, because rare heat, rain, and wind events can be genuine.
- Wind direction is measured in degrees (0–360); wind speed units follow the source dataset.

## Outputs
- `data/cleaned_weather.csv`
- `models/climate_models.joblib`
- `models/model_metrics.csv`

## Troubleshooting
- Run commands from the folder containing `app.py`.
- If `streamlit` is not recognized, use `python -m streamlit run app.py`.
- If the model file does not exist, open Weather Prediction and click Train and evaluate models, or run `python model_training.py`.
- If your CSV has a different name, update the path in `data_cleaning.py` / `load_raw_data()`.
