"""
Hybrid spatio-temporal price-per-sqft model.
  1. TREND component: per-city log-linear regression of price_per_sqft vs time.
     Captures market-level appreciation (extrapolatable).
  2. RESIDUAL component: HistGradientBoostingRegressor on residuals, using
     property_type, location, bedrooms, baths, area_sqft, city.
     Captures how a specific listing compares to its city's time-trend average.
  Final: price_per_sqft = exp(trend + residual); price = price_per_sqft * area_sqft.

Validation: TEMPORAL split within 2018-2019 (train on first 80% of the date
range, test on the most recent ~20%) - genuine forecasting test, not random.

Then: STRESS TEST on 2023 Karachi data (never seen at training time, 4 years
later) - tests whether the model's trend extrapolation holds up over a longer
horizon, mirroring the food-price project's structural-break finding.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
INTERIM_DIR = BASE_DIR / "interim"
for _d in (MODELS_DIR, RESULTS_DIR, INTERIM_DIR): _d.mkdir(parents=True, exist_ok=True)


hp = pd.read_pickle(INTERIM_DIR / 'hp_clean.pkl')
h23 = pd.read_pickle(INTERIM_DIR / 'h23_clean.pkl')

hp = hp.sort_values('date_added').reset_index(drop=True)
hp['time_idx'] = (hp['date_added'] - hp['date_added'].min()).dt.days
hp['log_pps'] = np.log(hp['price_per_sqft'])

# Temporal split: last ~20% of the date range as test
cutoff = hp['date_added'].quantile(0.80)
train = hp[hp['date_added'] < cutoff].copy()
test = hp[hp['date_added'] >= cutoff].copy()
print(f"Train: {len(train)} rows (< {cutoff.date()})")
print(f"Test:  {len(test)} rows (>= {cutoff.date()})")

# ---- 1. TREND: per-city log-linear trend of price_per_sqft over time ----
trend_models = {}
for city in train['city'].unique():
    sub = train[train['city'] == city]
    lr = LinearRegression()
    lr.fit(sub[['time_idx']], sub['log_pps'])
    trend_models[city] = lr

def predict_trend_vec(df_):
    preds = np.full(len(df_), np.nan)
    for c, m in trend_models.items():
        mask = (df_['city'] == c).values
        if mask.any():
            preds[mask] = m.predict(df_.loc[mask, ['time_idx']].values)
    return preds

train['trend_pred'] = predict_trend_vec(train)
test['trend_pred'] = predict_trend_vec(test)
train['residual'] = train['log_pps'] - train['trend_pred']

# ---- 2. RESIDUAL: spatio-temporal + property-feature ML model ----
cat_features = ['property_type', 'city', 'province_name']
num_features = ['bedrooms', 'baths', 'area_sqft', 'latitude', 'longitude']
features = cat_features + num_features

X_train = train[features].copy()
X_test = test[features].copy()
for c in cat_features:
    X_train[c] = X_train[c].astype('category')
    X_test[c] = pd.Categorical(X_test[c], categories=X_train[c].cat.categories)
cat_idx = [X_train.columns.get_loc(c) for c in cat_features]

residual_model = HistGradientBoostingRegressor(
    max_iter=400, max_depth=8, learning_rate=0.05,
    categorical_features=cat_idx, random_state=42
)
residual_model.fit(X_train, train['residual'])

train['residual_pred'] = residual_model.predict(X_train)
test['residual_pred'] = residual_model.predict(X_test)

train['pps_pred'] = np.exp(train['trend_pred'] + train['residual_pred'])
test['pps_pred'] = np.exp(test['trend_pred'] + test['residual_pred'])
train['price_pred'] = train['pps_pred'] * train['area_sqft']
test['price_pred'] = test['pps_pred'] * test['area_sqft']

def metrics(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"{label}: RMSE={rmse:,.0f} PKR | MAE={mae:,.0f} PKR | MAPE={mape:.1f}% | R2={r2:.3f}")
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}

print("\n--- Main model performance (price prediction) ---")
test_overall = metrics(test['price'], test['price_pred'], "TEST (most recent ~20%, 2019)")
train_overall = metrics(train['price'], train['price_pred'], "TRAIN (in-sample)")

per_city = {}
for c in test['city'].unique():
    sub = test[test['city'] == c]
    per_city[c] = metrics(sub['price'], sub['price_pred'], f"  {c}")

# ============================================================
# STRESS TEST: apply model (trained only on 2018-19 data) to 2023 Karachi data
# ============================================================
print("\n--- STRESS TEST: 2018-19-trained model applied to 2023 Karachi data ---")
h23['time_idx'] = (h23['date_added'] - hp['date_added'].min()).dt.days  # same origin as training
h23['bedrooms'] = pd.to_numeric(h23['bedrooms'], errors='coerce')
h23['baths'] = pd.to_numeric(h23['baths'], errors='coerce')
_n23_before = len(h23)
h23 = h23.dropna(subset=['bedrooms', 'baths'])
_n23_dropped = _n23_before - len(h23)
print(f"2023 stress test: {_n23_before} cleaned rows, {_n23_dropped} dropped for missing "
      f"bedrooms/baths, {len(h23)} evaluated")
karachi_lat = train.loc[train['city'] == 'Karachi', 'latitude'].mean()
karachi_lon = train.loc[train['city'] == 'Karachi', 'longitude'].mean()
h23['latitude'] = karachi_lat
h23['longitude'] = karachi_lon

h23_trend = trend_models['Karachi'].predict(h23[['time_idx']])
X_23 = h23[cat_features + num_features].copy()
for c in cat_features:
    X_23[c] = pd.Categorical(X_23[c], categories=X_train[c].cat.categories)
h23_residual = residual_model.predict(X_23)
h23['pps_pred'] = np.exp(h23_trend + h23_residual)
h23['price_pred'] = h23['pps_pred'] * h23['area_sqft']

stress_metrics = metrics(h23['price_numeric'], h23['price_pred'], "2023 Karachi (4yr extrapolation)")

# ---- Save everything ----
joblib.dump({"trend_models": trend_models, "residual_model": residual_model,
             "cat_features": cat_features, "num_features": num_features,
             "date_origin": hp['date_added'].min(),
             "train_categories": {c: list(X_train[c].cat.categories) for c in cat_features}},
            MODELS_DIR / 'hybrid_housing_model.joblib')

test[['date_added','city','property_type','location','bedrooms','baths','area_sqft','price','price_pred']].to_csv(
    RESULTS_DIR / 'test_predictions_2019.csv', index=False)
h23[['date_added','city','property_type','location','bedrooms','baths','area_sqft','price_numeric','price_pred']].rename(
    columns={'price_numeric':'price'}).to_csv(RESULTS_DIR / 'stress_test_2023_predictions.csv', index=False)

with open(RESULTS_DIR / 'housing_model_metrics.json', 'w') as f:
    json.dump({"test_2019": test_overall, "train_in_sample": train_overall,
               "per_city_2019_test": per_city, "stress_test_2023_karachi": stress_metrics,
               "cutoff_date": str(cutoff), "n_train": len(train), "n_test": len(test),
               "n_stress_test_cleaned": _n23_before,
               "n_stress_test_dropped_missing_beds_baths": _n23_dropped,
               "n_stress_test_evaluated": len(h23)}, f, indent=2, default=str)

print("\nSaved model, predictions, and metrics.")
