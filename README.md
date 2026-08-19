# Pakistan House Price — Hybrid Spatio-Temporal Forecasting Model

A hybrid trend-plus-residual model for residential property prices across five
Pakistani cities, built on Zameen.com listing data, and stress-tested four years
forward against genuinely unseen 2023 Karachi listings.

**Headline result:** the model predicts well within its training period
(R² = 0.790 on a temporal holdout) but underpredicts 2023 prices by 13.4%,
missing a policy-driven market acceleration that the pre-2020 trend gave no
warning of. That failure is the point of the study, not a defect in it.

## Data

| File | Source | Raw rows | After cleaning |
|---|---|---|---|
| `data/House_Price_dataset.csv` | Zameen.com listings, Aug 2018 – Jul 2019, 5 cities | 168,446 | **119,371** |
| `data/Housing_Prices_in_Pakistan_2023.csv` | Zameen.com listings, Karachi, Jun–Jul 2023 | 5,862 | **5,701** |

Cleaning drops rental listings, zero prices, unparseable areas or dates, missing
coordinates, areas under 50 sq ft, and the top/bottom 0.5% of price-per-square-foot.
**The 168,446 figure is the raw download; 119,371 is what the model actually sees.**

City distribution after cleaning: Karachi 45,961 · Lahore 41,143 · Islamabad 15,961 · Rawalpindi 11,887 · Faisalabad 4,440.

## Architecture

1. **Trend component** — per-city log-linear regression of price-per-square-foot
   against a continuous time index. This exists because gradient-boosted trees
   predict by averaging over training partitions and therefore *cannot* extrapolate
   beyond the price range seen in training. Any target with a genuine long-run
   trend needs a component that can.
2. **Residual component** — `HistGradientBoostingRegressor` on
   `log(price_per_sqft) − trend`, using property type, city, province, bedrooms,
   baths, area, latitude and longitude. Captures how a specific listing differs
   from its city's time-trend average.
3. **Prediction** — `price = exp(trend + residual) × area_sqft`.

The residual model is deliberately given no temporal feature: all time-varying
signal is carried by the trend component, so the two parts cannot compete.

## Validation

**Temporal split, not random.** Training uses listings before 2019-07-09 (the
80th percentile of the date distribution); the test set is everything after.
A random split would let the model see contemporaneous listings during training
and inflate apparent accuracy.

Outlier trim bounds are computed **on the training window only** and then applied
to the whole dataset, so no test-period information reaches the preprocessing step.

## Results

### Temporal holdout (2019)

| Metric | Test | Train (in-sample) |
|---|---|---|
| R² | **0.790** | 0.883 |
| MAPE | 20.2% | 18.7% |
| RMSE (PKR) | 13,911,442 | 13,459,255 |
| MAE (PKR) | 3,908,361 | 4,499,899 |

### Per city — all five reported

| City | R² | MAPE |
|---|---|---|
| Karachi | 0.922 | 21.9% |
| Faisalabad | 0.864 | 24.2% |
| Rawalpindi | 0.845 | 22.2% |
| Lahore | 0.780 | 16.7% |
| **Islamabad** | **0.394** | 19.7% |

Islamabad is the clear outlier. Its RMSE (29.4M PKR) is more than three times
Karachi's despite a comparable MAPE, which points to a small number of very
high-value properties the model misses badly rather than uniformly poor fit —
plausibly farmhouse and luxury-sector heterogeneity not captured by the
available features.

### Four-year forward stress test — 2023 Karachi

Model trained only on 2018–19 data, applied to 5,701 cleaned 2023 listings.
338 rows are dropped for missing bedroom/bathroom values, leaving **5,363
evaluated**.

| Metric | Value |
|---|---|
| R² | 0.601 |
| MAPE | 43.1% |
| RMSE (PKR) | 25,165,029 |

Karachi's actual mean price per square foot rose from 8,970 PKR (2018–19) to
16,890 PKR (2023) — **an 88.3% increase in four years**. The trend component
correctly predicted continued appreciation but underestimated its magnitude:
predicted mean price came in **13.4% below actual** (9.4% below on a
price-per-square-foot basis).

This is consistent with a documented acceleration in Pakistani real estate driven
by the 2020 construction-sector amnesty scheme and currency-driven demand for
real assets as an inflation hedge — neither of which is present anywhere in
2018–19 price history.

## Known limitations

- The 2023 dataset carries no coordinates. Karachi's training-set mean latitude
  and longitude are substituted for every stress-test row, which disables the
  spatial component of the residual model for that evaluation specifically.
- The 2023 window spans one month (27 Jun – 25 Jul 2023), so it measures a price
  level at a point in time, not a 2023 trajectory.
- Listing prices are asking prices, not transaction prices.
- Islamabad's weak fit is diagnosed but not resolved.

## Reproducing

```bash
git clone https://github.com/Gedoni-coder/PAKISTAN-HOUSE-PRICE-ST-MODEL
cd PAKISTAN-HOUSE-PRICE-ST-MODEL
pip install -r requirements.txt
python src/prep_housing_data.py     # writes interim/*.pkl
python src/train_housing_model.py   # writes models/ and results/
```

Both scripts resolve paths relative to the repository root and fail loudly if a
data file is missing. `results/housing_model_metrics.json` is regenerated on every
run; the committed copy was produced by exactly the commands above.

## Repository structure

```
data/      Zameen.com listing data (2018–19 primary, 2023 stress test)
src/       prep_housing_data.py, train_housing_model.py
models/    hybrid_housing_model.joblib (trend models + residual model + categories)
results/   metrics JSON, per-listing predictions, charts
interim/   cleaned pickles (gitignored, regenerated by prep)
```
