"""
Hybrid spatio-temporal price-per-sqft model for Pakistani real estate.
Primary data: House_Price_dataset.csv (2018-2019, 5 cities, 168K listings)
Stress test: Housing_Prices_in_Pakistan_2023.csv (2023, Karachi only) - genuinely
unseen data to test whether a 2018-19-trained model holds up 4 years later.
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
INTERIM_DIR = BASE_DIR / "interim"
for _d in (MODELS_DIR, RESULTS_DIR, INTERIM_DIR): _d.mkdir(parents=True, exist_ok=True)


AREA_UNIT_TO_SQFT = {
    'marla': 272.25,
    'kanal': 5445.0,
    'sq. yd.': 9.0,
    'sq.yd.': 9.0,
    'sq yd': 9.0,
    'sq. ft.': 1.0,
    'sq.ft.': 1.0,
    'sq ft': 1.0,
}

def parse_area_to_sqft(area_str):
    if pd.isna(area_str):
        return np.nan
    s = str(area_str).strip().lower()
    m = re.match(r'([\d.]+)\s*(.+)', s)
    if not m:
        return np.nan
    val, unit = float(m.group(1)), m.group(2).strip()
    for key, factor in AREA_UNIT_TO_SQFT.items():
        if key in unit:
            return val * factor
    return np.nan

def parse_pkr_price(price_str):
    """Parse 'PKR5.2 Crore' / 'PKR93 Lakh' style strings to raw PKR numeric."""
    if pd.isna(price_str):
        return np.nan
    s = str(price_str).replace('PKR', '').strip()
    m = re.match(r'([\d.]+)\s*(Crore|Lakh)?', s, re.IGNORECASE)
    if not m:
        return np.nan
    val = float(m.group(1))
    unit = (m.group(2) or '').lower()
    if unit == 'crore':
        return val * 10_000_000
    elif unit == 'lakh':
        return val * 100_000
    return val

# ---------------- Load primary 2018-2019 dataset ----------------
hp = pd.read_csv(DATA_DIR / 'House_Price_dataset.csv')
hp = hp[hp['purpose'] == 'For Sale'].copy()
hp = hp[hp['price'] > 0].copy()
hp['area_sqft'] = hp['area'].apply(parse_area_to_sqft)
hp['date_added'] = pd.to_datetime(hp['date_added'], errors='coerce')
hp = hp.dropna(subset=['area_sqft', 'date_added', 'latitude', 'longitude'])
hp = hp[hp['area_sqft'] > 50]  # drop implausible tiny areas (data errors)
hp['price_per_sqft'] = hp['price'] / hp['area_sqft']
# Drop extreme outliers (top/bottom 0.5%) in price_per_sqft - likely data entry errors
_cut = hp['date_added'].quantile(0.80)
lo, hi = hp.loc[hp['date_added'] < _cut, 'price_per_sqft'].quantile([0.005, 0.995])
hp = hp[(hp['price_per_sqft'] >= lo) & (hp['price_per_sqft'] <= hi)]

print(f"Primary (2018-2019) dataset after cleaning: {len(hp)} rows")
print(hp['city'].value_counts())
print(f"Date range: {hp['date_added'].min().date()} to {hp['date_added'].max().date()}")

# ---------------- Load 2023 Karachi stress-test dataset ----------------
h23 = pd.read_csv(DATA_DIR / 'Housing_Prices_in_Pakistan_2023.csv')
h23 = h23[h23['purpose'] == 'For Sale'].copy()
h23['price_numeric'] = h23['price'].apply(parse_pkr_price)
h23['area_sqft'] = h23['area'].apply(parse_area_to_sqft)
h23['date_added'] = pd.to_datetime(h23['date_added'], errors='coerce')
h23 = h23.dropna(subset=['area_sqft', 'date_added', 'price_numeric'])
h23 = h23[h23['area_sqft'] > 50]
h23['price_per_sqft'] = h23['price_numeric'] / h23['area_sqft']
h23.rename(columns={'prop_type': 'property_type', 'bathrooms': 'baths', 'province': 'province_name'}, inplace=True)

print(f"\n2023 Karachi stress-test dataset after cleaning: {len(h23)} rows")
print(f"Date range: {h23['date_added'].min().date()} to {h23['date_added'].max().date()}")

hp.to_pickle(INTERIM_DIR / 'hp_clean.pkl')
h23.to_pickle(INTERIM_DIR / 'h23_clean.pkl')
