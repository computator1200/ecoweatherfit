"""
EcoWeatherFit — Central Configuration & Settings
==================================================
This module centralises every tuneable parameter, API key reference,
file path, and domain constant used across the system.  Keeping
configuration separate from logic follows the **Single Responsibility
Principle** and makes the project easy to audit during a Viva Q&A.

Academic Note
-------------
* API keys are loaded from environment variables — **never** hard‑coded.
* All time‑related constants respect UTC to avoid daylight‑saving
  ambiguities across UK / European datasets.
* Feature‑window and horizon constants are defined here so that
  every module (data, models, UI) shares the same ground truth,
  preventing subtle data‑leakage bugs.
"""

import os
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────
# 1. PROJECT PATHS
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
CACHE_DIR    = PROJECT_ROOT / "cache"
MODEL_DIR    = PROJECT_ROOT / "models" / "saved"
LOG_DIR      = PROJECT_ROOT / "logs"

# Ensure directories exist at import time
for _dir in (DATA_DIR, CACHE_DIR, MODEL_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# 2. API KEYS  (loaded from environment)
# ──────────────────────────────────────────────
OPENWEATHERMAP_API_KEY = os.getenv("OWM_API_KEY", "")
GEMINI_API_KEY         = os.getenv("GEMINI_API_KEY", "")

# ──────────────────────────────────────────────
# 3. LOCATION DEFAULTS — UK‑Focused
# ──────────────────────────────────────────────
DEFAULT_LOCATIONS = {
    "London":     {"lat": 51.5074, "lon": -0.1278, "meteostat_id": "03772"},
    "Manchester": {"lat": 53.4808, "lon": -2.2426, "meteostat_id": "03334"},
    "Edinburgh":  {"lat": 55.9533, "lon": -3.1883, "meteostat_id": "03160"},
    "Birmingham": {"lat": 52.4862, "lon": -1.8904, "meteostat_id": "03535"},
    "Cardiff":    {"lat": 51.4816, "lon": -3.1791, "meteostat_id": "03715"},
}

DEFAULT_CITY = "London"

# ──────────────────────────────────────────────
# 4. TEMPORAL PARAMETERS
# ──────────────────────────────────────────────
# Historical window used for training (years of data from Meteostat)
HISTORICAL_YEARS       = 5          # ≥ 3 years recommended for seasonality
HISTORICAL_START_DATE  = "2020-01-01"
HISTORICAL_END_DATE    = "2025-12-31"

# Forecasting parameters
FORECAST_HORIZON_DAYS  = 7          # 7‑day ahead forecast
LSTM_LOOKBACK_WINDOW   = 30         # sliding‑window length for LSTM input
RF_LAG_FEATURES        = [1, 2, 3, 5, 7, 14]  # lag days for Random Forest

# ──────────────────────────────────────────────
# 5. TARGET & FEATURE VARIABLES
# ──────────────────────────────────────────────
# These are the core meteorological variables the system predicts.
TARGET_VARIABLES = ["temp", "humidity", "wind_speed", "precipitation"]

# Additional features derived during feature engineering
TEMPORAL_FEATURES = [
    "day_of_year_sin", "day_of_year_cos",
    "month_sin", "month_cos",
    "day_of_week_sin", "day_of_week_cos",
]

# ──────────────────────────────────────────────
# 6. DATA‑QUALITY THRESHOLDS
# ──────────────────────────────────────────────
# Maximum allowable fraction of missing values per column before
# the column is flagged (not silently dropped — academic transparency).
MAX_MISSING_RATIO = 0.15            # 15 %

# Physical plausibility bounds for UK weather (used in cleaning)
PLAUSIBILITY_BOUNDS = {
    "temp":          (-30.0, 45.0),   # °C
    "humidity":      (0.0,  100.0),   # %
    "wind_speed":    (0.0,  200.0),   # km/h (accommodates storm gusts)
    "precipitation": (0.0,  300.0),   # mm/day
}

# ──────────────────────────────────────────────
# 7. TRAIN / VALIDATION / TEST SPLIT POLICY
# ──────────────────────────────────────────────
# Strict chronological split — no shuffle, no future information.
# Ratios are approximate; exact boundaries snap to month starts.
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ──────────────────────────────────────────────
# 8. MODEL HYPER‑PARAMETERS (defaults)
# ──────────────────────────────────────────────
LSTM_CONFIG = {
    "units":          64,
    "dropout":        0.2,
    "recurrent_dropout": 0.1,
    "epochs":         100,
    "batch_size":     32,
    "patience":       10,       # early‑stopping patience
    "learning_rate":  0.001,
}

RF_CONFIG = {
    "n_estimators":   300,
    "max_depth":      15,
    "min_samples_leaf": 5,
    "random_state":   42,
}

# ──────────────────────────────────────────────
# 9. RECOMMENDATION ENGINE SETTINGS
# ──────────────────────────────────────────────
HEURISTIC_THRESHOLDS = {
    "cold_temp":        10.0,   # °C — below this ➜ warm clothing
    "hot_temp":         25.0,   # °C — above this ➜ light clothing
    "rain_prob":         0.4,   # 40 % ➜ suggest waterproofs
    "high_wind":        40.0,   # km/h ➜ suggest wind‑resistant layers
    "uv_high":           6,     # UV index ➜ sun protection advisory
}

SUSTAINABILITY_NUDGES = {
    "reuse_first":      True,   # Always suggest "shop your wardrobe" before buying
    "eco_links_opt_in": True,   # Affiliate links shown only if user opts in
    "care_tips":        True,   # Include garment longevity / care advice
}

# ──────────────────────────────────────────────
# 10. GEMINI API SETTINGS
# ──────────────────────────────────────────────
GEMINI_MODEL_NAME       = "gemini-2.0-flash"
GEMINI_MAX_TOKENS       = 512
GEMINI_TEMPERATURE      = 0.4      # lower = more deterministic
GEMINI_TIMEOUT_SECONDS  = 15
GEMINI_MAX_RETRIES      = 2

# ──────────────────────────────────────────────
# 11. LOGGING
# ──────────────────────────────────────────────
LOG_LEVEL  = os.getenv("ECOWEATHERFIT_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s"

# ──────────────────────────────────────────────
# 12. UI / STREAMLIT
# ──────────────────────────────────────────────
STREAMLIT_PAGE_TITLE = "EcoWeatherFit"
STREAMLIT_PAGE_ICON  = "🌿"
STREAMLIT_LAYOUT     = "wide"
