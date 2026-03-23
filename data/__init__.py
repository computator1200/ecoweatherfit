# EcoWeatherFit - Data Package
"""
data/
=====
This package implements the complete Data Acquisition & Preprocessing
Pipeline (Step 1 of the EcoWeatherFit architecture).

Modules
-------
acquisition        – API integrations (OpenWeatherMap, Meteostat)
preprocessing      – Cleaning, alignment, plausibility filtering
feature_engineering – Lag features, cyclical encodings, rolling stats
splitting          – Strict chronological train/val/test splits
"""

from data.acquisition import (
    fetch_meteostat_historical,
    fetch_owm_current,
    fetch_owm_forecast,
)
from data.preprocessing import preprocess_pipeline
from data.feature_engineering import engineer_features
from data.splitting import time_based_split
