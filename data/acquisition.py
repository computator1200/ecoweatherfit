"""
EcoWeatherFit — Data Acquisition Module
========================================
Responsible for retrieving weather data from two complementary sources:

1. **Meteostat** — bulk historical daily observations (training data).
2. **OpenWeatherMap (OWM)** — real‑time current conditions and 7‑day
   forecast (inference / live dashboard data).

Design Decisions
----------------
* Every function returns a **pandas DataFrame** with a `DatetimeIndex`
  (UTC) so downstream modules never need to worry about parsing.
* Raw API responses are cached locally (``cache/`` directory) to
  respect rate limits and allow offline development / testing.
* All network errors are caught, logged, and re‑raised as a custom
  ``DataAcquisitionError`` so the UI can present a friendly message.

Academic Integrity Note
-----------------------
Meteostat data is used **exclusively** for model training & evaluation.
OWM data is used **only** for live inference.  There is **no** mixing
of future OWM data into the training set — the pipeline enforces this
by construction (see ``splitting.py``).
"""

import json
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

from config.settings import (
    CACHE_DIR,
    DEFAULT_LOCATIONS,
    HISTORICAL_START_DATE,
    HISTORICAL_END_DATE,
    OPENWEATHERMAP_API_KEY,
    TARGET_VARIABLES,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Custom Exception
# ──────────────────────────────────────────────
class DataAcquisitionError(Exception):
    """Raised when an API call fails after retries."""
    pass


# ──────────────────────────────────────────────
# Caching Helpers
# ──────────────────────────────────────────────
def _cache_key(prefix: str, **kwargs) -> Path:
    """
    Generate a deterministic cache file path from the call parameters.

    Uses an MD5 hash of sorted keyword arguments to create unique,
    collision‑free file names.  This lets us cache multiple cities /
    date ranges independently.

    Parameters
    ----------
    prefix : str
        A human‑readable prefix (e.g. ``"meteostat"``).
    **kwargs
        Arbitrary key‑value pairs that define the request identity.

    Returns
    -------
    pathlib.Path
        Full path to the cache file (CSV).
    """
    raw = json.dumps(kwargs, sort_keys=True, default=str)
    digest = hashlib.md5(raw.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{prefix}_{digest}.csv"


def _load_cache(path: Path, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
    """
    Load a cached DataFrame if it exists and is fresh enough.

    Parameters
    ----------
    path : pathlib.Path
        Cache file path.
    max_age_hours : int
        Maximum acceptable age of the cache in hours.

    Returns
    -------
    pd.DataFrame or None
        Cached data, or None if stale / missing.
    """
    if not path.exists():
        return None
    age = datetime.now().timestamp() - path.stat().st_mtime
    if age > max_age_hours * 3600:
        logger.info("Cache expired: %s (%.1f h old)", path.name, age / 3600)
        return None
    logger.info("Cache hit: %s", path.name)
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    return df


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame to the cache directory."""
    df.to_csv(path)
    logger.debug("Cached → %s (%d rows)", path.name, len(df))


# ──────────────────────────────────────────────
# 1. METEOSTAT — Historical Daily Data
# ──────────────────────────────────────────────
def fetch_meteostat_historical(
    city: str = "London",
    start: str = HISTORICAL_START_DATE,
    end: str = HISTORICAL_END_DATE,
    use_cache: bool = True,
    cache_age_hours: int = 168,       # 7 days — historical data rarely changes
) -> pd.DataFrame:
    """
    Retrieve daily historical weather observations from the Meteostat
    JSON API for a given UK city.

    The function maps the city name to its WMO station ID via
    ``config.settings.DEFAULT_LOCATIONS``, then queries the Meteostat
    ``/stations/daily`` endpoint.

    Parameters
    ----------
    city : str
        City name (must exist in ``DEFAULT_LOCATIONS``).
    start : str
        ISO‑format start date, e.g. ``"2020-01-01"``.
    end : str
        ISO‑format end date, e.g. ``"2025-12-31"``.
    use_cache : bool
        Whether to read/write the local CSV cache.
    cache_age_hours : int
        Maximum cache age in hours before a refresh is triggered.

    Returns
    -------
    pd.DataFrame
        Columns: ``temp`` (°C mean), ``temp_min``, ``temp_max``,
        ``humidity`` (%), ``wind_speed`` (km/h), ``precipitation``
        (mm), plus the raw Meteostat fields.  Index is a UTC
        ``DatetimeIndex`` named ``"time"``.

    Raises
    ------
    DataAcquisitionError
        If the API call fails or returns no data.

    Notes
    -----
    Meteostat provides a free JSON API (https://dev.meteostat.net/).
    A RapidAPI key is required; set it in the ``METEOSTAT_API_KEY``
    environment variable.  For development without a key, the
    ``meteostat`` Python package can be used as a fallback (see below).
    """
    if city not in DEFAULT_LOCATIONS:
        raise ValueError(
            f"Unknown city '{city}'. Available: {list(DEFAULT_LOCATIONS)}"
        )

    cache_path = _cache_key("meteostat", city=city, start=start, end=end)
    if use_cache:
        cached = _load_cache(cache_path, max_age_hours=cache_age_hours)
        if cached is not None:
            return cached

    logger.info("Fetching Meteostat data for %s (%s → %s)…", city, start, end)

    # ── Primary Path: meteostat Python library (no API key needed) ──
    try:
        from meteostat import Daily, Stations

        loc = DEFAULT_LOCATIONS[city]
        stations = Stations()
        stations = stations.nearby(loc["lat"], loc["lon"])
        station = stations.fetch(1)

        if station.empty:
            raise DataAcquisitionError(
                f"No Meteostat station found near {city}"
            )

        station_id = station.index[0]
        logger.info("Using Meteostat station: %s", station_id)

        data = Daily(
            station_id,
            start=datetime.fromisoformat(start),
            end=datetime.fromisoformat(end),
        )
        df = data.fetch()

        if df.empty:
            raise DataAcquisitionError(
                f"Meteostat returned no data for station {station_id}"
            )

        # Standardise column names to internal conventions
        rename_map = {
            "tavg": "temp",
            "tmin": "temp_min",
            "tmax": "temp_max",
            "rhum": "humidity",        # not present on all stations
            "wspd": "wind_speed",
            "prcp": "precipitation",
            "pres": "pressure",
            "snow": "snow_depth",
        }

        # Keep only columns that exist in the response
        rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        # Ensure index is named 'time' and is timezone‑naive (UTC implied)
        df.index.name = "time"
        df.index = pd.to_datetime(df.index)

        if use_cache:
            _save_cache(df, cache_path)

        logger.info(
            "Meteostat: retrieved %d rows, %d columns for %s",
            len(df), len(df.columns), city,
        )
        return df

    except ImportError:
        logger.warning(
            "meteostat package not installed; falling back to synthetic data "
            "for development.  Install with: pip install meteostat"
        )
        return _generate_synthetic_historical(city, start, end, cache_path, use_cache)

    except Exception as exc:
        logger.error("Meteostat fetch failed: %s", exc)
        raise DataAcquisitionError(str(exc)) from exc


# ──────────────────────────────────────────────
# 2. OPENWEATHERMAP — Current Weather
# ──────────────────────────────────────────────
def fetch_owm_current(
    city: str = "London",
    use_cache: bool = True,
    cache_age_hours: int = 1,
) -> pd.DataFrame:
    """
    Fetch current weather conditions from the OpenWeatherMap
    *Current Weather Data* endpoint.

    Parameters
    ----------
    city : str
        City name (must exist in ``DEFAULT_LOCATIONS``).
    use_cache : bool
        If True, re‑use a cached response within ``cache_age_hours``.
    cache_age_hours : int
        Cache freshness window (default 1 hour).

    Returns
    -------
    pd.DataFrame
        Single‑row DataFrame with columns matching ``TARGET_VARIABLES``
        plus extras (``feels_like``, ``pressure``, ``clouds``, etc.).
        Index is a UTC ``DatetimeIndex``.

    Raises
    ------
    DataAcquisitionError
        On HTTP errors or missing API key.
    """
    if city not in DEFAULT_LOCATIONS:
        raise ValueError(
            f"Unknown city '{city}'. Available: {list(DEFAULT_LOCATIONS)}"
        )

    cache_path = _cache_key("owm_current", city=city)
    if use_cache:
        cached = _load_cache(cache_path, max_age_hours=cache_age_hours)
        if cached is not None:
            return cached

    if not OPENWEATHERMAP_API_KEY:
        logger.warning(
            "OWM_API_KEY not set — returning synthetic current weather."
        )
        return _generate_synthetic_current(city, cache_path, use_cache)

    loc = DEFAULT_LOCATIONS[city]
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat":   loc["lat"],
        "lon":   loc["lon"],
        "appid": OPENWEATHERMAP_API_KEY,
        "units": "metric",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise DataAcquisitionError(
            f"OWM current‑weather request failed: {exc}"
        ) from exc

    # Parse the flat JSON into a single‑row DataFrame
    record = {
        "time":          pd.Timestamp.now("UTC").floor("h").tz_localize(None),
        "temp":          data["main"]["temp"],
        "feels_like":    data["main"]["feels_like"],
        "temp_min":      data["main"]["temp_min"],
        "temp_max":      data["main"]["temp_max"],
        "humidity":      data["main"]["humidity"],
        "pressure":      data["main"]["pressure"],
        "wind_speed":    data["wind"]["speed"] * 3.6,  # m/s → km/h
        "wind_deg":      data["wind"].get("deg", np.nan),
        "clouds":        data["clouds"]["all"],
        "precipitation": (
            data.get("rain", {}).get("1h", 0.0)
            + data.get("snow", {}).get("1h", 0.0)
        ),
        "weather_main":  data["weather"][0]["main"],
        "weather_desc":  data["weather"][0]["description"],
    }

    df = pd.DataFrame([record]).set_index("time")
    df.index.name = "time"

    if use_cache:
        _save_cache(df, cache_path)

    logger.info("OWM current: %.1f °C, %s in %s", record["temp"],
                record["weather_desc"], city)
    return df


# ──────────────────────────────────────────────
# 3. OPENWEATHERMAP — 7‑Day Forecast
# ──────────────────────────────────────────────
def fetch_owm_forecast(
    city: str = "London",
    use_cache: bool = True,
    cache_age_hours: int = 3,
) -> pd.DataFrame:
    """
    Fetch the 7‑day daily forecast from the OpenWeatherMap
    *One Call API 3.0* (or free 5-day/3-hour endpoint, aggregated).

    Returns one row per forecast day with columns aligned to
    ``TARGET_VARIABLES``.

    Parameters
    ----------
    city : str
        City name.
    use_cache : bool
        Cache toggle.
    cache_age_hours : int
        Hours before cache refresh (default 3).

    Returns
    -------
    pd.DataFrame
        7 rows (one per day), indexed by forecast date (UTC).

    Raises
    ------
    DataAcquisitionError
        On HTTP errors, missing key, or unexpected response schema.
    """
    if city not in DEFAULT_LOCATIONS:
        raise ValueError(
            f"Unknown city '{city}'. Available: {list(DEFAULT_LOCATIONS)}"
        )

    cache_path = _cache_key("owm_forecast", city=city)
    if use_cache:
        cached = _load_cache(cache_path, max_age_hours=cache_age_hours)
        if cached is not None:
            return cached

    if not OPENWEATHERMAP_API_KEY:
        logger.warning(
            "OWM_API_KEY not set — returning synthetic 7‑day forecast."
        )
        return _generate_synthetic_forecast(city, cache_path, use_cache)

    loc = DEFAULT_LOCATIONS[city]

    # ── Try One Call API 3.0 first (requires subscription) ──
    # Fallback: 5‑day / 3‑hour free endpoint, aggregated to daily.
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "lat":   loc["lat"],
        "lon":   loc["lon"],
        "appid": OPENWEATHERMAP_API_KEY,
        "units": "metric",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise DataAcquisitionError(
            f"OWM forecast request failed: {exc}"
        ) from exc

    # Parse 3‑hourly entries and aggregate to daily
    records = []
    for entry in data.get("list", []):
        records.append({
            "time":          pd.to_datetime(entry["dt"], unit="s", utc=True),
            "temp":          entry["main"]["temp"],
            "feels_like":    entry["main"]["feels_like"],
            "temp_min":      entry["main"]["temp_min"],
            "temp_max":      entry["main"]["temp_max"],
            "humidity":      entry["main"]["humidity"],
            "pressure":      entry["main"]["pressure"],
            "wind_speed":    entry["wind"]["speed"] * 3.6,
            "clouds":        entry["clouds"]["all"],
            "precipitation": (
                entry.get("rain", {}).get("3h", 0.0)
                + entry.get("snow", {}).get("3h", 0.0)
            ),
            "pop":           entry.get("pop", 0.0),  # probability of precip
            "weather_main":  entry["weather"][0]["main"],
        })

    df_3h = pd.DataFrame(records).set_index("time")

    # Aggregate to daily: mean for most, sum for precipitation, max for pop
    daily = df_3h.resample("D").agg({
        "temp":          "mean",
        "feels_like":    "mean",
        "temp_min":      "min",
        "temp_max":      "max",
        "humidity":      "mean",
        "pressure":      "mean",
        "wind_speed":    "mean",
        "clouds":        "mean",
        "precipitation": "sum",
        "pop":           "max",
    }).dropna(how="all")

    # Keep only 7 days (the free endpoint gives ≤5, One Call gives 8)
    daily = daily.head(7)
    daily.index.name = "time"

    # Attempt to add dominant weather description per day
    try:
        weather_daily = df_3h.groupby(df_3h.index.date)["weather_main"].agg(
            lambda x: x.mode()[0] if not x.mode().empty else "Unknown"
        )
        daily["weather_main"] = weather_daily.values[:len(daily)]
    except Exception:
        daily["weather_main"] = "Unknown"

    if use_cache:
        _save_cache(daily, cache_path)

    logger.info("OWM forecast: %d days retrieved for %s", len(daily), city)
    return daily


# ──────────────────────────────────────────────
# 4. SYNTHETIC DATA GENERATORS (dev / testing)
# ──────────────────────────────────────────────
def _generate_synthetic_historical(
    city: str,
    start: str,
    end: str,
    cache_path: Path,
    use_cache: bool,
) -> pd.DataFrame:
    """
    Generate realistic synthetic daily weather data for development
    and testing when API access is unavailable.

    The generator uses sinusoidal seasonal patterns with added Gaussian
    noise to approximate UK weather distributions.  This is explicitly
    documented to ensure **academic transparency** — synthetic data is
    never passed off as real observations.

    Parameters
    ----------
    city : str
        City name (used for reproducible seeding).
    start, end : str
        Date range.
    cache_path : pathlib.Path
        Where to persist the synthetic data.
    use_cache : bool
        Whether to save to cache.

    Returns
    -------
    pd.DataFrame
        Synthetic daily weather data with the standard schema.
    """
    logger.warning("⚠ Generating SYNTHETIC historical data for '%s'.", city)

    rng = np.random.default_rng(seed=hash(city) % 2**32)
    dates = pd.date_range(start=start, end=end, freq="D")
    n = len(dates)

    # Day‑of‑year for seasonal cycle (0–365)
    doy = dates.dayofyear.values.astype(float)

    # Seasonal temperature: UK mean ~10 °C, amplitude ~8 °C
    temp_mean = 10.0 + 8.0 * np.sin(2 * np.pi * (doy - 80) / 365)
    temp = temp_mean + rng.normal(0, 2.5, n)

    df = pd.DataFrame({
        "temp":          np.round(temp, 1),
        "temp_min":      np.round(temp - rng.uniform(2, 5, n), 1),
        "temp_max":      np.round(temp + rng.uniform(2, 5, n), 1),
        "humidity":      np.clip(rng.normal(75, 12, n), 20, 100).round(1),
        "wind_speed":    np.clip(rng.exponential(12, n), 0, 100).round(1),
        "precipitation": np.clip(rng.exponential(2.5, n), 0, 80).round(1),
        "pressure":      rng.normal(1013, 10, n).round(1),
        "snow_depth":    np.where(temp < 2, rng.exponential(1, n).round(1), 0.0),
    }, index=dates)

    df.index.name = "time"

    if use_cache:
        _save_cache(df, cache_path)

    logger.info("Synthetic historical data: %d rows for %s", len(df), city)
    return df


def _generate_synthetic_current(
    city: str,
    cache_path: Path,
    use_cache: bool,
) -> pd.DataFrame:
    """Generate a single synthetic current‑weather row."""
    rng = np.random.default_rng(seed=int(datetime.now(timezone.utc).timestamp()) % 2**32)
    now = pd.Timestamp.now("UTC").floor("h").tz_localize(None)

    doy = now.dayofyear
    temp_base = 10.0 + 8.0 * np.sin(2 * np.pi * (doy - 80) / 365)
    temp = round(temp_base + rng.normal(0, 2.5), 1)

    record = {
        "time":          now,
        "temp":          temp,
        "feels_like":    round(temp - rng.uniform(0, 3), 1),
        "temp_min":      round(temp - rng.uniform(1, 3), 1),
        "temp_max":      round(temp + rng.uniform(1, 3), 1),
        "humidity":      round(np.clip(rng.normal(75, 12), 20, 100), 1),
        "pressure":      round(rng.normal(1013, 10), 1),
        "wind_speed":    round(np.clip(rng.exponential(12), 0, 80), 1),
        "wind_deg":      round(rng.uniform(0, 360), 0),
        "clouds":        round(np.clip(rng.normal(60, 25), 0, 100), 0),
        "precipitation": round(np.clip(rng.exponential(1), 0, 30), 1),
        "weather_main":  rng.choice(["Clouds", "Rain", "Clear", "Drizzle"]),
        "weather_desc":  "synthetic data",
    }

    df = pd.DataFrame([record]).set_index("time")
    df.index.name = "time"

    if use_cache:
        _save_cache(df, cache_path)

    return df


def _generate_synthetic_forecast(
    city: str,
    cache_path: Path,
    use_cache: bool,
) -> pd.DataFrame:
    """Generate a 7‑day synthetic forecast."""
    rng = np.random.default_rng(seed=int(datetime.now(timezone.utc).timestamp()) % 2**32)
    today = pd.Timestamp.now("UTC").normalize().tz_localize(None)
    dates = pd.date_range(start=today, periods=7, freq="D")

    doy = dates.dayofyear.values.astype(float)
    temp_base = 10.0 + 8.0 * np.sin(2 * np.pi * (doy - 80) / 365)
    temp = np.round(temp_base + rng.normal(0, 2.5, 7), 1)

    df = pd.DataFrame({
        "temp":          temp,
        "feels_like":    np.round(temp - rng.uniform(0, 3, 7), 1),
        "temp_min":      np.round(temp - rng.uniform(2, 5, 7), 1),
        "temp_max":      np.round(temp + rng.uniform(2, 5, 7), 1),
        "humidity":      np.clip(rng.normal(75, 12, 7), 20, 100).round(1),
        "pressure":      rng.normal(1013, 10, 7).round(1),
        "wind_speed":    np.clip(rng.exponential(12, 7), 0, 80).round(1),
        "clouds":        np.clip(rng.normal(60, 25, 7), 0, 100).round(0),
        "precipitation": np.clip(rng.exponential(2.5, 7), 0, 30).round(1),
        "pop":           np.clip(rng.uniform(0, 1, 7), 0, 1).round(2),
        "weather_main":  rng.choice(
            ["Clouds", "Rain", "Clear", "Drizzle", "Overcast"], 7
        ),
    }, index=dates)

    df.index.name = "time"

    if use_cache:
        _save_cache(df, cache_path)

    logger.info("Synthetic forecast: %d days for %s", len(df), city)
    return df
