"""
EcoWeatherFit — Feature Engineering Module
============================================
Transforms the cleaned daily weather DataFrame into a rich feature
matrix suitable for both the Random Forest (tabular) and LSTM
(sequential) forecasting models.

Feature Categories
------------------
1. **Lag features** — past observations shifted by configurable offsets
   (e.g. ``temp_lag_1``, ``wind_speed_lag_7``).  These give the RF
   model an explicit view of recent history.
2. **Rolling statistics** — moving averages and standard deviations
   over windows of 3, 7, and 14 days.  These capture local trends and
   volatility.
3. **Cyclical temporal encodings** — sine/cosine transforms of
   day‑of‑year, month, and day‑of‑week.  This avoids the arbitrary
   discontinuity of raw integer encodings (Dec 31 → Jan 1).
4. **Interaction features** — wind chill index, discomfort index
   (heat index), and a binary ``is_rainy`` flag.

Data‑Leakage Prevention
------------------------
* **Lag and rolling features are strictly backward‑looking**: no future
  information is used in their computation.
* Rolling windows use ``min_periods=1`` to avoid NaN inflation at the
  dataset boundary, but the first ``window-1`` rows should be
  considered "warm‑up" and excluded from test evaluation.
* All feature engineering is defined as a **stateless transformation**
  — it does not fit any parameters.  Scaling (which *does* fit) is
  handled separately in ``splitting.py`` to guarantee train‑only
  fitting.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from config.settings import (
    RF_LAG_FEATURES,
    TARGET_VARIABLES,
    TEMPORAL_FEATURES,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────
def engineer_features(
    df: pd.DataFrame,
    target_cols: Optional[List[str]] = None,
    lag_days: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    add_temporal: bool = True,
    add_interactions: bool = True,
    drop_na_rows: bool = False,
) -> pd.DataFrame:
    """
    Apply the full feature‑engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned daily DataFrame with ``DatetimeIndex``.
    target_cols : list of str, optional
        Columns for which lag/rolling features are created.
        Defaults to ``TARGET_VARIABLES``.
    lag_days : list of int, optional
        Lag offsets.  Defaults to ``RF_LAG_FEATURES`` from settings.
    rolling_windows : list of int, optional
        Window sizes for rolling stats (default ``[3, 7, 14]``).
    add_temporal : bool
        Whether to add cyclical time encodings.
    add_interactions : bool
        Whether to add interaction / derived features.
    drop_na_rows : bool
        If True, drop rows that contain any NaN produced by lagging.
        Default False — the caller (e.g. ``splitting.py``) manages
        NaN removal to preserve full control.

    Returns
    -------
    pd.DataFrame
        Enhanced DataFrame with all new features appended.
    """
    if target_cols is None:
        target_cols = [c for c in TARGET_VARIABLES if c in df.columns]
    if lag_days is None:
        lag_days = RF_LAG_FEATURES
    if rolling_windows is None:
        rolling_windows = [3, 7, 14]

    df = df.copy()
    n_before = len(df.columns)

    # 1. Lag features
    df = _add_lag_features(df, target_cols, lag_days)

    # 2. Rolling statistics
    df = _add_rolling_stats(df, target_cols, rolling_windows)

    # 3. Cyclical temporal encodings
    if add_temporal:
        df = _add_temporal_features(df)

    # 4. Interaction / derived features
    if add_interactions:
        df = _add_interaction_features(df)

    n_after = len(df.columns)
    logger.info(
        "Feature engineering: %d → %d columns (+%d features).",
        n_before, n_after, n_after - n_before,
    )

    if drop_na_rows:
        before = len(df)
        df = df.dropna()
        logger.info(
            "Dropped %d rows with NaN after feature engineering.", before - len(df)
        )

    return df


# ──────────────────────────────────────────────
# 1. Lag Features
# ──────────────────────────────────────────────
def _add_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
) -> pd.DataFrame:
    """
    Create lagged copies of specified columns.

    For each column ``c`` and lag ``k``, a new column ``{c}_lag_{k}``
    is created containing the value from ``k`` days prior.

    These features are the primary mechanism by which the Random Forest
    model receives temporal context — since RF treats each row
    independently, explicit history must be encoded as features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Columns to lag.
    lags : list of int
        Day offsets (positive integers).

    Returns
    -------
    pd.DataFrame
        DataFrame with lag columns appended.
    """
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


# ──────────────────────────────────────────────
# 2. Rolling Statistics
# ──────────────────────────────────────────────
def _add_rolling_stats(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
) -> pd.DataFrame:
    """
    Compute backward‑looking rolling mean and standard deviation.

    * ``{col}_rmean_{w}`` — rolling mean over ``w`` days.
    * ``{col}_rstd_{w}``  — rolling standard deviation over ``w`` days.

    ``min_periods=1`` ensures the first few rows produce partial (but
    valid) statistics rather than NaN.  The standard deviation for a
    single observation is defined as 0 (no variance).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Target columns.
    windows : list of int
        Rolling window sizes in days.

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling statistics appended.
    """
    for col in columns:
        if col not in df.columns:
            continue
        for w in windows:
            roll = df[col].rolling(window=w, min_periods=1)
            df[f"{col}_rmean_{w}"]  = roll.mean()
            df[f"{col}_rstd_{w}"]   = roll.std().fillna(0.0)
    return df


# ──────────────────────────────────────────────
# 3. Cyclical Temporal Encodings
# ──────────────────────────────────────────────
def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode calendar information as sine / cosine pairs.

    Why cyclical?
    ~~~~~~~~~~~~~
    Raw integers (month = 12, then 1) introduce a false discontinuity.
    Sine–cosine encoding maps December and January close together in
    the feature space, reflecting their true seasonal proximity.

    Features added:
    * ``day_of_year_sin``, ``day_of_year_cos`` — annual cycle
    * ``month_sin``, ``month_cos``             — annual cycle (coarser)
    * ``day_of_week_sin``, ``day_of_week_cos`` — weekly cycle

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with temporal features appended.
    """
    idx = df.index

    # Day of year (1–366)
    doy = idx.dayofyear.values.astype(float)
    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Month (1–12)
    month = idx.month.values.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Day of week (0=Mon, 6=Sun)
    dow = idx.dayofweek.values.astype(float)
    df["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)

    return df


# ──────────────────────────────────────────────
# 4. Interaction / Derived Features
# ──────────────────────────────────────────────
def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain‑informed derived features.

    Features
    --------
    * **wind_chill** — Apparent temperature factoring wind.
      Uses the standard North American wind chill formula when
      temp < 10 °C and wind > 4.8 km/h; otherwise equals temp.

    * **heat_index** — Simplified discomfort index for warm/humid
      conditions (Steadman, 1979).

    * **is_rainy** — Binary flag: 1 if precipitation > 0.5 mm/day.

    * **temp_range** — Daily temperature range (temp_max - temp_min)
      when both are available.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least ``temp`` and ``wind_speed``.

    Returns
    -------
    pd.DataFrame
        DataFrame with interaction features appended.
    """
    if "temp" in df.columns and "wind_speed" in df.columns:
        # Wind Chill Index (Environment Canada formula)
        t = df["temp"]
        w = df["wind_speed"]
        wc = 13.12 + 0.6215 * t - 11.37 * w**0.16 + 0.3965 * t * w**0.16
        # Only valid when temp < 10 and wind > 4.8 km/h
        df["wind_chill"] = np.where(
            (t < 10) & (w > 4.8), wc, t
        )

    if "temp" in df.columns and "humidity" in df.columns:
        # Simplified Heat Index (valid for temp > 20 °C)
        t = df["temp"]
        rh = df["humidity"]
        hi = (
            -8.7847 + 1.6114 * t + 2.3385 * rh
            - 0.1461 * t * rh - 0.0123 * t**2
            - 0.0164 * rh**2 + 0.0022 * t**2 * rh
            + 0.0007 * t * rh**2 - 0.0000036 * t**2 * rh**2
        )
        df["heat_index"] = np.where(t > 20, hi, t)

    if "precipitation" in df.columns:
        df["is_rainy"] = (df["precipitation"] > 0.5).astype(np.float32)

    if "temp_max" in df.columns and "temp_min" in df.columns:
        df["temp_range"] = df["temp_max"] - df["temp_min"]

    return df


# ──────────────────────────────────────────────
# Utility: Feature Name Lists
# ──────────────────────────────────────────────
def get_feature_names(
    df: pd.DataFrame,
    exclude_targets: bool = True,
) -> List[str]:
    """
    Return an ordered list of feature column names, optionally
    excluding raw target variables.

    Useful for selecting ``X`` columns before model training.

    Parameters
    ----------
    df : pd.DataFrame
        Feature‑engineered DataFrame.
    exclude_targets : bool
        If True, exclude columns in ``TARGET_VARIABLES``.

    Returns
    -------
    list of str
        Column names suitable for model input.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if exclude_targets:
        numeric_cols = [c for c in numeric_cols if c not in TARGET_VARIABLES]

    return numeric_cols
