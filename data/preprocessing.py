"""
EcoWeatherFit — Data Preprocessing Pipeline
=============================================
This module is the quality gate between raw API data and the feature‑
engineering / modelling stages.  Every row that reaches the models has
passed through a documented, reproducible chain of transformations:

Pipeline Steps (in order)
-------------------------
1. **Schema enforcement** — ensure expected columns exist.
2. **Duplicate removal** — drop exact duplicate timestamps.
3. **Plausibility filtering** — flag/remove physically impossible
   values using domain bounds from ``config.settings``.
4. **Missing‑value audit** — quantify and log gaps; apply interpolation
   where appropriate; reject columns exceeding the threshold.
5. **Temporal alignment** — resample to a strict daily frequency with
   no gaps, forward‑filling at most 3 consecutive days.
6. **Type coercion** — cast all numeric columns to ``float32`` for
   memory efficiency and GPU compatibility.

Academic Note — Transparency
----------------------------
* No silent dropping: every exclusion is logged with a reason.
* A ``cleaning_report`` dict is attached to the returned DataFrame as
  ``df.attrs["cleaning_report"]`` for programmatic inspection.
* This design supports full reproducibility and Viva scrutiny.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    MAX_MISSING_RATIO,
    PLAUSIBILITY_BOUNDS,
    TARGET_VARIABLES,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────
def preprocess_pipeline(
    df: pd.DataFrame,
    target_columns: Optional[List[str]] = None,
    max_missing_ratio: float = MAX_MISSING_RATIO,
    max_consecutive_fill: int = 3,
    resample_freq: str = "D",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline on a raw weather DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with a ``DatetimeIndex`` named ``"time"``.
    target_columns : list of str, optional
        Columns to retain.  Defaults to ``TARGET_VARIABLES`` from
        settings.  Extra columns are dropped to keep the dataset lean.
    max_missing_ratio : float
        Maximum fraction of NaN allowed per column (0–1).  Columns
        exceeding this are **logged and dropped** with a warning.
    max_consecutive_fill : int
        Maximum number of consecutive missing days that will be
        forward‑filled.  Longer gaps are left as NaN (and later
        handled by the feature‑engineering module).
    resample_freq : str
        Pandas offset alias for resampling (default ``"D"`` = daily).
    verbose : bool
        If True, print a summary table to the logger.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a complete daily DatetimeIndex, float32
        numeric types, and a ``cleaning_report`` dict in ``df.attrs``.

    Raises
    ------
    ValueError
        If the input DataFrame is empty or has no valid DatetimeIndex.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty — cannot preprocess.")

    report: Dict[str, object] = {
        "rows_in":               len(df),
        "duplicates_removed":    0,
        "implausible_nullified": 0,
        "columns_dropped":       [],
        "missing_filled":        0,
        "rows_out":              0,
    }

    # Resolve target columns
    if target_columns is None:
        target_columns = TARGET_VARIABLES

    # ── Step 0: Copy to avoid mutating the caller's data ──
    df = df.copy()

    # ── Step 1: Schema enforcement ──
    df = _enforce_schema(df, target_columns)

    # ── Step 2: Duplicate removal ──
    df, n_dup = _remove_duplicates(df)
    report["duplicates_removed"] = n_dup

    # ── Step 3: Plausibility filtering ──
    df, n_impl = _plausibility_filter(df)
    report["implausible_nullified"] = n_impl

    # ── Step 4: Missing‑value audit & conditional column drop ──
    df, dropped = _audit_missing(df, max_missing_ratio)
    report["columns_dropped"] = dropped

    # ── Step 5: Temporal alignment (resample + limited fill) ──
    df, n_filled = _temporal_align(df, resample_freq, max_consecutive_fill)
    report["missing_filled"] = n_filled

    # ── Step 6: Type coercion ──
    df = _coerce_types(df)

    report["rows_out"] = len(df)

    # Attach report as DataFrame metadata
    df.attrs["cleaning_report"] = report

    if verbose:
        _log_report(report)

    return df


# ──────────────────────────────────────────────
# Pipeline Sub‑Steps
# ──────────────────────────────────────────────

def _enforce_schema(
    df: pd.DataFrame,
    required_cols: List[str],
) -> pd.DataFrame:
    """
    Ensure the DataFrame has the expected columns and a proper
    DatetimeIndex.

    * Missing required columns are created and filled with ``NaN``
      (rather than raising — this allows partial data to proceed).
    * Extra columns beyond the required set + known auxiliaries are
      preserved but logged so the user is aware.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame.
    required_cols : list of str
        Columns that **must** be present (will be created if absent).

    Returns
    -------
    pd.DataFrame
        DataFrame with guaranteed required columns and DatetimeIndex.
    """
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time")
        else:
            raise ValueError(
                "DataFrame must have a DatetimeIndex or a 'time' column."
            )

    df.index.name = "time"

    # Add missing required columns as NaN
    for col in required_cols:
        if col not in df.columns:
            logger.warning("Schema: required column '%s' missing — added as NaN.", col)
            df[col] = np.nan

    present = set(df.columns)
    extra = present - set(required_cols)
    if extra:
        logger.debug("Schema: extra columns retained: %s", sorted(extra))

    return df


def _remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove rows with duplicate timestamps, keeping the first occurrence.

    Returns
    -------
    tuple of (pd.DataFrame, int)
        Cleaned DataFrame and count of duplicates removed.
    """
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info("Duplicates: removed %d duplicate timestamps.", n_removed)
    return df, n_removed


def _plausibility_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Replace values outside physically plausible bounds with NaN.

    Bounds are defined in ``config.settings.PLAUSIBILITY_BOUNDS``.
    Every replacement is counted and logged — **no silent data loss**.

    Returns
    -------
    tuple of (pd.DataFrame, int)
        DataFrame with implausible values replaced, and total count of
        values nullified.
    """
    total_nullified = 0

    for col, (lo, hi) in PLAUSIBILITY_BOUNDS.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        n_bad = mask.sum()
        if n_bad > 0:
            logger.warning(
                "Plausibility: %d values in '%s' outside [%.1f, %.1f] → NaN.",
                n_bad, col, lo, hi,
            )
            df.loc[mask, col] = np.nan
            total_nullified += n_bad

    return df, total_nullified


def _audit_missing(
    df: pd.DataFrame,
    max_ratio: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Audit missing values across all columns.  Columns exceeding the
    maximum allowed missing ratio are dropped with a logged warning.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    max_ratio : float
        Maximum fraction of NaN per column before dropping.

    Returns
    -------
    tuple of (pd.DataFrame, list of str)
        Cleaned DataFrame and list of dropped column names.
    """
    missing = df.isnull().mean()
    dropped: List[str] = []

    for col, ratio in missing.items():
        if ratio > max_ratio:
            logger.warning(
                "Missing audit: '%s' has %.1f%% missing (threshold %.1f%%) → DROPPED.",
                col, ratio * 100, max_ratio * 100,
            )
            dropped.append(str(col))
        elif ratio > 0:
            logger.info(
                "Missing audit: '%s' has %.1f%% missing — will be interpolated.",
                col, ratio * 100,
            )

    if dropped:
        df = df.drop(columns=dropped)

    return df, dropped


def _temporal_align(
    df: pd.DataFrame,
    freq: str,
    max_fill: int,
) -> Tuple[pd.DataFrame, int]:
    """
    Resample to a strict frequency and fill short gaps.

    Gaps longer than ``max_fill`` consecutive periods are left as NaN.
    Forward‑fill first, then backward‑fill (at most 1 period) to handle
    leading NaNs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex.
    freq : str
        Pandas offset alias (e.g. ``"D"``).
    max_fill : int
        Maximum consecutive periods to fill.

    Returns
    -------
    tuple of (pd.DataFrame, int)
        Aligned DataFrame and count of values filled.
    """
    # Sort chronologically
    df = df.sort_index()

    # Resample: use mean aggregation if multiple observations per period
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    df_resampled = df[numeric_cols].resample(freq).mean()

    # Carry forward non‑numeric columns (e.g. weather_main)
    for col in non_numeric_cols:
        if col in df.columns:
            df_resampled[col] = df[col].resample(freq).first()

    missing_before = df_resampled[numeric_cols].isnull().sum().sum()

    # Limited forward‑fill + 1‑period back‑fill
    df_resampled[numeric_cols] = (
        df_resampled[numeric_cols]
        .ffill(limit=max_fill)
        .bfill(limit=1)
    )

    missing_after = df_resampled[numeric_cols].isnull().sum().sum()
    n_filled = int(missing_before - missing_after)

    if n_filled > 0:
        logger.info("Temporal align: filled %d missing values.", n_filled)

    remaining = int(missing_after)
    if remaining > 0:
        logger.warning(
            "Temporal align: %d NaN values remain (long gaps).", remaining,
        )

    return df_resampled, n_filled


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast all numeric columns to ``float32`` for memory efficiency and
    consistent tensor conversion downstream.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].astype(np.float32)
    return df


def _log_report(report: Dict[str, object]) -> None:
    """Pretty‑print the cleaning report to the logger."""
    logger.info("=" * 55)
    logger.info("  PREPROCESSING REPORT")
    logger.info("=" * 55)
    logger.info("  Rows in:               %d", report["rows_in"])
    logger.info("  Duplicates removed:     %d", report["duplicates_removed"])
    logger.info("  Implausible → NaN:      %d", report["implausible_nullified"])
    logger.info("  Columns dropped:        %s", report["columns_dropped"] or "none")
    logger.info("  Missing values filled:  %d", report["missing_filled"])
    logger.info("  Rows out:               %d", report["rows_out"])
    logger.info("=" * 55)


# ──────────────────────────────────────────────
# Convenience — Quick Quality Summary
# ──────────────────────────────────────────────
def quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a per‑column quality summary useful for EDA notebooks
    and Viva presentations.

    Columns
    -------
    dtype, count, missing, missing_pct, min, max, mean, std

    Returns
    -------
    pd.DataFrame
        Summary table.
    """
    summary = pd.DataFrame({
        "dtype":       df.dtypes,
        "count":       df.count(),
        "missing":     df.isnull().sum(),
        "missing_pct": (df.isnull().mean() * 100).round(2),
        "min":         df.min(numeric_only=True),
        "max":         df.max(numeric_only=True),
        "mean":        df.mean(numeric_only=True),
        "std":         df.std(numeric_only=True),
    })
    return summary
