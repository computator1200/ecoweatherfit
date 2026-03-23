"""
EcoWeatherFit — Data Splitting & Scaling Module
=================================================
Implements strict **chronological** train / validation / test splits
with **zero data leakage**.  This is the single most critical academic
requirement in the pipeline.

Data‑Leakage Prevention Strategy
---------------------------------
1. **No shuffling**: all splits respect temporal order.  The training
   set always precedes the validation set, which always precedes the
   test set.
2. **Scalers are fit on training data only**: ``fit()`` is called
   exclusively on ``X_train``.  ``transform()`` is then applied to
   validation and test sets.  This prevents future distributional
   information from leaking into the training features.
3. **Gap option**: an optional gap (in days) between train and
   validation, and between validation and test, prevents any residual
   autocorrelation from bridging the split boundaries.
4. **Rolling‑origin evaluation** (optional): for rigorous walk‑forward
   validation, the module provides an expanding‑window generator.

All choices are logged and returned as metadata so they can be
presented during a Viva Q&A.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config.settings import (
    FORECAST_HORIZON_DAYS,
    LSTM_LOOKBACK_WINDOW,
    TARGET_VARIABLES,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data‑Class for Split Results
# ──────────────────────────────────────────────
@dataclass
class TimeSeriesSplit:
    """
    Container for a chronological train/val/test split.

    Attributes
    ----------
    X_train, y_train : pd.DataFrame
        Training features and targets.
    X_val, y_val : pd.DataFrame
        Validation features and targets.
    X_test, y_test : pd.DataFrame
        Test features and targets.
    feature_scaler : sklearn scaler
        Fitted on X_train only.
    target_scaler : sklearn scaler
        Fitted on y_train only.
    metadata : dict
        Split boundaries, sizes, and configuration.
    """
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_val:   pd.DataFrame
    y_val:   pd.DataFrame
    X_test:  pd.DataFrame
    y_test:  pd.DataFrame
    feature_scaler: object = None
    target_scaler:  object = None
    metadata: Dict = field(default_factory=dict)


# ──────────────────────────────────────────────
# Main Split Function
# ──────────────────────────────────────────────
def time_based_split(
    df: pd.DataFrame,
    target_cols: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    test_ratio:  float = TEST_RATIO,
    gap_days:    int = 0,
    scale:       bool = True,
    scaler_type: str = "standard",
    drop_na:     bool = True,
) -> TimeSeriesSplit:
    """
    Perform a strict chronological split with leakage‑safe scaling.

    Parameters
    ----------
    df : pd.DataFrame
        Feature‑engineered DataFrame with DatetimeIndex.
    target_cols : list of str, optional
        Columns to predict.  Defaults to ``TARGET_VARIABLES``.
    feature_cols : list of str, optional
        Columns to use as inputs.  If None, all numeric columns
        **except** targets are used.
    train_ratio, val_ratio, test_ratio : float
        Approximate proportions (must sum to ≤ 1.0).
    gap_days : int
        Number of days to skip between splits to mitigate temporal
        autocorrelation leakage.
    scale : bool
        If True, apply scaling (fit on train only).
    scaler_type : str
        ``"standard"`` (zero‑mean, unit‑variance) or ``"minmax"``
        (0–1 range).
    drop_na : bool
        If True, drop rows with any NaN before splitting.

    Returns
    -------
    TimeSeriesSplit
        Dataclass containing all splits, scalers, and metadata.

    Raises
    ------
    ValueError
        If ratios are invalid or data is too short.
    """
    # ── Validate ratios ──
    total = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total <= 1.01):
        raise ValueError(
            f"Split ratios must sum to ~1.0, got {total:.3f}."
        )

    # ── Resolve columns ──
    if target_cols is None:
        target_cols = [c for c in TARGET_VARIABLES if c in df.columns]
    if feature_cols is None:
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in target_cols
        ]

    # ── Sort and optionally drop NaN ──
    df = df.sort_index()
    if drop_na:
        before = len(df)
        df = df.dropna(subset=feature_cols + target_cols)
        dropped = before - len(df)
        if dropped > 0:
            logger.info("Split prep: dropped %d NaN rows.", dropped)

    n = len(df)
    if n < 60:
        raise ValueError(
            f"Only {n} rows after NaN removal — too few for a meaningful split."
        )

    # ── Compute split indices ──
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    # Test gets the remainder to avoid rounding loss
    n_test  = n - n_train - n_val

    # Apply gap
    train_end = n_train
    val_start = train_end + gap_days
    val_end   = val_start + n_val
    test_start = val_end + gap_days

    if test_start >= n:
        logger.warning(
            "Gap too large for dataset size — reducing gap to 0."
        )
        gap_days = 0
        val_start = train_end
        val_end   = val_start + n_val
        test_start = val_end

    # ── Slice ──
    X = df[feature_cols]
    y = df[target_cols]

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val   = X.iloc[val_start:val_end]
    y_val   = y.iloc[val_start:val_end]
    X_test  = X.iloc[test_start:]
    y_test  = y.iloc[test_start:]

    # ── Scaling (fit on TRAIN only) ──
    feature_scaler = None
    target_scaler  = None

    if scale:
        ScalerClass = StandardScaler if scaler_type == "standard" else MinMaxScaler

        feature_scaler = ScalerClass()
        feature_scaler.fit(X_train)                           # ← TRAIN ONLY
        X_train = pd.DataFrame(
            feature_scaler.transform(X_train),
            index=X_train.index, columns=X_train.columns,
        )
        X_val = pd.DataFrame(
            feature_scaler.transform(X_val),
            index=X_val.index, columns=X_val.columns,
        )
        X_test = pd.DataFrame(
            feature_scaler.transform(X_test),
            index=X_test.index, columns=X_test.columns,
        )

        target_scaler = ScalerClass()
        target_scaler.fit(y_train)                            # ← TRAIN ONLY
        y_train = pd.DataFrame(
            target_scaler.transform(y_train),
            index=y_train.index, columns=y_train.columns,
        )
        y_val = pd.DataFrame(
            target_scaler.transform(y_val),
            index=y_val.index, columns=y_val.columns,
        )
        y_test = pd.DataFrame(
            target_scaler.transform(y_test),
            index=y_test.index, columns=y_test.columns,
        )

        logger.info(
            "Scaling (%s): fit on %d training rows, applied to val/test.",
            scaler_type, len(X_train),
        )

    # ── Build metadata ──
    metadata = {
        "train_start":  str(X_train.index.min()),
        "train_end":    str(X_train.index.max()),
        "val_start":    str(X_val.index.min()),
        "val_end":      str(X_val.index.max()),
        "test_start":   str(X_test.index.min()),
        "test_end":     str(X_test.index.max()),
        "n_train":      len(X_train),
        "n_val":        len(X_val),
        "n_test":       len(X_test),
        "n_features":   len(feature_cols),
        "n_targets":    len(target_cols),
        "gap_days":     gap_days,
        "scaler_type":  scaler_type if scale else "none",
        "feature_cols": feature_cols,
        "target_cols":  target_cols,
    }

    _log_split_summary(metadata)

    return TimeSeriesSplit(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        metadata=metadata,
    )


# ──────────────────────────────────────────────
# LSTM Sequence Builder
# ──────────────────────────────────────────────
def create_lstm_sequences(
    X: pd.DataFrame,
    y: pd.DataFrame,
    lookback: int = LSTM_LOOKBACK_WINDOW,
    horizon: int = FORECAST_HORIZON_DAYS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert tabular time‑series into 3‑D arrays for LSTM input.

    Constructs non‑overlapping (sliding) windows of shape:
    * ``X_seq``: ``(n_samples, lookback, n_features)``
    * ``y_seq``: ``(n_samples, horizon, n_targets)``

    This is the standard "sliding window" approach described in the
    project specification.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (scaled).
    y : pd.DataFrame
        Target matrix (scaled).
    lookback : int
        Number of past days used as input context.
    horizon : int
        Number of future days to predict.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Input sequences and corresponding target sequences.

    Raises
    ------
    ValueError
        If the data is shorter than ``lookback + horizon``.
    """
    X_arr = X.values
    y_arr = y.values
    n = len(X_arr)

    if n < lookback + horizon:
        raise ValueError(
            f"Data length ({n}) < lookback ({lookback}) + horizon ({horizon})."
        )

    X_seq, y_seq = [], []
    for i in range(n - lookback - horizon + 1):
        X_seq.append(X_arr[i : i + lookback])
        y_seq.append(y_arr[i + lookback : i + lookback + horizon])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    logger.info(
        "LSTM sequences: X=%s, y=%s (lookback=%d, horizon=%d)",
        X_seq.shape, y_seq.shape, lookback, horizon,
    )
    return X_seq, y_seq


# ──────────────────────────────────────────────
# Rolling‑Origin (Walk‑Forward) Generator
# ──────────────────────────────────────────────
def walk_forward_splits(
    df: pd.DataFrame,
    target_cols: List[str],
    feature_cols: List[str],
    min_train_size: int = 365,
    step_size: int = 30,
    test_size: int = FORECAST_HORIZON_DAYS,
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], None, None]:
    """
    Generate expanding‑window walk‑forward validation splits.

    At each step, the training window grows by ``step_size`` days and
    the test window is the next ``test_size`` days.  This mimics
    real‑world deployment where the model is periodically retrained
    on all available past data.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature‑engineered DataFrame.
    target_cols : list of str
        Target column names.
    feature_cols : list of str
        Feature column names.
    min_train_size : int
        Minimum training set size (days) for the first fold.
    step_size : int
        Number of days the training window expands per fold.
    test_size : int
        Number of days in each test fold.

    Yields
    ------
    tuple of (X_train, y_train, X_test, y_test)
        DataFrames for each fold.
    """
    n = len(df)
    X = df[feature_cols]
    y = df[target_cols]

    fold = 0
    train_end = min_train_size

    while train_end + test_size <= n:
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test  = X.iloc[train_end : train_end + test_size]
        y_test  = y.iloc[train_end : train_end + test_size]

        fold += 1
        logger.debug(
            "Walk‑forward fold %d: train=%d, test=%d (%s → %s)",
            fold, len(X_train), len(X_test),
            X_test.index.min().date(), X_test.index.max().date(),
        )

        yield X_train, y_train, X_test, y_test

        train_end += step_size

    logger.info("Walk‑forward: generated %d folds.", fold)


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
def _log_split_summary(meta: Dict) -> None:
    """Pretty‑print the split configuration."""
    logger.info("=" * 55)
    logger.info("  TIME‑SERIES SPLIT SUMMARY")
    logger.info("=" * 55)
    logger.info("  Train : %s → %s  (%d rows)", meta["train_start"], meta["train_end"], meta["n_train"])
    logger.info("  Val   : %s → %s  (%d rows)", meta["val_start"], meta["val_end"], meta["n_val"])
    logger.info("  Test  : %s → %s  (%d rows)", meta["test_start"], meta["test_end"], meta["n_test"])
    logger.info("  Gap   : %d days", meta["gap_days"])
    logger.info("  Features: %d,  Targets: %d", meta["n_features"], meta["n_targets"])
    logger.info("  Scaler: %s (fit on train only)", meta["scaler_type"])
    logger.info("=" * 55)
