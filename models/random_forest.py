"""
EcoWeatherFit — Random Forest Weather Forecasting Model
========================================================
A multi-output Random Forest regressor that serves as the classical
ML baseline for weather prediction.  It learns the mapping from
engineered features (lags, rolling stats, temporal encodings) to the
four target variables: temp, humidity, wind_speed, precipitation.

For multi-step (7-day) forecasting, the model uses **recursive
prediction**: predict one day ahead, feed the prediction back as
lag features, and repeat.  This approach is standard practice for
tabular models operating on time-series data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.settings import (
    FORECAST_HORIZON_DAYS,
    MODEL_DIR,
    RF_CONFIG,
    TARGET_VARIABLES,
)

logger = logging.getLogger(__name__)

SAVE_PATH = MODEL_DIR / "random_forest.joblib"


class WeatherRandomForest:
    """
    Multi-output Random Forest for weather forecasting.

    Wraps ``sklearn.ensemble.RandomForestRegressor`` with convenience
    methods for training, evaluation, serialisation, and recursive
    multi-step prediction.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or RF_CONFIG
        self.model: Optional[RandomForestRegressor] = None
        self.is_trained: bool = False
        self.metrics: Dict = {}
        self.feature_names: List[str] = []
        self.target_names: List[str] = []

    # ──────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.DataFrame] = None,
    ) -> "WeatherRandomForest":
        """Train the Random Forest on the provided split data."""
        self.feature_names = list(X_train.columns)
        self.target_names = list(y_train.columns)

        self.model = RandomForestRegressor(
            n_estimators=self.config["n_estimators"],
            max_depth=self.config["max_depth"],
            min_samples_leaf=self.config["min_samples_leaf"],
            random_state=self.config["random_state"],
            n_jobs=-1,
        )

        logger.info(
            "Training RF: %d samples, %d features → %d targets …",
            len(X_train), len(self.feature_names), len(self.target_names),
        )
        self.model.fit(X_train.values, y_train.values)
        self.is_trained = True

        if X_val is not None and y_val is not None:
            self.metrics = self.evaluate(X_val, y_val)
            _log_metrics("RF Validation", self.metrics)

        return self

    # ──────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets from a feature matrix (2-D numpy array)."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)

    def forecast_7day(
        self,
        recent_raw: pd.DataFrame,
        feature_scaler,
        target_scaler,
        feature_cols: List[str],
        target_cols: List[str],
    ) -> pd.DataFrame:
        """
        Generate a 7-day recursive forecast.

        Parameters
        ----------
        recent_raw : pd.DataFrame
            The last ~60 days of *preprocessed + feature-engineered* data
            (unscaled, with all feature columns present).
        feature_scaler : fitted sklearn scaler
        target_scaler : fitted sklearn scaler
        feature_cols : list of str
        target_cols : list of str

        Returns
        -------
        pd.DataFrame
            7 rows of inverse-scaled predictions with columns = target_cols.
        """
        from data.feature_engineering import engineer_features

        working = recent_raw.copy()
        predictions = []
        last_date = working.index.max()

        # Persistence fallback for non-target columns: use the trailing 7-day
        # mean of the most recent observed window. Without this, every
        # appended prediction row has NaN for fields like temp_min, temp_max,
        # pressure, wpgt — and the downstream lag/rolling features collapse
        # toward zero, dragging the recursive forecast toward a constant.
        recent_window = working.tail(7).copy()
        non_target_means = {}
        for col in working.columns:
            if col in target_cols:
                continue
            try:
                non_target_means[col] = float(recent_window[col].mean(skipna=True))
            except (TypeError, ValueError):
                non_target_means[col] = np.nan

        for day_offset in range(1, FORECAST_HORIZON_DAYS + 1):
            # Re-engineer features from the growing working set
            featured = engineer_features(working, drop_na_rows=False)

            # Ensure column alignment
            missing_cols = [c for c in feature_cols if c not in featured.columns]
            for c in missing_cols:
                featured[c] = 0.0

            last_row = featured[feature_cols].iloc[[-1]].fillna(0.0)
            X_scaled = feature_scaler.transform(last_row.values)
            y_scaled = self.predict(X_scaled)
            y_inv = target_scaler.inverse_transform(y_scaled)

            pred_dict = {col: y_inv[0, i] for i, col in enumerate(target_cols)}
            pred_date = last_date + pd.Timedelta(days=day_offset)

            predictions.append(pred_dict)

            # Build the next-day row: predicted targets + persistence for the
            # rest. This keeps lag/rolling features in-distribution for the
            # next recursive step instead of letting NaNs cascade.
            row_data = {}
            for col in working.columns:
                if col in target_cols:
                    row_data[col] = pred_dict.get(col, np.nan)
                else:
                    row_data[col] = non_target_means.get(col, np.nan)
            new_row = pd.DataFrame(row_data, index=[pred_date])
            new_row.index.name = "time"
            working = pd.concat([working, new_row])

        forecast_dates = pd.date_range(
            last_date + pd.Timedelta(days=1), periods=FORECAST_HORIZON_DAYS, freq="D"
        )
        return pd.DataFrame(predictions, index=forecast_dates, columns=target_cols)

    # ──────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────
    def evaluate(self, X: pd.DataFrame, y_true: pd.DataFrame) -> Dict:
        """Compute per-target MAE, RMSE, and R² on a given set."""
        y_pred = self.predict(X.values)
        metrics = {}
        for i, col in enumerate(y_true.columns):
            yt = y_true.iloc[:, i].values
            yp = y_pred[:, i]
            metrics[col] = {
                "mae": round(float(mean_absolute_error(yt, yp)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
                "r2": round(float(r2_score(yt, yp)), 4),
            }
        return metrics

    # ──────────────────────────────────────────────
    # Serialisation
    # ──────────────────────────────────────────────
    def save(self, path: Optional[Path] = None) -> None:
        path = path or SAVE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "metrics": self.metrics,
                "feature_names": self.feature_names,
                "target_names": self.target_names,
            },
            path,
        )
        logger.info("RF model saved → %s", path)

    def load(self, path: Optional[Path] = None) -> bool:
        path = path or SAVE_PATH
        if not path.exists():
            return False
        data = joblib.load(path)
        self.model = data["model"]
        self.metrics = data.get("metrics", {})
        self.feature_names = data.get("feature_names", [])
        self.target_names = data.get("target_names", [])
        self.is_trained = True
        logger.info("RF model loaded ← %s", path)
        return True


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _log_metrics(label: str, metrics: Dict) -> None:
    logger.info("── %s Metrics ──", label)
    for col, m in metrics.items():
        logger.info(
            "  %s  →  MAE=%.4f  RMSE=%.4f  R²=%.4f",
            col, m["mae"], m["rmse"], m["r2"],
        )
