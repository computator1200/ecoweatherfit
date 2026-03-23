"""
EcoWeatherFit — LSTM Deep Learning Weather Forecasting Model
=============================================================
A sliding-window LSTM that maps 30 days of historical features to a
7-day multi-target weather forecast.

Architecture
------------
Input  → (batch, 30, n_features)
LSTM(64) → Dropout → Dense(32) → Dense(7 × n_targets)
Output → (batch, 7, n_targets)

This is a *direct multi-step* forecaster: it outputs all 7 horizon
days in a single forward pass, avoiding the error-accumulation
problem of recursive approaches.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    FORECAST_HORIZON_DAYS,
    LSTM_CONFIG,
    LSTM_LOOKBACK_WINDOW,
    MODEL_DIR,
    TARGET_VARIABLES,
)

logger = logging.getLogger(__name__)

SAVE_DIR = MODEL_DIR / "lstm"


class WeatherLSTM:
    """
    Sliding-window LSTM for 7-day weather forecasting.

    Wraps a TensorFlow/Keras Sequential model.  Degrades gracefully
    if TensorFlow is not installed (returns None from build).
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or LSTM_CONFIG
        self.model = None
        self.is_trained: bool = False
        self.metrics: Dict = {}
        self.n_features: int = 0
        self.n_targets: int = 0
        self.lookback: int = LSTM_LOOKBACK_WINDOW
        self.horizon: int = FORECAST_HORIZON_DAYS
        self._tf_available = _check_tensorflow()

    # ──────────────────────────────────────────────
    # Model Construction
    # ──────────────────────────────────────────────
    def build(self, n_features: int, n_targets: int) -> bool:
        """Build the Keras model.  Returns False if TF unavailable."""
        if not self._tf_available:
            logger.warning("TensorFlow not available — LSTM disabled.")
            return False

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
        from tensorflow.keras.optimizers import Adam

        self.n_features = n_features
        self.n_targets = n_targets

        model = Sequential(name="EcoWeatherFit_LSTM")
        model.add(
            LSTM(
                self.config["units"],
                input_shape=(self.lookback, n_features),
                name="lstm_encoder",
            )
        )
        model.add(Dropout(self.config["dropout"], name="dropout"))
        model.add(Dense(32, activation="relu", name="dense_hidden"))
        model.add(Dense(self.horizon * n_targets, name="dense_output"))
        model.add(Reshape((self.horizon, n_targets), name="reshape_output"))

        model.compile(
            optimizer=Adam(learning_rate=self.config["learning_rate"]),
            loss="mse",
            metrics=["mae"],
        )

        self.model = model
        logger.info("LSTM model built: %d params", model.count_params())
        return True

    # ──────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "WeatherLSTM":
        """Train the LSTM on 3-D sequence arrays."""
        if self.model is None:
            raise RuntimeError("Call build() before train().")

        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        callbacks = [
            EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=self.config["patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        logger.info(
            "Training LSTM: %d samples, lookback=%d, horizon=%d …",
            len(X_train), self.lookback, self.horizon,
        )

        history = self.model.fit(
            X_train,
            y_train,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
        )

        self.is_trained = True
        self.metrics = {
            "final_loss": float(history.history["loss"][-1]),
            "final_mae": float(history.history["mae"][-1]),
            "epochs_run": len(history.history["loss"]),
        }
        if "val_loss" in history.history:
            self.metrics["val_loss"] = float(history.history["val_loss"][-1])
            self.metrics["val_mae"] = float(history.history["val_mae"][-1])

        logger.info("LSTM training complete: %s", self.metrics)
        return self

    # ──────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict from a 3-D array (batch, lookback, features).

        Returns shape (batch, horizon, n_targets).
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained.")
        return self.model.predict(X, verbose=0)

    def forecast_7day(
        self,
        recent_features_scaled: np.ndarray,
        target_scaler,
        target_cols: List[str],
        last_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Generate a 7-day forecast from the last 30 days of scaled features.

        Parameters
        ----------
        recent_features_scaled : np.ndarray
            Shape (lookback, n_features) — the most recent window of
            scaled feature values.
        target_scaler : fitted sklearn scaler
        target_cols : list of str
        last_date : pd.Timestamp
            Date of the last known observation.

        Returns
        -------
        pd.DataFrame
            7 rows of inverse-scaled predictions.
        """
        X = recent_features_scaled.reshape(1, self.lookback, -1)
        y_scaled = self.predict(X)  # (1, 7, n_targets)
        y_flat = y_scaled.reshape(self.horizon, len(target_cols))

        if target_scaler is not None:
            y_inv = target_scaler.inverse_transform(y_flat)
        else:
            y_inv = y_flat

        forecast_dates = pd.date_range(
            last_date + pd.Timedelta(days=1),
            periods=self.horizon,
            freq="D",
        )
        return pd.DataFrame(y_inv, index=forecast_dates, columns=target_cols)

    # ──────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────
    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        target_cols: Optional[List[str]] = None,
    ) -> Dict:
        """Compute per-target MAE, RMSE, R² averaged over all horizon steps."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        y_pred = self.predict(X)  # (batch, horizon, n_targets)
        if target_cols is None:
            target_cols = [f"target_{i}" for i in range(y_true.shape[-1])]

        metrics = {}
        for i, col in enumerate(target_cols):
            yt = y_true[:, :, i].flatten()
            yp = y_pred[:, :, i].flatten()
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
        import joblib

        path = path or SAVE_DIR
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "model.keras")
        joblib.dump(
            {
                "metrics": self.metrics,
                "n_features": self.n_features,
                "n_targets": self.n_targets,
                "lookback": self.lookback,
                "horizon": self.horizon,
                "config": self.config,
            },
            path / "metadata.joblib",
        )
        logger.info("LSTM model saved → %s", path)

    def load(self, path: Optional[Path] = None) -> bool:
        if not self._tf_available:
            return False

        import joblib
        from tensorflow.keras.models import load_model

        path = path or SAVE_DIR
        model_path = path / "model.keras"
        meta_path = path / "metadata.joblib"

        if not model_path.exists() or not meta_path.exists():
            return False

        self.model = load_model(model_path)
        meta = joblib.load(meta_path)
        self.metrics = meta.get("metrics", {})
        self.n_features = meta.get("n_features", 0)
        self.n_targets = meta.get("n_targets", 0)
        self.lookback = meta.get("lookback", LSTM_LOOKBACK_WINDOW)
        self.horizon = meta.get("horizon", FORECAST_HORIZON_DAYS)
        self.is_trained = True
        logger.info("LSTM model loaded ← %s", path)
        return True


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────
def _check_tensorflow() -> bool:
    """Return True if TensorFlow can be imported."""
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False
