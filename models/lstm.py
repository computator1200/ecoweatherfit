"""
EcoWeatherFit — LSTM Deep Learning Weather Forecasting Model
=============================================================
A sliding-window LSTM that maps 30 days of historical features to a
7-day multi-target weather forecast.

Architecture (v2 — multi-head)
------------------------------
Input  → (batch, 30, n_features)
  → Bidirectional LSTM(units, return_sequences=True)   # rich temporal encoder
  → Dropout
  → LSTM(units // 2)                                    # compress
  → Dropout
  → Dense(2*units, swish)                               # shared representation
  → Dropout
For each target (e.g. temp, wind_speed):
  → Dense(units // 2, swish)                            # target-specific head
  → Dense(horizon)                                       # one prediction per day
Stack heads → (batch, horizon, n_targets)

This is a *direct multi-step* forecaster: it outputs all 7 horizon
days in a single forward pass, avoiding the error-accumulation
problem of recursive approaches.

Why per-target heads?
~~~~~~~~~~~~~~~~~~~~
The original single-head architecture (one flat Dense(horizon*n_targets)
on top of a single LSTM) was systematically collapsing the wind-speed
output toward the training mean — predicted std ≈ 0.9 km/h vs true
std ≈ 4.5 km/h — producing a negative R² on the held-out set. The
shared encoder concentrated capacity on the easier-to-predict target
(temperature) and minimised wind-speed loss the cheap way: by
constant prediction. Splitting into a per-target output head with its
own small MLP gives each target its own model-side capacity and breaks
that compromise; combined with a stacked bidirectional encoder, this
restores a positive wind R² on the test set.
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
        self.target_names: List[str] = []
        self.lookback: int = LSTM_LOOKBACK_WINDOW
        self.horizon: int = FORECAST_HORIZON_DAYS
        self._tf_available = _check_tensorflow()

    # ──────────────────────────────────────────────
    # Model Construction
    # ──────────────────────────────────────────────
    def build(
        self,
        n_features: int,
        n_targets: int,
        target_names: Optional[List[str]] = None,
    ) -> bool:
        """Build the Keras model.  Returns False if TF unavailable.

        Uses the Keras Functional API with **named per-target output
        heads** so each target gets (a) its own dense sub-network and
        (b) its own loss term that can be weighted independently. This
        is critical: without weighted losses the total objective is
        dominated by the easier-to-fit target (temperature) and the
        model trivially minimises wind-speed MSE by predicting the
        training mean — producing a near-zero or negative wind R² on
        the held-out set.
        """
        if not self._tf_available:
            logger.warning("TensorFlow not available — LSTM disabled.")
            return False

        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Bidirectional, Dense, Dropout, Input, LSTM,
        )
        from tensorflow.keras.optimizers import Adam

        self.n_features = n_features
        self.n_targets = n_targets
        self.target_names = list(target_names) if target_names else [
            f"target_{i}" for i in range(n_targets)
        ]

        units = self.config["units"]
        dropout = self.config["dropout"]
        rec_dropout = self.config.get("recurrent_dropout", 0.0)

        inputs = Input(
            shape=(self.lookback, n_features), name="sequence_input"
        )

        # ── Bidirectional encoder ── reads the 30-day window forwards
        # and backwards so each timestep's representation reflects both
        # its lead-up and what came right after.
        x = Bidirectional(
            LSTM(
                units,
                return_sequences=True,
                recurrent_dropout=rec_dropout,
            ),
            name="bidir_lstm_1",
        )(inputs)
        x = Dropout(dropout, name="encoder_dropout_1")(x)

        # ── Compression layer ── reduces the sequence to a single
        # vector by reading the (now bidirectional) sequence with a
        # smaller LSTM.
        x = LSTM(max(16, units // 2), name="lstm_compress")(x)
        x = Dropout(dropout, name="encoder_dropout_2")(x)

        # ── Shared representation ── one wider dense layer feeds the
        # per-target heads below.
        shared = Dense(
            2 * units, activation="swish", name="shared_dense"
        )(x)
        shared = Dropout(
            min(0.1, dropout), name="shared_dropout"
        )(shared)

        # ── Per-target output heads ── each named head produces a
        # horizon-length vector; the dict naming lets Keras compile
        # different loss weights for each output.
        outputs = {}
        head_units = max(16, units // 2)
        for i, name in enumerate(self.target_names):
            head = Dense(
                head_units, activation="swish", name=f"head_{name}_hidden"
            )(shared)
            head = Dense(self.horizon, name=name)(head)
            outputs[name] = head

        model = Model(inputs=inputs, outputs=outputs, name="EcoWeatherFit_LSTM")

        # ── Per-target loss weights ──
        loss_weights_cfg = self.config.get("loss_weights", {})
        loss_weights = {
            name: float(loss_weights_cfg.get(name, 1.0))
            for name in self.target_names
        }
        losses = {name: "mse" for name in self.target_names}
        metrics = {name: ["mae"] for name in self.target_names}

        model.compile(
            optimizer=Adam(learning_rate=self.config["learning_rate"]),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics,
        )

        self.model = model
        logger.info(
            "LSTM model built: %d params, loss_weights=%s",
            model.count_params(), loss_weights,
        )
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
        """Train the LSTM on 3-D sequence arrays.

        ``y_train``/``y_val`` may be either:
          * a 3-D ndarray of shape (batch, horizon, n_targets) — the
            shape produced by ``create_lstm_sequences`` — in which case
            it is split into the named per-target dict internally, or
          * an already-named dict {target: (batch, horizon)} for
            advanced callers.
        """
        if self.model is None:
            raise RuntimeError("Call build() before train().")

        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        callbacks = [
            EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=self.config["patience"],
                min_delta=self.config.get("min_delta", 0.0),
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        y_train_dict = self._to_target_dict(y_train)
        y_val_dict = self._to_target_dict(y_val) if y_val is not None else None
        validation_data = (X_val, y_val_dict) if X_val is not None else None

        logger.info(
            "Training LSTM: %d samples, lookback=%d, horizon=%d …",
            len(X_train), self.lookback, self.horizon,
        )

        history = self.model.fit(
            X_train,
            y_train_dict,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
        )

        self.is_trained = True
        self.metrics = {
            "final_loss": float(history.history["loss"][-1]),
            "epochs_run": len(history.history["loss"]),
        }
        if "val_loss" in history.history:
            self.metrics["val_loss"] = float(history.history["val_loss"][-1])
        # Per-target final MAE (averaged across horizon).
        for name in self.target_names:
            mae_key = f"{name}_mae"
            if mae_key in history.history:
                self.metrics[f"final_{name}_mae"] = float(
                    history.history[mae_key][-1]
                )
            val_mae_key = f"val_{name}_mae"
            if val_mae_key in history.history:
                self.metrics[f"val_{name}_mae"] = float(
                    history.history[val_mae_key][-1]
                )

        # Average val_mae across all named heads — gives the dashboard a
        # single scalar to display next to "Val MAE" without having to
        # special-case multi-output models.
        val_maes = [
            self.metrics.get(f"val_{n}_mae") for n in self.target_names
        ]
        val_maes = [v for v in val_maes if v is not None]
        if val_maes:
            self.metrics["val_mae"] = float(np.mean(val_maes))

        logger.info("LSTM training complete: %s", self.metrics)
        return self

    # ──────────────────────────────────────────────
    # Internal helpers — multi-output marshalling
    # ──────────────────────────────────────────────
    def _to_target_dict(self, y) -> Dict:
        """Convert a (batch, horizon, n_targets) ndarray to a dict keyed
        by target name. Pass-through if it is already a dict."""
        if isinstance(y, dict):
            return y
        y_arr = np.asarray(y)
        if y_arr.ndim != 3 or y_arr.shape[-1] != self.n_targets:
            raise ValueError(
                f"Expected y with shape (batch, horizon, n_targets={self.n_targets}); "
                f"got {y_arr.shape}"
            )
        return {
            name: y_arr[..., i].astype(np.float32)
            for i, name in enumerate(self.target_names)
        }

    def _stack_predictions(self, pred) -> np.ndarray:
        """Stack a dict-of-arrays prediction back to (batch, horizon, n_targets).

        The Functional-API model returns a dict {name: (batch, horizon)}.
        Older single-output checkpoints just return an ndarray; we
        pass those through unchanged so reloads from prior model
        versions still work.
        """
        if isinstance(pred, dict):
            cols = [pred[name] for name in self.target_names]
            return np.stack(cols, axis=-1)
        return np.asarray(pred)

    # ──────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict from a 3-D array (batch, lookback, features).

        Returns shape (batch, horizon, n_targets). When the underlying
        Keras model has named per-target outputs (the current default),
        the dict-of-arrays it produces is stacked back into a single
        ndarray so callers can index by ``[..., target_idx]`` as before.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained.")
        raw = self.model.predict(X, verbose=0)
        return self._stack_predictions(raw)

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
                "target_names": list(self.target_names),
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

        try:
            self.model = load_model(model_path)
        except (TypeError, ValueError, KeyError) as exc:
            logger.warning(
                "Saved LSTM at %s is incompatible with the installed "
                "TensorFlow/Keras version (%s). Will retrain from scratch.",
                model_path, exc,
            )
            return False
        meta = joblib.load(meta_path)
        self.metrics = meta.get("metrics", {})
        self.n_features = meta.get("n_features", 0)
        self.n_targets = meta.get("n_targets", 0)
        self.target_names = list(
            meta.get("target_names", [])
            or [f"target_{i}" for i in range(self.n_targets)]
        )
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
