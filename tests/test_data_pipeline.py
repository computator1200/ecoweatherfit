"""
EcoWeatherFit — Unit Tests for Data Pipeline
==============================================
Tests cover the full data pipeline: acquisition (synthetic mode),
preprocessing, feature engineering, and chronological splitting.

Each test is documented so that it serves double duty as a
**verification artefact** for academic assessment.

Run with:
    pytest tests/test_data_pipeline.py -v
"""

import logging
import numpy as np
import pandas as pd
import pytest

# Configure logging so test output is visible
logging.basicConfig(level=logging.INFO)

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def synthetic_raw_df() -> pd.DataFrame:
    """
    Create a minimal synthetic daily weather DataFrame that mimics
    the schema produced by ``data.acquisition``.

    Returns ~3 years of daily data for robust split testing.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", "2023-12-31", freq="D")
    n = len(dates)

    doy = dates.dayofyear.values.astype(float)
    temp_mean = 10.0 + 8.0 * np.sin(2 * np.pi * (doy - 80) / 365)

    df = pd.DataFrame({
        "temp":          np.round(temp_mean + rng.normal(0, 2.5, n), 1),
        "temp_min":      np.round(temp_mean - rng.uniform(2, 5, n), 1),
        "temp_max":      np.round(temp_mean + rng.uniform(2, 5, n), 1),
        "humidity":      np.clip(rng.normal(75, 12, n), 20, 100).round(1),
        "wind_speed":    np.clip(rng.exponential(12, n), 0, 100).round(1),
        "precipitation": np.clip(rng.exponential(2.5, n), 0, 80).round(1),
        "pressure":      rng.normal(1013, 10, n).round(1),
    }, index=dates)
    df.index.name = "time"
    return df


@pytest.fixture
def dirty_df(synthetic_raw_df) -> pd.DataFrame:
    """
    Introduce controlled impurities into the synthetic data:
    * 5 duplicate timestamps
    * 10 physically implausible temperature values
    * 20 random NaN insertions
    """
    df = synthetic_raw_df.copy()

    # Duplicate 5 rows
    dup_rows = df.iloc[:5].copy()
    df = pd.concat([df, dup_rows])

    # Implausible temperatures (outside -30 to 45 °C)
    df.iloc[100:105, df.columns.get_loc("temp")] = 60.0   # too hot
    df.iloc[200:205, df.columns.get_loc("temp")] = -50.0  # too cold

    # Random NaN
    rng = np.random.default_rng(99)
    for _ in range(20):
        row = rng.integers(0, len(df))
        col = rng.choice(["humidity", "wind_speed", "precipitation"])
        df.iloc[row, df.columns.get_loc(col)] = np.nan

    return df


# ──────────────────────────────────────────────
# 1. Acquisition Tests
# ──────────────────────────────────────────────
class TestAcquisition:
    """Tests for the data acquisition module (synthetic fallback)."""

    def test_synthetic_historical_returns_dataframe(self):
        """
        VERIFY: When no API key is set, ``fetch_meteostat_historical``
        falls back to synthetic data and returns a valid DataFrame.
        """
        from data.acquisition import fetch_meteostat_historical

        df = fetch_meteostat_historical(
            city="London",
            start="2022-01-01",
            end="2022-12-31",
            use_cache=False,
        )
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert df.index.name == "time"
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_synthetic_historical_contains_target_columns(self):
        """
        VERIFY: Synthetic data includes all four target variables.
        """
        from data.acquisition import fetch_meteostat_historical

        df = fetch_meteostat_historical(
            city="London", start="2022-01-01", end="2022-12-31",
            use_cache=False,
        )
        for col in ["temp", "humidity", "wind_speed", "precipitation"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_synthetic_historical_date_range(self):
        """
        VERIFY: The returned data spans the requested date range.
        """
        from data.acquisition import fetch_meteostat_historical

        df = fetch_meteostat_historical(
            city="Manchester", start="2023-01-01", end="2023-06-30",
            use_cache=False,
        )
        assert df.index.min().date() == pd.Timestamp("2023-01-01").date()
        assert df.index.max().date() == pd.Timestamp("2023-06-30").date()

    def test_unknown_city_raises_error(self):
        """
        VERIFY: Requesting an unsupported city raises ValueError.
        """
        from data.acquisition import fetch_meteostat_historical

        with pytest.raises(ValueError, match="Unknown city"):
            fetch_meteostat_historical(city="Atlantis", use_cache=False)

    def test_synthetic_current_weather(self):
        """
        VERIFY: Synthetic current weather returns a single‑row DataFrame.
        """
        from data.acquisition import fetch_owm_current

        df = fetch_owm_current(city="London", use_cache=False)
        assert len(df) == 1
        assert "temp" in df.columns

    def test_synthetic_forecast_7_days(self):
        """
        VERIFY: Synthetic forecast returns exactly 7 rows.
        """
        from data.acquisition import fetch_owm_forecast

        df = fetch_owm_forecast(city="London", use_cache=False)
        assert len(df) == 7
        assert "temp" in df.columns


# ──────────────────────────────────────────────
# 2. Preprocessing Tests
# ──────────────────────────────────────────────
class TestPreprocessing:
    """Tests for the data preprocessing pipeline."""

    def test_pipeline_runs_without_error(self, synthetic_raw_df):
        """
        VERIFY: The pipeline completes on clean synthetic data.
        """
        from data.preprocessing import preprocess_pipeline

        result = preprocess_pipeline(synthetic_raw_df)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_duplicates_removed(self, dirty_df):
        """
        VERIFY: Duplicate timestamps are removed during preprocessing.
        """
        from data.preprocessing import preprocess_pipeline

        result = preprocess_pipeline(dirty_df)
        assert not result.index.duplicated().any(), "Duplicates remain!"

    def test_implausible_values_replaced(self, dirty_df):
        """
        VERIFY: Temperatures outside [-30, 45] °C are replaced with NaN
        or interpolated — none survive as raw implausible values.
        """
        from data.preprocessing import preprocess_pipeline

        result = preprocess_pipeline(dirty_df)
        if "temp" in result.columns:
            assert result["temp"].max() <= 45.0, "Implausible high temp survived"
            assert result["temp"].min() >= -30.0, "Implausible low temp survived"

    def test_cleaning_report_attached(self, dirty_df):
        """
        VERIFY: A cleaning report dict is attached to df.attrs.
        """
        from data.preprocessing import preprocess_pipeline

        result = preprocess_pipeline(dirty_df)
        report = result.attrs.get("cleaning_report")
        assert report is not None
        assert "rows_in" in report
        assert "duplicates_removed" in report

    def test_no_gaps_in_date_index(self, synthetic_raw_df):
        """
        VERIFY: After temporal alignment, the DatetimeIndex has no gaps.
        """
        from data.preprocessing import preprocess_pipeline

        result = preprocess_pipeline(synthetic_raw_df)
        expected_dates = pd.date_range(
            result.index.min(), result.index.max(), freq="D"
        )
        assert len(result) == len(expected_dates), "Date gaps detected!"

    def test_empty_dataframe_raises(self):
        """
        VERIFY: Passing an empty DataFrame raises ValueError.
        """
        from data.preprocessing import preprocess_pipeline

        with pytest.raises(ValueError, match="empty"):
            preprocess_pipeline(pd.DataFrame())

    def test_output_types_float32(self, synthetic_raw_df):
        """
        VERIFY: Numeric columns are coerced to float32.
        """
        from data.preprocessing import preprocess_pipeline

        result = preprocess_pipeline(synthetic_raw_df)
        for col in result.select_dtypes(include=[np.number]).columns:
            assert result[col].dtype == np.float32, (
                f"Column '{col}' is {result[col].dtype}, expected float32"
            )


# ──────────────────────────────────────────────
# 3. Feature Engineering Tests
# ──────────────────────────────────────────────
class TestFeatureEngineering:
    """Tests for the feature engineering module."""

    def test_lag_features_created(self, synthetic_raw_df):
        """
        VERIFY: Lag features are created for target variables.
        """
        from data.feature_engineering import engineer_features

        result = engineer_features(synthetic_raw_df)
        assert "temp_lag_1" in result.columns
        assert "temp_lag_7" in result.columns
        assert "wind_speed_lag_1" in result.columns

    def test_rolling_stats_created(self, synthetic_raw_df):
        """
        VERIFY: Rolling mean and std features are present.
        """
        from data.feature_engineering import engineer_features

        result = engineer_features(synthetic_raw_df)
        assert "temp_rmean_7" in result.columns
        assert "temp_rstd_14" in result.columns

    def test_temporal_encodings_bounded(self, synthetic_raw_df):
        """
        VERIFY: Sine/cosine encodings are in [-1, 1].
        """
        from data.feature_engineering import engineer_features

        result = engineer_features(synthetic_raw_df)
        for col in ["day_of_year_sin", "day_of_year_cos", "month_sin", "month_cos"]:
            assert result[col].min() >= -1.0, f"{col} below -1"
            assert result[col].max() <=  1.0, f"{col} above 1"

    def test_is_rainy_flag(self, synthetic_raw_df):
        """
        VERIFY: Binary is_rainy flag is 0 or 1.
        """
        from data.feature_engineering import engineer_features

        result = engineer_features(synthetic_raw_df)
        assert "is_rainy" in result.columns
        assert set(result["is_rainy"].dropna().unique()).issubset({0.0, 1.0})

    def test_feature_count_increases(self, synthetic_raw_df):
        """
        VERIFY: Feature engineering strictly increases column count.
        """
        from data.feature_engineering import engineer_features

        n_before = len(synthetic_raw_df.columns)
        result = engineer_features(synthetic_raw_df)
        assert len(result.columns) > n_before

    def test_no_future_leakage_in_lags(self, synthetic_raw_df):
        """
        VERIFY: Lag‑1 of temperature at row i equals the raw
        temperature at row i-1 (strict backward‑looking).
        """
        from data.feature_engineering import engineer_features

        result = engineer_features(synthetic_raw_df)
        # Check row index 10 (well past warm‑up)
        lag1_val = result["temp_lag_1"].iloc[10]
        actual_prev = synthetic_raw_df["temp"].iloc[9]
        assert lag1_val == actual_prev, (
            f"Lag‑1 leakage! lag={lag1_val}, expected={actual_prev}"
        )


# ──────────────────────────────────────────────
# 4. Splitting & Scaling Tests
# ──────────────────────────────────────────────
class TestSplitting:
    """Tests for time‑based splitting and leakage‑safe scaling."""

    def _prepare(self, synthetic_raw_df):
        """Helper: preprocess + feature‑engineer the fixture."""
        from data.preprocessing import preprocess_pipeline
        from data.feature_engineering import engineer_features

        cleaned = preprocess_pipeline(synthetic_raw_df, verbose=False)
        featured = engineer_features(cleaned, drop_na_rows=True)
        return featured

    def test_split_returns_correct_type(self, synthetic_raw_df):
        """
        VERIFY: ``time_based_split`` returns a ``TimeSeriesSplit``.
        """
        from data.splitting import time_based_split, TimeSeriesSplit

        df = self._prepare(synthetic_raw_df)
        split = time_based_split(df, scale=False)
        assert isinstance(split, TimeSeriesSplit)

    def test_chronological_order(self, synthetic_raw_df):
        """
        VERIFY: Train end < Val start < Val end < Test start.
        This is the single most critical academic requirement.
        """
        from data.splitting import time_based_split

        df = self._prepare(synthetic_raw_df)
        split = time_based_split(df, scale=False)

        assert split.X_train.index.max() < split.X_val.index.min(), \
            "LEAKAGE: Train overlaps validation!"
        assert split.X_val.index.max() < split.X_test.index.min(), \
            "LEAKAGE: Validation overlaps test!"

    def test_no_data_leakage_in_scaling(self, synthetic_raw_df):
        """
        VERIFY: The feature scaler's mean is computed from training
        data only — it should not equal the global mean.
        """
        from data.splitting import time_based_split

        df = self._prepare(synthetic_raw_df)
        split = time_based_split(df, scale=True, scaler_type="standard")

        # The scaler was fit on train; its mean_ should differ from
        # the full‑dataset mean (with high probability).
        global_mean = df[split.metadata["feature_cols"]].mean().values
        scaler_mean = split.feature_scaler.mean_

        # They should be close but NOT identical (different subsets).
        # With 3 years of data and 70/15/15 split, this is reliable.
        assert not np.allclose(scaler_mean, global_mean, atol=0.01), \
            "Scaler mean matches global mean — possible leakage!"

    def test_split_sizes_approximate(self, synthetic_raw_df):
        """
        VERIFY: Split proportions are approximately 70/15/15.
        """
        from data.splitting import time_based_split

        df = self._prepare(synthetic_raw_df)
        split = time_based_split(df, scale=False)

        total = split.metadata["n_train"] + split.metadata["n_val"] + split.metadata["n_test"]
        train_pct = split.metadata["n_train"] / total
        assert 0.65 <= train_pct <= 0.75, f"Train ratio {train_pct:.2f} out of range"

    def test_scaled_train_mean_near_zero(self, synthetic_raw_df):
        """
        VERIFY: After standard scaling, the training set has
        approximately zero mean (by construction).
        """
        from data.splitting import time_based_split

        df = self._prepare(synthetic_raw_df)
        split = time_based_split(df, scale=True, scaler_type="standard")

        train_means = split.X_train.mean()
        assert (train_means.abs() < 0.01).all(), (
            f"Scaled train means not near zero: {train_means}"
        )

    def test_metadata_contains_boundaries(self, synthetic_raw_df):
        """
        VERIFY: Split metadata includes all expected fields.
        """
        from data.splitting import time_based_split

        df = self._prepare(synthetic_raw_df)
        split = time_based_split(df, scale=False)

        required_keys = [
            "train_start", "train_end", "val_start", "val_end",
            "test_start", "test_end", "n_train", "n_val", "n_test",
        ]
        for key in required_keys:
            assert key in split.metadata, f"Missing metadata key: {key}"


# ──────────────────────────────────────────────
# 5. LSTM Sequence Builder Tests
# ──────────────────────────────────────────────
class TestLSTMSequences:
    """Tests for the sliding‑window LSTM sequence builder."""

    def test_sequence_shapes(self, synthetic_raw_df):
        """
        VERIFY: Output arrays have correct 3‑D / 3‑D shapes.
        """
        from data.preprocessing import preprocess_pipeline
        from data.feature_engineering import engineer_features
        from data.splitting import time_based_split, create_lstm_sequences

        cleaned = preprocess_pipeline(synthetic_raw_df, verbose=False)
        featured = engineer_features(cleaned, drop_na_rows=True)
        split = time_based_split(featured, scale=True)

        X_seq, y_seq = create_lstm_sequences(
            split.X_train, split.y_train,
            lookback=30, horizon=7,
        )

        assert X_seq.ndim == 3
        assert y_seq.ndim == 3
        assert X_seq.shape[1] == 30  # lookback
        assert y_seq.shape[1] == 7   # horizon
        assert X_seq.shape[2] == split.X_train.shape[1]  # n_features

    def test_too_short_data_raises(self):
        """
        VERIFY: Data shorter than lookback + horizon raises ValueError.
        """
        from data.splitting import create_lstm_sequences

        X = pd.DataFrame(np.random.randn(10, 5))
        y = pd.DataFrame(np.random.randn(10, 2))

        with pytest.raises(ValueError, match="Data length"):
            create_lstm_sequences(X, y, lookback=30, horizon=7)


# ──────────────────────────────────────────────
# Run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
