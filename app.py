"""
EcoWeatherFit — Main Streamlit Application
============================================
Launch with:  streamlit run app.py

Orchestrates the full pipeline:
  Data Acquisition → Preprocessing → Feature Engineering →
  Model Training/Loading → Forecasting → Hybrid Recommendations →
  Streamlit Dashboard Rendering

The application supports both live API data and synthetic fallbacks,
making it fully functional for demonstration without API keys.
"""

import sys
import logging
from pathlib import Path

# Ensure project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import (
    FORECAST_HORIZON_DAYS,
    LOG_FORMAT,
    LOG_LEVEL,
    LSTM_LOOKBACK_WINDOW,
    STREAMLIT_LAYOUT,
    STREAMLIT_PAGE_ICON,
    STREAMLIT_PAGE_TITLE,
    TARGET_VARIABLES,
)

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("EcoWeatherFit")

# ──────────────────────────────────────────────
# Streamlit Page Config (must be first st call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon=STREAMLIT_PAGE_ICON,
    layout=STREAMLIT_LAYOUT,
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Cached Data & Model Functions
# ──────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Fetching historical data…")
def load_historical_data(city: str) -> pd.DataFrame:
    """Fetch, preprocess, and feature-engineer historical data."""
    from data.acquisition import fetch_meteostat_historical
    from data.preprocessing import preprocess_pipeline
    from data.feature_engineering import engineer_features

    raw = fetch_meteostat_historical(city=city, use_cache=True)
    cleaned = preprocess_pipeline(raw, verbose=False)
    featured = engineer_features(cleaned, drop_na_rows=True)
    return featured


@st.cache_data(ttl=1800, show_spinner="Fetching current weather…")
def load_current_weather(city: str) -> pd.DataFrame:
    from data.acquisition import fetch_owm_current
    return fetch_owm_current(city=city, use_cache=True)


@st.cache_data(ttl=1800, show_spinner="Fetching OWM forecast…")
def load_owm_forecast(city: str) -> pd.DataFrame:
    from data.acquisition import fetch_owm_forecast
    return fetch_owm_forecast(city=city, use_cache=True)


@st.cache_resource(show_spinner="Training models… This may take a moment on first run.")
def train_models(city: str):
    """
    Train both RF and LSTM models on historical data.
    Returns (rf_model, lstm_model, split_data, artifacts).
    """
    from data.splitting import time_based_split, create_lstm_sequences
    from models.random_forest import WeatherRandomForest
    from models.lstm import WeatherLSTM

    featured = load_historical_data(city)
    split = time_based_split(featured, scale=True)

    # ── Random Forest ──
    rf = WeatherRandomForest()
    if not rf.load():
        rf.train(split.X_train, split.y_train, split.X_val, split.y_val)
        rf.save()
    rf_test_metrics = rf.evaluate(split.X_test, split.y_test)

    # ── LSTM ──
    lstm = WeatherLSTM()
    lstm_test_metrics = {}

    if not lstm.load():
        if lstm.build(n_features=split.X_train.shape[1], n_targets=split.y_train.shape[1]):
            X_train_seq, y_train_seq = create_lstm_sequences(split.X_train, split.y_train)
            X_val_seq, y_val_seq = create_lstm_sequences(split.X_val, split.y_val)
            lstm.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
            lstm.save()

    if lstm.is_trained:
        X_test_seq, y_test_seq = create_lstm_sequences(split.X_test, split.y_test)
        if len(X_test_seq) > 0:
            lstm_test_metrics = lstm.evaluate(
                X_test_seq, y_test_seq,
                target_cols=list(split.y_test.columns),
            )

    artifacts = {
        "feature_scaler": split.feature_scaler,
        "target_scaler": split.target_scaler,
        "feature_cols": split.metadata["feature_cols"],
        "target_cols": split.metadata["target_cols"],
        "split_metadata": split.metadata,
        "featured_data": featured,
        "X_train": split.X_train,
    }

    return rf, lstm, rf_test_metrics, lstm_test_metrics, artifacts


# ──────────────────────────────────────────────
# Forecast Generation
# ──────────────────────────────────────────────
def generate_rf_forecast(rf, featured_data, artifacts):
    """Generate 7-day RF forecast using recursive prediction."""
    try:
        # Use the last 60 days of raw featured data
        recent = featured_data.tail(60).copy()
        forecast = rf.forecast_7day(
            recent_raw=recent,
            feature_scaler=artifacts["feature_scaler"],
            target_scaler=artifacts["target_scaler"],
            feature_cols=artifacts["feature_cols"],
            target_cols=artifacts["target_cols"],
        )
        return forecast
    except Exception as exc:
        logger.error("RF forecast failed: %s", exc)
        return None


def generate_lstm_forecast(lstm, artifacts):
    """Generate 7-day LSTM forecast from last 30 days."""
    try:
        X_train = artifacts["X_train"]
        last_window = X_train.iloc[-LSTM_LOOKBACK_WINDOW:].values.astype(np.float32)

        if len(last_window) < LSTM_LOOKBACK_WINDOW:
            logger.warning("Not enough data for LSTM lookback window.")
            return None

        last_date = X_train.index.max()
        forecast = lstm.forecast_7day(
            recent_features_scaled=last_window,
            target_scaler=artifacts["target_scaler"],
            target_cols=artifacts["target_cols"],
            last_date=last_date,
        )
        return forecast
    except Exception as exc:
        logger.error("LSTM forecast failed: %s", exc)
        return None


# ──────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────
def main():
    from ui.dashboard import (
        render_header,
        render_sidebar,
        render_current_weather,
        render_forecast_comparison,
        render_forecast_table,
        render_model_metrics,
        render_recommendation,
        render_week_recommendation,
        render_sustainability,
        render_training_status,
    )
    from recommendations.gemini_integration import GeminiAdvisor
    from recommendations.sustainability import SustainabilityEngine

    render_header()
    user = render_sidebar()

    city = user["city"]
    user_profile = {
        "gender": user["gender"],
        "style_preference": user["style_preference"],
    }

    # Always show current weather
    try:
        current = load_current_weather(city)
        render_current_weather(current, city)
    except Exception as exc:
        st.error(f"Could not fetch current weather: {exc}")
        current = None

    # Main forecast flow — triggered by button or session state
    if user["forecast_btn"]:
        st.session_state["run_forecast"] = True

    if st.session_state.get("run_forecast"):
        st.markdown("---")

        # ── Step 1: Train / Load Models ──
        with st.spinner("Preparing models…"):
            try:
                rf, lstm, rf_metrics, lstm_metrics, artifacts = train_models(city)
            except Exception as exc:
                st.error(f"Model training failed: {exc}")
                logger.exception("Model training error")
                return

        render_training_status(rf.is_trained, lstm.is_trained)

        # ── Step 2: Generate Forecasts ──
        st.markdown("---")

        owm_forecast = None
        rf_forecast = None
        lstm_forecast = None

        try:
            owm_forecast = load_owm_forecast(city)
        except Exception as exc:
            st.warning(f"OWM forecast unavailable: {exc}")

        if rf.is_trained:
            rf_forecast = generate_rf_forecast(rf, artifacts["featured_data"], artifacts)

        if lstm.is_trained:
            lstm_forecast = generate_lstm_forecast(lstm, artifacts)

        # ── Step 3: Display Forecasts ──
        render_forecast_comparison(owm_forecast, rf_forecast, lstm_forecast)

        # Show raw data tables
        if owm_forecast is not None:
            render_forecast_table(owm_forecast, "OpenWeatherMap Forecast")
        if rf_forecast is not None:
            render_forecast_table(rf_forecast, "Random Forest Forecast")
        if lstm_forecast is not None:
            render_forecast_table(lstm_forecast, "LSTM Forecast")

        # ── Step 4: Model Metrics ──
        render_model_metrics(rf_metrics, lstm_metrics)

        # ── Step 5: Hybrid Recommendations ──
        st.markdown("---")

        # Use the best available forecast for recommendations
        rec_forecast = owm_forecast
        if rec_forecast is None:
            rec_forecast = rf_forecast
        if rec_forecast is None:
            rec_forecast = lstm_forecast

        if rec_forecast is not None:
            advisor = GeminiAdvisor()

            # Daily recommendation (tomorrow)
            with st.spinner("Generating personalised outfit advice…"):
                recommendation = advisor.generate_recommendation(
                    forecast_df=rec_forecast,
                    user_profile=user_profile,
                    day_index=0,
                )
            render_recommendation(recommendation)

            # Week summary
            with st.spinner("Generating weekly wardrobe plan…"):
                week_rec = advisor.generate_week_summary(
                    forecast_df=rec_forecast,
                    user_profile=user_profile,
                )
            render_week_recommendation(week_rec)

            # ── Step 6: Sustainability Section ──
            st.markdown("---")
            constraints = recommendation.get("constraints")
            if constraints and user["show_eco"]:
                sustainability = SustainabilityEngine()
                advice = sustainability.generate_advice(
                    temperature_category=constraints.temperature_category,
                    day_index=0,
                )
                render_sustainability(advice, show_loop=user["show_loop"])
        else:
            st.warning("No forecast data available to generate recommendations.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #999; font-size: 0.8rem; padding: 1rem;">
            <p>EcoWeatherFit — A Sustainable Fashion & Weather Platform</p>
            <p>Built with Streamlit • Powered by Random Forest & LSTM models • Enhanced by Gemini AI</p>
            <p>♻️ <em>Reuse > Layer > Repair > Buy Sustainable</em></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
