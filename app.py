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
    """Fetch, preprocess, and feature-engineer historical data.

    Optionally bridges the gap between the most recent Meteostat observation
    and today by querying the OWM One Call 3.0 timemachine endpoint for the
    intervening days. This requires the paid One Call by Call subscription
    and is gated on ``OWM_HISTORY_BRIDGE_ENABLED``. If the bridge call fails
    or returns no rows, the pipeline silently falls back to Meteostat-only
    history.
    """
    from data.acquisition import (
        fetch_meteostat_historical, fetch_owm_history_range,
        DataAcquisitionError,
    )
    from data.preprocessing import preprocess_pipeline
    from data.feature_engineering import engineer_features
    from config.settings import (
        OWM_HISTORY_BRIDGE_ENABLED,
        OWM_HISTORY_BRIDGE_MAX_DAYS,
        OPENWEATHERMAP_API_KEY,
    )

    raw = fetch_meteostat_historical(city=city, use_cache=True)

    if OWM_HISTORY_BRIDGE_ENABLED and OPENWEATHERMAP_API_KEY:
        meteostat_end = raw.index.max()
        today = pd.Timestamp.utcnow().tz_convert(None).normalize()
        if pd.notna(meteostat_end):
            gap_start = (meteostat_end + pd.Timedelta(days=1)).normalize()
            # Cap bridge length so a misconfigured account doesn't fan out
            # into thousands of API calls.
            gap_days = (today - gap_start).days + 1
            if 1 <= gap_days <= OWM_HISTORY_BRIDGE_MAX_DAYS:
                logger.info(
                    "OWM history bridge: requesting %d days for %s "
                    "(%s to %s)",
                    gap_days, city, gap_start.date(), today.date(),
                )
                try:
                    bridge = fetch_owm_history_range(
                        city=city,
                        start_date=gap_start,
                        end_date=today,
                        use_cache=True,
                    )
                    if not bridge.empty:
                        # Drop the `source` column before merging into the
                        # main frame so downstream preprocessing sees a
                        # consistent schema.
                        bridge_clean = bridge.drop(
                            columns=[c for c in ["source"] if c in bridge.columns]
                        )
                        # Align columns: keep the union of Meteostat + OWM
                        # columns, fill missing with NaN so the cleaning
                        # pipeline's plausibility filter still applies.
                        all_cols = list(
                            dict.fromkeys(list(raw.columns) + list(bridge_clean.columns))
                        )
                        raw = raw.reindex(columns=all_cols)
                        bridge_clean = bridge_clean.reindex(columns=all_cols)
                        raw = pd.concat([raw, bridge_clean]).sort_index()
                except DataAcquisitionError as exc:
                    logger.warning("OWM history bridge unavailable: %s", exc)
                except Exception as exc:
                    logger.warning("OWM history bridge failed: %s", exc)
            elif gap_days > OWM_HISTORY_BRIDGE_MAX_DAYS:
                logger.warning(
                    "OWM history gap (%d days) exceeds cap %d — bridge skipped.",
                    gap_days, OWM_HISTORY_BRIDGE_MAX_DAYS,
                )

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
        if lstm.build(
            n_features=split.X_train.shape[1],
            n_targets=split.y_train.shape[1],
            target_names=list(split.y_train.columns),
        ):
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

    lstm_training_metrics = dict(lstm.metrics) if lstm.metrics else {}

    artifacts = {
        "feature_scaler": split.feature_scaler,
        "target_scaler": split.target_scaler,
        "feature_cols": split.metadata["feature_cols"],
        "target_cols": split.metadata["target_cols"],
        "split_metadata": split.metadata,
        "featured_data": featured,
        "X_train": split.X_train,
        "lstm_training_metrics": lstm_training_metrics,
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
    """Generate 7-day LSTM forecast from the last 30 days of full historical data.

    Important: the lookback window is taken from the **full feature-engineered
    history** (which ends at the most recent available observation), NOT from
    the training split. Using X_train here would anchor the forecast to the
    end of training data (early 2024 with a 70/15/15 split), which would
    misalign with the RF and OWM forecasts that start from "today".
    """
    try:
        featured = artifacts["featured_data"]
        feature_cols = artifacts["feature_cols"]
        feature_scaler = artifacts["feature_scaler"]

        # Take the most recent `lookback` rows of engineered features
        recent = featured[feature_cols].tail(LSTM_LOOKBACK_WINDOW)
        if len(recent) < LSTM_LOOKBACK_WINDOW:
            logger.warning("Not enough data for LSTM lookback window.")
            return None

        # Scale with the training-fitted scaler (no leakage — scaler was
        # fit on train rows only inside splitting.py).
        recent_scaled = feature_scaler.transform(recent.values).astype(np.float32)

        last_date = featured.index.max()
        forecast = lstm.forecast_7day(
            recent_features_scaled=recent_scaled,
            target_scaler=artifacts["target_scaler"],
            target_cols=artifacts["target_cols"],
            last_date=last_date,
        )
        return forecast
    except Exception as exc:
        logger.error("LSTM forecast failed: %s", exc)
        return None


# ──────────────────────────────────────────────
# Forecast-history persistence helper
# ──────────────────────────────────────────────
def _persist_forecast_history(user_id, city, forecast_df, recommendation):
    """Serialise the day's recommendation + constraints into the auth.ForecastHistory row.

    Tolerates partial data — any missing field collapses to an empty string
    rather than aborting the call. The constraints object is a
    ``DailyConstraints`` dataclass; we ``asdict`` it (lists of primitives only,
    so the resulting JSON is small and stable).
    """
    import json
    from dataclasses import asdict, is_dataclass
    from auth import record_forecast

    forecast_summary = ""
    if forecast_df is not None and not forecast_df.empty:
        head = forecast_df.head(1).to_dict(orient="records")[0]
        parts = []
        for k in ("temp", "wind_speed", "humidity", "precipitation", "pop"):
            if k in head and head[k] is not None:
                v = head[k]
                parts.append(f"{k}={float(v):.1f}" if isinstance(v, (int, float)) else f"{k}={v}")
        forecast_summary = ", ".join(parts)

    rec_text = (
        recommendation.get("recommendation")
        or recommendation.get("text")
        or ""
    )

    constraints_obj = recommendation.get("constraints")
    constraints_json = None
    if is_dataclass(constraints_obj):
        try:
            constraints_json = json.dumps(asdict(constraints_obj), default=str)
        except (TypeError, ValueError):
            constraints_json = None

    record_forecast(
        user_id=user_id,
        city=city,
        forecast_summary=forecast_summary,
        recommendation_text=str(rec_text)[:4000],
        constraints_json=constraints_json,
    )


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
    from auth import (
        init_db,
        render_auth_gate,
        render_user_menu,
        render_wardrobe_manager,
        render_history_panel,
        record_forecast,
        update_preferences,
    )

    # One-off database initialisation. Idempotent (CREATE TABLE IF NOT EXISTS).
    init_db()

    render_header()

    # Auth gate: short-circuits the rest of the app when no valid session exists.
    # Also handles ?verify_token=<uuid> email-verification deep links.
    user_id = render_auth_gate()
    if user_id is None:
        return

    # User is authenticated — render the per-user sidebar block (account info,
    # logout, persisted preferences) and use those preferences as defaults
    # for the rest of the sidebar.
    user_prefs = render_user_menu(user_id) or {}
    user = render_sidebar(defaults=user_prefs)

    # Persist any sidebar changes back to the DB so they survive the next visit.
    try:
        update_preferences(
            user_id,
            default_city=user["city"],
            gender_presentation=user["gender"].title(),
            style_preference=user["style_preference"].title(),
            show_eco_tips=user["show_eco"],
            show_circular_loop=user["show_loop"],
        )
    except Exception as exc:
        logger.warning("Could not persist preferences for user %s: %s", user_id, exc)

    # Wardrobe + history live in the sidebar under expanders so they don't crowd
    # the main panel when collapsed.
    with st.sidebar:
        render_wardrobe_manager(user_id)
        render_history_panel(user_id)

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
        render_model_metrics(
            rf_metrics,
            lstm_metrics,
            lstm_training_metrics=artifacts.get("lstm_training_metrics"),
        )

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

            # Persist this recommendation to the user's forecast history so
            # they can review it on a future visit.
            try:
                _persist_forecast_history(
                    user_id=user_id,
                    city=city,
                    forecast_df=rec_forecast,
                    recommendation=recommendation,
                )
            except Exception as exc:
                logger.warning(
                    "Could not persist forecast history for user %s: %s", user_id, exc
                )

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
