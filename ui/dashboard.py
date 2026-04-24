"""
EcoWeatherFit — Streamlit Dashboard Components
================================================
All UI rendering functions for the EcoWeatherFit application.
Each function is responsible for a discrete section of the dashboard,
keeping the main ``app.py`` clean and focused on orchestration.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from recommendations.heuristic_engine import WeekConstraints
from recommendations.sustainability import SustainabilityAdvice, SustainabilityEngine

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Page Config & Header
# ──────────────────────────────────────────────
def render_header():
    """Render the main page header and description."""
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0;">
            <h1>🌿 EcoWeatherFit</h1>
            <p style="font-size: 1.1rem; color: #666;">
                Sustainable Weather Forecasting & Clothing Recommendations
            </p>
            <p style="font-size: 0.9rem; color: #888;">
                Combating fast fashion through intelligent wardrobe advice •
                Reuse > Layer > Repair > Buy Sustainable
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
def render_sidebar() -> Dict:
    """Render the sidebar and return user selections."""
    from config.settings import DEFAULT_LOCATIONS

    st.sidebar.markdown("## 🌍 Location & Profile")

    city = st.sidebar.selectbox(
        "Select City",
        options=list(DEFAULT_LOCATIONS.keys()),
        index=0,
        help="Choose your UK city for local weather forecasting.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Your Profile")

    gender = st.sidebar.selectbox(
        "Gender Presentation",
        ["Unisex", "Feminine", "Masculine"],
        index=0,
    )

    style = st.sidebar.selectbox(
        "Style Preference",
        ["Casual", "Smart Casual", "Formal", "Sporty", "Bohemian"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ♻️ Sustainability")

    show_eco = st.sidebar.checkbox("Show eco-facts & tips", value=True)
    show_loop = st.sidebar.checkbox("Show Circular Fashion Loop", value=True)

    st.sidebar.markdown("---")

    forecast_btn = st.sidebar.button(
        "🔍 Get Forecast & Advice",
        type="primary",
        use_container_width=True,
    )

    return {
        "city": city,
        "gender": gender.lower(),
        "style_preference": style.lower(),
        "show_eco": show_eco,
        "show_loop": show_loop,
        "forecast_btn": forecast_btn,
    }


# ──────────────────────────────────────────────
# Current Weather Card
# ──────────────────────────────────────────────
def render_current_weather(current_df: pd.DataFrame, city: str):
    """Render the current weather conditions card."""
    st.markdown("## 🌡️ Current Conditions")

    if current_df is None or current_df.empty:
        st.warning("Current weather data unavailable.")
        return

    row = current_df.iloc[0]
    temp = row.get("temp", "N/A")
    feels = row.get("feels_like", temp)
    humidity = row.get("humidity", "N/A")
    wind = row.get("wind_speed", "N/A")
    precip = row.get("precipitation", 0)
    desc = row.get("weather_desc", row.get("weather_main", ""))

    cols = st.columns(5)
    with cols[0]:
        st.metric("🌡️ Temperature", f"{temp:.1f}°C", f"Feels like {feels:.1f}°C")
    with cols[1]:
        st.metric("💧 Humidity", f"{humidity:.0f}%")
    with cols[2]:
        st.metric("💨 Wind", f"{wind:.1f} km/h")
    with cols[3]:
        st.metric("🌧️ Precipitation", f"{precip:.1f} mm")
    with cols[4]:
        st.metric("📍 Location", city, desc)


# ──────────────────────────────────────────────
# 7-Day Forecast Chart
# ──────────────────────────────────────────────
def render_forecast_comparison(
    owm_forecast: pd.DataFrame,
    rf_forecast: Optional[pd.DataFrame],
    lstm_forecast: Optional[pd.DataFrame],
):
    """Render an interactive comparison chart of the three forecast sources."""
    st.markdown("## 📊 7-Day Forecast Comparison")

    if owm_forecast is None or owm_forecast.empty:
        st.warning("Forecast data unavailable.")
        return

    tabs = st.tabs(["Temperature", "Precipitation", "Humidity", "Wind Speed"])

    variables = [
        ("temp", "Temperature (°C)", tabs[0]),
        ("precipitation", "Precipitation (mm)", tabs[1]),
        ("humidity", "Humidity (%)", tabs[2]),
        ("wind_speed", "Wind Speed (km/h)", tabs[3]),
    ]

    for var, ylabel, tab in variables:
        with tab:
            fig = go.Figure()

            # OWM (API) forecast
            if var in owm_forecast.columns:
                dates = [d.strftime("%a %d %b") if hasattr(d, "strftime") else str(d)
                         for d in owm_forecast.index]
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=owm_forecast[var],
                    name="OpenWeatherMap API",
                    mode="lines+markers",
                    line=dict(color="#2196F3", width=2),
                    marker=dict(size=8),
                ))

            # RF forecast
            if rf_forecast is not None and var in rf_forecast.columns:
                dates_rf = [d.strftime("%a %d %b") if hasattr(d, "strftime") else str(d)
                            for d in rf_forecast.index]
                fig.add_trace(go.Scatter(
                    x=dates_rf,
                    y=rf_forecast[var],
                    name="Random Forest",
                    mode="lines+markers",
                    line=dict(color="#FF9800", width=2, dash="dash"),
                    marker=dict(size=8, symbol="diamond"),
                ))

            # LSTM forecast
            if lstm_forecast is not None and var in lstm_forecast.columns:
                dates_lstm = [d.strftime("%a %d %b") if hasattr(d, "strftime") else str(d)
                              for d in lstm_forecast.index]
                fig.add_trace(go.Scatter(
                    x=dates_lstm,
                    y=lstm_forecast[var],
                    name="LSTM Neural Network",
                    mode="lines+markers",
                    line=dict(color="#4CAF50", width=2, dash="dot"),
                    marker=dict(size=8, symbol="square"),
                ))

            fig.update_layout(
                yaxis_title=ylabel,
                xaxis_title="",
                template="plotly_white",
                height=350,
                margin=dict(l=60, r=20, t=30, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                ),
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# Model Performance Metrics
# ──────────────────────────────────────────────
def render_model_metrics(
    rf_metrics: Dict,
    lstm_metrics: Dict,
    lstm_training_metrics: Optional[Dict] = None,
):
    """Display model performance comparison table.

    Parameters
    ----------
    rf_metrics : per-target {mae, rmse, r2} dict for Random Forest
    lstm_metrics : per-target {mae, rmse, r2} dict for LSTM (test set)
    lstm_training_metrics : optional {epochs_run, val_loss, val_mae, ...} from training
    """
    st.markdown("## 📈 Model Performance (Test Set)")

    if not rf_metrics and not lstm_metrics:
        st.info("Train models to see performance metrics.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌲 Random Forest")
        if rf_metrics:
            rows = []
            for var, m in rf_metrics.items():
                rows.append({
                    "Variable": var,
                    "MAE": f"{m['mae']:.3f}",
                    "RMSE": f"{m['rmse']:.3f}",
                    "R²": f"{m['r2']:.3f}",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("RF model not yet trained.")

    with col2:
        st.markdown("### 🧠 LSTM Neural Network")
        if lstm_metrics:
            rows = []
            for var, m in lstm_metrics.items():
                rows.append({
                    "Variable": var,
                    "MAE": f"{m['mae']:.3f}",
                    "RMSE": f"{m['rmse']:.3f}",
                    "R²": f"{m['r2']:.3f}",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("LSTM test metrics unavailable (test sequences may be too short).")

        if lstm_training_metrics:
            sub1, sub2, sub3 = st.columns(3)
            with sub1:
                st.metric("Epochs Trained", lstm_training_metrics.get("epochs_run", "N/A"))
            with sub2:
                val_loss = lstm_training_metrics.get("val_loss")
                st.metric(
                    "Val Loss (MSE)",
                    f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else "N/A",
                )
            with sub3:
                val_mae = lstm_training_metrics.get("val_mae")
                st.metric(
                    "Val MAE",
                    f"{val_mae:.4f}" if isinstance(val_mae, (int, float)) else "N/A",
                )


# ──────────────────────────────────────────────
# Outfit Recommendation
# ──────────────────────────────────────────────
def render_recommendation(recommendation: Dict):
    """Display the hybrid recommendation with guardrail indicators."""
    st.markdown("## 👗 Your Personalised Outfit Recommendation")

    if not recommendation:
        st.info("Click 'Get Forecast & Advice' to generate recommendations.")
        return

    # Source badge
    source = recommendation.get("source", "unknown")
    if source == "hybrid":
        st.success("🤖 **Powered by Gemini AI + Weather Safety Guardrails**")
    else:
        st.info("📋 **Powered by Weather-Based Heuristic Rules**")

    # Guardrail indicator
    if recommendation.get("was_validated"):
        st.warning(
            f"⚠️ **Safety guardrails activated**: {len(recommendation['missing_items'])} "
            f"mandatory items were added to ensure weather-appropriate advice."
        )

    # Main recommendation text
    st.markdown(recommendation.get("recommendation", "No recommendation available."))

    # Constraints summary (collapsible)
    constraints = recommendation.get("constraints")
    if constraints:
        with st.expander("🔍 View Weather Constraints (Heuristic Analysis)", expanded=True):
            summary = (
                f"**Date:** {constraints.date}  •  "
                f"**Category:** {constraints.temperature_category.replace('_', ' ').title()}  •  "
                f"**Temp:** {constraints.temp:.1f}°C  •  "
                f"**Rain Prob:** {constraints.rain_probability * 100:.0f}%  •  "
                f"**Wind:** {constraints.wind_speed:.1f} km/h"
            )
            st.markdown(summary)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**✅ Mandatory Items**")
                if constraints.mandatory_items:
                    for item in constraints.mandatory_items:
                        st.markdown(f"- {item}")
                else:
                    st.markdown("_No mandatory items (mild conditions)_")
            with c2:
                st.markdown("**❌ Avoid**")
                if constraints.avoid_items:
                    for item in constraints.avoid_items:
                        st.markdown(f"- {item}")
                else:
                    st.markdown("_No restrictions_")
            with c3:
                st.markdown("**⚠️ Warnings**")
                if constraints.warnings:
                    for w in constraints.warnings:
                        st.markdown(f"- {w}")
                else:
                    st.markdown("_No weather warnings_")

            if constraints.layering_advice:
                st.markdown("**🧥 Layering Guide**")
                for layer in constraints.layering_advice:
                    st.markdown(f"- {layer}")


# ──────────────────────────────────────────────
# Week Summary
# ──────────────────────────────────────────────
def render_week_recommendation(week_rec: Dict):
    """Display the 7-day wardrobe plan."""
    if not week_rec:
        return

    text = (week_rec.get("recommendation") or "").strip()
    source = week_rec.get("source", "unknown")

    with st.expander("📅 View Full 7-Day Wardrobe Plan", expanded=True):
        if source == "hybrid":
            st.caption("🤖 Generated by Gemini AI + heuristic guardrails")
        else:
            st.caption("📋 Generated from weather-based heuristic rules")

        if text:
            st.markdown(text)
        else:
            st.warning(
                "Weekly plan could not be generated. Check the daily recommendation above "
                "for guidance, or review the forecast table below."
            )

        # Always surface the structured per-day breakdown — it is never empty.
        week = week_rec.get("week_constraints")
        if week and getattr(week, "daily", None):
            st.markdown("**Day-by-day essentials:**")
            for d in week.daily:
                emoji = _week_emoji(d.temperature_category, d.precipitation)
                items = ", ".join(d.mandatory_items[:3]) if d.mandatory_items else "flexible outfit day"
                st.markdown(
                    f"{emoji} **{d.date}** — {d.temp:.0f}°C, "
                    f"{d.temperature_category.replace('_', ' ')} — {items}"
                )


def _week_emoji(category: str, precip: float) -> str:
    if precip > 5:
        return "🌧️"
    if precip > 1:
        return "🌦️"
    mapping = {
        "freezing": "🥶",
        "very_cold": "❄️",
        "cold": "🌬️",
        "mild": "🌤️",
        "warm": "☀️",
        "hot": "🔥",
    }
    return mapping.get(category, "🌤️")


# ──────────────────────────────────────────────
# Sustainability Section
# ──────────────────────────────────────────────
def render_sustainability(advice: SustainabilityAdvice, show_loop: bool = True):
    """Render the sustainability advice section."""
    st.markdown("## ♻️ Sustainability Corner")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔄 Shop Your Wardrobe")
        st.markdown(advice.wardrobe_tip)

        st.markdown("### 🧥 Layering Guide")
        st.markdown(advice.layering_tip)

        if advice.swap_suggestion:
            st.markdown("### 🤝 Swap & Share")
            st.markdown(advice.swap_suggestion)

    with col2:
        st.markdown("### 🧹 Garment Care Tip")
        st.info(advice.care_tip)

        st.markdown("### 🌍 Did You Know?")
        st.warning(advice.eco_fact)

    if show_loop:
        render_circular_loop()


def render_circular_loop():
    """Display the Circular Fashion Loop infographic."""
    st.markdown("### The Circular Fashion Loop")

    engine = SustainabilityEngine()
    loop = engine.get_circular_loop_display()

    cols = st.columns(len(loop))
    for i, step in enumerate(loop):
        with cols[i]:
            colour = "#4CAF50" if step["is_priority"] else "#666"
            st.markdown(
                f"""
                <div style="text-align: center; padding: 0.5rem;
                     border: 2px solid {colour}; border-radius: 10px;
                     background: {'#E8F5E9' if step['is_priority'] else '#f9f9f9'};">
                    <div style="font-size: 1.8rem;">{step['icon']}</div>
                    <div style="font-weight: bold; font-size: 0.85rem;">{step['name']}</div>
                    <div style="font-size: 0.7rem; color: #666; margin-top: 4px;">
                        {step['description'][:60]}...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "Priority flows left to right: always start by reusing what you own. "
        "Buying new is the absolute last resort."
    )


# ──────────────────────────────────────────────
# Forecast Data Table
# ──────────────────────────────────────────────
def render_forecast_table(forecast_df: pd.DataFrame, label: str = "Forecast"):
    """Show a formatted data table of the forecast."""
    with st.expander(f"📋 {label} — Raw Data Table"):
        display_df = forecast_df.copy()
        # Format numeric columns
        for col in display_df.select_dtypes(include=[np.number]).columns:
            display_df[col] = display_df[col].round(1)
        # Format index
        if hasattr(display_df.index, "strftime"):
            display_df.index = display_df.index.strftime("%a %d %b %Y")
        st.dataframe(display_df, use_container_width=True)


# ──────────────────────────────────────────────
# Training Progress
# ──────────────────────────────────────────────
def render_training_status(rf_trained: bool, lstm_trained: bool):
    """Show model training status indicators."""
    c1, c2 = st.columns(2)
    with c1:
        if rf_trained:
            st.success("🌲 Random Forest — Trained ✓")
        else:
            st.warning("🌲 Random Forest — Not yet trained")
    with c2:
        if lstm_trained:
            st.success("🧠 LSTM — Trained ✓")
        else:
            st.warning("🧠 LSTM — Not yet trained")
