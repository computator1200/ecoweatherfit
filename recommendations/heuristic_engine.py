"""
EcoWeatherFit — Heuristic Rule Engine (Deterministic Guardrails)
=================================================================
This module implements the **deterministic safety layer** that
constrains all clothing recommendations.  No matter what the GenAI
model suggests, these rules act as hard guardrails to prevent
dangerous or nonsensical advice (e.g. suggesting shorts in sub-zero
temperatures, or omitting waterproofs in a storm).

The engine analyses each day of the forecast and produces a
structured ``DailyConstraints`` object containing:
* **mandatory_items** — clothing that *must* be recommended
* **avoid_items** — clothing that *must not* be recommended
* **warnings** — weather alerts to display prominently
* **temperature_category** — semantic label (freezing/cold/mild/warm/hot)
* **layering_advice** — circular-fashion layering suggestions

These constraints are fed into the Gemini prompt *and* used as a
post-generation validation filter (see ``gemini_integration.py``).

Academic Note
-------------
This hybrid approach (GenAI + deterministic rules) is explicitly
required by the project specification to prevent hallucinations
and ensure factual grounding in the weather data.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import HEURISTIC_THRESHOLDS

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────
@dataclass
class DailyConstraints:
    """Deterministic clothing constraints for a single forecast day."""
    date: str
    temp: float
    humidity: float
    wind_speed: float
    precipitation: float
    rain_probability: float = 0.0
    temperature_category: str = "mild"
    mandatory_items: List[str] = field(default_factory=list)
    avoid_items: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    layering_advice: List[str] = field(default_factory=list)


@dataclass
class WeekConstraints:
    """Aggregated constraints for a 7-day forecast."""
    daily: List[DailyConstraints] = field(default_factory=list)
    summary_warnings: List[str] = field(default_factory=list)
    overall_category: str = "mixed"


# ──────────────────────────────────────────────
# Main Engine
# ──────────────────────────────────────────────
class HeuristicEngine:
    """
    Analyse a weather forecast and produce deterministic clothing
    constraints based on configurable thresholds.
    """

    def __init__(self, thresholds: Optional[Dict] = None):
        self.thresholds = thresholds or HEURISTIC_THRESHOLDS

    def analyse_forecast(self, forecast_df: pd.DataFrame) -> WeekConstraints:
        """
        Analyse a 7-day forecast DataFrame and return structured constraints.

        Parameters
        ----------
        forecast_df : pd.DataFrame
            Must contain columns: temp, humidity, wind_speed, precipitation.
            Optionally: pop (probability of precipitation), weather_main.

        Returns
        -------
        WeekConstraints
        """
        week = WeekConstraints()

        for idx, row in forecast_df.iterrows():
            daily = self._analyse_day(idx, row)
            week.daily.append(daily)

        # Aggregate summary
        week.overall_category = self._overall_category(week.daily)
        week.summary_warnings = self._summary_warnings(week.daily)

        logger.info(
            "Heuristic analysis: %d days, overall=%s, warnings=%d",
            len(week.daily), week.overall_category, len(week.summary_warnings),
        )
        return week

    def _analyse_day(self, date_idx, row: pd.Series) -> DailyConstraints:
        """Apply all heuristic rules to a single forecast day."""
        temp = float(row.get("temp", 15.0))
        humidity = float(row.get("humidity", 50.0))
        wind = float(row.get("wind_speed", 0.0))
        precip = float(row.get("precipitation", 0.0))
        pop = float(row.get("pop", 0.0))

        date_str = str(date_idx.date()) if hasattr(date_idx, "date") else str(date_idx)

        dc = DailyConstraints(
            date=date_str,
            temp=temp,
            humidity=humidity,
            wind_speed=wind,
            precipitation=precip,
            rain_probability=pop,
        )

        # ── Temperature Rules ──
        dc.temperature_category = self._classify_temp(temp)

        if temp < 0:
            dc.mandatory_items.extend([
                "thermal base layer", "insulated winter coat",
                "warm hat", "gloves", "scarf",
            ])
            dc.avoid_items.extend(["t-shirt only", "shorts", "sandals"])
            dc.warnings.append(f"Freezing conditions ({temp:.0f}°C) — dress in warm layers")
            dc.layering_advice.extend([
                "Layer 1: Thermal base layer (merino or synthetic)",
                "Layer 2: Fleece or wool mid-layer",
                "Layer 3: Insulated winter coat",
            ])

        elif temp < 5:
            dc.mandatory_items.extend([
                "thermal base layer", "warm coat", "scarf",
            ])
            dc.avoid_items.extend(["t-shirt only", "shorts", "light dress"])
            dc.warnings.append(f"Very cold ({temp:.0f}°C) — thermal layers essential")
            dc.layering_advice.extend([
                "Layer 1: Thermal or long-sleeve base",
                "Layer 2: Jumper or fleece",
                "Layer 3: Warm winter coat",
            ])

        elif temp < self.thresholds["cold_temp"]:
            dc.mandatory_items.extend(["warm jumper or fleece", "jacket"])
            dc.layering_advice.extend([
                "Layer 1: Long-sleeve top",
                "Layer 2: Jumper or cardigan",
                "Layer 3: Jacket or light coat",
            ])

        elif temp <= self.thresholds["hot_temp"]:
            dc.layering_advice.append("Versatile layers that can be removed if warm")

        else:
            dc.mandatory_items.extend(["light breathable clothing", "sun hat"])
            dc.avoid_items.extend(["heavy coat", "thick jumper"])
            dc.layering_advice.append("Light, breathable fabrics — cotton or linen")
            if temp > 30:
                dc.warnings.append(f"Hot conditions ({temp:.0f}°C) — stay hydrated, seek shade")

        # ── Rain Rules ──
        if pop > 0.8 or precip > 10:
            dc.mandatory_items.append("waterproof jacket")
            dc.mandatory_items.append("waterproof footwear")
            dc.warnings.append(
                f"High rain likelihood ({pop*100:.0f}%, {precip:.1f}mm) — waterproofs essential"
            )
        elif pop > self.thresholds["rain_prob"] or precip > 2:
            dc.mandatory_items.append("waterproof jacket or umbrella")
            dc.warnings.append(f"Rain possible ({pop*100:.0f}%) — carry waterproofs")

        # ── Wind Rules ──
        if wind > self.thresholds["high_wind"]:
            dc.mandatory_items.append("wind-resistant outer layer")
            dc.warnings.append(f"Strong winds ({wind:.0f} km/h) — secure loose clothing")
            if temp < 10:
                dc.warnings.append("Wind chill will make it feel significantly colder")

        # ── Humidity Rules ──
        if humidity > 85 and temp > 20:
            dc.warnings.append("High humidity — choose moisture-wicking fabrics")

        # Deduplicate
        dc.mandatory_items = list(dict.fromkeys(dc.mandatory_items))
        dc.avoid_items = list(dict.fromkeys(dc.avoid_items))
        dc.warnings = list(dict.fromkeys(dc.warnings))

        return dc

    def _classify_temp(self, temp: float) -> str:
        if temp < 0:
            return "freezing"
        elif temp < 5:
            return "very_cold"
        elif temp < self.thresholds["cold_temp"]:
            return "cold"
        elif temp <= self.thresholds["hot_temp"]:
            return "mild"
        elif temp <= 30:
            return "warm"
        else:
            return "hot"

    def _overall_category(self, daily: List[DailyConstraints]) -> str:
        cats = [d.temperature_category for d in daily]
        if all(c == cats[0] for c in cats):
            return cats[0]
        return "mixed"

    def _summary_warnings(self, daily: List[DailyConstraints]) -> List[str]:
        all_warnings = []
        rain_days = sum(1 for d in daily if d.rain_probability > 0.4 or d.precipitation > 2)
        cold_days = sum(1 for d in daily if d.temp < 5)
        hot_days = sum(1 for d in daily if d.temp > 25)

        if rain_days >= 3:
            all_warnings.append(f"Rain expected on {rain_days} of 7 days — keep waterproofs handy all week")
        if cold_days >= 3:
            all_warnings.append(f"Cold spell: {cold_days} days below 5°C — plan warm layers")
        if hot_days >= 3:
            all_warnings.append(f"Warm spell: {hot_days} days above 25°C — light clothing recommended")

        return all_warnings

    def validate_recommendation(
        self,
        text: str,
        constraints: DailyConstraints,
    ) -> tuple:
        """
        Post-generation guardrail: verify that a recommendation text
        honours the mandatory constraints.

        Returns
        -------
        (is_valid, missing_items, text_with_fixes)
        """
        text_lower = text.lower()
        missing = []

        for item in constraints.mandatory_items:
            # Check if any key word from the item appears in the text
            keywords = item.lower().split()
            if not any(kw in text_lower for kw in keywords if len(kw) > 3):
                missing.append(item)

        if not missing:
            return True, [], text

        # Append missing mandatory items
        fix = "\n\n**Weather Safety Reminder:**\n"
        fix += "Based on the forecast, please also ensure you have:\n"
        for item in missing:
            fix += f"- {item.title()}\n"
        for warning in constraints.warnings:
            fix += f"- ⚠️ {warning}\n"

        return False, missing, text + fix
