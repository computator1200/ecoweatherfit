"""
EcoWeatherFit — Gemini GenAI Integration (Hybrid Recommendation Engine)
========================================================================
Connects to Google Gemini to generate personalised, conversational
outfit recommendations.  Crucially, Gemini's output is **always**
constrained by the deterministic heuristic rules — this is the
"Hybrid Architecture" required by the project specification.

Flow
----
1. Heuristic engine produces ``DailyConstraints`` from the forecast.
2. A structured prompt is built that includes the constraints as
   hard requirements (the GenAI *must* include them).
3. Gemini generates a natural-language recommendation.
4. Post-processing validates the output against the constraints.
5. Any missing mandatory items are appended automatically.

This ensures that even if Gemini hallucinates or drifts, the final
recommendation is always factually grounded in the weather data.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from config.settings import (
    GEMINI_API_KEY,
    GEMINI_MAX_TOKENS,
    GEMINI_MODEL_NAME,
    GEMINI_TEMPERATURE,
    GEMINI_TIMEOUT_SECONDS,
)
from recommendations.heuristic_engine import (
    DailyConstraints,
    HeuristicEngine,
    WeekConstraints,
)

logger = logging.getLogger(__name__)


class GeminiAdvisor:
    """
    Hybrid recommendation engine: Gemini GenAI + heuristic guardrails.
    """

    def __init__(self):
        self.heuristic = HeuristicEngine()
        self._gemini_available = False
        self._model = None
        self._init_gemini()

    def _init_gemini(self) -> None:
        """Initialise the Gemini client if API key is available."""
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set — using heuristic-only mode.")
            return

        try:
            import google.generativeai as genai

            genai.configure(api_key=GEMINI_API_KEY)
            self._model = genai.GenerativeModel(
                GEMINI_MODEL_NAME,
                generation_config={
                    "max_output_tokens": GEMINI_MAX_TOKENS,
                    "temperature": GEMINI_TEMPERATURE,
                },
            )
            self._gemini_available = True
            logger.info("Gemini API initialised (model: %s)", GEMINI_MODEL_NAME)
        except Exception as exc:
            logger.warning("Gemini init failed: %s — falling back to heuristic mode.", exc)

    # ──────────────────────────────────────────────
    # Main Entry Point
    # ──────────────────────────────────────────────
    def generate_recommendation(
        self,
        forecast_df: pd.DataFrame,
        user_profile: Optional[Dict] = None,
        day_index: int = 0,
    ) -> Dict:
        """
        Generate a hybrid recommendation for a specific forecast day.

        Parameters
        ----------
        forecast_df : pd.DataFrame
            7-day forecast with target variable columns.
        user_profile : dict, optional
            Keys: gender, style_preference, age_group.
        day_index : int
            Which forecast day to generate advice for (0 = tomorrow).

        Returns
        -------
        dict with keys:
            recommendation : str — the final outfit advice text
            constraints : DailyConstraints — the heuristic constraints
            week_constraints : WeekConstraints — full week analysis
            was_validated : bool — whether guardrails were applied
            missing_items : list — items that were appended by guardrails
            source : str — "hybrid" or "heuristic_only"
        """
        if user_profile is None:
            user_profile = {"gender": "unisex", "style_preference": "casual"}

        # Step 1: Heuristic analysis
        week = self.heuristic.analyse_forecast(forecast_df)
        day_idx = min(day_index, len(week.daily) - 1)
        constraints = week.daily[day_idx]

        # Step 2: Generate recommendation
        if self._gemini_available:
            text, gemini_ok = self._call_gemini(constraints, week, user_profile)
            source = "hybrid" if gemini_ok else "heuristic_only"
        else:
            text = self._heuristic_only_text(constraints, user_profile)
            source = "heuristic_only"

        # Step 3: Validate with guardrails
        is_valid, missing, final_text = self.heuristic.validate_recommendation(
            text, constraints
        )

        if not is_valid:
            logger.info(
                "Guardrail triggered: %d mandatory items were missing → appended.",
                len(missing),
            )

        return {
            "recommendation": final_text,
            "constraints": constraints,
            "week_constraints": week,
            "was_validated": not is_valid,
            "missing_items": missing,
            "source": source,
        }

    def generate_week_summary(
        self,
        forecast_df: pd.DataFrame,
        user_profile: Optional[Dict] = None,
    ) -> Dict:
        """Generate a summary recommendation for the entire week."""
        if user_profile is None:
            user_profile = {"gender": "unisex", "style_preference": "casual"}

        week = self.heuristic.analyse_forecast(forecast_df)

        if self._gemini_available:
            text = self._call_gemini_week(week, user_profile)
        else:
            text = self._heuristic_week_text(week, user_profile)

        return {
            "recommendation": text,
            "week_constraints": week,
            "source": "hybrid" if self._gemini_available else "heuristic_only",
        }

    # ──────────────────────────────────────────────
    # Gemini Prompt Engineering
    # ──────────────────────────────────────────────
    def _build_prompt(
        self,
        constraints: DailyConstraints,
        week: WeekConstraints,
        user_profile: Dict,
    ) -> str:
        """Build a structured prompt with embedded heuristic constraints."""
        gender = user_profile.get("gender", "unisex")
        style = user_profile.get("style_preference", "casual")

        prompt = f"""You are EcoWeatherFit, a sustainable fashion advisor that helps people
dress appropriately for the weather using clothes they ALREADY OWN.

WEATHER FORECAST for {constraints.date}:
- Temperature: {constraints.temp:.1f}°C (Category: {constraints.temperature_category})
- Humidity: {constraints.humidity:.0f}%
- Wind Speed: {constraints.wind_speed:.1f} km/h
- Precipitation: {constraints.precipitation:.1f} mm
- Rain Probability: {constraints.rain_probability*100:.0f}%

USER PROFILE:
- Gender presentation: {gender}
- Style preference: {style}

MANDATORY REQUIREMENTS (these MUST appear in your recommendation):
{self._format_mandatory(constraints)}

ITEMS TO AVOID RECOMMENDING:
{self._format_avoid(constraints)}

WEATHER WARNINGS:
{self._format_warnings(constraints)}

LAYERING GUIDE:
{self._format_layering(constraints)}

SUSTAINABILITY RULES (CRITICAL — you must follow these):
1. ALWAYS suggest "shopping your wardrobe" first — recommend items people likely already own
2. Suggest creative layering with existing clothes before buying anything new
3. If you must suggest a purchase, recommend second-hand or sustainable brands
4. Include one garment care tip to extend clothing life
5. Frame everything through the circular fashion lens: Reuse > Repair > Recycle > Buy New

Generate a friendly, conversational outfit recommendation (150-200 words) that:
1. Addresses the user by considering their style preference ({style})
2. Includes ALL mandatory items naturally woven into the advice
3. Prioritises reusing existing wardrobe items
4. Provides specific layering guidance
5. Ends with a brief sustainability tip
"""
        return prompt

    def _build_week_prompt(self, week: WeekConstraints, user_profile: Dict) -> str:
        gender = user_profile.get("gender", "unisex")
        style = user_profile.get("style_preference", "casual")

        days_summary = ""
        for d in week.daily:
            days_summary += (
                f"- {d.date}: {d.temp:.0f}°C, {d.precipitation:.1f}mm rain, "
                f"{d.wind_speed:.0f}km/h wind ({d.temperature_category})\n"
            )

        prompt = f"""You are EcoWeatherFit, a sustainable fashion advisor.

7-DAY FORECAST SUMMARY:
{days_summary}

Overall pattern: {week.overall_category}
Warnings: {'; '.join(week.summary_warnings) if week.summary_warnings else 'None'}

USER: {gender}, {style} style

Generate a concise 7-day wardrobe plan (200-250 words) that:
1. Identifies key wardrobe items needed for the whole week
2. Suggests a minimal, versatile capsule from the user's EXISTING wardrobe
3. Highlights days that need special attention (rain, cold snaps, etc.)
4. Promotes the Circular Fashion Loop: Reuse > Layer > Repair > Buy Sustainable
5. Includes one weekly garment care tip

IMPORTANT: Prioritise "Shop Your Wardrobe" — help them use what they already have.
"""
        return prompt

    # ──────────────────────────────────────────────
    # Gemini API Call
    # ──────────────────────────────────────────────
    def _call_gemini(
        self,
        constraints: DailyConstraints,
        week: WeekConstraints,
        user_profile: Dict,
    ) -> tuple:
        """Call Gemini with the structured prompt.  Returns (text, success)."""
        prompt = self._build_prompt(constraints, week, user_profile)

        try:
            response = self._model.generate_content(prompt)
            text = _safe_response_text(response)
            if not text or not text.strip():
                logger.warning("Gemini returned empty day response — falling back to heuristic.")
                return self._heuristic_only_text(constraints, user_profile), False
            logger.info("Gemini response received (%d chars)", len(text))
            return text, True
        except Exception as exc:
            logger.error("Gemini API call failed: %s — falling back to heuristic.", exc)
            return self._heuristic_only_text(constraints, user_profile), False

    def _call_gemini_week(self, week: WeekConstraints, user_profile: Dict) -> str:
        prompt = self._build_week_prompt(week, user_profile)
        try:
            response = self._model.generate_content(prompt)
            text = _safe_response_text(response)
            if not text or not text.strip():
                logger.warning("Gemini returned empty week response — falling back to heuristic.")
                return self._heuristic_week_text(week, user_profile)
            return text
        except Exception as exc:
            logger.error("Gemini week call failed: %s", exc)
            return self._heuristic_week_text(week, user_profile)

    # ──────────────────────────────────────────────
    # Heuristic-Only Fallback
    # ──────────────────────────────────────────────
    def _heuristic_only_text(
        self,
        constraints: DailyConstraints,
        user_profile: Dict,
    ) -> str:
        """Generate a structured recommendation without GenAI."""
        style = user_profile.get("style_preference", "casual")
        lines = [
            f"**Your Outfit Plan for {constraints.date}**",
            f"Weather: {constraints.temp:.0f}°C, {constraints.temperature_category.replace('_', ' ').title()}",
            "",
        ]

        if constraints.warnings:
            lines.append("**Weather Alerts:**")
            for w in constraints.warnings:
                lines.append(f"  ⚠️ {w}")
            lines.append("")

        if constraints.layering_advice:
            lines.append("**Recommended Layers** (from your existing wardrobe!):")
            for layer in constraints.layering_advice:
                lines.append(f"  🧥 {layer}")
            lines.append("")

        if constraints.mandatory_items:
            lines.append("**Essential Items:**")
            for item in constraints.mandatory_items:
                lines.append(f"  ✅ {item.title()}")
            lines.append("")

        if constraints.avoid_items:
            lines.append("**Avoid Today:**")
            for item in constraints.avoid_items:
                lines.append(f"  ❌ {item.title()}")
            lines.append("")

        lines.extend([
            "**♻️ Sustainability Tip:**",
            "Before reaching for anything new, check what you already have!",
            "Creative layering with existing pieces is the most sustainable choice.",
            f"Style suggestion: pair items in your {style} wardrobe for a fresh look.",
        ])

        return "\n".join(lines)

    def _heuristic_week_text(self, week: WeekConstraints, user_profile: Dict) -> str:
        style = user_profile.get("style_preference", "casual")
        lines = ["**Your 7-Day Wardrobe Plan**\n"]

        for d in week.daily:
            emoji = _weather_emoji(d.temperature_category, d.precipitation)
            lines.append(
                f"{emoji} **{d.date}**: {d.temp:.0f}°C — "
                + (", ".join(d.mandatory_items[:3]) if d.mandatory_items else "flexible outfit day")
            )

        lines.append("")
        if week.summary_warnings:
            lines.append("**Week Alerts:**")
            for w in week.summary_warnings:
                lines.append(f"  ⚠️ {w}")
            lines.append("")

        lines.extend([
            "**♻️ Weekly Sustainability Tip:**",
            "Plan your outfits at the start of the week to reduce decision fatigue",
            f"and maximise the use of your existing {style} wardrobe pieces.",
            "Remember: Reuse > Layer > Repair > Buy Sustainable",
        ])
        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # Formatting Helpers
    # ──────────────────────────────────────────────
    @staticmethod
    def _format_mandatory(c: DailyConstraints) -> str:
        if not c.mandatory_items:
            return "- No specific mandatory items (mild conditions)"
        return "\n".join(f"- {item}" for item in c.mandatory_items)

    @staticmethod
    def _format_avoid(c: DailyConstraints) -> str:
        if not c.avoid_items:
            return "- No specific restrictions"
        return "\n".join(f"- {item}" for item in c.avoid_items)

    @staticmethod
    def _format_warnings(c: DailyConstraints) -> str:
        if not c.warnings:
            return "- No weather warnings"
        return "\n".join(f"- {w}" for w in c.warnings)

    @staticmethod
    def _format_layering(c: DailyConstraints) -> str:
        if not c.layering_advice:
            return "- Standard layering appropriate"
        return "\n".join(f"- {l}" for l in c.layering_advice)


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────
def _safe_response_text(response) -> str:
    """
    Extract text from a Gemini response without raising on empty outputs.

    The google-generativeai SDK raises ValueError on `.text` when the
    response has no text parts (e.g. the whole token budget was spent on
    reasoning tokens, or content was blocked). This helper returns an
    empty string in those cases so the caller can fall back gracefully.
    """
    try:
        text = response.text
        if text is not None:
            return text
    except (ValueError, AttributeError):
        pass

    parts = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            piece = getattr(part, "text", "") or ""
            if piece:
                parts.append(piece)
    return "".join(parts)


def _weather_emoji(category: str, precip: float) -> str:
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
