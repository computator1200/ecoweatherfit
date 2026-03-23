# EcoWeatherFit - Recommendations Package
"""
recommendations/
=================
Hybrid recommendation engine combining GenAI with deterministic guardrails.

Modules
-------
heuristic_engine     – Rule-based safety constraints
gemini_integration   – Gemini GenAI with hybrid architecture
sustainability       – Circular Fashion Loop advice
"""

from recommendations.heuristic_engine import HeuristicEngine, DailyConstraints, WeekConstraints
from recommendations.gemini_integration import GeminiAdvisor
from recommendations.sustainability import SustainabilityEngine, SustainabilityAdvice
