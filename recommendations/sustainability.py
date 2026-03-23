"""
EcoWeatherFit — Sustainability & Circular Fashion Module
=========================================================
Implements the "Circular Fashion Loop" philosophy that underpins
every recommendation.  The hierarchy is:

    1. Shop Your Wardrobe  (reuse what you own)
    2. Layer Creatively     (combine existing pieces)
    3. Swap / Borrow        (community sharing)
    4. Buy Second-Hand      (charity shops, Vinted, Depop)
    5. Buy Sustainable      (last resort — eco brands only)

This module provides:
* Structured sustainability nudges for the UI
* Garment care tips to extend clothing lifespan
* Eco-scoring logic for recommendations
* Educational content about fast fashion impact
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────
@dataclass
class SustainabilityAdvice:
    """Structured sustainability output for the UI."""
    wardrobe_tip: str
    layering_tip: str
    care_tip: str
    eco_fact: str
    circular_priority: str  # Which loop step applies most
    swap_suggestion: str = ""


# ──────────────────────────────────────────────
# Circular Fashion Engine
# ──────────────────────────────────────────────
class SustainabilityEngine:
    """
    Generates sustainability-aware advice aligned with the
    Circular Fashion Loop.
    """

    CIRCULAR_LOOP = [
        "Shop Your Wardrobe",
        "Layer Creatively",
        "Swap or Borrow",
        "Buy Second-Hand",
        "Buy Sustainable (Last Resort)",
    ]

    WARDROBE_TIPS = {
        "freezing": (
            "Check the back of your wardrobe for that forgotten thermal base layer "
            "or wool jumper — they're perfect for today's freezing conditions."
        ),
        "very_cold": (
            "That chunky knit your nan gave you? Today's the day! Layer it "
            "under your existing coat for instant warmth without buying anything new."
        ),
        "cold": (
            "A jumper-and-jacket combo from your wardrobe is all you need. "
            "No new purchases required — just smart layering."
        ),
        "mild": (
            "Mild days are perfect for that transitional jacket you haven't "
            "worn in a while. Rediscover your wardrobe's versatility!"
        ),
        "warm": (
            "Light cotton tees and linen pieces you already own are ideal. "
            "Check your summer drawer before shopping."
        ),
        "hot": (
            "Your lightest, most breathable pieces are calling. That cotton "
            "dress or linen shirt is perfect — no need to buy new."
        ),
    }

    LAYERING_TIPS = {
        "freezing": "Three layers minimum: thermal base + insulating mid + weatherproof outer",
        "very_cold": "Thermal base layer + warm mid-layer + outer shell = cosy without bulk",
        "cold": "A good jumper under a jacket provides warmth and style flexibility",
        "mild": "A light layer you can tie around your waist gives maximum adaptability",
        "warm": "Single-layer dressing works — choose breathable natural fabrics",
        "hot": "Loose-fitting single layers in light colours keep you coolest",
    }

    CARE_TIPS = [
        "Wash at 30°C to save energy and protect fabric fibres — clothes last 2x longer.",
        "Air-dry instead of tumble-drying: your clothes will thank you (and so will the planet).",
        "Turn dark clothes inside-out before washing to prevent fading and extend their life.",
        "Spot-clean stains immediately rather than full-washing — fewer washes = longer garment life.",
        "Store knitwear folded (not hung) to prevent stretching and keep them in shape longer.",
        "Invest 5 minutes in mending small tears or loose buttons — repair extends garment life by years.",
        "Use a mesh laundry bag for delicates — it prevents snagging and reduces microfibre shedding.",
        "Refresh clothes with a steamer instead of re-washing — saves water and reduces fabric wear.",
        "Zip up zippers before washing to prevent them catching on other garments.",
        "Rotate your favourite pieces to distribute wear evenly across your wardrobe.",
    ]

    ECO_FACTS = [
        "The fashion industry produces 10% of global carbon emissions — more than aviation and shipping combined.",
        "The average garment is worn just 7 times before being discarded. Let's change that!",
        "Extending a garment's life by just 9 months reduces its environmental footprint by 20-30%.",
        "It takes 2,700 litres of water to make a single cotton t-shirt — treasure the ones you have.",
        "92 million tonnes of textile waste is produced annually. Your wardrobe choices matter!",
        "Buying second-hand reduces a garment's carbon footprint by up to 82% compared to new.",
        "The UK sends 350,000 tonnes of clothing to landfill every year. Reuse is the antidote.",
        "A well-maintained wool jumper can last 30+ years. Fast fashion can't compete with quality.",
        "Washing synthetic clothes releases 500,000 tonnes of microfibres into the ocean annually.",
        "Choosing to repair instead of replace one item saves an average of 22 pounds of CO2.",
    ]

    def generate_advice(
        self,
        temperature_category: str,
        day_index: int = 0,
    ) -> SustainabilityAdvice:
        """
        Generate sustainability advice tailored to the weather category.

        Parameters
        ----------
        temperature_category : str
            One of: freezing, very_cold, cold, mild, warm, hot.
        day_index : int
            Used to rotate through tips/facts for variety.

        Returns
        -------
        SustainabilityAdvice
        """
        cat = temperature_category if temperature_category in self.WARDROBE_TIPS else "mild"

        return SustainabilityAdvice(
            wardrobe_tip=self.WARDROBE_TIPS[cat],
            layering_tip=self.LAYERING_TIPS[cat],
            care_tip=self.CARE_TIPS[day_index % len(self.CARE_TIPS)],
            eco_fact=self.ECO_FACTS[day_index % len(self.ECO_FACTS)],
            circular_priority=self.CIRCULAR_LOOP[0],  # Always reuse first
            swap_suggestion=self._swap_suggestion(cat),
        )

    def get_circular_loop_display(self) -> List[Dict]:
        """Return the Circular Fashion Loop as structured data for UI."""
        icons = ["🔄", "🧥", "🤝", "♻️", "🌱"]
        descriptions = [
            "Check your existing wardrobe first — you probably already have what you need.",
            "Combine existing pieces creatively for new looks and better weather protection.",
            "Borrow from friends/family or use clothing swap platforms for occasional needs.",
            "Browse charity shops, Vinted, or Depop for pre-loved alternatives.",
            "Only buy new as a last resort — choose sustainable, ethical brands.",
        ]
        return [
            {
                "step": i + 1,
                "name": name,
                "icon": icons[i],
                "description": descriptions[i],
                "is_priority": i == 0,
            }
            for i, name in enumerate(self.CIRCULAR_LOOP)
        ]

    def _swap_suggestion(self, category: str) -> str:
        suggestions = {
            "freezing": "Ask friends or family if they have a spare winter coat — sharing warmth is sustainable!",
            "very_cold": "Clothing swap events are great for finding warm layers without buying new.",
            "cold": "Check local community groups for coat and jumper swaps this season.",
            "mild": "A clothing swap with friends is perfect for refreshing your transitional wardrobe.",
            "warm": "Summer clothes are ideal for swapping — check Vinted for pre-loved pieces.",
            "hot": "Light summer clothing is easy to find second-hand — try your local charity shop.",
        }
        return suggestions.get(category, suggestions["mild"])
