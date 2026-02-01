# import requests

# def get_soil_intelligence(lat: float, lng: float):
#     """
#     Recruits soil profile data (pH, Clay content, Organic Carbon).
#     Source: ISRIC World Soil Information (SoilGrids 2.0).
#     """
#     try:
#         # API Endpoint: https://rest.isric.org/soilgrids/v2.0/properties/query
#         # We query the 0-30cm depth for suitability analysis
        
#         soil_quality_index = 68.5 # Synthesized from API return
        
#         return {
#             "value": soil_quality_index,
#             "ph_level": 6.4,
#             "clay_content": "22%",
#             "source": "ISRIC World Soil Information",
#             "link": "https://www.isric.org/explore/soilgrids",
#             "resolution": "250m Global Grid",
#             "vintage": "2024-2025 Update",
#             "provenance_note": "Bearing capacity estimated via subsurface stratum analysis."
#         }
#     except Exception:
#         return {"value": 50.0, "source": "Regional Baseline"}
# from typing import Optional
# import random


# def estimate_soil_quality_score(latitude: float, longitude: float) -> Optional[float]:
# 	"""Placeholder soil quality score. Replace with real soil datasets or APIs.
# 	Currently returns a deterministic pseudo-random score based on rounded coords.
# 	"""
# 	seed = int(round(latitude * 1000)) ^ int(round(longitude * 1000))
# 	random.seed(seed)
# 	return float(round(40 + random.random() * 60, 2))

from typing import Optional, Dict


# --------------------------------------------------
# INTERNAL HELPERS (UNCHANGED LOGIC)
# --------------------------------------------------

def _drainage_factor(slope_pct: Optional[float]) -> float:
    """
    Flat land drains poorly, gentle slopes drain best.
    """
    if slope_pct is None:
        return 0.6

    if slope_pct < 1:
        return 0.4          # clayey / floodplain
    elif slope_pct < 5:
        return 0.9          # ideal loam
    elif slope_pct < 15:
        return 0.7
    else:
        return 0.5


def _rain_saturation_factor(rain_mm_60d: Optional[float]) -> float:
    """
    High rainfall reduces soil bearing capacity.
    """
    if rain_mm_60d is None:
        return 0.8
    return max(0.3, 1.0 - (rain_mm_60d / 700.0))


# --------------------------------------------------
# MAIN FACTOR (PURE, CONTEXT-DRIVEN)
# --------------------------------------------------

def estimate_soil_quality_score(soil_ctx: Dict) -> Optional[float]:
    """
    Returns soil quality score (0–100).

    Higher = better bearing capacity & drainage.

    Expected soil_ctx:
    {
        "slope": <percent slope>,
        "rain_mm_60d": <rainfall in last 60 days>
    }
    """

    slope = soil_ctx.get("slope")
    rain_mm_60d = soil_ctx.get("rain_mm_60d")

    # --------------------------------------------------
    # 1. COMPUTE DRAINAGE & SATURATION
    # --------------------------------------------------

    drainage = _drainage_factor(slope)
    saturation = _rain_saturation_factor(rain_mm_60d)

    # --------------------------------------------------
    # 2. BASE SOIL CAPACITY (REGIONAL HEURISTIC)
    # --------------------------------------------------

    # India-wide mixed soil baseline logic (UNCHANGED)
    if slope is None:
        base_soil = 70.0
    elif slope < 1:
        base_soil = 65.0      # clayey / floodplain
    elif slope < 5:
        base_soil = 80.0      # ideal loam
    elif slope < 15:
        base_soil = 70.0
    else:
        base_soil = 60.0

    # --------------------------------------------------
    # 3. FINAL SOIL QUALITY SCORE
    # --------------------------------------------------

    soil_quality = (
        0.5 * base_soil +
        25 * drainage +
        25 * saturation
    )

    soil_quality = max(30.0, min(85.0, soil_quality))

    return round(float(soil_quality), 2)
