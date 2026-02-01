# import requests
# from ..physical_terrain.elevation_adapter import get_elevation_data

# def estimate_flood_risk(lat: float, lng: float):
#     """
#     Synthesizes Inundation Hazard by checking Topographical Depressions 
#     relative to recent 60-day rainfall patterns.
#     """
#     try:
#         # 1. Get current 60-day rainfall sum from Open-Meteo
#         rain_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lng}&start_date=2025-12-01&end_date=2026-01-30&daily=precipitation_sum&timezone=auto"
#         rain_res = requests.get(rain_url).json()
#         total_rain = sum(rain_res.get('daily', {}).get('precipitation_sum', []))

#         # 2. Get local elevation to determine if site is in a "bowl"
#         elevation = get_elevation_data(lat, lng)['value']
        
#         # Scoring Logic: High rainfall + low elevation relative to surroundings
#         # 0 = Safe, 100 = Catastrophic Risk
#         risk_score = 10.0
#         if total_rain > 200: risk_score += 30
#         if elevation < 50: risk_score += 20 # Coastal/Lowland penalty

#         return {
#             "value": round(risk_score, 1),
#             "safety_score": 100 - risk_score, # Used by aggregator
#             "recent_rainfall_mm": round(total_rain, 1),
#             "source": "Open-Meteo Global Surface Summary + NASA SRTM",
#             "link": "https://open-meteo.com/en/docs/historical-weather-api",
#             "provenance_note": "Risk based on cumulative 60-day precipitation and elevation profile."
#         }
#     except Exception:
#         return {"value": 20.0, "safety_score": 80.0, "source": "Regional Hydrological Baseline"}
from typing import Dict, Optional


def _rain_factor(rain_mm_60d: Optional[float]) -> float:
    """
    Normalizes rainfall into 0–1 risk space.
    600mm in 60 days ≈ extreme monsoon.
    """
    if rain_mm_60d is None:
        return 0.35
    return min(1.0, rain_mm_60d / 600.0)


def _water_distance_factor(w_dist_km: Optional[float]) -> float:
    """
    Normalizes distance to water into 0–1 risk space.
    0 km   = max flood amplification
    3 km+  = negligible effect
    """
    if w_dist_km is None:
        return 0.35
    return max(0.0, 1.0 - (w_dist_km / 3.0))


def estimate_flood_risk(hydrology_ctx: Dict) -> float:
    """
    PURE flood suitability evaluator.

    hydrology_ctx = {
        "rain_mm_60d": float | None,
        "water_distance_km": float | None
    }

    Returns:
        Flood suitability score (0–100)
        Higher = safer
    """

    rain_mm_60d = hydrology_ctx.get("rain_mm_60d")
    water_dist = hydrology_ctx.get("water_distance_km")

    # SAME LOGIC AS BEFORE (UNCHANGED)
    rain_factor = _rain_factor(rain_mm_60d)
    dist_factor = _water_distance_factor(water_dist)

    flood_risk = rain_factor * dist_factor
    flood_suitability = 100.0 * (1.0 - flood_risk ** 0.6)

    flood_suitability = max(20.0, min(90.0, flood_suitability))

    return round(flood_suitability, 2)
