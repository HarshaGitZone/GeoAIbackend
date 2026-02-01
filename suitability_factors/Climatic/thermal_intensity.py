# # backend/suitability_factors/Climatic/thermal_intensity.py
# import requests

# def get_thermal_intensity(lat: float, lng: float):
#     """
#     Measures Land Surface Temperature (LST) and Heat Stress.
#     Source: Copernicus Sentinel-3 (SLSTR).
#     """
#     try:
#         # Fetching real-world LST data
#         avg_temp_c = 28.4  # Sample processed return
        
#         return {
#             "value": round(avg_temp_c, 1),
#             "unit": "°C",
#             "source": "Copernicus Sentinel-3 (SLSTR)",
#             "link": "https://sentinel.esa.int/web/sentinel/missions/sentinel-3",
#             "vintage": "2026 Monthly Average",
#             "provenance_note": "Radiative surface temperature derived from thermal infrared bands."
#         }
#     except Exception:
#         return {"value": 25.0, "source": "Climatic Baseline"}

import requests
from typing import Dict

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def _thermal_intensity_label(value: float) -> str:
    if value < 25:
        return "Low heat stress"
    elif value < 45:
        return "Moderate heat stress"
    elif value < 65:
        return "High heat stress"
    else:
        return "Extreme heat stress"


def get_thermal_intensity(lat: float, lng: float) -> Dict:
    """
    Heat stress intensity derived from real forecasted max temperature.
    Fully dynamic, no hard-coded assumptions.
    """

    params = {
        "latitude": lat,
        "longitude": lng,
        "daily": "temperature_2m_max",
        "forecast_days": 7,
        "timezone": "auto"
    }

    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=12)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})

        temps = daily.get("temperature_2m_max")

        if not temps:
            return {
                "status": "unavailable",
                "reason": "Temperature data not available",
                "source": "Open-Meteo"
            }

        avg_max_temp = sum(temps) / len(temps)

        # Heat stress transformation (transparent, linear)
        intensity = max(0.0, min(100.0, (avg_max_temp - 18.0) * 4.0))
        intensity = round(intensity, 2)

        return {
            "value": intensity,
            "label": _thermal_intensity_label(intensity),
            "raw": round(avg_max_temp, 2),
            "unit": "heat-index",
            "confidence": 90,
            "source": "Open-Meteo Meteorological Network",
            "note": "Derived from 7-day average daily maximum temperature"
        }

    except Exception as e:
        return {
            "status": "unavailable",
            "reason": str(e),
            "source": "Open-Meteo"
        }
