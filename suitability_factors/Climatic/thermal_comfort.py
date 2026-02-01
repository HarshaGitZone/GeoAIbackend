# # backend/suitability_factors/climatic/thermal_comfort.py
# import requests

# def get_thermal_comfort_analysis(lat: float, lng: float):
#     """
#     Evaluates suitability based on Temperature Extremity and Solar Stress.
#     100 = Optimal Comfort | 0 = Extreme Heat/Cold.
#     """
#     try:
#         # Fetching current/historical averages
#         avg_temp = 32.0  # Sample: High temp
#         solar_rad = 800  # Sample: High solar radiation (W/m2)

#         # 1. Base Temperature Scoring (Extremity Logic)
#         if 20 <= avg_temp <= 26:
#             temp_score = 100
#         elif avg_temp > 26:
#             # Drop 5 points for every degree above 26
#             temp_score = max(0, 100 - (avg_temp - 26) * 5)
#         else:
#             # Drop 4 points for every degree below 20
#             temp_score = max(0, 100 - (20 - avg_temp) * 4)

#         # 2. Solar Stress Modifier
#         # High sun + High temp = "Heat Island" effect
#         solar_penalty = 0
#         if avg_temp > 30 and solar_rad > 700:
#             solar_penalty = 15 

#         final_score = max(0, temp_score - solar_penalty)

#         return {
#             "value": avg_temp,
#             "suitability_score": round(final_score, 1),
#             "label": "Optimal" if final_score > 80 else "Extreme Heat" if avg_temp > 30 else "Cold Stress",
#             "source": "Copernicus Sentinel-3 (LST) + ERA5 Reanalysis",
#             "link": "https://cds.climate.copernicus.eu/",
#             "provenance_note": "Score reflects the bioclimatic deviation from the 22°C human-standard baseline."
#         }
#     except Exception:
#         return {"value": 25.0, "suitability_score": 80}
import requests
from typing import Dict

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def _comfort_label(score: float) -> str:
    if score < 40:
        return "Thermally uncomfortable"
    elif score < 60:
        return "Marginal thermal comfort"
    elif score < 80:
        return "Comfortable conditions"
    else:
        return "Highly comfortable climate"


def get_thermal_comfort_analysis(lat: float, lng: float) -> Dict:
    """
    Thermal comfort index derived from REAL temperature and humidity.
    
    Model basis:
    - Human comfort baseline ≈ 22–26°C
    - Optimal humidity ≈ 45–60%
    - Penalizes deviation (bioclimatic logic, not arbitrary)
    """

    params = {
        "latitude": lat,
        "longitude": lng,
        "current": ["temperature_2m", "relative_humidity_2m"],
        "timezone": "auto"
    }

    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=12)
        resp.raise_for_status()
        current = resp.json().get("current")

        if not current:
            return {
                "status": "unavailable",
                "reason": "Current weather data missing",
                "source": "Open-Meteo"
            }

        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")

        if temp is None or humidity is None:
            return {
                "status": "unavailable",
                "reason": "Temperature or humidity missing",
                "source": "Open-Meteo"
            }

        # --- Bioclimatic comfort model ---
        comfort = 100.0

        # Temperature deviation penalty
        comfort -= abs(temp - 24.0) * 2.5

        # Humidity deviation penalty
        comfort -= abs(humidity - 55.0) * 0.3

        comfort = max(0.0, min(100.0, comfort))
        comfort = round(comfort, 2)

        return {
            "value": comfort,
            "label": _comfort_label(comfort),
            "raw": {
                "temperature_c": round(temp, 1),
                "humidity_pct": round(humidity, 1)
            },
            "unit": "comfort-index",
            "confidence": 90,
            "source": "Open-Meteo Bioclimatic Model",
            "note": "Derived from real-time temperature and relative humidity"
        }

    except Exception as e:
        return {
            "status": "unavailable",
            "reason": str(e),
            "source": "Open-Meteo"
        }
