# # backend/suitability_factors/physical_terrain/elevation_adapter.py
# import requests

# # Get your free API key at https://opentopography.org/
# OPENTopo_API_KEY = "YOUR_OPENTOPOGRAPHY_API_KEY"

# def get_elevation_data(lat: float, lng: float):
#     """
#     Fetches ground elevation (altitude) in meters.
#     Source: NASA SRTM GL1 (30m Resolution).
#     """
#     url = "https://portal.opentopography.org/api/languages/python/getElevation"
#     params = {
#         "demtype": "SRTMGL1",
#         "lat": lat,
#         "lon": lng,
#         "outputFormat": "json",
#         "API_Key": OPENTopo_API_KEY
#     }
    
#     try:
#         response = requests.get(url, params=params, timeout=10)
#         response.raise_for_status()
#         elevation = response.json().get("height", 0.0)
        
#         return {
#             "value": float(elevation),
#             "unit": "meters",
#             "source": "NASA SRTM v3.0 (Global 30m)",
#             "link": "https://opentopography.org/",
#             "vintage": "2025-2026 Baseline",
#             "provenance_note": "Digital Elevation Model (DEM) point query."
#         }
#     except Exception as e:
#         return {"value": 0.0, "error": str(e), "source": "API Fallback"}
# import requests
# from typing import Dict

# OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"


# def _elevation_label(elevation_m: float) -> str:
#     if elevation_m < 200:
#         return "Low elevation"
#     elif elevation_m < 600:
#         return "Moderate elevation"
#     elif elevation_m < 1500:
#         return "High elevation"
#     else:
#         return "Very high elevation"


# def get_elevation_data(lat: float, lng: float) -> Dict:
#     """
#     Fetches elevation in meters above sea level.
#     Source: NASA SRTM (Open-Elevation).
#     """

#     try:
#         resp = requests.get(
#             OPEN_ELEVATION_URL,
#             params={"locations": f"{lat},{lng}"},
#             timeout=8
#         )
#         resp.raise_for_status()

#         elevation = float(resp.json()["results"][0]["elevation"])

#         return {
#             "value": elevation,
#             "label": _elevation_label(elevation),
#             "raw": elevation,
#             "unit": "m",
#             "confidence": 95,
#             "source": "NASA SRTM (Open-Elevation)"
#         }

#     except Exception:
#         return {
#             "value": 100.0,
#             "label": "Low elevation",
#             "raw": None,
#             "unit": "m",
#             "confidence": 60,
#             "source": "Elevation Baseline Fallback"
#         }



import requests
from typing import Dict

OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"


def _elevation_label(elevation_m: float) -> str:
    if elevation_m < 200:
        return "Low elevation"
    elif elevation_m < 600:
        return "Moderate elevation"
    elif elevation_m < 1500:
        return "High elevation"
    else:
        return "Very high elevation"


def get_elevation_data(lat: float, lng: float) -> Dict:
    """
    Fetches REAL elevation (meters above sea level).
    Source: NASA SRTM (30m).
    """

    try:
        resp = requests.get(
            OPEN_ELEVATION_URL,
            params={"locations": f"{lat},{lng}"},
            timeout=8
        )
        resp.raise_for_status()

        elevation = float(resp.json()["results"][0]["elevation"])

        return {
            "value": elevation,
            "label": _elevation_label(elevation),
            "raw": elevation,
            "unit": "m",
            "confidence": 95,
            "source": "NASA SRTM (Open-Elevation)"
        }

    except Exception as e:
        return {
            "value": None,
            "label": "Elevation unavailable",
            "raw": None,
            "unit": "m",
            "confidence": 40,
            "source": "DEM fetch failed",
            "error": str(e)
        }
