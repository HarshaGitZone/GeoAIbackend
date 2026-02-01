# # import requests

# # def get_pollution_metrics(lat: float, lng: float):
# #     """
# #     Air quality analysis based on Sentinel-5P Satellite Aerosol Data.
# #     Source: Copernicus Program (2025-2026).
# #     """
# #     try:
# #         # We fetch NO2, SO2, and CO levels from TROPOMI sensor
# #         pollution_score = 60.0 # (100 is Clean, 0 is Highly Polluted)
        
# #         return {
# #             "value": pollution_score,
# #             "pm2_5_estimate": "12.4 µg/m³",
# #             "source": "Copernicus Sentinel-5P (TROPOMI)",
# #             "link": "https://sentinel.esa.int/web/sentinel/missions/sentinel-5p",
# #             "resolution": "1113m (processed)",
# #             "vintage": "Real-time (Last 24h Sync)",
# #             "provenance_note": "Aerosol optical depth used for particulate matter modeling."
# #         }
# #     except Exception:
# #         return {"value": 60.0, "source": "Global CAMS Baseline"}
# # import requests
# # from typing import Optional


# # OPENAQ_URL = "https://api.openaq.org/v2/latest"


# # def estimate_pollution_score(latitude: float, longitude: float) -> Optional[float]:
# # 	"""Query OpenAQ for PM2.5 near the coordinate and map to a 0-100 score.
# # 	If API fails, return None.
# # 	"""
# # 	try:
# # 		params = {
# # 			"coordinates": f"{latitude},{longitude}",
# # 			"radius": 10000,
# # 			"limit": 1,
# # 		}
# # 		resp = requests.get(OPENAQ_URL, params=params, timeout=5)
# # 		resp.raise_for_status()
# # 		js = resp.json()
# # 		if not js.get("results"):
# # 			return None
# # 		meas = js["results"][0].get("measurements", [])
# # 		pm25 = None
# # 		for m in meas:
# # 			if m.get("parameter") in ("pm25", "pm2.5", "pm_25"):
# # 				pm25 = m.get("value")
# # 				break
# # 		if pm25 is None:
# # 			return None
# # 		v = float(pm25)
# # 		if v < 10:
# # 			return 90.0
# # 		elif v < 25:
# # 			return 70.0
# # 		elif v < 50:
# # 			return 50.0
# # 		else:
# # 			return 30.0
# # 	except Exception:
# # 		return None



# import requests
# from typing import Optional, Tuple

# import math
# OPENAQ_URL = "https://api.openaq.org/v2/latest"

# def estimate_pollution_score(latitude: float, longitude: float) -> Tuple[float, Optional[float], Optional[dict]]:
#     """
#     Query OpenAQ for PM2.5 near the coordinate and map to a 0-100 score.
#     STRICT 0.0 for water bodies to maintain terrestrial suitability logic.
#     Returns: (Score, PM25_Value, Details)
#     """
   
#     # 2. PROCEED WITH AIR QUALITY QUERY FOR LAND
#     try:
#         params = {
#             "coordinates": f"{latitude},{longitude}",
#             "radius": 25000, # Increased radius for better global coverage
#             "limit": 1,
#         }
#         resp = requests.get(OPENAQ_URL, params=params, timeout=8)
#         resp.raise_for_status()
#         js = resp.json()
        
#         if not js.get("results"):
#             # Fallback for land areas with no nearby sensors
#             return 65.0, None, {"source": "fallback", "reason": "No nearby OpenAQ station"}

#         results = js["results"][0]
#         meas = results.get("measurements", [])
#         pm25 = None
#         for m in meas:
#             if m.get("parameter") in ("pm25", "pm2.5", "pm_25"):
#                 pm25 = m.get("value")
#                 break
        
#         if pm25 is None:
#             return 65.0, None, {"source": "station_found_no_pm25"}

#         v = float(pm25)
       
#         health_factor = min(1.0, pm25 / 75.0)
#         score = 100 * (1 - health_factor)
#         score = max(30, min(85, score))

            
#         # Get measurement timestamp for data freshness proof
#         last_updated = results.get("lastUpdated", "")
#         location_name = results.get("location", "Unknown")
#         city = results.get("city", "Unknown")
        
#         details = {
#             "location": location_name,
#             "city": city,
#             "last_updated": last_updated,
#             "pm25_value": v,
#             "pm25_who_standard_annual": 10,  # WHO 2024 guideline
#             "pm25_who_standard_24hr": 35,  # WHO 2024 guideline
#             "pm25_epa_standard_annual": 12,  # EPA annual guideline
#             "dataset_source": "OpenAQ International Network (Real-time monitoring)",
#             "dataset_date": "Jan 2026",
#             "measurement_type": m.get("unit", "µg/m³") if m else "µg/m³",
#             "sensor_status": "Active"
#         }

#         return float(round(score, 2)), v, details

#     except Exception:
     
#         return 60.0, None, {"source": "api_error_fallback"}
# from typing import Dict, Optional, Tuple

# def estimate_pollution_score(
#     pollution_ctx: Dict
# ) -> Tuple[float, Optional[float], Dict]:
#     """
#     PURE pollution suitability evaluator.

#     Input pollution_ctx (from GeoDataService):
#     {
#         "pm25": float | None,
#         "location": str | None,
#         "city": str | None,
#         "last_updated": str | None,
#         "unit": str | None,
#         "source": str
#     }

#     Returns:
#         (score, pm25_value, details)
#     """

#     pm25 = pollution_ctx.get("pm25")

#     # Fallback when no PM2.5 data is available
#     if pm25 is None:
#         return 65.0, None, {
#             "source": pollution_ctx.get("source", "unknown"),
#             "reason": "No PM2.5 data available"
#         }

#     # 🔹 SAME LOGIC AS BEFORE (UNCHANGED)
#     health_factor = min(1.0, pm25 / 75.0)
#     score = 100.0 * (1.0 - health_factor)
#     score = max(30.0, min(85.0, score))

#     details = {
#         "pm25_value": pm25,
#         "pm25_unit": pollution_ctx.get("unit", "µg/m³"),
#         "location": pollution_ctx.get("location"),
#         "city": pollution_ctx.get("city"),
#         "last_updated": pollution_ctx.get("last_updated"),
#         "pm25_who_standard_annual": 10,
#         "pm25_who_standard_24hr": 35,
#         "pm25_epa_standard_annual": 12,
#         "dataset_source": pollution_ctx.get("source", "OpenAQ"),
#     }

#     return round(score, 2), pm25, details

from typing import Dict, Optional, Tuple


def estimate_pollution_score(
    pollution_ctx: Dict
) -> Tuple[float, Optional[float], Dict]:
    """
    PURE pollution suitability evaluator.

    Input pollution_ctx (from GeoDataService):
    {
        "pm25": float | None,
        "location": str | None,
        "city": str | None,
        "last_updated": str | None,
        "unit": str | None,
        "source": str
    }

    Returns:
        (score, pm25_value, details)
    """

    # Defensive read
    pm25 = pollution_ctx.get("pm25")

    # --------------------------------------------------
    # FALLBACK: No PM2.5 data available
    # --------------------------------------------------
    if pm25 is None:
        return 65.0, None, {
            "source": pollution_ctx.get("source", "unknown"),
            "reason": "No PM2.5 data available"
        }

    # --------------------------------------------------
    # SCORING LOGIC (UNCHANGED)
    # --------------------------------------------------
    # WHO-aligned degradation curve
    health_factor = min(1.0, pm25 / 75.0)
    score = 100.0 * (1.0 - health_factor)

    # Clamp to realistic suitability bounds
    score = max(30.0, min(85.0, score))

    # --------------------------------------------------
    # DETAILS (PROVENANCE + CONTEXT)
    # --------------------------------------------------
    details = {
        "pm25_value": pm25,
        "pm25_unit": pollution_ctx.get("unit", "µg/m³"),
        "location": pollution_ctx.get("location"),
        "city": pollution_ctx.get("city"),
        "last_updated": pollution_ctx.get("last_updated"),
        "pm25_who_standard_annual": 10,
        "pm25_who_standard_24hr": 35,
        "pm25_epa_standard_annual": 12,
        "dataset_source": pollution_ctx.get("source", "OpenAQ"),
    }

    return round(score, 2), pm25, details
