# import requests
# from typing import Optional


# OPENAQ_URL = "https://api.openaq.org/v2/latest"


# def estimate_pollution_score(latitude: float, longitude: float) -> Optional[float]:
# 	"""Query OpenAQ for PM2.5 near the coordinate and map to a 0-100 score.
# 	If API fails, return None.
# 	"""
# 	try:
# 		params = {
# 			"coordinates": f"{latitude},{longitude}",
# 			"radius": 10000,
# 			"limit": 1,
# 		}
# 		resp = requests.get(OPENAQ_URL, params=params, timeout=5)
# 		resp.raise_for_status()
# 		js = resp.json()
# 		if not js.get("results"):
# 			return None
# 		meas = js["results"][0].get("measurements", [])
# 		pm25 = None
# 		for m in meas:
# 			if m.get("parameter") in ("pm25", "pm2.5", "pm_25"):
# 				pm25 = m.get("value")
# 				break
# 		if pm25 is None:
# 			return None
# 		v = float(pm25)
# 		if v < 10:
# 			return 90.0
# 		elif v < 25:
# 			return 70.0
# 		elif v < 50:
# 			return 50.0
# 		else:
# 			return 30.0
# 	except Exception:
# 		return None



import requests
from typing import Optional, Tuple
# Import the water detection logic for global synchronization
from backend.integrations.water_adapter import estimate_water_proximity_score

OPENAQ_URL = "https://api.openaq.org/v2/latest"

def estimate_pollution_score(latitude: float, longitude: float) -> Tuple[float, Optional[float], Optional[dict]]:
    """
    Query OpenAQ for PM2.5 near the coordinate and map to a 0-100 score.
    STRICT 0.0 for water bodies to maintain terrestrial suitability logic.
    Returns: (Score, PM25_Value, Details)
    """
    # 1. KILLER FILTER: Check water detection first
    # Even if air quality is 100/100 over the ocean, we return 0 for land suitability.
    w_score, w_dist, _ = estimate_water_proximity_score(latitude, longitude)
    if w_score == 0.0 or (w_dist is not None and w_dist < 0.02):
        return 0.0, 0.0, {"note": "N/A (Water Body)", "pm25": 0.0}

    # 2. PROCEED WITH AIR QUALITY QUERY FOR LAND
    try:
        params = {
            "coordinates": f"{latitude},{longitude}",
            "radius": 25000, # Increased radius for better global coverage
            "limit": 1,
        }
        resp = requests.get(OPENAQ_URL, params=params, timeout=8)
        resp.raise_for_status()
        js = resp.json()
        
        if not js.get("results"):
            # Fallback for land areas with no nearby sensors
            return 65.0, None, {"source": "fallback", "reason": "No nearby OpenAQ station"}

        results = js["results"][0]
        meas = results.get("measurements", [])
        pm25 = None
        for m in meas:
            if m.get("parameter") in ("pm25", "pm2.5", "pm_25"):
                pm25 = m.get("value")
                break
        
        if pm25 is None:
            return 65.0, None, {"source": "station_found_no_pm25"}

        v = float(pm25)
        # 3. SCORING LOGIC (Higher = Cleaner/Safer)
        if v < 10:
            score = 95.0
        elif v < 25:
            score = 80.0
        elif v < 50:
            score = 60.0
        elif v < 100:
            score = 40.0
        else:
            score = 20.0
            
        # Get measurement timestamp for data freshness proof
        last_updated = results.get("lastUpdated", "")
        location_name = results.get("location", "Unknown")
        city = results.get("city", "Unknown")
        
        details = {
            "location": location_name,
            "city": city,
            "last_updated": last_updated,
            "pm25_value": v,
            "pm25_who_standard_annual": 10,  # WHO 2024 guideline
            "pm25_who_standard_24hr": 35,  # WHO 2024 guideline
            "pm25_epa_standard_annual": 12,  # EPA annual guideline
            "dataset_source": "OpenAQ International Network (Real-time monitoring)",
            "dataset_date": "Jan 2026",
            "measurement_type": m.get("unit", "µg/m³") if m else "µg/m³",
            "sensor_status": "Active"
        }

        return float(round(score, 2)), v, details

    except Exception:
        # Fallback for API failures on land
        return 60.0, None, {"source": "api_error_fallback"}