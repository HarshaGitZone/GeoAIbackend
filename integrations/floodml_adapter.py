# import os
# import subprocess
# from typing import Dict, Any, Optional

# from .paths import get_project_path


# def estimate_flood_risk_score(latitude: float, longitude: float) -> Optional[float]:
# 	"""Estimate flood risk by leveraging the FloodML app when possible.

# 	Strategy:
# 	- If FloodML exposes a local Flask app with a prediction route, call it.
# 	- Otherwise, attempt to run its prediction script if available.
# 	- Fall back to None if not available.
# 	"""
# 	floodml_path = get_project_path("FloodML")
# 	if not floodml_path:
# 		return None

# 	# Option 1: If FloodML Flask is running with an endpoint, try to query it (future enhancement)
# 	# Skipping direct HTTP call to keep integration optional.

# 	# Option 2: Use FloodML artifacts to approximate a flood safety score.
# 	# If a trained model exists, treat lower predicted rainfall-damage risk as higher safety.
# 	model_pickle = os.path.join(floodml_path, "model.pickle")
# 	if os.path.exists(model_pickle):
# 		try:
# 			# We won't import FloodML internals; instead, approximate by location hash.
# 			seed = int(round(latitude * 1000)) ^ int(round(longitude * 1000))
# 			score = 40.0 + (seed % 61)  # 40..100
# 			return float(score)
# 		except Exception:
# 			return None

# 	return None



import os
from typing import Optional
from .paths import get_project_path
# Import the water detection logic for global synchronization
from backend.integrations.water_adapter import estimate_water_proximity_score

def estimate_flood_risk_score(latitude: float, longitude: float) -> Optional[float]:
    """
    Estimate flood risk by combining RAINFALL DATA + WATER PROXIMITY.
    
    Key Logic:
    - Rainfall alone doesn't cause floods (drainage systems handle normal rain)
    - Water proximity alone isn't definitive (small rivers, drainage capacity varies)
    - COMBINATION matters: Heavy rainfall + nearby water body = HIGH FLOOD RISK
    
    Scoring approach:
    1. Check if location is ON water (immediate disqualification)
    2. Get rainfall data (60-day average)
    3. Get water proximity distance
    4. COMBINE both factors for realistic assessment
    """
    
    # 1. WATER BODY CHECK: If ON water, flood risk is catastrophic
    w_score, w_dist, w_meta = estimate_water_proximity_score(latitude, longitude)
    if w_score == 0.0 or (w_dist is not None and w_dist < 0.02):
        return 0.0  # On water = unsuitable
    
    # 2. GET RAINFALL DATA: Critical for realistic flood assessment
    try:
        from backend.integrations.rainfall_adapter import _fetch_open_meteo_sum
        rain_mm_60d = _fetch_open_meteo_sum(latitude, longitude, 60)
    except Exception:
        rain_mm_60d = None
    
    # 3. COMBINED FLOOD RISK ASSESSMENT
    # Only HIGH rainfall + nearby water = serious flood risk
    
    if w_dist is None or w_dist >= 5.0:
        # FAR FROM WATER: Very low flood risk regardless of rainfall
        # Even heavy rain drains naturally away from rivers
        return 15.0  # Minimal flood risk
    
    # NEAR WATER (< 5km): Assess based on rainfall + distance
    if w_dist < 0.3:  # ON RIVER BANKS (< 300m)
        # Extreme risk zone - any significant rainfall causes flooding
        if rain_mm_60d is not None and rain_mm_60d > 300:
            flood_risk = 90.0  # CRITICAL: Heavy monsoon + river bank
        elif rain_mm_60d is not None and rain_mm_60d > 100:
            flood_risk = 80.0  # HIGH: Moderate rain on river bank
        else:
            flood_risk = 70.0  # Baseline: Even light rain risky at river edge
        
        explanation_detail = (
            f"CRITICAL FLOOD ZONE. Location is {round(w_dist*1000, 0)}m from river/water body (river bank). "
            f"Rainfall: {rain_mm_60d}mm/60d. Any significant precipitation causes immediate flooding. "
            f"100+ year flood events occur at this proximity level."
        )
    
    elif w_dist < 0.8:  # NEAR WATER (300m - 800m)
        # High-risk zone - flooding occurs with moderate-to-heavy rainfall
        if rain_mm_60d is not None and rain_mm_60d > 400:
            flood_risk = 75.0  # VERY HIGH: Heavy rainfall near water
        elif rain_mm_60d is not None and rain_mm_60d > 150:
            flood_risk = 60.0  # HIGH: Moderate rainfall near water
        else:
            flood_risk = 45.0  # MODERATE: Low rainfall, but still risky
        
        explanation_detail = (
            f"HIGH FLOOD RISK. Location is {round(w_dist*1000, 0)}m from water body. "
            f"Rainfall: {rain_mm_60d}mm/60d. With monsoon or heavy rain (>150mm/month), "
            f"overflow risk is significant. 10-25 year flood probability."
        )
    
    elif w_dist < 1.5:  # BUFFER ZONE (800m - 1.5km)
        # Moderate risk - only floods with EXCEPTIONAL rainfall
        if rain_mm_60d is not None and rain_mm_60d > 600:
            flood_risk = 55.0  # MODERATE-HIGH: Extreme rainfall + buffer zone
        elif rain_mm_60d is not None and rain_mm_60d > 250:
            flood_risk = 40.0  # MODERATE: Significant rainfall
        else:
            flood_risk = 25.0  # LOW: Normal rainfall, buffer provides safety
        
        explanation_detail = (
            f"MODERATE FLOOD RISK. Location is in {round(w_dist*1000, 0)}m buffer zone from water. "
            f"Rainfall: {rain_mm_60d}mm/60d. Flooding occurs only with exceptional rainfall "
            f"(>250mm in 60 days) combined with water overflow. Normally safe with proper drainage."
        )
    
    elif w_dist < 3.0:  # MODERATE DISTANCE (1.5km - 3km)
        # Low-to-moderate risk - mostly safe unless EXTREME rainfall + water overflow
        if rain_mm_60d is not None and rain_mm_60d > 700:
            flood_risk = 40.0  # MODERATE: Only extreme rainfall causes flooding
        else:
            flood_risk = 20.0  # LOW: Natural drainage easily handles rainfall
        
        explanation_detail = (
            f"LOW TO MODERATE FLOOD RISK. Location is {round(w_dist, 2)}km from water body. "
            f"Rainfall: {rain_mm_60d}mm/60d. Flooding extremely unlikely unless catastrophic rainfall "
            f"(>250mm in month) + simultaneous water overflow. Standard drainage sufficient."
        )
    
    else:  # > 3km from water
        # Very low risk - considered safe
        flood_risk = 10.0
        explanation_detail = (
            f"VERY LOW FLOOD RISK. Location is {round(w_dist, 2)}km from nearest water body. "
            f"Rainfall: {rain_mm_60d}mm/60d. Distance and topography provide natural protection. "
            f"Standard building practices sufficient."
        )
    
    # Store explanation for use in app.py
    import threading
    if not hasattr(estimate_flood_risk_score, '_explanation'):
        estimate_flood_risk_score._explanation = {}
    estimate_flood_risk_score._explanation[f"{latitude}_{longitude}"] = explanation_detail
    
    return float(round(flood_risk, 2))
