# import requests

# def get_water_utility(lat: float, lng: float):
#     """
#     Measures Hydrological Utility based on proximity to surface water 
#     for irrigation and consumption.
#     """
#     query = f"""
#     [out:json][timeout:15];
#     (
#       node["waterway"](around:3000,{lat},{lng});
#       way["waterway"](around:3000,{lat},{lng});
#       node["natural"="water"](around:3000,{lat},{lng});
#     );
#     out body;
#     """
#     try:
#         resp = requests.post("https://overpass-api.de/api/interpreter", data={"data": query})
#         data = resp.json()
        
#         # Calculate distance to nearest (simplified for 0-100 score)
#         # 100 = Right next to water, 0 = No water within 3km
#         utility_score = 0.0
#         if data.get("elements"):
#             utility_score = 85.0 # High access detected
            
#         return {
#             "value": utility_score,
#             "source": "OpenStreetMap Vector Water Layers",
#             "link": "https://wiki.openstreetmap.org/wiki/Key:waterway",
#             "resolution": "Vector/Real-time",
#             "provenance_note": "Utility represents physical accessibility to permanent water infrastructure."
#         }
#     except Exception:
#         return {"value": 50.0, "source": "OSM Fallback"}
import requests
from math import radians, sin, cos, sqrt, atan2
from typing import Optional, Tuple, Dict

# --------------------------------------------------
# CONSTANTS & CONFIG
# --------------------------------------------------

NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

_HEADERS = {
    "User-Agent": "GeoAI_WaterUtility/11.0",
    "Accept": "application/json"
}

# --------------------------------------------------
# LOW-LEVEL HELPERS
# --------------------------------------------------

def _is_in_hardcoded_ocean(lat: float, lon: float) -> Tuple[bool, Optional[str]]:
    """
    Deep-ocean geometric fail-safe.
    Used ONLY when map-based detection is inconclusive.
    """

    # Deep Pacific Ocean
    if -50.0 <= lat <= 50.0 and (140.0 <= lon <= 180.0 or -180.0 <= lon <= -80.0):
        return True, "Deep Pacific Ocean"

    # Deep Atlantic Ocean
    if -50.0 <= lat <= 50.0 and -50.0 <= lon <= -20.0:
        return True, "Deep Atlantic Ocean"

    # Deep Indian Ocean (below 5°N to avoid India false positives)
    if -45.0 <= lat <= 5.0 and 50.0 <= lon <= 100.0:
        return True, "Deep Indian Ocean"

    return False, None


def _multi_scale_search(lat: float, lon: float) -> Tuple[bool, Optional[Dict]]:
    """
    High-precision named water detection.
    Scans local → global to catch lakes, rivers, reservoirs, seas.
    """
    for zoom in [18, 14, 8, 3]:
        try:
            params = {
                "format": "jsonv2",
                "lat": lat,
                "lon": lon,
                "zoom": zoom,
                "addressdetails": 1
            }
            resp = requests.get(
                NOMINATIM_REVERSE_URL,
                params=params,
                headers=_HEADERS,
                timeout=6
            )
            data = resp.json()
            if "error" in data:
                continue

            name = (data.get("display_name") or "").lower()
            category = (data.get("category") or data.get("class") or "").lower()

            triggers = [
                "ocean", "sea", "lake", "river", "sagar",
                "reservoir", "water", "bay", "gulf"
            ]

            if any(t in name for t in triggers) or category in ["natural", "water", "waterway"]:
                return True, {
                    "source": f"Nominatim Z{zoom}",
                    "name": data.get("display_name"),
                    "detail": f"Directly located on {data.get('display_name')}"
                }
        except Exception:
            continue

    return False, None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return R * (2 * atan2(sqrt(a), sqrt(1 - a)))


# --------------------------------------------------
# MAIN WATER UTILITY FACTOR
# --------------------------------------------------

def get_water_utility(
    latitude: float,
    longitude: float
) -> Dict[str, Optional[float]]:
    """
    Authoritative water proximity & water presence detector.

    Returns:
    {
        "value": <0–100 suitability>,
        "distance_km": <distance to nearest water>,
        "normalized_water_risk": <0–1>,
        "details": {...}
    }
    """

    # --------------------------------------------------
    # 1. DIRECT NAMED WATER DETECTION (HIGHEST CONFIDENCE)
    # --------------------------------------------------
    found, details = _multi_scale_search(latitude, longitude)
    if found:
        return {
            "value": 0.0,
            "distance_km": 0.0,
            "normalized_water_risk": 1.0,
            "details": details
        }

    # --------------------------------------------------
    # 2. DEEP OCEAN FAIL-SAFE (SECONDARY)
    # --------------------------------------------------
    is_ocean, ocean_name = _is_in_hardcoded_ocean(latitude, longitude)
    if is_ocean:
        return {
            "value": 0.0,
            "distance_km": 0.0,
            "normalized_water_risk": 1.0,
            "details": {
                "source": "Geometric Fail-Safe",
                "name": ocean_name,
                "detail": f"Located within bounds of {ocean_name}"
            }
        }

    # --------------------------------------------------
    # 3. PROXIMITY SCAN (RIVERS, LAKES, WATERWAYS)
    # --------------------------------------------------
    for radius_m in [1000, 3000, 5000]:
        try:
            query = f"""
            [out:json][timeout:15];
            (
              node["natural"="water"](around:{radius_m},{latitude},{longitude});
              way["waterway"](around:{radius_m},{latitude},{longitude});
              relation["natural"="water"](around:{radius_m},{latitude},{longitude});
            );
            out center 1;
            """
            resp = requests.post(
                OVERPASS_URLS[0],
                data={"data": query},
                headers=_HEADERS,
                timeout=12
            )
            elements = resp.json().get("elements")

            if elements:
                el = elements[0]
                e_lat = el.get("lat") or el.get("center", {}).get("lat")
                e_lon = el.get("lon") or el.get("center", {}).get("lon")
                dist_km = _haversine_km(latitude, longitude, e_lat, e_lon)
                water_name = el.get("tags", {}).get("name", "Unnamed Waterway")

                # ✔ CONFIRMED SCORING (UNCHANGED)
                if dist_km < 0.3:
                    score = 15.0
                elif dist_km < 0.7:
                    score = 30.0
                elif dist_km < 1.0:
                    score = 45.0
                elif dist_km < 2.5:
                    score = 65.0
                elif dist_km < 5.0:
                    score = 75.0
                elif dist_km < 7.0:
                    score = 90.0
                else:
                    score = 45.0

                return {
                    "value": score,
                    "distance_km": round(dist_km, 3),
                    "normalized_water_risk": round(1.0 - (score / 100.0), 3),
                    "details": {
                        "source": "Overpass Waterway Scan",
                        "name": water_name,
                        "detail": f"Approximately {round(dist_km, 2)} km from {water_name}"
                    }
                }
        except Exception:
            continue

    # --------------------------------------------------
    # 4. VERIFIED LAND FALLBACK
    # --------------------------------------------------
    return {
        "value": 50.0,
        "distance_km": None,
        "normalized_water_risk": 0.5,
        "details": {
            "source": "Safety Fallback",
            "confidence": 0.4,
            "detail": "No major water bodies detected within 5 km. Status unverified."
        }
    }
