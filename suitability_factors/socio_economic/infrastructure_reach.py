# # backend/suitability_factors/socio_econ/infrastructure_reach.py
# import requests

# def get_infrastructure_score(lat: float, lng: float):
#     """
#     Calculates proximity to roads, power grids, and urban hubs.
#     Source: OpenStreetMap (OSM) Vector Data.
#     """
#     # Query for highways and power lines within a 5km radius
#     query = f"""
#     [out:json][timeout:20];
#     (
#       way["highway"~"^(motorway|trunk|primary|secondary)$"](around:5000,{lat},{lng});
#       way["power"="line"](around:5000,{lat},{lng});
#     );
#     out tags;
#     """
#     try:
#         resp = requests.post("https://overpass-api.de/api/interpreter", data={"data": query})
#         elements = resp.json().get("elements", [])
        
#         # Scoring Logic: 100 = Urban/Connected, 0 = Off-grid/Remote
#         count = len(elements)
#         reach_score = min(100, count * 5) # Simple density-based scaling
        
#         return {
#             "value": float(reach_score),
#             "element_count": count,
#             "source": "OpenStreetMap Vector Infrastructure",
#             "link": "https://www.openstreetmap.org/",
#             "resolution": "Real-time Vector",
#             "vintage": "2026 Live Sync",
#             "provenance_note": "Measures spatial density of transport and utility networks."
#         }
#     except Exception:
#         return {"value": 50.0, "source": "Regional Infrastructure Baseline"}
import time
import requests
import math
from typing import Optional, Tuple, Dict

from suitability_factors.hydrology.water_utility import get_water_utility

# --------------------------------------------------
# OVERPASS CONFIG
# --------------------------------------------------

_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

_HEADERS = {
    "User-Agent": "GeoAI_Suitability_Engine/2.0 (contact: support@example.com)",
    "Accept": "application/json",
}

# Road class buffers (meters) for noise / pollution exposure
ROAD_CLASS_BUFFER_M = {
    "motorway": 500,
    "trunk": 400,
    "primary": 350,
    "secondary": 250,
    "tertiary": 150,
    "residential": 100,
    "living_street": 50,
    "unclassified": 100,
    "service": 80,
    "road": 100,
}

DEFAULT_ROAD_BUFFER_M = 100


# --------------------------------------------------
# OVERPASS HELPERS
# --------------------------------------------------

def _build_roads_query(lat: float, lon: float, radius_m: int) -> str:
    """Query major road infrastructure for accessibility analysis."""
    return f"""
    [out:json][timeout:25];
    (
      way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"](around:{radius_m},{lat},{lon});
      node["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"](around:{radius_m},{lat},{lon});
    );
    out center 20;
    """


def _query_roads(lat: float, lon: float, radius_m: int) -> Optional[dict]:
    query = _build_roads_query(lat, lon, radius_m)
    for attempt in range(2):
        for base in _MIRRORS:
            try:
                resp = requests.post(
                    base,
                    data={"data": query},
                    headers=_HEADERS,
                    timeout=12
                )
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                continue
        time.sleep(1)
    return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = (sin(dphi / 2) ** 2) + cos(phi1) * cos(phi2) * (sin(dlambda / 2) ** 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# --------------------------------------------------
# SCORING CURVES
# --------------------------------------------------

def _proximity_benefit(distance_km: Optional[float], optimal_km: float = 0.8) -> float:
    """
    Accessibility benefit peaks near optimal distance.
    Gaussian-like curve.
    """
    if distance_km is None:
        return 0.3
    return math.exp(-((distance_km - optimal_km) ** 2) / (2 * 0.8 ** 2))


def _noise_penalty(distance_km: float, buffer_km: float) -> float:
    """
    Noise & pollution penalty when too close to roads.
    """
    if distance_km < buffer_km:
        return 1.0 - (distance_km / buffer_km)
    return 0.0


# --------------------------------------------------
# MAIN FACTOR
# --------------------------------------------------

def get_infrastructure_score(
    latitude: float,
    longitude: float
) -> Tuple[float, Optional[float], Optional[Dict]]:
    """
    Calculates proximity to major roads.

    Returns:
        (score, distance_km, details)

    Notes:
    - Continuous suitability factor
    - Penalized near water (not hard-killed)
    """

    # --------------------------------------------------
    # 1. WATER CONTEXT (FROM HYDROLOGY)
    # --------------------------------------------------
    water_ctx = get_water_utility(latitude, longitude)
    water_distance = water_ctx.get("distance_km")

    # If point lies directly on water, infrastructure is irrelevant
    if water_distance is not None and water_distance < 0.02:
        return {
            "value": 0.0,
            "distance_km": 0.0,
            "label": "Not Applicable (Water Body)",
            "confidence": 95,
            "source": "Hydrology Override",
            "details": {
                "reason": "Location lies on a water body. Road proximity is not applicable."
            }
        }


    # --------------------------------------------------
    # 2. SEARCH FOR NEAREST MAJOR ROADS
    # --------------------------------------------------
    elements = None
    search_radius = 0

    for radius in (1000, 3000, 7000):
        data = _query_roads(latitude, longitude, radius)
        if data and data.get("elements"):
            elements = data["elements"]
            search_radius = radius
            break

    if not elements:
        return {
            "value": 30.0,
            "distance_km": None,
            "label": "Remote Area",
            "confidence": 70,
            "source": "OpenStreetMap (Overpass API)",
            "details": {
                "reason": "No major roads found within search radius."
            }
        }


    # --------------------------------------------------
    # 3. FIND CLOSEST ROAD FEATURE
    # --------------------------------------------------
    min_km = 999.0
    closest_tags = {}

    for el in elements:
        if "lat" in el and "lon" in el:
            d = _haversine_km(latitude, longitude, el["lat"], el["lon"])
        elif "center" in el:
            d = _haversine_km(
                latitude,
                longitude,
                el["center"]["lat"],
                el["center"]["lon"]
            )
        else:
            continue

        # 🚨 ON-ROAD DETECTION
        if d < 0.04:
            return {
                "value": 25.0,
                "distance_km": round(d, 3),
                "label": "On Road Corridor",
                "confidence": 90,
                "source": "OpenStreetMap (Overpass API)",
                "details": {
                    "nearest_road_name": el.get("tags", {}).get("name", "Unnamed Road"),
                    "road_type": el.get("tags", {}).get("highway", "unknown"),
                    "distance_km": round(d, 3),
                    "explanation": (
                        "Location lies directly on a road corridor. "
                        "High noise, air pollution, and safety risk."
                    )
                }
            }


        if d < min_km:
            min_km = d
            closest_tags = el.get("tags", {})

    # --------------------------------------------------
    # 4. ROAD TYPE & BUFFER
    # --------------------------------------------------
    road_type = closest_tags.get("highway", "unknown")
    road_buffer_km = ROAD_CLASS_BUFFER_M.get(
        road_type,
        DEFAULT_ROAD_BUFFER_M
    ) / 1000.0

    if min_km < 0.03:
        return 25.0, round(min_km, 3), {
            "nearest_road_name": closest_tags.get("name", "Unnamed Road"),
            "road_type": road_type,
            "distance_km": round(min_km, 3),
            "buffer_m": ROAD_CLASS_BUFFER_M.get(road_type, DEFAULT_ROAD_BUFFER_M),
            "in_buffer_zone": True,
            "search_radius_m": search_radius,
            "explanation": (
                f"Location lies directly on a {road_type} road. "
                f"High noise and safety risk."
            )
        }

    # --------------------------------------------------
    # 5. CONTINUOUS SCORING
    # --------------------------------------------------
    benefit = _proximity_benefit(min_km)
    penalty = _noise_penalty(min_km, road_buffer_km)

    proximity_signal = (benefit * 0.85) - (penalty * 0.4)

    score = 20.0 + (proximity_signal * 80.0)
    score = min(score, 80.0)

    # ⚠️ Soft penalty near water bodies
    if water_distance is not None and water_distance < 0.5:
        score *= 0.8

    explanation = (
        f"Nearest road ({road_type}) at {round(min_km, 2)}km. "
        f"Accessibility benefit={round(benefit, 2)}, "
        f"noise penalty={round(penalty, 2)}."
    )

    details = {
        "nearest_road_name": closest_tags.get("name", "Unnamed Road"),
        "road_type": road_type,
        "distance_km": round(min_km, 3),
        "buffer_m": ROAD_CLASS_BUFFER_M.get(road_type, DEFAULT_ROAD_BUFFER_M),
        "in_buffer_zone": min_km < road_buffer_km,
        "search_radius_m": search_radius,
        "explanation": explanation
    }
    return {
    "value": float(round(score, 2)),
    "distance_km": float(round(min_km, 3)),
    "label": "High Access" if score >= 65 else "Moderate Access" if score >= 40 else "Low Access",
    "confidence": 85,
    "source": "OpenStreetMap (Overpass API)",
    "details": details
}

