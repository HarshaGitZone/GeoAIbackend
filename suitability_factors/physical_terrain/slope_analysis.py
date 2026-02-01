# # backend/suitability_factors/physical_terrain/slope_adapter.py
# import math
# from .elevation_adapter import get_elevation_data

# def get_slope_analysis(lat: float, lng: float):
#     """
#     Calculates Terrain Slope (Gradient) in Degrees.
#     Uses Horn's Method by sampling neighboring elevation points.
#     """
#     # 0.0003 degrees is roughly 30 meters at the equator
#     delta = 0.0003 

#     try:
#         # Fetch a 3x3 grid of elevations (simplified logic)
#         # In a high-perf system, you'd use a single 'getArea' call, 
#         # but for this logic we'll sample the center and cross-points.
#         z_center = get_elevation_data(lat, lng)['value']
#         z_north = get_elevation_data(lat + delta, lng)['value']
#         z_south = get_elevation_data(lat - delta, lng)['value']
#         z_east = get_elevation_data(lat, lng + delta)['value']
#         z_west = get_elevation_data(lat, lng - delta)['value']

#         # Calculate Rise over Run (Horn's Algorithm simplified for 4-neighbors)
#         dz_dx = (z_east - z_west) / (2 * 30.0)
#         dz_dy = (z_north - z_south) / (2 * 30.0)
        
#         slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
#         slope_deg = math.degrees(slope_rad)

#         # Scale for Aggregator (100 = Perfect Flat, 0 = 45+ degree cliff)
#         # Construction is very difficult above 15 degrees.
#         scaled_score = max(0, 100 - (slope_deg * 2.2))

#         return {
#             "value": round(slope_deg, 2),
#             "scaled_score": round(scaled_score, 1),
#             "label": _classify_slope(slope_deg),
#             "source": "NASA SRTM 30m Topographic Analysis",
#             "link": "https://portal.opentopography.org/apidocs/"
#         }
#     except Exception:
#         return {"value": 0.0, "scaled_score": 70.0, "label": "Unknown"}

# def _classify_slope(deg):
#     if deg < 5: return "Flat (Ideal)"
#     if deg < 15: return "Moderate (Developable)"
#     if deg < 30: return "Steep (High Cost)"
#     return "Extreme (Non-Buildable)"
# import requests
# from typing import Dict

# OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"


# def _slope_label(slope: float) -> str:
#     if slope < 5:
#         return "Flat terrain"
#     elif slope < 15:
#         return "Gentle slope"
#     elif slope < 30:
#         return "Moderate slope"
#     else:
#         return "Steep terrain"


# def get_slope_analysis(lat: float, lng: float) -> Dict:
#     """
#     Estimates slope (%) using local elevation gradient.
#     Source: NASA SRTM via Open-Elevation.
#     """

#     try:
#         delta = 0.0003  # ~30m

#         points = [
#             {"latitude": lat, "longitude": lng},
#             {"latitude": lat + delta, "longitude": lng},
#             {"latitude": lat, "longitude": lng + delta},
#         ]

#         resp = requests.post(
#             OPEN_ELEVATION_URL,
#             json={"locations": points},
#             timeout=10
#         )
#         data = resp.json().get("results", [])

#         if len(data) < 3:
#             raise ValueError("Insufficient elevation points")

#         e0 = data[0]["elevation"]
#         e1 = data[1]["elevation"]
#         e2 = data[2]["elevation"]

#         dz = max(abs(e1 - e0), abs(e2 - e0))
#         dx = 30.0  # meters

#         slope_pct = (dz / dx) * 100.0
#         slope_pct = round(slope_pct, 2)

#         return {
#             "value": slope_pct,
#             "label": _slope_label(slope_pct),   # ✅ FIX
#             "raw": slope_pct,
#             "unit": "%",
#             "source": "NASA SRTM (Open-Elevation)"
#         }

#     except Exception:
#         return {
#             "value": 5.0,
#             "label": "Gentle slope",
#             "raw": None,
#             "unit": "%",
#             "source": "Slope Baseline Fallback"
#         }


import requests
import math
from typing import Dict

# Multiple API endpoints for redundancy
OPEN_ELEVATION_URLS = [
    "https://api.open-elevation.com/api/v1/lookup",
    "https://api.opentopodata.org/v1/srtm30m"
]


def _slope_label(slope_pct: float) -> str:
    if slope_pct is None:
        return "Slope unavailable"
    if slope_pct < 3:
        return "Very flat terrain"
    elif slope_pct < 8:
        return "Gentle slope"
    elif slope_pct < 15:
        return "Moderate slope"
    elif slope_pct < 30:
        return "Steep terrain"
    else:
        return "Very steep terrain"


def _fetch_elevations_open_elevation(points: list) -> list:
    """Fetch from Open-Elevation API"""
    resp = requests.post(
        "https://api.open-elevation.com/api/v1/lookup",
        json={"locations": points},
        timeout=15
    )
    resp.raise_for_status()
    return [r["elevation"] for r in resp.json()["results"]]


def _fetch_elevations_opentopodata(lat: float, lng: float, delta: float) -> list:
    """Fetch from OpenTopoData API (alternative)"""
    locations = f"{lat},{lng}|{lat+delta},{lng}|{lat-delta},{lng}|{lat},{lng+delta}|{lat},{lng-delta}"
    resp = requests.get(
        f"https://api.opentopodata.org/v1/srtm30m?locations={locations}",
        timeout=15
    )
    resp.raise_for_status()
    return [r["elevation"] for r in resp.json()["results"]]


def get_slope_analysis(lat: float, lng: float) -> Dict:
    """
    Computes slope (%) from REAL elevation gradients.
    Source: NASA SRTM (30m) via multiple API providers for reliability.
    Uses Horn's algorithm for accurate slope calculation.
    """
    delta = 0.0003  # ~30m spacing
    
    elevations = None
    source_used = "NASA SRTM"
    
    # Try Open-Elevation first
    try:
        points = [
            {"latitude": lat, "longitude": lng},
            {"latitude": lat + delta, "longitude": lng},
            {"latitude": lat - delta, "longitude": lng},
            {"latitude": lat, "longitude": lng + delta},
            {"latitude": lat, "longitude": lng - delta},
        ]
        elevations = _fetch_elevations_open_elevation(points)
        source_used = "NASA SRTM (Open-Elevation)"
    except Exception:
        pass
    
    # Fallback to OpenTopoData
    if elevations is None:
        try:
            elevations = _fetch_elevations_opentopodata(lat, lng, delta)
            source_used = "NASA SRTM (OpenTopoData)"
        except Exception:
            pass
    
    # If all APIs failed, return with appropriate fallback
    if elevations is None or len(elevations) < 5:
        # Use a reasonable regional estimate based on latitude
        # Coastal/low areas tend to be flatter
        estimated_slope = 2.5 if abs(lat) < 25 else 5.0
        suitability = max(0.0, min(100.0, 100.0 - estimated_slope * 2.22))
        return {
            "value": estimated_slope,
            "scaled_score": round(suitability, 1),
            "label": _slope_label(estimated_slope),
            "raw": {
                "center_elevation_m": None,
                "estimated": True
            },
            "unit": "%",
            "confidence": 50,
            "source": "Regional Terrain Baseline (API temporarily unavailable)"
        }
    
    # Extract elevations
    zc = elevations[0]  # center
    zn = elevations[1]  # north
    zs = elevations[2]  # south
    ze = elevations[3]  # east
    zw = elevations[4]  # west
    
    # Horn's algorithm for slope calculation
    dz_dx = (ze - zw) / (2 * 30.0)  # 30m cell size
    dz_dy = (zn - zs) / (2 * 30.0)
    
    slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
    slope_pct = math.tan(slope_rad) * 100.0
    slope_pct = round(max(0, slope_pct), 2)
    
    # Suitability score: 0% slope = 100 (ideal), ~45% slope = 0 (not suitable)
    slope_suitability = max(0.0, min(100.0, 100.0 - slope_pct * 2.22))
    
    return {
        "value": slope_pct,
        "scaled_score": round(slope_suitability, 1),
        "label": _slope_label(slope_pct),
        "raw": {
            "center_elevation_m": zc,
            "gradient_dx": round(dz_dx, 4),
            "gradient_dy": round(dz_dy, 4),
            "north_m": zn,
            "south_m": zs,
            "east_m": ze,
            "west_m": zw
        },
        "unit": "%",
        "confidence": 95,
        "source": source_used
    }
