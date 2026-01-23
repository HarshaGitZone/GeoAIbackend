import requests
from math import radians, sin, cos, sqrt, atan2
from typing import Optional, Tuple

NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.openstreetmap.ru/api/interpreter"]
_HEADERS = {"User-Agent": "GeoAI_Universal_Hydrology/11.0", "Accept": "application/json"}

def _is_in_hardcoded_ocean(lat: float, lon: float) -> Tuple[bool, Optional[str]]:
    if -50.0 <= lat <= 50.0 and (140.0 <= lon <= 180.0 or -180.0 <= lon <= -80.0):
        return True, "Deep Pacific Ocean"
    if -50.0 <= lat <= 50.0 and -50.0 <= lon <= -20.0:
        return True, "Deep Atlantic Ocean"
    if -45.0 <= lat <= 5.0 and 50.0 <= lon <= 100.0:
        return True, "Deep Indian Ocean"
    return False, None

def _multi_scale_search(lat: float, lon: float) -> Tuple[bool, Optional[dict]]:
    # Scans local to global to identify named bodies like 'Ganga' or 'Hussain Sagar'
    for zoom in [18, 14, 8, 3]:
        try:
            params = {"format": "jsonv2", "lat": lat, "lon": lon, "zoom": zoom, "addressdetails": 1}
            resp = requests.get(NOMINATIM_REVERSE_URL, params=params, headers=_HEADERS, timeout=6)
            data = resp.json()
            if "error" in data: continue

            name = (data.get("display_name") or "").lower()
            cat = (data.get("category") or data.get("class") or "").lower()
            
            triggers = ["ocean", "sea", "lake", "river", "sagar", "reservoir", "water", "bay", "gulf"]
            if any(t in name for t in triggers) or cat in ["natural", "water", "waterway"]:
                return True, {
                    "source": f"Scale Z{zoom} Map Search",
                    "name": data.get("display_name"),
                    "detail": f"Directly located on {data.get('display_name')}"
                }
        except: continue
    return False, None

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2, dphi, dlamb = radians(lat1), radians(lat2), radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlamb/2)**2
    return R * (2 * atan2(sqrt(a), sqrt(1-a)))

def estimate_water_proximity_score(latitude: float, longitude: float) -> Tuple[float, Optional[float], Optional[dict]]:
    """Final Precise Logic with Evidence Metadata"""
    
    # 1. Direct Named Water Check
    found, details = _multi_scale_search(latitude, longitude)
    if found:
        return 0.0, 0.0, details

    # 2. Hardcoded Ocean Fail-Safe
    is_ocean, ocean_name = _is_in_hardcoded_ocean(latitude, longitude)
    if is_ocean:
        return 0.0, 0.0, {
            "source": "Geometric Fail-Safe",
            "name": ocean_name,
            "detail": f"Located within the coordinates of the {ocean_name}"
        }

    # 3. ADVANCED PROXIMITY SCAN (The logic you confirmed works well)
    for rad in [1000, 3000, 5000]:
        try:
            query = f"""
            [out:json][timeout:15];
            (
              node["natural"="water"](around:{rad},{latitude},{longitude});
              way["waterway"](around:{rad},{latitude},{longitude});
              relation["natural"="water"](around:{rad},{latitude},{longitude});
            );
            out center 1;
            """
            resp = requests.post(OVERPASS_URLS[0], data={"data": query}, headers=_HEADERS, timeout=12)
            elements = resp.json().get("elements")
            
            if elements:
                el = elements[0]
                e_lat = el.get("lat") or el.get("center", {}).get("lat")
                e_lon = el.get("lon") or el.get("center", {}).get("lon")
                dist = haversine_km(latitude, longitude, e_lat, e_lon)
                water_name = el.get("tags", {}).get("name", "Unnamed Waterway")
                
                # Your confirmed Refined Scoring
                if dist < 0.3: score = 15.0
                elif dist < 0.8: score = 35.0
                elif dist < 1.5: score = 55.0
                elif dist < 3.0: score = 75.0
                else: score = 90.0
                
                return score, round(dist, 3), {
                    "source": "Overpass Poly Engine",
                    "name": water_name,
                    "detail": f"Located approximately {round(dist, 2)} km from {water_name}"
                }
        except: continue

    # 4. Fallback for unverified areas
    return 50.0, None, {
        "source": "Safety Fallback",
        "detail": "No major water bodies detected within 5km. Status unverified."
    }