# import requests
# from math import radians, sin, cos, sqrt, atan2
# from typing import Optional, Tuple

# NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
# OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.openstreetmap.ru/api/interpreter"]
# _HEADERS = {"User-Agent": "GeoAI_Universal_Hydrology/6.0", "Accept": "application/json"}

# def _is_in_hardcoded_ocean(lat: float, lon: float) -> Tuple[bool, Optional[str]]:
#     """Math-based fail-safe for Pacific, Atlantic, and Indian Oceans."""
#     # Pacific Ocean (Covering the entire basin)
#     if -60.0 <= lat <= 60.0 and (120.0 <= lon <= 180.0 or -180.0 <= lon <= -70.0):
#         return True, "Pacific Ocean (Geometric Fail-Safe)"
#     # Atlantic Ocean
#     if -60.0 <= lat <= 60.0 and -60.0 <= lon <= -5.0:
#         return True, "Atlantic Ocean (Geometric Fail-Safe)"
#     # Indian Ocean / Arabian Sea / Bay of Bengal
#     if -45.0 <= lat <= 26.0 and 40.0 <= lon <= 110.0:
#         return True, "Indian Ocean Region (Geometric Fail-Safe)"
#     return False, None

# def _multi_scale_search(lat: float, lon: float) -> Tuple[bool, Optional[dict]]:
#     """Checks 5 scales of zoom to ensure Hussain Sagar and Pacific are caught."""
#     # Zoom 3: Global Oceans | Zoom 18: Local features like 'Sagar'
#     for zoom in [3, 8, 14, 18]:
#         try:
#             params = {"format": "jsonv2", "lat": lat, "lon": lon, "zoom": zoom, "addressdetails": 1}
#             resp = requests.get(NOMINATIM_REVERSE_URL, params=params, headers=_HEADERS, timeout=6)
#             data = resp.json()
#             if "error" in data: continue

#             name = (data.get("display_name") or "").lower()
#             cat = (data.get("category") or data.get("class") or "").lower()
            
#             # TRIGGER KEYWORDS: 'Sagar' is critical for Hussain Sagar
#             triggers = ["ocean", "sea", "lake", "river", "sagar", "reservoir", "water", "bay", "gulf", "tank", "pond"]
#             if any(t in name for t in triggers) or cat in ["natural", "water", "waterway"]:
#                 return True, {"source": f"scale_z{zoom}", "name": data.get("display_name")}
#         except: continue
#     return False, None

# def haversine_km(lat1, lon1, lat2, lon2):
#     R = 6371.0
#     phi1, phi2, dphi, dlamb = radians(lat1), radians(lat2), radians(lat2-lat1), radians(lon2-lon1)
#     a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlamb/2)**2
#     return R * (2 * atan2(sqrt(a), sqrt(1-a)))

# def estimate_water_proximity_score(latitude: float, longitude: float) -> Tuple[float, Optional[float], Optional[dict]]:
#     """Optimized global-to-local detection with a 50.0 fallback."""
#     # 1. Macro Global Check (Pacific/Atlantic/Indian)
#     is_ocean, ocean_name = _is_in_hardcoded_ocean(latitude, longitude)
#     if is_ocean:
#         return 0.0, 0.0, {"source": "geo_bounds", "name": ocean_name}

#     # 2. Named Water Body Check (Zoom 3 to 18)
#     found, details = _multi_scale_search(latitude, longitude)
#     if found:
#         return 0.0, 0.0, details

#     # 3. Dynamic Proximity Logic (Nearby Rivers/Lakes)
#     for rad in [1000, 5000]:
#         try:
#             query = f'[out:json][timeout:10];node["natural"="water"](around:{rad},{latitude},{longitude});out center 1;'
#             resp = requests.post(OVERPASS_URLS[0], data={"data": query}, headers=_HEADERS, timeout=8)
#             elements = resp.json().get("elements")
#             if elements:
#                 el = elements[0]
#                 e_lat = el.get("lat") or el.get("center", {}).get("lat")
#                 e_lon = el.get("lon") or el.get("center", {}).get("lon")
#                 dist = haversine_km(latitude, longitude, e_lat, e_lon)
                
#                 if dist < 0.05: return 0.0, 0.0, {"source": "overpass", "name": "On Water"}
                
#                 # Dynamic scoring based on distance
#                 if dist < 0.5: score = 30.0
#                 elif dist < 1.0: score = 60.0
#                 elif dist < 3.0: score = 80.0
#                 else: score = 90.0
                
#                 return score, round(dist, 3), {"source": "overpass", "name": "Nearby Water"}
#         except: continue

#     # 4. FINAL FALLBACK: If all checks find 'nothing', return 50.0 (Uncertain)
#     return 50.0, None, {"source": "uncertainty_fallback", "note": "Water status unverified."}













# import requests
# from math import radians, sin, cos, sqrt, atan2
# from typing import Optional, Tuple

# NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
# OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.openstreetmap.ru/api/interpreter"]
# _HEADERS = {"User-Agent": "GeoAI_Universal_Hydrology/7.0", "Accept": "application/json"}

# def _is_in_hardcoded_ocean(lat: float, lon: float) -> Tuple[bool, Optional[str]]:
#     """
#     Layer 1: Tightened Fail-Safe.
#     We only use this for deep, undisputed water far from the coastline.
#     """
#     # DEEP PACIFIC (Far from land)
#     if -50.0 <= lat <= 50.0 and (140.0 <= lon <= 180.0 or -180.0 <= lon <= -80.0):
#         return True, "Deep Pacific Ocean"
    
#     # DEEP ATLANTIC (Far from land)
#     if -50.0 <= lat <= 50.0 and -50.0 <= lon <= -20.0:
#         return True, "Deep Atlantic Ocean"
    
#     # DEEP INDIAN OCEAN (Below 5°N to avoid covering India)
#     # This prevents the "everything is water" error in India.
#     if -45.0 <= lat <= 5.0 and 50.0 <= lon <= 100.0:
#         return True, "Deep Indian Ocean"
        
#     return False, None

# def _multi_scale_search(lat: float, lon: float) -> Tuple[bool, Optional[dict]]:
#     """Checks zoom scales to identify Hussain Sagar and named rivers."""
#     # We check high detail (18) first to ensure land features are prioritized
#     for zoom in [18, 14, 8, 3]:
#         try:
#             params = {"format": "jsonv2", "lat": lat, "lon": lon, "zoom": zoom, "addressdetails": 1}
#             resp = requests.get(NOMINATIM_REVERSE_URL, params=params, headers=_HEADERS, timeout=6)
#             data = resp.json()
#             if "error" in data: continue

#             name = (data.get("display_name") or "").lower()
#             cat = (data.get("category") or data.get("class") or "").lower()
            
#             # TRIGGER KEYWORDS
#             triggers = ["ocean", "sea", "lake", "river", "sagar", "reservoir", "water", "bay", "gulf"]
#             if any(t in name for t in triggers) or cat in ["natural", "water", "waterway"]:
#                 return True, {"source": f"scale_z{zoom}", "name": data.get("display_name")}
#         except: continue
#     return False, None

# def haversine_km(lat1, lon1, lat2, lon2):
#     R = 6371.0
#     phi1, phi2, dphi, dlamb = radians(lat1), radians(lat2), radians(lat2-lat1), radians(lon2-lon1)
#     a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlamb/2)**2
#     return R * (2 * atan2(sqrt(a), sqrt(1-a)))

# def estimate_water_proximity_score(latitude: float, longitude: float) -> Tuple[float, Optional[float], Optional[dict]]:
#     """Strictly prioritized water detection."""
    
#     # 1. Named/Local Water Check (Highest Accuracy)
#     # We run the actual MAP search first so land doesn't get flagged by accident
#     found, details = _multi_scale_search(latitude, longitude)
#     if found:
#         return 0.0, 0.0, details

#     # 2. Hardcoded Bounding Boxes (Deep Water Fail-Safe)
#     # Only runs if the maps didn't return a clear result.
#     is_ocean, ocean_name = _is_in_hardcoded_ocean(latitude, longitude)
#     if is_ocean:
#         return 0.0, 0.0, {"source": "geo_bounds", "name": ocean_name}

#     # 3. Proximity Search (Nearby feature nodes)
#     # for rad in [1000, 5000]:
#     #     try:
#     #         query = f'[out:json][timeout:10];node["natural"="water"](around:{rad},{latitude},{longitude});out center 1;'
#     #         resp = requests.post(OVERPASS_URLS[0], data={"data": query}, headers=_HEADERS, timeout=8)
#     #         elements = resp.json().get("elements")
#     #         if elements:
#     #             el = elements[0]
#     #             dist = haversine_km(latitude, longitude, el.get("lat"), el.get("lon"))
#     #             if dist < 0.05: return 0.0, 0.0, {"source": "overpass", "name": "On Water"}
                
#     #             # Dynamic scoring
#     #             if dist < 0.5: score = 30.0
#     #             elif dist < 1.0: score = 60.0
#     #             elif dist < 3.0: score = 80.0
#     #             else: score = 90.0
#     #             return score, round(dist, 3), {"source": "overpass", "name": "Nearby Water"}
#     #     except: continue

#     # 3. ADVANCED PROXIMITY SCAN (Nodes, Ways, and Areas)
#     # We increase the search intensity to catch large river banks
#     for rad in [1000, 3000, 5000]:
#         try:
#             # Expanded query to include 'way' and 'rel' for major river polygons
#             query = f"""
#             [out:json][timeout:15];
#             (
#               node["natural"="water"](around:{rad},{latitude},{longitude});
#               way["waterway"](around:{rad},{latitude},{longitude});
#               relation["natural"="water"](around:{rad},{latitude},{longitude});
#             );
#             out center 1;
#             """
#             resp = requests.post(OVERPASS_URLS[0], data={"data": query}, headers=_HEADERS, timeout=12)
#             elements = resp.json().get("elements")
            
#             if elements:
#                 el = elements[0]
#                 e_lat = el.get("lat") or el.get("center", {}).get("lat")
#                 e_lon = el.get("lon") or el.get("center", {}).get("lon")
#                 dist = haversine_km(latitude, longitude, e_lat, e_lon)
                
#                 # REFINED SCORING FOR NEAR-RIVER AREAS
#                 if dist < 0.3: score = 15.0   # Extremely high flood risk (Banks)
#                 elif dist < 0.8: score = 35.0 # High risk (Near river)
#                 elif dist < 1.5: score = 55.0 # Moderate (Buffer zone)
#                 elif dist < 3.0: score = 75.0 # Safe
#                 else: score = 90.0            # Optimal
                
#                 return score, round(dist, 3), {"source": "overpass_poly", "name": "Nearby Waterway Area"}
#         except: continue

#     # 4. Fallback for clear land
#     return 50.0, None, {"source": "verified_land", "note": "Safe terrestrial location."}




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