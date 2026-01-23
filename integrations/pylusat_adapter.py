# from typing import Optional
# import time
# import requests

# _MIRRORS = [
# 	"https://overpass-api.de/api/interpreter",
# 	"https://overpass.kumi.systems/api/interpreter",
# 	"https://overpass.openstreetmap.ru/api/interpreter",
# ]

# _HEADERS = {
# 	"User-Agent": "GeoAI/1.0 (contact: support@example.com)",
# 	"Accept": "application/json",
# }

# def _build_roads_query(lat: float, lon: float, radius_m: int) -> str:
	
# 	return f"""
# 	[out:json][timeout:25];
# 	(
# 	  way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"](around:{radius_m},{lat},{lon});
# 	  node["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"](around:{radius_m},{lat},{lon});
# 	);
# 	out center 20;
# 	"""

# def _query_roads(lat: float, lon: float, radius_m: int) -> Optional[dict]:
# 	q = _build_roads_query(lat, lon, radius_m)
# 	last_err: Optional[Exception] = None
# 	for attempt in range(3):
# 		for base in _MIRRORS:
# 			try:
# 				resp = requests.post(base, data={"data": q}, headers=_HEADERS, timeout=15)
# 				if resp.status_code == 429:
# 					last_err = Exception("429 Too Many Requests")
# 					continue
# 				resp.raise_for_status()
# 				return resp.json()
# 			except Exception as e:
# 				last_err = e
# 				continue
# 		time.sleep(0.8 * (attempt + 1))
# 	return None

# def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
# 	from math import radians, sin, cos, sqrt, atan2
# 	R = 6371.0
# 	phi1, phi2 = radians(lat1), radians(lat2)
# 	dphi = radians(lat2 - lat1)
# 	dlambda = radians(lon2 - lon1)
# 	a = (sin(dphi / 2) ** 2) + cos(phi1) * cos(phi2) * (sin(dlambda / 2) ** 2)
# 	c = 2 * atan2(sqrt(a), sqrt(1 - a))
# 	return R * c

# def compute_proximity_score(latitude: float, longitude: float) -> Optional[float]:
# 	"""Estimate access proximity to major roads.

# 	Closer to major roads is considered better for access/markets.
# 	Returns a score in [0, 100].
# 	"""
# 	elements = None
# 	for radius in (1000, 3000, 6000):
# 		data = _query_roads(latitude, longitude, radius)
# 		if data and data.get("elements"):
# 			elements = data["elements"]
# 			break
# 	if not elements:
# 		return None

# 	min_km = None
# 	for el in elements:
# 		if "lat" in el and "lon" in el:
# 			d = _haversine_km(latitude, longitude, el["lat"], el["lon"])
# 		elif "center" in el and "lat" in el["center"] and "lon" in el["center"]:
# 			d = _haversine_km(latitude, longitude, el["center"]["lat"], el["center"]["lon"])
# 		else:
# 			continue
# 		min_km = d if min_km is None else min(min_km, d)
# 	if min_km is None:
# 		return None

# 	# Map distance to score (closer = better access)
# 	if min_km < 0.1:
# 		score = 92.0
# 	elif min_km < 0.3:
# 		score = 85.0
# 	elif min_km < 0.8:
# 		score = 70.0
# 	elif min_km < 2.0:
# 		score = 55.0
# 	else:
# 		score = 45.0
# 	return score




import time
import requests
from typing import Optional, Tuple, Dict
# Import the water detection logic for global synchronization
from backend.integrations.water_adapter import estimate_water_proximity_score

_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

_HEADERS = {
    "User-Agent": "GeoAI_Suitability_Engine/2.0 (contact: support@example.com)",
    "Accept": "application/json",
}

# Road class buffers: distances within which noise/pollution affects suitability
ROAD_CLASS_BUFFER_M = {
    "motorway": 500,      # 500m - major noise/pollution exposure
    "trunk": 400,         # 400m - high traffic
    "primary": 350,       # 350m - significant traffic
    "secondary": 250,     # 250m - moderate traffic
    "tertiary": 150,      # 150m - light to moderate traffic
    "residential": 100,   # 100m - local traffic
    "living_street": 50,  # 50m - minimal traffic
    "unclassified": 100,
    "service": 80,
    "road": 100,
}
DEFAULT_ROAD_BUFFER_M = 100  # Default 100m buffer

def _build_roads_query(lat: float, lon: float, radius_m: int) -> str:
    """Queries for major transport infrastructure to determine accessibility."""
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
                resp = requests.post(base, data={"data": query}, headers=_HEADERS, timeout=12)
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

def compute_proximity_score(latitude: float, longitude: float) -> Tuple[float, Optional[float], Optional[Dict]]:
    """
    Calculates proximity to major roads and returns score, distance, and details.
    ENFORCES 0.0 for water bodies.
    """
    # 1. KILLER FILTER: Check if the location is on water first
    # Proximity to roads is irrelevant if the site is in the Atlantic Ocean or a lake.
    w_score, w_dist, _ = estimate_water_proximity_score(latitude, longitude)
    if w_score == 0.0 or (w_dist is not None and w_dist < 0.02):
        return 0.0, 0.0, {"name": "N/A (Water Body)", "type": "water_override"}

    # 2. PROXIMITY SEARCH: Locate nearest major infrastructure
    elements = None
    search_radius = 0
    for radius in (1000, 3000, 7000):
        data = _query_roads(latitude, longitude, radius)
        if data and data.get("elements"):
            elements = data["elements"]
            search_radius = radius
            break
            
    if not elements:
        # Default low score for extremely remote land
        return 30.0, None, {"name": "Remote Area", "type": "no_major_roads_found"}

    # 3. PRECISION CALCULATION: Identify the single closest road
    min_km = 999.0
    closest_feature = {}

    for el in elements:
        # Get coordinates from node or way center
        e_lat = el.get("lat") or el.get("center", {}).get("lat")
        e_lon = el.get("lon") or el.get("center", {}).get("lon")
        
        if e_lat and e_lon:
            d = _haversine_km(latitude, longitude, e_lat, e_lon)
            if d < min_km:
                min_km = d
                closest_feature = el.get("tags", {})

    # 4. DETERMINE ROAD TYPE AND BUFFER
    road_type = closest_feature.get("highway", "unknown")
    road_buffer_km = ROAD_CLASS_BUFFER_M.get(road_type, DEFAULT_ROAD_BUFFER_M) / 1000.0  # Convert to km

    # 5. OPTIMAL ZONE SCORING LOGIC
    # Bell curve approach: Sweet spot at 0.3-1.5km gives highest scores
    # Too close (on road/in buffer): Lower score due to noise
    # Farther away: Still good initially, then declines as accessibility drops
    
    if min_km < road_buffer_km:
        # ZONE 1: TOO CLOSE (within noise buffer)
        # Score improves as you move away from the road
        penalty_ratio = min_km / road_buffer_km  # 0 to 1
        
        if road_type in ("motorway", "trunk"):
            score = 25.0 + (penalty_ratio * 20.0)  # 25-45 range (poor to acceptable)
            explanation = (
                f"Location is {round(min_km*1000, 0)}m from a {road_type} road (directly on/near road). "
                f"High noise and pollution impact. Severe environmental constraints requiring extensive mitigation."
            )
        elif road_type in ("primary", "secondary"):
            score = 35.0 + (penalty_ratio * 20.0)  # 35-55 range
            explanation = (
                f"Location is {round(min_km*1000, 0)}m from a {road_type} road (in noise buffer zone). "
                f"Moderate to high noise/pollution. Mitigation measures essential for habitability."
            )
        else:
            # Minor roads have less impact
            score = 45.0 + (penalty_ratio * 20.0)  # 45-65 range
            explanation = (
                f"Location is {round(min_km*1000, 0)}m from a {road_type} road (light traffic exposure). "
                f"Minimal noise impact. Land value moderate with good proximity advantage."
            )
    
    elif min_km < 1.5:
        # ZONE 2: OPTIMAL (Sweet spot for land value + low noise)
        # Best balance: accessible but not too noisy
        if road_type in ("motorway", "trunk"):
            score = 85.0
            explanation = (
                f"Location is {round(min_km, 2)}km from a major {road_type} road. "
                f"EXCELLENT for land value: Strong connectivity with negligible noise pollution. "
                f"Optimal distance for commercial/industrial development."
            )
        elif road_type in ("primary", "secondary"):
            score = 88.0
            explanation = (
                f"Location is {round(min_km, 2)}km from a {road_type} road. "
                f"PRIME LOCATION: Ideal balance of accessibility and environmental quality. "
                f"High land value with excellent road connectivity."
            )
        else:
            score = 82.0
            explanation = (
                f"Location is {round(min_km, 2)}km from nearby {road_type} road. "
                f"Good accessibility with strong land value potential and low noise impact."
            )
    
    elif min_km < 3.0:
        # ZONE 3: GOOD (Accessible but farther)
        # Accessibility decreases, value decreases
        score = 72.0
        explanation = (
            f"Location is {round(min_km, 2)}km from nearest road. "
            f"Good road connectivity with strong environmental quality. "
            f"Moderate land development potential with excellent living conditions."
        )
    
    elif min_km < 5.0:
        # ZONE 4: MODERATE (Limited access)
        # Accessibility significantly impacted
        score = 55.0
        explanation = (
            f"Location is {round(min_km, 2)}km from nearest road. "
            f"Limited road access reduces land value and development potential. "
            f"Excellent environmental quality but requires planned infrastructure."
        )
    
    else:
        # ZONE 5: REMOTE (Poor accessibility)
        # Very limited accessibility, poor land value
        score = 35.0
        explanation = (
            f"Location is {round(min_km, 2)}km from nearest road (remote area). "
            f"Severely limited accessibility impacts land development potential. "
            f"Poor infrastructure connectivity significantly reduces suitability."
        )

    # 6. DETAILED EVIDENCE RETURN
    details = {
        "nearest_road_name": closest_feature.get("name", "Unnamed Road"),
        "road_type": road_type,
        "distance_km": round(min_km, 3),
        "buffer_m": ROAD_CLASS_BUFFER_M.get(road_type, DEFAULT_ROAD_BUFFER_M),
        "in_buffer_zone": min_km < road_buffer_km,
        "search_radius_m": search_radius,
        "explanation": explanation
    }

    return float(round(score, 2)), float(round(min_km, 3)), details