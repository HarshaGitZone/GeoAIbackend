# """
# PyLandslide Adapter: Now with Google Elevation for high-res slope (~3m vs. 1km).
# Get free key: console.cloud.google.com/apis/library/elevation-backend.googleapis.com
# Fallback to Open-Meteo if no key.
# """

# from typing import Optional
# import requests
# import json
# import math

# def get_elevation(lat: float, lon: float, google_key: Optional[str] = None) -> Optional[float]:
#     """Primary: Google (high-res); fallback Open-Meteo."""
#     if google_key:
#         url = "https://maps.googleapis.com/maps/api/elevation/json"
#         params = {'locations': f"{lat},{lon}", 'key': google_key}
#         try:
#             resp = requests.get(url, params=params, timeout=5)
#             data = resp.json()
#             if data['status'] == 'OK':
#                 return data['results'][0]['elevation']
#         except requests.RequestException:
#             pass
#     url = "https://api.open-meteo.com/v1/elevation"
#     params = {'latitude': lat, 'longitude': lon, 'format': 'json'}
#     try:
#         resp = requests.get(url, params=params, timeout=5)
#         resp.raise_for_status()
#         elevation_list = resp.json().get('elevation')
#         return float(elevation_list[0]) if elevation_list else None
#     except (requests.RequestException, IndexError, ValueError):
#         return None

# def estimate_slope(lat: float, lon: float, google_key: Optional[str] = None) -> float:
#     """High-res approx with smaller span (0.001° ~111m)."""
#     delta = 0.001
#     points = [(lat, lon), (lat + delta, lon), (lat, lon + delta), (lat - delta, lon), (lat, lon - delta)]
#     elevations = [get_elevation(p_lat, p_lon, google_key) for p_lat, p_lon in points]
#     elevations = [e for e in elevations if e is not None]
#     if len(elevations) < 2:
#         return 0.0
#     center_elev = elevations[0]
#     dist_m = delta * 111000
#     deltas = [abs(e - center_elev) / dist_m for e in elevations[1:]]
#     avg_gradient = sum(deltas) / len(deltas)
#     return round(avg_gradient * 100, 2)

# def estimate_landslide_risk_score(latitude: float, longitude: float, api_key: Optional[str] = None, google_key: Optional[str] = None) -> Optional[float]:
#     delta_bbox = 0.2
#     bbox = f"{longitude - delta_bbox},{latitude - delta_bbox},{longitude + delta_bbox},{latitude + delta_bbox}"
#     url = "https://eonet.gsfc.nasa.gov/api/v3/events"
#     params = {'category': 'landslides', 'bbox': bbox, 'days': 3650, 'limit': 50}
#     num_events = 0
#     try:
#         resp = requests.get(url, params=params, timeout=10)
#         resp.raise_for_status()
#         events = [e for e in resp.json().get('events', []) if e.get('geometry')]
#         num_events = len(events)
#         recent = sum(1 for e in events if int(e.get('date', '2025')[:4]) >= 2023)
#         event_penalty = min((num_events * 8) + (recent * 4), 40)
#     except requests.RequestException:
#         event_penalty = 0
#     score = 85 - event_penalty
#     if num_events == 0:
#         slope = estimate_slope(latitude, longitude, google_key)
#         slope_penalty = min(slope * 2.5, 60)  
#         score = max(0, 85 - slope_penalty)
#     return float(score)


# import requests
# import math
# from typing import Optional
# # Import the water detection logic
# from backend.integrations.water_adapter import estimate_water_proximity_score

# def get_elevations_batch(points: list, google_key: Optional[str] = None) -> list:
#     """Fetches multiple elevations in one go to save time and API limits."""
#     if google_key:
#         locations = "|".join([f"{p[0]},{p[1]}" for p in points])
#         url = "https://maps.googleapis.com/maps/api/elevation/json"
#         try:
#             resp = requests.get(url, params={'locations': locations, 'key': google_key}, timeout=5)
#             data = resp.json()
#             if data['status'] == 'OK':
#                 return [r['elevation'] for r in data['results']]
#         except: pass
    
#     # Fallback: Open-Meteo Batch
#     lats = ",".join([str(p[0]) for p in points])
#     lons = ",".join([str(p[1]) for p in points])
#     url = f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}"
#     try:
#         resp = requests.get(url, timeout=5)
#         return resp.json().get('elevation', [])
#     except:
#         return []

# def estimate_slope(lat: float, lon: float, google_key: Optional[str] = None) -> float:
#     """Approximates slope gradient (%) using 5-point sampling."""
#     delta = 0.001
#     points = [
#         (lat, lon),             # Center
#         (lat + delta, lon),     # North
#         (lat, lon + delta),     # East
#         (lat - delta, lon),     # South
#         (lat, lon - delta)      # West
#     ]
    
#     elevations = get_elevations_batch(points, google_key)
#     if len(elevations) < 2:
#         return 0.0
    
#     center_elev = elevations[0]
#     dist_m = delta * 111000  # Approx meters per degree
    
#     # Calculate gradients relative to center
#     gradients = [abs(e - center_elev) / dist_m for e in elevations[1:]]
#     avg_gradient = sum(gradients) / len(gradients)
    
#     return round(avg_gradient * 100, 2)

# def estimate_landslide_risk_score(latitude: float, longitude: float, google_key: Optional[str] = None) -> Optional[float]:
#     """
#     Returns a safety score (0-100). Higher = Safer.
#     FORCES 0.0 if location is a water body.
#     """
#     # 1. KILLER FILTER: Check water first
#     w_score, w_dist, _ = estimate_water_proximity_score(latitude, longitude)
#     if w_score == 0.0 or (w_dist is not None and w_dist < 0.02):
#         return 0.0

#     # 2. Historical Event Check (NASA EONET)
#     delta_bbox = 0.2
#     bbox = f"{longitude - delta_bbox},{latitude - delta_bbox},{longitude + delta_bbox},{latitude + delta_bbox}"
#     url = "https://eonet.gsfc.nasa.gov/api/v3/events"
#     params = {'category': 'landslides', 'bbox': bbox, 'days': 3650, 'limit': 20}
    
#     event_penalty = 0
#     try:
#         resp = requests.get(url, params=params, timeout=8)
#         if resp.status_code == 200:
#             events = resp.json().get('events', [])
#             num_events = len(events)
#             recent = sum(1 for e in events if "2023" in e.get('geometry', [{}])[0].get('date', ''))
#             event_penalty = min((num_events * 8) + (recent * 5), 45)
#     except: pass

#     # 3. Slope-based Risk (If no major history)
#     slope = estimate_slope(latitude, longitude, google_key)
#     # Heuristic: 0% slope = 0 penalty, 30% slope = 60 penalty (max)
#     slope_penalty = min(slope * 2.0, 60)
    
#     # Start at 90 (very safe) and subtract penalties
#     final_score = max(0, 90 - event_penalty - slope_penalty)
    
#     return float(round(final_score, 2))


import requests
import math
from typing import Optional
# Import the water detection logic to ensure 100% synchronization
from backend.integrations.water_adapter import estimate_water_proximity_score

def get_elevations_batch(points: list, google_key: Optional[str] = None) -> list:
    """
    Fetches multiple elevations in one call to optimize speed and API limits.
    Uses Google Elevation API as primary and Open-Meteo as fallback.
    """
    if google_key:
        locations = "|".join([f"{p[0]},{p[1]}" for p in points])
        url = "https://maps.googleapis.com/maps/api/elevation/json"
        try:
            resp = requests.get(url, params={'locations': locations, 'key': google_key}, timeout=5)
            data = resp.json()
            if data['status'] == 'OK':
                return [r['elevation'] for r in data['results']]
        except: pass
    
    # Fallback: Open-Meteo Batch API
    lats = ",".join([str(p[0]) for p in points])
    lons = ",".join([str(p[1]) for p in points])
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}"
    try:
        resp = requests.get(url, timeout=5)
        return resp.json().get('elevation', [])
    except:
        return []

def estimate_slope(lat: float, lon: float, google_key: Optional[str] = None) -> float:
    """
    Approximates slope gradient (%) using a 5-point sampling grid (~111m span).
    """
    delta = 0.001
    points = [
        (lat, lon),             # Center point
        (lat + delta, lon),     # North
        (lat, lon + delta),     # East
        (lat - delta, lon),     # South
        (lat, lon - delta)      # West
    ]
    
    elevations = get_elevations_batch(points, google_key)
    if len(elevations) < 2:
        return 0.0
    
    center_elev = elevations[0]
    dist_m = delta * 111000  # Conversion for roughly 111 meters per 0.001 degree
    
    # Calculate gradients relative to the center coordinate
    gradients = [abs(e - center_elev) / dist_m for e in elevations[1:]]
    avg_gradient = sum(gradients) / len(gradients)
    
    # Return as percentage (e.g., 0.15 gradient = 15%)
    return round(avg_gradient * 100, 2)

def estimate_landslide_risk_score(latitude: float, longitude: float, google_key: Optional[str] = None) -> Optional[float]:
    """
    Returns a landslide safety score (0-100). Higher = Safer/Less Risk.
    
    STRICT WATER LOGIC:
    If the location is a water body, construction safety is 0.0.
    """
    # 1. KILLER FILTER: Check water detection first
    # This prevents 'Safe' scores from appearing in the middle of the Ocean/Sea.
    w_score, w_dist, _ = estimate_water_proximity_score(latitude, longitude)
    
    # If the location is water (score 0) or extremely close (<20m)
    if w_score == 0.0 or (w_dist is not None and w_dist < 0.02):
        return 0.0

    # 2. Historical Event Penalty (NASA EONET API)
    # Checks for reported landslide events within a 0.2-degree bounding box over 10 years.
    delta_bbox = 0.2
    bbox = f"{longitude - delta_bbox},{latitude - delta_bbox},{longitude + delta_bbox},{latitude + delta_bbox}"
    url = "https://eonet.gsfc.nasa.gov/api/v3/events"
    params = {'category': 'landslides', 'bbox': bbox, 'days': 3650, 'limit': 20}
    
    event_penalty = 0
    try:
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code == 200:
            events = resp.json().get('events', [])
            num_events = len(events)
            # Higher penalty for recent events (recorded since 2023)
            recent = sum(1 for e in events if "2023" in str(e.get('geometry', [{}])[0].get('date', '')))
            event_penalty = min((num_events * 8) + (recent * 5), 45)
    except: pass

    # 3. Slope-based Penalty (Calculated if location is confirmed land)
    slope = estimate_slope(latitude, longitude, google_key)
    # Heuristic: 0% slope = 0 penalty; 30%+ slope = 60 penalty (max)
    slope_penalty = min(slope * 2.0, 60)
    
    # Base Safety starts at 90 (High Safety)
    final_score = max(0, 90 - event_penalty - slope_penalty)
    
    return float(round(final_score, 2))