import requests
import math
from typing import Optional
# Import the water detection logic to ensure 100% synchronization
from integrations.water_adapter import estimate_water_proximity_score

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