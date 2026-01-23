import datetime as _dt
from typing import Optional, Tuple
import requests

# Standard headers to identify the application to Open-Meteo
_HEADERS = {
    "User-Agent": "GeoAI/1.0 (contact: support@example.com)",
    "Accept": "application/json",
}

def _daterange_days(days: int) -> Tuple[str, str]:
    """Helper to return start and end dates in ISO format."""
    end = _dt.date.today()
    start = end - _dt.timedelta(days=days)
    return start.isoformat(), end.isoformat()

def _fetch_open_meteo_sum(lat: float, lon: float, days: int = 60) -> Optional[float]:
    """
    Internal helper to fetch cumulative precipitation from Open-Meteo Archive API.
    Note: The Archive API usually has a 2-day delay for the most recent data.
    """
    start, end = _daterange_days(days)
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
        "&daily=precipitation_sum&timezone=auto"
    )
    try:
        # Increased timeout to 20s as API can be slow for large date ranges
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json() or {}
        
        daily = (data.get("daily") or {})
        values = daily.get("precipitation_sum") or []
        
        if not values:
            return None
            
        # Sum only non-None values to avoid errors
        total_mm = float(sum(v for v in values if v is not None))
        return total_mm
    except Exception as e:
        # Log the error if necessary and return None for fallback handling
        return None

def estimate_rainfall_score(latitude: float, longitude: float) -> Tuple[float, Optional[float]]:
    """
    Returns (suitability_score_0_100, total_mm_60d).
    Scoring heuristic: Lower rainfall is generally safer for construction/flood prevention.
      - > 800 mm in 60 days => 20 (High Risk)
      - 400–800 mm => 40
      - 100–400 mm => 70
      - < 100 mm => 85 (Ideal)
    """
    total_mm = _fetch_open_meteo_sum(latitude, longitude, 60)
    
    # If API fails, return neutral 50.0 score
    if total_mm is None:
        return 50.0, None
        
    if total_mm > 800:
        score = 20.0
    elif total_mm > 400:
        score = 40.0
    elif total_mm > 100:
        score = 70.0
    else:
        score = 85.0
        
    return score, round(total_mm, 1)

def get_rainfall_totals(latitude: float, longitude: float, days_list=(7, 30, 60)) -> dict:
    """
    Returns a dictionary mapping time windows to total rainfall in mm.
    Example: {7: 12.3, 30: 45.0, 60: 120.4}
    Used for frontend 'Evidence Details' and temporal summaries.
    """
    out = {}
    for d in days_list:
        try:
            tot = _fetch_open_meteo_sum(latitude, longitude, d)
            out[int(d)] = (round(tot, 1) if tot is not None else None)
        except Exception:
            out[int(d)] = None
    return out