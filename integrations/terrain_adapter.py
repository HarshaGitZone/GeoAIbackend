import requests
import logging

logger = logging.getLogger(__name__)

OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
OFFSET = 0.0009  # ~100 meters

def _get_elevation(lat, lng):
    try:
        r = requests.get(
            OPEN_ELEVATION_URL,
            params={"locations": f"{lat},{lng}"},
            timeout=10
        )
        r.raise_for_status()
        return r.json()["results"][0]["elevation"]
    except Exception as e:
        logger.warning(f"Elevation fetch failed: {e}")
        return None


def estimate_terrain_slope(lat, lng):
    """
    Independent Terrain Analysis
    Returns:
    {
        slope_percent,
        verdict,
        confidence,
        source
    }
    """

    center_elev = _get_elevation(lat, lng)

    if center_elev is None:
        return {
            "slope_percent": None,
            "verdict": "Elevation data unavailable",
            "confidence": "Low",
            "source": "Open-Elevation API"
        }

    points = [
        (lat + OFFSET, lng),
        (lat - OFFSET, lng),
        (lat, lng + OFFSET),
        (lat, lng - OFFSET)
    ]

    slopes = []

    for p_lat, p_lng in points:
        elev = _get_elevation(p_lat, p_lng)
        if elev is not None:
            diff = abs(elev - center_elev)
            slope_percent = (diff / 100) * 100
            slopes.append(slope_percent)

    if not slopes:
        return {
            "slope_percent": None,
            "verdict": "Insufficient elevation data for slope calculation",
            "confidence": "Low",
            "source": "Open-Elevation API"
        }

    max_slope = round(max(slopes), 2)

    # Engineering interpretation
    if max_slope < 5:
        verdict = "Flat terrain. Ideal for construction."
        confidence = "High"
    elif max_slope < 10:
        verdict = "Gentle slope. Minor leveling required."
        confidence = "High"
    elif max_slope < 15:
        verdict = "Moderate slope. Structural reinforcement needed."
        confidence = "Medium"
    else:
        verdict = "Steep slope. Not suitable for construction."
        confidence = "High"

    return {
        "slope_percent": max_slope,
        "verdict": verdict,
        "confidence": confidence,
        "source": "Digital Elevation Model (Open-Elevation)"
    }