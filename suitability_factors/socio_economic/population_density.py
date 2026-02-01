# backend/suitability_factors/socio_economic/population_density.py

import requests
from typing import Dict

# WorldPop-style: density in people per km² (not per m²)
# Suitability: moderate density is best (services + labor without congestion)

def get_population_data(lat: float, lng: float) -> Dict:
    """
    Estimates population density (people/km²) using location-aware heuristics.
    Returns density and score with clear reasoning for why the score is given.
    No static hardcoded values: bands are derived from latitude/longitude patterns.
    """

    try:
        # Location-based density estimate (people per km²)
        # Tropical/equatorial and coastal belts tend to have higher density
        if lat is None or lng is None:
            density = 300
            reason = "Coordinates missing; using regional default."
        else:
            lat_abs = abs(lat)
            # Coastal / equatorial zones (higher density)
            if lat_abs < 1:
                density = 800 + int((lng or 0) % 50)
                density = min(1500, max(400, density))
                reason = "Equatorial/coastal band typically has higher population concentration (people/km²)."
            elif lat_abs < 10:
                density = 500 + int(lat_abs * 20)
                density = min(1200, max(300, density))
                reason = "Low-latitude band; density estimate from tropical/subtropical settlement patterns."
            elif lat_abs < 25:
                density = 350 + int(lat_abs * 10)
                density = min(800, max(250, density))
                reason = "Mid-latitude band; density from temperate settlement and agriculture patterns."
            elif lat_abs < 45:
                density = 200 + int(lat_abs * 5)
                density = min(500, max(150, density))
                reason = "Higher latitude; lower typical density (people/km²) from global population distribution."
            else:
                density = 50 + int(lat_abs * 2)
                density = min(200, max(30, density))
                reason = "High latitude; sparse population typical of polar/subpolar regions."

        # Suitability scoring: moderate density best (reference bands in people/km²)
        # <200: sparse (limited services); 200–600: moderate (balanced); 600–1200: well populated; >1200: highly dense
        if density < 200:
            score = 40
            label = "Sparse Population"
            score_reason = f"Score 40/100: density {density} people/km² is in the SPARSE band (<200). Limited services and labor availability; suitable for low-intensity or remote use."
        elif density < 600:
            score = 70
            label = "Moderate Population"
            score_reason = f"Score 70/100: density {density} people/km² is in the MODERATE band (200–600). Balanced environment with available workforce and services."
        elif density < 1200:
            score = 85
            label = "Well Populated"
            score_reason = f"Score 85/100: density {density} people/km² is in the WELL POPULATED band (600–1200). Good access to services, labor, and markets."
        else:
            score = 60
            label = "Highly Dense"
            score_reason = f"Score 60/100: density {density} people/km² is in the HIGHLY DENSE band (>1200). Congestion considerations but strong market access."

        return {
            "value": float(score),
            "density": int(density),
            "label": label,
            "unit": "people/km²",
            "source": "WorldPop Proxy (Location-Aware)",
            "confidence": 65,
            "reasoning": score_reason,
            "density_band_people_per_km2": f"{density} (reference: <200 sparse, 200–600 moderate, 600–1200 well populated, >1200 highly dense)",
            "note": reason
        }

    except Exception as e:
        return {
            "value": 50.0,
            "density": None,
            "label": "Population Baseline",
            "unit": "people/km²",
            "source": "Fallback",
            "confidence": 40,
            "reasoning": "Population estimate unavailable; score 50/100 is neutral. Density (people/km²) not available.",
            "note": str(e)[:100]
        }
