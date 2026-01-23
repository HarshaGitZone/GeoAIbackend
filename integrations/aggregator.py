from typing import Dict, Optional

def _normalize_optional(value: Optional[float], default: float) -> float:
    """Safely convert and clamp values between 0 and 100."""
    if value is None:
        return default
    try:
        v = float(value)
        return max(0.0, min(100.0, v))
    except (ValueError, TypeError):
        return default

def compute_suitability_score(
    *,
    rainfall_score: Optional[float],
    flood_risk_score: Optional[float],
    landslide_risk_score: Optional[float],
    soil_quality_score: Optional[float],
    proximity_score: Optional[float],
    water_proximity_score: Optional[float] = None,
    pollution_score: Optional[float] = None,
    landuse_score: Optional[float] = None,
) -> Dict:
    """
    Combines factors into a score. 
    ENFORCES STRICT 0.0 for all terrestrial factors if on a water body.
    """
    
    factors = {
        "rainfall": _normalize_optional(rainfall_score, 50.0),
        "flood": _normalize_optional(flood_risk_score, 50.0),
        "landslide": _normalize_optional(landslide_risk_score, 50.0),
        "soil": _normalize_optional(soil_quality_score, 50.0),
        "proximity": _normalize_optional(proximity_score, 50.0),
        "water": _normalize_optional(water_proximity_score, 50.0),
        "pollution": _normalize_optional(pollution_score, 50.0),
        "landuse": _normalize_optional(landuse_score, 50.0),
    }

    
    is_hard_unsuitable = factors["water"] == 0
    

    if is_hard_unsuitable:
        
        for key in factors:
            factors[key] = 0.0
        score = 0.0
    else:
        weights = {
            "rainfall": 0.10, "flood": 0.16, "landslide": 0.10, "soil": 0.16,
            "proximity": 0.10, "water": 0.18, "pollution": 0.10, "landuse": 0.10,
        }
        score = round(sum(factors[k] * weights[k] for k in weights), 2)
        
        # 🚨 SAFETY CLAMP: Forest / Protected land is NON-BUILDABLE
        # If landuse_score indicates forest/protected (<=20), clamp total score
        if landuse_score is not None:
            try:
                lu_raw = float(landuse_score)
            except (ValueError, TypeError):
                lu_raw = None

            if lu_raw is not None and lu_raw <= 20:
                score = min(score, 20.0)
    return {"score": score, "factors": factors, "is_hard_unsuitable": is_hard_unsuitable}