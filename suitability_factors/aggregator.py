# backend/suitability_factors/aggregator.py
from typing import Dict, Any, Optional


def _slope_to_suitability(slope_data: Any) -> float:
    """Convert slope (percent or dict with value/scaled_score) to 0-100 suitability. Flat=100, steep=0."""
    if slope_data is None:
        return 50.0
    if isinstance(slope_data, dict):
        scaled = slope_data.get("scaled_score")
        if scaled is not None:
            return max(0.0, min(100.0, float(scaled)))
        pct = slope_data.get("value")
        if pct is not None:
            return max(0.0, min(100.0, 100.0 - float(pct) * 2.22))
    return 50.0


def _elevation_to_suitability(elev_m: float) -> float:
    """Convert elevation (m) to 0-100 suitability. Optimal band 200-600m; very low/high penalized."""
    try:
        m = float(elev_m)
        if m < 50:
            s = max(0, min(100, 50 + m))
        elif m < 200:
            s = 70 + (m - 50) / 7.5
        elif m < 600:
            s = 85 + (600 - m) / 40
        elif m < 1500:
            s = max(0, 85 - (m - 600) / 15)
        else:
            s = max(0, 30 - (m - 1500) / 100)
        return max(0.0, min(100.0, s))
    except (TypeError, ValueError):
        return 50.0


class Aggregator:
    @staticmethod
    def _normalize(val: Any, default: float = 50.0) -> float:
        """Safely convert and clamp values between 0 and 100."""
        if val is None:
            return default
        try:
            # Check if the value is a dictionary (new structure) or raw number (fallback)
            if isinstance(val, dict):
                v = val.get("value") or val.get("scaled_score") or val.get("suitability_score") or val.get("safety_score") or default
            else:
                v = val
            
            v_float = float(v)
            return max(0.0, min(100.0, v_float))
        except (ValueError, TypeError, AttributeError):
            return default

    @classmethod
    def compute_suitability_score(cls, package: Dict[str, Any]) -> Dict[str, Any]:
        """
        MASTER SCORING ENGINE
        Processes 15 factors across 5 categories.
        Enforces logical hard-stops for water bodies and protected zones.
        """
        raw = package.get("raw_factors", {})
        
        # --- 1. PHYSICAL (2 Factors: slope, elevation; ruggedness not in 15-factor set) ---
        p = raw.get("physical", {})
        slope_data = p.get("slope")
        slope_score = _slope_to_suitability(slope_data) if slope_data else 50.0
        elev_data = p.get("elevation", {})
        elev_val = elev_data.get("value") if isinstance(elev_data, dict) else elev_data
        elev_score = _elevation_to_suitability(elev_val) if elev_val is not None else 50.0
        cat_physical = (slope_score + elev_score) / 2

        # --- 2. ENVIRONMENTAL (3 Factors) ---
        e = raw.get("environmental", {})
        # FIX for KeyError: 'ndvi_index' - Checks multiple potential keys
        veg_data = e.get("vegetation", {})
        ndvi_val = veg_data.get("ndvi_index") or veg_data.get("value") or 0.5
        
        cat_environmental = (
            cls._normalize(ndvi_val * 100) + 
            cls._normalize(e.get("soil")) + 
            cls._normalize(e.get("pollution"))
        ) / 3

        # --- 3. HYDROLOGY (3 Factors) ---
        h = raw.get("hydrology", {})
        water_val = cls._normalize(h.get("water"))
        cat_hydrology = (
            water_val +
            cls._normalize(h.get("drainage"))
        ) / 2
        flood_safety = cls._normalize(h.get("flood"))

        # --- 4. CLIMATIC (2 Factors) ---
        c = raw.get("climatic", {})
        cat_climatic = (
            cls._normalize(c.get("rainfall")) +
            cls._normalize(c.get("thermal"))
        ) / 2

        # --- 5. SOCIO-ECONOMIC (3 Factors) ---
        s = raw.get("socio_econ", {})
        landuse_raw = s.get("landuse")
        landuse_val = cls._normalize(landuse_raw)
        landuse_class = landuse_raw.get("classification", "Unknown") if isinstance(landuse_raw, dict) else "Unknown"
        cat_socio = (
            cls._normalize(s.get("infrastructure")) +
            landuse_val +
            cls._normalize(s.get("population"))
        ) / 3

        # --- Water body detection (before aggregation) ---
        water_details = h.get("water", {}) if isinstance(h.get("water"), dict) else {}
        water_dist = water_details.get("distance_km")
        is_on_water = water_val <= 5 or (water_dist is not None and float(water_dist) < 0.02)
        water_body_name = None
        if is_on_water and isinstance(water_details.get("details"), dict):
            water_body_name = water_details.get("details", {}).get("name") or "identified water body"
        elif is_on_water:
            water_body_name = "identified water body"

        # When on water: physical contribution = 0 (flat on water = flood prone, not suitable)
        if is_on_water:
            cat_physical = 0.0
            cat_hydrology = 0.0
            flood_safety = 0.0

        # --- FINAL AGGREGATION ---
        weights = {"phys": 0.2, "env": 0.2, "hydro": 0.2, "clim": 0.2, "socio": 0.2}
        base_score = (
            (cat_physical * weights["phys"]) +
            (cat_environmental * weights["env"]) +
            (cat_hydrology * weights["hydro"]) +
            (cat_climatic * weights["clim"]) +
            (cat_socio * weights["socio"])
        )

        # 🚨 MASTER PENALTY LOGIC
        final_score = base_score
        penalty_note = "None"
        is_hard_unsuitable = False
        label = "Highly Suitable" if final_score > 75 else "Suitable" if final_score > 40 else "High Risk"
        water_body_snippet = None
        protected_snippet = None

        # Water Body — low score (not zero) so other factors still visible; label/snippet remain clear
        if is_on_water:
            final_score = min(final_score, 12.0)
            penalty_note = "Non-Terrestrial (Open Water)"
            is_hard_unsuitable = True
            label = "Not Suitable (Water Body)"
            water_body_snippet = water_body_name or "Open water"

        # Flood Hazard Multiplier (only when not on water)
        if not is_on_water and flood_safety < 40:
            final_score *= 0.5
            penalty_note = "High Flood Inundation Hazard"

        # Forest/Protected Area — immediate detail and low score
        if landuse_val <= 20:
            final_score = min(final_score, 20.0)
            penalty_note = "Protected Environmental Zone"
            if not is_on_water:
                label = "Not Suitable (Protected/Forest Area)"
                protected_snippet = landuse_class if landuse_class and landuse_class != "Unknown" else "Protected zone"

        flat_factors = {
            "rainfall": cls._normalize(c.get("rainfall")),
            "flood": flood_safety,
            "landslide": slope_score,
            "soil": cls._normalize(e.get("soil")),
            "proximity": cls._normalize(s.get("infrastructure")),
            "water": water_val,
            "pollution": cls._normalize(e.get("pollution")),
            "landuse": landuse_val
        }

        return {
            "score": round(final_score, 1),
            "label": label,
            "is_hard_unsuitable": is_hard_unsuitable,
            "water_body_snippet": water_body_snippet,
            "protected_snippet": protected_snippet,
            "category_scores": {
                "physical": round(cat_physical, 1),
                "environmental": round(cat_environmental, 1),
                "hydrology": round(cat_hydrology, 1),
                "climatic": round(cat_climatic, 1),
                "socio_econ": round(cat_socio, 1)
            },
            "factors": flat_factors,
            "penalty": penalty_note
        }