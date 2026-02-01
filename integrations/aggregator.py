# backend/suitability_factors/aggregator.py
from typing import Dict, Any, Optional

class Aggregator:
    @staticmethod
    def _normalize(val: Any, default: float = 50.0) -> float:
        """Strictly ensures every factor is a 0.0 - 100.0 float."""
        if val is None:
            return default
        try:
            # Check if the value is a dictionary (new structure) or raw number (fallback)
            if isinstance(val, dict):
                # Priority order for keys returned by your 15 adapters
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
        Processes 15 factors across 5 categories averaged into a 100% total.
        Applies Strategic Weightage: Phys(30%), Hydro(25%), Env(15%), Clim(15%), Socio(15%).
        """
        raw = package.get("raw_factors", {})
        
        # --- LEVEL 1: SUB-FACTOR AVERAGING (Each factor scored out of 100) ---
        
        # ⛰️ PHYSICAL (3 Factors) - Averaged out of 100
        p = raw.get("physical", {})
        cat_physical = (
            cls._normalize(p.get("slope")) + 
            cls._normalize(p.get("elevation")) + 
            cls._normalize(p.get("ruggedness"))
        ) / 3

        # 🌿 ENVIRONMENTAL (3 Factors) - Averaged out of 100
        e = raw.get("environmental", {})
        veg_data = e.get("vegetation", {})
        # Safety check for potential 'ndvi_index' or 'value' naming
        ndvi_val = veg_data.get("ndvi_index") or veg_data.get("value") or 0.5
        
        cat_environmental = (
            cls._normalize(ndvi_val * 100) + 
            cls._normalize(e.get("soil")) + 
            cls._normalize(e.get("pollution"))
        ) / 3

        # 💧 HYDROLOGY (3 Factors) - Averaged out of 100
        h = raw.get("hydrology", {})
        cat_hydrology = (
            cls._normalize(h.get("water")) + 
            cls._normalize(h.get("drainage")) +
            cls._normalize(h.get("flood"))
        ) / 3
        # Extract flood safety specifically for the master penalty logic
        flood_safety = cls._normalize(h.get("flood"))

        # 🌤️ CLIMATIC (3 Factors) - Averaged out of 100
        c = raw.get("climatic", {})
        cat_climatic = (
            cls._normalize(c.get("rainfall")) + 
            cls._normalize(c.get("thermal")) +
            cls._normalize(c.get("intensity"))
        ) / 3

        # 🏗️ SOCIO-ECONOMIC (3 Factors) - Averaged out of 100
        s = raw.get("socio_econ", {})
        landuse_val = cls._normalize(s.get("landuse"))
        cat_socio = (
            cls._normalize(s.get("infrastructure")) + 
            landuse_val + 
            cls._normalize(s.get("population"))
        ) / 3

        # --- LEVEL 2: STRATEGIC CATEGORY WEIGHTING (Total 100%) ---
        
        # Adjusted weights to prioritize Safety and Physical constraints
        weights = {
            "physical": 0.30, 
            "hydrology": 0.25, 
            "environmental": 0.15, 
            "climatic": 0.15, 
            "socio_econ": 0.15
        }
        
        base_score = (
            (cat_physical * weights["physical"]) + 
            (cat_environmental * weights["environmental"]) + 
            (cat_hydrology * weights["hydrology"]) + 
            (cat_climatic * weights["climatic"]) + 
            (cat_socio * weights["socio_econ"])
        )

        # 🚨 LEVEL 3: MASTER PENALTY LOGIC (Hard Overrides)
        final_score = base_score
        penalty_note = "None"
        is_hard_unsuitable = False

        # Water Body Detection (Total Zero Override)
        if cls._normalize(h.get("water")) <= 5:
            final_score = 0.0
            penalty_note = "Non-Terrestrial (Open Water)"
            is_hard_unsuitable = True

        # Flood Hazard Penalty (Safety Override)
        if flood_safety < 35:
            final_score *= 0.6
            penalty_note = "High Flood Inundation Hazard"

        # Forest/Protected Area Safety Clamp (Conservation Override)
        if landuse_val <= 20:
            final_score = min(final_score, 20.0)
            penalty_note = "Protected Environmental Zone"

        # --- LEGACY COMPATIBILITY MAPPING ---
        # Flattening structure for app.py strategic intelligence helpers
        flat_factors = {
            "rainfall": cls._normalize(c.get("rainfall")),
            "flood": flood_safety,
            "landslide": cls._normalize(p.get("slope")), 
            "soil": cls._normalize(e.get("soil")),
            "proximity": cls._normalize(s.get("infrastructure")),
            "water": cls._normalize(h.get("water")),
            "pollution": cls._normalize(e.get("pollution")),
            "landuse": landuse_val
        }

        return {
            "score": round(final_score, 1),
            "label": "Optimal" if final_score > 75 else "Viable" if final_score > 45 else "Restricted",
            "is_hard_unsuitable": is_hard_unsuitable,
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