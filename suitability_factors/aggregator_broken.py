# backend/suitability_factors/aggregator.py
from typing import Dict, Any, Optional
import math


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
    def _normalize(val: Any, default: float = 50.0, factor_name: str = None, lat: float = None, lng: float = None) -> float:
        """Safely convert and clamp values between 0 and 100 with dynamic context-aware defaults."""
        if val is None:
            # Dynamic defaults based on geographic context
            if lat is not None and lng is not None:
                dynamic_default = Aggregator._get_dynamic_default(factor_name, lat, lng)
                if dynamic_default is not None:
                    return dynamic_default
            
            # Fallback to reasonable defaults
            fallback_defaults = {
                "pollution": 55.0,      # Moderate pollution baseline
                "infrastructure": 45.0,  # Developing infrastructure baseline
                "population": 40.0,     # Moderate population baseline
                "soil": 65.0,          # Good soil baseline
                "vegetation": 45.0,     # Moderate vegetation baseline
                "drainage": 55.0,       # Moderate drainage baseline
                "rainfall": 60.0,       # Moderate rainfall baseline
                "thermal": 65.0,        # Good thermal comfort baseline
                "water": 70.0,          # Good water access baseline
                "flood": 70.0,          # Moderate flood safety baseline
                "landuse": 60.0,        # Mixed landuse baseline
                "ruggedness": 60.0,     # Moderate terrain ruggedness baseline
                "stability": 65.0       # Moderate land stability baseline
            }
            return fallback_defaults.get(factor_name, default)
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
        
    @staticmethod
    def _get_dynamic_default(factor_name: str, lat: float, lng: float) -> float:
        """Calculate dynamic default values based on geographic context."""
        try:
            # Geographic context analysis
            is_urban = Aggregator._estimate_urban_density(lat, lng)
            is_coastal = Aggregator._is_coastal(lat, lng)
            climate_zone = Aggregator._get_climate_zone(lat)
            region = Aggregator._get_geographic_region(lat, lng)
            
            # Dynamic defaults based on context
            if factor_name == "pollution":
                return Aggregator._get_pollution_default(is_urban, region)
            elif factor_name == "infrastructure":
                return Aggregator._get_infrastructure_default(is_urban, region)
            elif factor_name == "population":
                return Aggregator._get_population_default(is_urban, lat, lng)
            elif factor_name == "soil":
                return Aggregator._get_soil_default(climate_zone, is_coastal)
            elif factor_name == "vegetation":
                return Aggregator._get_vegetation_default(is_urban, climate_zone)
            elif factor_name == "drainage":
                return Aggregator._get_drainage_default(is_coastal, climate_zone)
            elif factor_name == "rainfall":
                return Aggregator._get_rainfall_default(climate_zone, lat)
            elif factor_name == "thermal":
                return Aggregator._get_thermal_default(climate_zone, lat)
            elif factor_name == "water":
                return Aggregator._get_water_default(is_coastal, is_urban)
            elif factor_name == "flood":
                return Aggregator._get_flood_default(is_coastal, climate_zone)
            elif factor_name == "landuse":
                return Aggregator._get_landuse_default(is_urban, region)
            elif factor_name == "ruggedness":
                return Aggregator._get_ruggedness_default(is_urban, region, climate_zone)
            elif factor_name == "stability":
                return Aggregator._get_stability_default(is_urban, region, climate_zone)
            
        except Exception:
            # Fallback on any error
            return None
    
    @staticmethod
    def _estimate_urban_density(lat: float, lng: float) -> str:
        """Estimate if location is in urban area based on coordinates."""
        # Major urban centers approximation (simplified)
        urban_centers = [
            # India: Delhi, Mumbai, Bangalore, Chennai, Kolkata
            (28.6, 77.2, 0.5), (19.1, 72.9, 0.5), (12.9, 77.6, 0.5),
            (13.1, 80.3, 0.5), (22.6, 88.4, 0.5),
            # Other major world cities
            (40.7, -74.0, 0.3), (51.5, -0.1, 0.3), (35.7, 139.7, 0.3),
            (-33.9, 151.2, 0.3), (37.8, -122.4, 0.3)
        ]
        
        for city_lat, city_lng, radius in urban_centers:
            distance = math.sqrt((lat - city_lat)**2 + (lng - city_lng)**2)
            if distance <= radius:
                return "high"
        
        # Medium density areas (within 2 degrees of major cities)
        for city_lat, city_lng, _ in urban_centers:
            distance = math.sqrt((lat - city_lat)**2 + (lng - city_lng)**2)
            if distance <= 2.0:
                return "medium"
        
        return "low"
    
    @staticmethod
    def _is_coastal(lat: float, lng: float) -> bool:
        """Check if location is near coast (simplified)."""
        # This is a simplified check - in production, use actual coastline data
        coastal_lat_ranges = [
            (8, 25),    # Indian coast
            (22, 27),   # Bay of Bengal
            (24, 32),   # Mediterranean
            (35, 42),   # US East Coast
            (32, 38),   # US West Coast
        ]
        
        for min_lat, max_lat in coastal_lat_ranges:
            if min_lat <= lat <= max_lat:
                return True
        
        return False
    
    @staticmethod
    def _get_climate_zone(lat: float) -> str:
        """Determine climate zone based on latitude."""
        if abs(lat) <= 10:
            return "tropical"
        elif abs(lat) <= 23.5:
            return "subtropical"
        elif abs(lat) <= 35:
            return "temperate"
        elif abs(lat) <= 50:
            return "cool"
        else:
            return "polar"
    
    @staticmethod
    def _get_geographic_region(lat: float, lng: float) -> str:
        """Determine geographic region."""
        if 60 <= lat <= 80 and -10 <= lng <= 40:
            return "europe"
        elif 25 <= lat <= 50 and -130 <= lng <= -60:
            return "north_america"
        elif -55 <= lat <= 15 and -80 <= lng <= -35:
            return "south_america"
        elif -35 <= lat <= 37 and 10 <= lng <= 50:
            return "africa"
        elif 5 <= lat <= 50 and 60 <= lng <= 150:
            return "asia"
        elif -10 <= lat <= -45 and 110 <= lng <= 180:
            return "oceania"
        else:
            return "other"
    
    # Dynamic default calculation methods
    @staticmethod
    def _get_pollution_default(is_urban: str, region: str) -> float:
        """Dynamic pollution default based on urban density and region."""
        base_pollution = {
            "high": {"asia": 25.0, "europe": 35.0, "north_america": 40.0, "other": 30.0},
            "medium": {"asia": 45.0, "europe": 55.0, "north_america": 60.0, "other": 50.0},
            "low": {"asia": 65.0, "europe": 70.0, "north_america": 75.0, "other": 70.0}
        }
        return base_pollution.get(is_urban, {}).get(region, 50.0)
    
    @staticmethod
    def _get_infrastructure_default(is_urban: str, region: str) -> float:
        """Dynamic infrastructure default based on urban density and region."""
        base_infra = {
            "high": {"asia": 75.0, "europe": 85.0, "north_america": 80.0, "other": 70.0},
            "medium": {"asia": 50.0, "europe": 65.0, "north_america": 60.0, "other": 45.0},
            "low": {"asia": 25.0, "europe": 40.0, "north_america": 35.0, "other": 20.0}
        }
        return base_infra.get(is_urban, {}).get(region, 45.0)
    
    @staticmethod
    def _get_population_default(is_urban: str, lat: float, lng: float) -> float:
        """Dynamic population default based on urban density and location."""
        if is_urban == "high":
            return 75.0
        elif is_urban == "medium":
            return 50.0
        else:
            # Rural areas - adjust based on agricultural potential
            if abs(lat) <= 35:  # Temperate/tropical good for agriculture
                return 35.0
            else:
                return 20.0
    
    @staticmethod
    def _get_soil_default(climate_zone: str, is_coastal: bool) -> float:
        """Dynamic soil default based on climate and coastal proximity."""
        if is_coastal:
            return 60.0  # Coastal soils often sandy/moderate
        
        soil_by_climate = {
            "tropical": 75.0,    # Lateritic soils, good for agriculture
            "subtropical": 70.0,  # Good diverse soils
            "temperate": 80.0,    # Excellent agricultural soils
            "cool": 65.0,         # Moderate soils
            "polar": 40.0          # Poor thin soils
        }
        return soil_by_climate.get(climate_zone, 65.0)
    
    @staticmethod
    def _get_vegetation_default(is_urban: str, climate_zone: str) -> float:
        """Dynamic vegetation default based on urban density and climate."""
        if is_urban == "high":
            return 25.0  # Low vegetation in cities
        elif is_urban == "medium":
            return 45.0  # Moderate vegetation
        
        # Rural vegetation by climate
        veg_by_climate = {
            "tropical": 85.0,    # Dense rainforests
            "subtropical": 75.0,  # Good vegetation
            "temperate": 70.0,    # Mixed forests
            "cool": 60.0,         # Coniferous forests
            "polar": 30.0          # Tundra vegetation
        }
        return veg_by_climate.get(climate_zone, 60.0)
    
    @staticmethod
    def _get_drainage_default(is_coastal: bool, climate_zone: str) -> float:
        """Dynamic drainage default based on coastal and climate factors."""
        if is_coastal:
            return 50.0  # Coastal areas often have drainage challenges
        
        drainage_by_climate = {
            "tropical": 45.0,    # High rainfall, drainage challenges
            "subtropical": 55.0,  # Moderate drainage
            "temperate": 65.0,    # Good drainage
            "cool": 60.0,         # Moderate drainage
            "polar": 70.0          # Good drainage (frozen ground)
        }
        return drainage_by_climate.get(climate_zone, 60.0)
    
    @staticmethod
    def _get_rainfall_default(climate_zone: str, lat: float) -> float:
        """Dynamic rainfall default based on climate zone."""
        rainfall_by_climate = {
            "tropical": 80.0,    # High rainfall
            "subtropical": 70.0,  # Good rainfall
            "temperate": 65.0,    # Moderate rainfall
            "cool": 55.0,         # Lower rainfall
            "polar": 30.0          # Very low rainfall
        }
        return rainfall_by_climate.get(climate_zone, 60.0)
    
    @staticmethod
    def _get_thermal_default(climate_zone: str, lat: float) -> float:
        """Dynamic thermal comfort default based on climate."""
        thermal_by_climate = {
            "tropical": 40.0,    # Hot and humid
            "subtropical": 55.0,  # Warm
            "temperate": 75.0,    # Comfortable
            "cool": 70.0,         # Cool but comfortable
            "polar": 50.0          # Cold
        }
        return thermal_by_climate.get(climate_zone, 65.0)
    
    @staticmethod
    def _get_water_default(is_coastal: bool, is_urban: str) -> float:
        """Dynamic water access default."""
        if is_urban == "high":
            return 80.0  # Good water infrastructure in cities
        elif is_coastal:
            return 75.0  # Good water access near coasts
        else:
            return 60.0  # Moderate water access in rural areas
    
    @staticmethod
    def _get_flood_default(is_coastal: bool, climate_zone: str) -> float:
        """Dynamic flood safety default."""
        if is_coastal:
            return 50.0  # Higher flood risk in coastal areas
        
        flood_by_climate = {
            "tropical": 45.0,    # High flood risk
            "subtropical": 55.0,  # Moderate flood risk
            "temperate": 65.0,    # Lower flood risk
            "cool": 70.0,         # Low flood risk
            "polar": 80.0          # Very low flood risk
        }
        return flood_by_climate.get(climate_zone, 60.0)
    
    @staticmethod
    def _get_landuse_default(is_urban: str, region: str) -> float:
        """Dynamic landuse default based on urban density and region."""
        if is_urban == "high":
            return 25.0  # Mostly developed land
        elif is_urban == "medium":
            return 50.0  # Mixed land use
        else:
            return 75.0  # Mostly natural/agricultural land
    
    @staticmethod
    def _get_ruggedness_default(is_urban: str, region: str, climate_zone: str) -> float:
        """Dynamic terrain ruggedness default based on context."""
        if is_urban == "high":
            return 75.0  # Urban areas often have flatter, graded terrain
        elif is_urban == "medium":
            return 60.0  # Moderate terrain in suburban areas
        else:
            # Rural areas - vary by geography
            if climate_zone == "mountainous":
                return 30.0  # Mountainous areas are rugged
            elif climate_zone == "polar":
                return 45.0  # Polar regions can be rugged
            else:
                return 55.0  # Moderate rural terrain
    
    @staticmethod
    def _get_stability_default(is_urban: str, region: str, climate_zone: str) -> float:
        """Dynamic land stability default based on context."""
        if is_urban == "high":
            return 70.0  # Urban areas often have engineered, stable ground
        elif is_urban == "medium":
            return 65.0  # Suburban areas have moderate stability
        else:
            # Rural areas - vary by climate
            if climate_zone == "tropical":
                return 55.0  # Tropical areas can have erosion issues
            elif climate_zone == "polar":
                return 75.0  # Polar areas often stable (frozen)
            else:
                return 65.0  # Moderate rural stability
    
    @classmethod
    def compute_suitability_score(cls, package: Dict[str, Any]) -> Dict[str, Any]:
        """
        MASTER SCORING ENGINE
        Processes 15 factors across 5 categories.
        Enforces logical hard-stops for water bodies and protected zones.
        """
        raw = package.get("raw_factors", {})
        
        # Get coordinates for dynamic defaults
        lat = package.get("latitude")
        lng = package.get("longitude")
        
        # --- 1. PHYSICAL (4 Factors: slope, elevation, ruggedness, stability) ---
        p = raw.get("physical", {})
        slope_data = p.get("slope")
        slope_score = _slope_to_suitability(slope_data) if slope_data else 50.0
        elev_data = p.get("elevation", {})
        elev_val = elev_data.get("value") if isinstance(elev_data, dict) else elev_data
        elev_score = _elevation_to_suitability(elev_val) if elev_val is not None else 50.0
        ruggedness_score = cls._normalize(p.get("ruggedness"), 50.0, "ruggedness", lat, lng)
        stability_score = cls._normalize(p.get("stability"), 50.0, "stability", lat, lng)
        cat_physical = (slope_score + elev_score + ruggedness_score + stability_score) / 4

        # --- 2. ENVIRONMENTAL (5 Factors) ---
        e = raw.get("environmental", {})
        # FIX for KeyError: 'ndvi_index' - Checks multiple potential keys
        veg_data = e.get("vegetation", {})
        ndvi_val = veg_data.get("ndvi_index") or veg_data.get("value") or 0.5
        
        # Proper NDVI normalization (NDVI is typically 0-1, convert to 0-100)
        if ndvi_val is not None and 0 <= ndvi_val <= 1.0:
            vegetation_score = ndvi_val * 100  # Convert NDVI 0-1 to 0-100
        else:
            vegetation_score = cls._normalize(ndvi_val, 50.0, "vegetation", lat, lng)
        
        cat_environmental = (
            vegetation_score + 
            cls._normalize(e.get("soil"), 50.0, "soil", lat, lng) + 
            cls._normalize(e.get("pollution"), 50.0, "pollution", lat, lng) +
            cls._normalize(e.get("biodiversity"), 50.0, "biodiversity", lat, lng) +
            cls._normalize(e.get("heat_island"), 50.0, "heat_island", lat, lng)
        ) / 5

        # --- 3. HYDROLOGY (4 Factors) ---
        h = raw.get("hydrology", {})
        water_val = cls._normalize(h.get("water"), 50.0, "water", lat, lng)
        drainage_val = cls._normalize(h.get("drainage"), 50.0, "drainage", lat, lng)
        flood_val = cls._normalize(h.get("flood"), 50.0, "flood", lat, lng)
        groundwater_val = cls._normalize(h.get("groundwater"), 50.0, "groundwater", lat, lng)
        
        # Include all 4 hydrology factors in average
        cat_hydrology = (water_val + drainage_val + flood_val + groundwater_val) / 4
        flood_safety = flood_val  # Keep separate for penalty logic

        # --- 4. CLIMATIC (3 Factors) ---
        c = raw.get("climatic", {})
        cat_climatic = (

@staticmethod
def _get_drainage_default(is_coastal: bool, climate_zone: str) -> float:
    """Dynamic drainage default based on coastal and climate factors."""
    if is_coastal:
        return 50.0  # Coastal areas often have drainage challenges

    drainage_by_climate = {
        "tropical": 45.0,    # High rainfall, drainage challenges
        "subtropical": 55.0,  # Moderate drainage
        "temperate": 65.0,    # Good drainage
        "cool": 60.0,         # Moderate drainage
        "polar": 70.0          # Good drainage (frozen ground)
    }
    return drainage_by_climate.get(climate_zone, 60.0)
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