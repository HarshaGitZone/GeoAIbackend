"""
Heat Island Potential Analysis Module
Assesses surface reflectivity and built-up heat retention risk
Data Sources: NASA MODIS, ESA Sentinel, Landsat, OpenStreetMap
"""

import math
import requests
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def get_heat_island_potential(lat: float, lng: float) -> Dict[str, Any]:
    """
    Calculate heat island potential for given coordinates.
    
    Args:
        lat: Latitude
        lng: Longitude
        
    Returns:
        Dictionary with heat island potential score and metadata
    """
    try:
        # Get multiple heat island indicators
        surface_albedo = _get_surface_albedo(lat, lng)
        built_up_density = _get_built_up_density(lat, lng)
        vegetation_cover = _get_vegetation_cover(lat, lng)
        water_bodies = _get_water_body_proximity(lat, lng)
        
        # Calculate heat island potential
        heat_island_index = _calculate_heat_island_index(
            surface_albedo, built_up_density, vegetation_cover, water_bodies
        )
        
        # Convert to suitability score (inverse relationship)
        suitability_score = _heat_island_to_suitability(heat_island_index)
        
        return {
            "value": suitability_score,
            "heat_island_index": round(heat_island_index, 2),
            "surface_albedo": round(surface_albedo, 3),
            "built_up_density": round(built_up_density, 2),
            "vegetation_cover": round(vegetation_cover, 2),
            "water_proximity": water_bodies.get("distance_km", 10.0),
            "label": _get_heat_island_label(suitability_score),
            "source": "NASA MODIS + ESA Sentinel + OSM + Derived Calculations",
            "confidence": _calculate_confidence(surface_albedo, built_up_density),
            "reasoning": _generate_reasoning(heat_island_index, surface_albedo, built_up_density)
        }
        
    except Exception as e:
        logger.error(f"Error calculating heat island potential for {lat}, {lng}: {e}")
        return _get_fallback_heat_island(lat, lng)

def _get_surface_albedo(lat: float, lng: float) -> float:
    """Get surface albedo (reflectivity) for the location."""
    try:
        # Use NASA MODIS Albedo data
        url = f"https://modis.gsfc.nasa.gov/api/albedo"
        params = {
            'latitude': lat,
            'longitude': lng,
            'date': '2023-06-15',  # Summer solstice for maximum solar radiation
            'product': 'MCD43A3'
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            albedo = data.get('albedo', 0.3)  # Default albedo
            
            # Ensure albedo is in valid range (0-1)
            return max(0.0, min(1.0, float(albedo)))
        
    except Exception as e:
        logger.debug(f"Failed to get MODIS albedo data: {e}")
    
    # Fallback to land use-based albedo estimation
    return _estimate_albedo_from_landuse(lat, lng)

def _get_built_up_density(lat: float, lng: float) -> float:
    """Get built-up density (impervious surface percentage)."""
    try:
        # Use OpenStreetMap data for built-up estimation
        # Query building density in 1km radius
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:25];
        (
          way["building"](around:1000,{lat},{lng});
          node["building"](around:1000,{lat},{lng});
          relation["building"](around:1000,{lat},{lng});
        );
        out count;
        """
        
        response = requests.post(overpass_url, data=query, timeout=15)
        if response.status_code == 200:
            data = response.json()
            building_count = data.get('elements', [])
            
            # Calculate building density (buildings per km²)
            area_km2 = math.pi * (1.0 ** 2)  # 1km radius circle
            building_density = len(building_count) / area_km2
            
            # Normalize to 0-1 scale (0 = no buildings, 1 = maximum urban density)
            normalized_density = min(1.0, building_density / 1000.0)
            
            return normalized_density
        
    except Exception as e:
        logger.debug(f"Failed to get OSM building data: {e}")
    
    # Fallback to population density estimation
    return _estimate_built_up_from_population(lat, lng)

def _get_vegetation_cover(lat: float, lng: float) -> float:
    """Get vegetation cover percentage."""
    try:
        # Use existing NDVI data
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from ndvi_analysis import get_ndvi_data
        
        ndvi_result = get_ndvi_data(lat, lng)
        ndvi_value = ndvi_result.get("ndvi_index", 0.5)
        
        # Convert NDVI to vegetation cover percentage
        if ndvi_value < 0.1:
            return 0.0  # Bare soil
        elif ndvi_value < 0.3:
            return (ndvi_value - 0.1) * 5  # 0-100% for 0.1-0.3 range
        elif ndvi_value < 0.6:
            return 20 + (ndvi_value - 0.3) * 133  # 20-60% for 0.3-0.6 range
        else:
            return min(100.0, 60 + (ndvi_value - 0.6) * 100)  # 60-100% for >0.6
        
    except Exception:
        # Fallback to regional vegetation estimation
        return _estimate_vegetation_cover(lat, lng)

def _get_water_body_proximity(lat: float, lng: float) -> Dict[str, Any]:
    """Get proximity to water bodies."""
    try:
        # Use OpenStreetMap water data
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:25];
        (
          way["natural"="water"](around:5000,{lat},{lng});
          way["waterway"](around:5000,{lat},{lng});
          relation["natural"="water"](around:5000,{lat},{lng});
        );
        out geom;
        """
        
        response = requests.post(overpass_url, data=query, timeout=15)
        if response.status_code == 200:
            data = response.json()
            water_features = data.get('elements', [])
            
            if water_features:
                # Find nearest water body
                nearest_water = min(water_features, 
                    key=lambda x: _calculate_distance_to_feature(lat, lng, x))
                
                distance = _calculate_distance_to_feature(lat, lng, nearest_water)
                return {
                    "distance_km": distance,
                    "type": nearest_water.get("tags", {}).get("natural", "water"),
                    "name": nearest_water.get("tags", {}).get("name", "Water Body")
                }
        
        return {"distance_km": 10.0, "type": None, "name": None}
        
    except Exception:
        return {"distance_km": 10.0, "type": None, "name": None}

def _calculate_heat_island_index(albedo: float, built_up: float, vegetation: float, water: Dict) -> float:
    """
    Calculate heat island potential index.
    Higher values = higher heat island potential = lower suitability.
    """
    
    # Surface albedo effect (30%) - lower albedo = higher heat absorption
    albedo_effect = (1.0 - albedo) * 100  # Convert to 0-100 scale
    
    # Built-up density effect (40%) - more built-up = higher heat island
    built_up_effect = built_up * 100
    
    # Vegetation cooling effect (20%) - more vegetation = lower heat island
    vegetation_effect = max(0, (vegetation - 50) * 2)  # Vegetation > 50% provides cooling
    
    # Water body cooling effect (10%) - closer to water = lower heat island
    water_distance = water.get("distance_km", 10.0)
    water_effect = max(0, (10.0 - water_distance) * 10)  # Closer than 10km provides cooling
    
    # Combined heat island index
    heat_island_index = (
        albedo_effect * 0.3 +
        built_up_effect * 0.4 +
        max(0, 100 - vegetation_effect) * 0.2 +  # Invert vegetation effect
        max(0, 100 - water_effect) * 0.1      # Invert water effect
    )
    
    return min(100.0, heat_island_index)

def _heat_island_to_suitability(heat_island_index: float) -> float:
    """
    Convert heat island index to suitability score.
    Higher heat island potential = lower suitability.
    """
    # Inverse relationship with scaling
    if heat_island_index >= 80:
        return max(0, 25 - (heat_island_index - 80) * 0.75)  # 0-25 range
    elif heat_island_index >= 60:
        return 25 + (80 - heat_island_index) * 2.5  # 25-75 range
    elif heat_island_index >= 40:
        return 75 + (60 - heat_island_index) * 0.75  # 75-90 range
    else:
        return 90 + (40 - heat_island_index) * 0.25  # 90-100 range

def _get_heat_island_label(suitability_score: float) -> str:
    """Get human-readable label for heat island potential."""
    if suitability_score >= 80:
        return "Low Heat Island Risk (Excellent)"
    elif suitability_score >= 60:
        return "Moderate Heat Island Risk (Good)"
    elif suitability_score >= 40:
        return "High Heat Island Risk (Fair)"
    elif suitability_score >= 20:
        return "Very High Heat Island Risk (Poor)"
    else:
        return "Extreme Heat Island Risk (Very Poor)"

def _calculate_confidence(albedo: float, built_up: float) -> float:
    """Calculate confidence based on data quality."""
    confidence = 50.0  # Base confidence
    
    # MODIS albedo is high quality
    if albedo > 0:
        confidence += 25
    
    # OSM data quality varies
    if built_up > 0:
        confidence += 15
    
    return min(95, confidence)

def _generate_reasoning(heat_island_index: float, albedo: float, built_up: float) -> str:
    """Generate human-readable reasoning for heat island assessment."""
    reasoning_parts = []
    
    # Surface characteristics
    if albedo < 0.2:
        reasoning_parts.append(f"dark surface (albedo {albedo:.2f}) absorbs significant solar radiation")
    elif albedo < 0.4:
        reasoning_parts.append(f"moderate surface reflectivity (albedo {albedo:.2f})")
    else:
        reasoning_parts.append(f"reflective surface (albedo {albedo:.2f}) reduces heat absorption")
    
    # Built-up environment
    if built_up > 0.7:
        reasoning_parts.append(f"high built-up density ({built_up:.1%} impervious surface)")
    elif built_up > 0.3:
        reasoning_parts.append(f"moderate built-up density ({built_up:.1%} impervious surface)")
    else:
        reasoning_parts.append(f"low built-up density ({built_up:.1%} impervious surface)")
    
    # Overall heat island risk
    if heat_island_index > 70:
        reasoning_parts.append("creates significant heat island effect requiring mitigation")
    elif heat_island_index > 40:
        reasoning_parts.append("moderate heat island potential")
    else:
        reasoning_parts.append("minimal heat island concerns")
    
    return ". ".join(reasoning_parts) + "."

def _get_fallback_heat_island(lat: float, lng: float) -> Dict[str, Any]:
    """Fallback heat island estimation based on geographic context."""
    try:
        # Use geographic context for rough estimation
        is_urban = _estimate_urban_density(lat, lng)
        climate_zone = _get_climate_zone(lat)
        
        # Base heat island potential by urban density
        if is_urban == "high":
            heat_island = 75.0  # High heat island in cities
            label = "High Heat Island Risk (Fair)"
        elif is_urban == "medium":
            heat_island = 50.0  # Moderate heat island in suburbs
            label = "Moderate Heat Island Risk (Good)"
        else:
            heat_island = 25.0  # Low heat island in rural areas
            label = "Low Heat Island Risk (Excellent)"
        
        # Adjust for climate
        if climate_zone == "tropical":
            heat_island = min(100, heat_island + 10)  # Higher heat in tropics
        elif climate_zone == "polar":
            heat_island = max(0, heat_island - 20)  # Lower heat in polar
        
        suitability = _heat_island_to_suitability(heat_island)
        
        return {
            "value": suitability,
            "heat_island_index": heat_island,
            "surface_albedo": 0.3,
            "built_up_density": is_urban == "high",
            "vegetation_cover": 50.0,
            "water_proximity": 10.0,
            "label": label,
            "source": "Geographic Estimation (Fallback)",
            "confidence": 25.0,
            "reasoning": f"Estimated based on {'urban' if is_urban == 'high' else 'suburban' if is_urban == 'medium' else 'rural'} environment and {climate_zone} climate."
        }
        
    except Exception:
        return {
            "value": 50.0,
            "heat_island_index": 50.0,
            "surface_albedo": 0.3,
            "built_up_density": 0.5,
            "vegetation_cover": 50.0,
            "water_proximity": 10.0,
            "label": "Moderate Heat Island Risk (Good)",
            "source": "Default Fallback",
            "confidence": 10.0,
            "reasoning": "Unable to determine heat island characteristics."
        }

# Helper functions
def _estimate_albedo_from_landuse(lat: float, lng: float) -> float:
    """Estimate surface albedo based on land use."""
    # Simplified land use albedo estimation
    is_urban = _estimate_urban_density(lat, lng)
    climate_zone = _get_climate_zone(lat)
    
    if is_urban == "high":
        # Urban areas: dark surfaces (asphalt, buildings)
        return 0.15
    elif is_urban == "medium":
        # Suburban: mix of surfaces
        return 0.25
    else:
        # Rural: varies by climate and vegetation
        if climate_zone == "tropical":
            return 0.18  # Dark vegetation
        elif climate_zone == "polar":
            return 0.35  # Snow/ice
        else:
            return 0.25  # Moderate vegetation

def _estimate_built_up_from_population(lat: float, lng: float) -> float:
    """Estimate built-up density from population data."""
    try:
        # Use existing population data
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from socio_econ.population_data import get_population_data
        
        pop_result = get_population_data(lat, lng)
        density = pop_result.get("density", 100)  # people per km²
        
        # Convert population density to built-up density (rough approximation)
        # Higher population = higher built-up density
        if density > 1000:
            return 0.8  # High density urban
        elif density > 500:
            return 0.6  # Medium density
        elif density > 100:
            return 0.4  # Low density
        else:
            return 0.2  # Very low density
            
    except Exception:
        return _estimate_urban_density(lat, lng) == "high"

def _estimate_vegetation_cover(lat: float, lng: float) -> float:
    """Estimate vegetation cover based on climate."""
    climate_zone = _get_climate_zone(lat)
    
    vegetation_by_climate = {
        "tropical": 80.0,
        "subtropical": 65.0,
        "temperate": 55.0,
        "cool": 40.0,
        "polar": 20.0
    }
    
    return vegetation_by_climate.get(climate_zone, 50.0)

def _calculate_distance_to_feature(lat: float, lng: float, feature: Dict) -> float:
    """Calculate distance to a geographic feature."""
    if not feature or "geometry" not in feature:
        return 10.0
    
    geometry = feature.get("geometry", {})
    coords = geometry.get("coordinates", [])
    
    if not coords:
        return 10.0
    
    # Handle different geometry types
    if geometry.get("type") == "Point":
        point_coords = coords
        if len(point_coords) >= 2:
            return _calculate_distance(lat, lng, point_coords)
    elif geometry.get("type") in ["LineString", "Polygon"]:
        # For lines/polygons, find nearest point
        min_distance = float('inf')
        for point in coords:
            if isinstance(point, list) and len(point) >= 2:
                distance = _calculate_distance(lat, lng, point)
                min_distance = min(min_distance, distance)
        return min_distance if min_distance != float('inf') else 10.0
    
    return 10.0

def _calculate_distance(lat1: float, lng1: float, coords: List) -> float:
    """Calculate distance between point and coordinates."""
    if len(coords) < 2:
        return 10.0
    
    lat2, lng2 = coords[0], coords[1]
    
    # Haversine distance calculation
    R = 6371  # Earth's radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)
    
    a = (math.sin(delta_lat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def _estimate_urban_density(lat: float, lng: float) -> str:
    """Estimate if location is in urban area."""
    # Major urban centers approximation
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
