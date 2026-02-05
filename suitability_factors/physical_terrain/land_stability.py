"""
Land Stability / Erosion Analysis Module
Calculates long-term land reliability and erosion risk
Data Sources: USGS, ESA, National Soil Surveys, Global Erosion Maps
"""

import math
import requests
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def get_land_stability(lat: float, lng: float) -> Dict[str, Any]:
    """
    Calculate land stability and erosion risk for given coordinates.
    
    Args:
        lat: Latitude
        lng: Longitude
        
    Returns:
        Dictionary with stability score and erosion risk assessment
    """
    try:
        # Get input factors for stability assessment
        slope_data = _get_slope_data(lat, lng)
        soil_type = _get_soil_type(lat, lng)
        rainfall_data = _get_rainfall_data(lat, lng)
        vegetation_cover = _get_vegetation_cover(lat, lng)
        geology_data = _get_geology_data(lat, lng)
        
        # Calculate erosion risk using USLE/RUSLE methodology
        erosion_risk = _calculate_erosion_risk(
            slope_data, soil_type, rainfall_data, vegetation_cover
        )
        
        # Calculate landslide susceptibility
        landslide_risk = _calculate_landslide_risk(
            slope_data, geology_data, rainfall_data
        )
        
        # Calculate overall stability index
        stability_index = _calculate_stability_index(erosion_risk, landslide_risk)
        
        # Convert to suitability score
        suitability_score = _stability_to_suitability(stability_index)
        
        return {
            "value": suitability_score,
            "stability_index": round(stability_index, 2),
            "erosion_risk": round(erosion_risk, 2),
            "landslide_risk": round(landslide_risk, 2),
            "slope_factor": round(slope_data.get("slope_percent", 0), 1),
            "soil_type": soil_type.get("type", "Unknown"),
            "erosivity": round(rainfall_data.get("erosivity", 0), 1),
            "vegetation_cover": round(vegetation_cover.get("cover_percent", 0), 1),
            "label": _get_stability_label(suitability_score),
            "source": "USGS + ESA + Derived Calculations",
            "confidence": _calculate_confidence(slope_data, soil_type, rainfall_data),
            "reasoning": _generate_reasoning(erosion_risk, landslide_risk, stability_index, suitability_score)
        }
        
    except Exception as e:
        logger.error(f"Error calculating land stability for {lat}, {lng}: {e}")
        return _get_fallback_stability(lat, lng)

def _get_slope_data(lat: float, lng: float) -> Dict[str, Any]:
    """Get slope data for the location."""
    try:
        # Import from the correct location
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from slope_analysis import get_slope_analysis
        slope_result = get_slope_analysis(lat, lng)
        
        return {
            "slope_percent": slope_result.get("value", 0),
            "slope_degrees": slope_result.get("degrees", 0),
            "source": slope_result.get("source", "Unknown")
        }
    except Exception:
        return {"slope_percent": 5.0, "slope_degrees": 2.9, "source": "Estimated"}

def _get_soil_type(lat: float, lng: float) -> Dict[str, Any]:
    """Get soil type and characteristics."""
    try:
        # Use ISRIC SoilGrids API for global soil data
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query"
        params = {
            'lon': lng,
            'lat': lat,
            'properties': ['wrb', 'bdod', 'cec', 'phh2o', 'sand', 'silt', 'clay'],
            'depth': '0-5cm',
            'values': 'mean'
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            properties = data.get('properties', {})
            
            # Extract soil classification and properties
            wrb_class = properties.get('wrb', {}).get('mean', {}).get('value', 'Unknown')
            
            # Extract physical properties
            sand_content = properties.get('sand', {}).get('mean', {}).get('value', 33)
            silt_content = properties.get('silt', {}).get('mean', {}).get('value', 33)
            clay_content = properties.get('clay', {}).get('mean', {}).get('value', 34)
            ph_value = properties.get('phh2o', {}).get('mean', {}).get('value', 6.5)
            bulk_density = properties.get('bdod', {}).get('mean', {}).get('value', 1.3)
            
            # Determine soil erodibility (K factor) based on soil properties
            k_factor = _calculate_k_factor(sand_content, silt_content, clay_content, organic_matter=2.5)
            
            return {
                "type": wrb_class,
                "sand_percent": sand_content,
                "silt_percent": silt_content,
                "clay_percent": clay_content,
                "ph": ph_value,
                "bulk_density": bulk_density,
                "k_factor": k_factor,
                "source": "ISRIC SoilGrids"
            }
        
    except Exception as e:
        logger.debug(f"Failed to get soil data from ISRIC: {e}")
    
    # Fallback to regional soil estimation
    return _estimate_soil_type(lat, lng)

def _estimate_soil_type(lat: float, lng: float) -> Dict[str, Any]:
    """Estimate soil type based on geographic region."""
    region = _get_geographic_region(lat, lng)
    
    soil_types = {
        "north_america": {"type": "Alfisols", "sand": 40, "silt": 35, "clay": 25, "k_factor": 0.25},
        "south_america": {"type": "Oxisols", "sand": 50, "silt": 20, "clay": 30, "k_factor": 0.30},
        "europe": {"type": "Luvisols", "sand": 35, "silt": 40, "clay": 25, "k_factor": 0.20},
        "africa": {"type": "Ferralsols", "sand": 45, "silt": 25, "clay": 30, "k_factor": 0.35},
        "asia": {"type": "Inceptisols", "sand": 38, "silt": 32, "clay": 30, "k_factor": 0.28},
        "oceania": {"type": "Acrisols", "sand": 42, "silt": 28, "clay": 30, "k_factor": 0.32}
    }
    
    soil_data = soil_types.get(region, soil_types["asia"])
    soil_data["source"] = "Regional Estimation"
    soil_data["ph"] = 6.5
    soil_data["bulk_density"] = 1.3
    
    return soil_data

def _get_rainfall_data(lat: float, lng: float) -> Dict[str, Any]:
    """Get rainfall data and calculate erosivity (R factor)."""
    try:
        # Use Open-Meteo API for historical rainfall data
        url = f"https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': lat,
            'longitude': lng,
            'start_date': '2020-01-01',
            'end_date': '2020-12-31',
            'daily': ['precipitation_sum'],
            'timezone': 'auto'
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            daily_data = data.get('daily', {})
            precipitation = daily_data.get('precipitation_sum', [])
            
            if precipitation and len(precipitation) > 0:
                # Calculate annual rainfall
                annual_rainfall = sum(p for p in precipitation if p is not None)
                
                # Calculate R factor (rainfall erosivity) based on annual rainfall
                # R = 0.0483 * P^1.61 (where P is annual rainfall in mm)
                r_factor = 0.0483 * (annual_rainfall ** 1.61) if annual_rainfall > 0 else 0
                
                return {
                    "annual_mm": annual_rainfall,
                    "erosivity": r_factor,
                    "source": "Open-Meteo Historical Data"
                }
    
    except Exception as e:
        logger.debug(f"Failed to get rainfall data: {e}")
    
    # Fallback to climate-based estimation
    return _estimate_rainfall_erosivity(lat, lng)

def _estimate_rainfall_erosivity(lat: float, lng: float) -> Dict[str, Any]:
    """Estimate rainfall erosivity based on climate zone."""
    climate_zone = _get_climate_zone(lat)
    
    climate_rainfall = {
        "tropical": {"annual_mm": 2000, "erosivity": 1200},
        "subtropical": {"annual_mm": 1200, "erosivity": 600},
        "temperate": {"annual_mm": 800, "erosivity": 350},
        "cool": {"annual_mm": 600, "erosivity": 200},
        "polar": {"annual_mm": 200, "erosivity": 50}
    }
    
    data = climate_rainfall.get(climate_zone, climate_rainfall["temperate"])
    data["source"] = "Climate Zone Estimation"
    
    return data

def _get_vegetation_cover(lat: float, lng: float) -> Dict[str, Any]:
    """Get vegetation cover data."""
    try:
        # Import NDVI data from the correct location
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from environmental.ndvi_analysis import get_ndvi_data
        ndvi_result = get_ndvi_data(lat, lng)
        
        ndvi_value = ndvi_result.get("ndvi_index", 0.5)
        
        # Convert NDVI to vegetation cover percentage
        # NDVI < 0.1 = bare soil (0% cover)
        # NDVI 0.1-0.3 = sparse vegetation (10-30% cover)
        # NDVI 0.3-0.6 = moderate vegetation (30-60% cover)
        # NDVI > 0.6 = dense vegetation (60-100% cover)
        
        if ndvi_value < 0.1:
            cover_percent = 0
        elif ndvi_value < 0.3:
            cover_percent = (ndvi_value - 0.1) * 100  # 0-20%
        elif ndvi_value < 0.6:
            cover_percent = 20 + (ndvi_value - 0.3) * 133  # 20-60%
        else:
            cover_percent = 60 + (ndvi_value - 0.6) * 100  # 60-100%
        
        cover_percent = min(100, max(0, cover_percent))
        
        return {
            "cover_percent": cover_percent,
            "ndvi": ndvi_value,
            "source": "Sentinel-2 NDVI"
        }
        
    except Exception:
        # Fallback to regional vegetation estimation
        return _estimate_vegetation_cover(lat, lng)

def _estimate_vegetation_cover(lat: float, lng: float) -> Dict[str, Any]:
    """Estimate vegetation cover based on climate and region."""
    climate_zone = _get_climate_zone(lat)
    region = _get_geographic_region(lat, lng)
    
    # Base vegetation by climate
    climate_veg = {
        "tropical": 80,
        "subtropical": 65,
        "temperate": 55,
        "cool": 40,
        "polar": 15
    }
    
    # Adjust for region (urbanization, agriculture, etc.)
    region_adjustments = {
        "north_america": -10,
        "europe": -5,
        "asia": -15,
        "south_america": 5,
        "africa": 0,
        "oceania": -5
    }
    
    base_cover = climate_veg.get(climate_zone, 50)
    adjustment = region_adjustments.get(region, 0)
    final_cover = max(0, min(100, base_cover + adjustment))
    
    return {
        "cover_percent": final_cover,
        "ndvi": final_cover / 100.0,
        "source": "Regional Estimation"
    }

def _get_geology_data(lat: float, lng: float) -> Dict[str, Any]:
    """Get geological data for landslide assessment."""
    try:
        # Use USGS Geological Map API
        url = f"https://earthquake.usgs.gov/ws/geoserve/regions.json"
        params = {'latitude': lat, 'longitude': lng}
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            region = data.get('region', {})
            
            # Extract geological information
            geology_type = region.get('name', 'Unknown')
            
            # Determine rock strength based on geology
            rock_strength = _estimate_rock_strength(geology_type)
            
            return {
                "type": geology_type,
                "rock_strength": rock_strength,
                "source": "USGS Geological Survey"
            }
    
    except Exception as e:
        logger.debug(f"Failed to get geology data: {e}")
    
    # Fallback to regional geology estimation
    return _estimate_geology(lat, lng)

def _estimate_geology(lat: float, lng: float) -> Dict[str, Any]:
    """Estimate geology based on terrain characteristics."""
    # Simple estimation based on elevation and slope
    is_mountainous = _is_mountainous_region(lat, lng)
    
    if is_mountainous:
        geology_type = "Metamorphic/Igneous"
        rock_strength = 0.8  # High strength
    else:
        geology_type = "Sedimentary"
        rock_strength = 0.5  # Medium strength
    
    return {
        "type": geology_type,
        "rock_strength": rock_strength,
        "source": "Terrain-based Estimation"
    }

def _calculate_erosion_risk(slope_data: Dict, soil_type: Dict, rainfall_data: Dict, vegetation: Dict) -> float:
    """
    Calculate erosion risk using USLE (Universal Soil Loss Equation).
    USLE: A = R × K × LS × C × P
    
    Where:
    A = Soil loss (tons/ha/year)
    R = Rainfall erosivity factor
    K = Soil erodibility factor  
    LS = Slope length and steepness factor
    C = Cover management factor
    P = Support practice factor (assumed 1.0)
    """
    
    # R factor (rainfall erosivity)
    r_factor = rainfall_data.get("erosivity", 350)
    
    # K factor (soil erodibility)
    k_factor = soil_type.get("k_factor", 0.25)
    
    # LS factor (slope length and steepness)
    slope_percent = slope_data.get("slope_percent", 5)
    ls_factor = _calculate_ls_factor(slope_percent)
    
    # C factor (cover management)
    vegetation_cover = vegetation.get("cover_percent", 50)
    c_factor = _calculate_c_factor(vegetation_cover)
    
    # P factor (support practice) - assume no conservation practices
    p_factor = 1.0
    
    # Calculate soil loss
    soil_loss = r_factor * k_factor * ls_factor * c_factor * p_factor
    
    # Normalize to 0-100 scale (higher = more erosion risk)
    # > 50 tons/ha/year = very high risk (90-100)
    # 20-50 = high risk (70-90)
    # 10-20 = moderate risk (50-70)
    # 5-10 = low risk (30-50)
    # < 5 = very low risk (0-30)
    
    if soil_loss >= 50:
        return min(100, 90 + (soil_loss - 50) * 0.2)
    elif soil_loss >= 20:
        return 70 + (soil_loss - 20) * 0.67
    elif soil_loss >= 10:
        return 50 + (soil_loss - 10) * 2
    elif soil_loss >= 5:
        return 30 + (soil_loss - 5) * 4
    else:
        return soil_loss * 6

def _calculate_landslide_risk(slope_data: Dict, geology_data: Dict, rainfall_data: Dict) -> float:
    """
    Calculate landslide susceptibility based on slope, geology, and rainfall.
    """
    
    # Base risk from slope
    slope_percent = slope_data.get("slope_percent", 5)
    slope_risk = _calculate_slope_landslide_risk(slope_percent)
    
    # Modify by geology (rock strength)
    rock_strength = geology_data.get("rock_strength", 0.5)
    geology_factor = 1.0 - rock_strength  # Lower strength = higher risk
    
    # Modify by rainfall (saturation factor)
    annual_rainfall = rainfall_data.get("annual_mm", 800)
    rainfall_factor = min(1.5, 1.0 + (annual_rainfall - 500) / 1000)  # Higher rain = higher risk
    
    # Calculate combined risk
    landslide_risk = slope_risk * geology_factor * rainfall_factor
    
    # Normalize to 0-100 scale
    return min(100, landslide_risk * 100)

def _calculate_stability_index(erosion_risk: float, landslide_risk: float) -> float:
    """
    Calculate overall stability index from erosion and landslide risks.
    Higher index = more stable.
    """
    # Convert risks to stability (inverse relationship)
    erosion_stability = max(0, 100 - erosion_risk)
    landslide_stability = max(0, 100 - landslide_risk)
    
    # Weighted combination (landslides are more critical)
    stability_index = (erosion_stability * 0.4 + landslide_stability * 0.6)
    
    return stability_index

def _stability_to_suitability(stability_index: float) -> float:
    """
    Convert stability index to suitability score.
    Higher stability = higher suitability.
    """
    # Direct relationship with some scaling for extreme values
    if stability_index >= 80:
        return min(100, stability_index + 10)
    elif stability_index >= 60:
        return stability_index + 5
    elif stability_index >= 40:
        return stability_index
    else:
        return max(0, stability_index - 10)

def _calculate_k_factor(sand: float, silt: float, clay: float, organic_matter: float) -> float:
    """Calculate soil erodibility K factor from soil composition."""
    # Simplified K factor calculation
    # High sand content = lower K (less erodible)
    # High silt/clay = higher K (more erodible)
    
    silt_clay = silt + clay
    if silt_clay <= 40:
        k_factor = 0.2 + 0.3 * (silt_clay / 40)
    else:
        k_factor = 0.5 - 0.2 * ((silt_clay - 40) / 60)
    
    # Adjust for organic matter (higher OM = lower K)
    if organic_matter > 4:
        k_factor *= 0.7
    elif organic_matter > 2:
        k_factor *= 0.85
    
    return max(0.1, min(0.4, k_factor))

def _calculate_ls_factor(slope_percent: float) -> float:
    """Calculate slope length and steepness (LS) factor."""
    # Simplified LS factor calculation
    # LS = (slope/9)^0.5 * (length/22.1)^m (where m varies with slope)
    # We'll use a simplified version based on slope only
    
    if slope_percent <= 1:
        return 0.1
    elif slope_percent <= 3:
        return 0.5
    elif slope_percent <= 5:
        return 1.0
    elif slope_percent <= 10:
        return 2.0
    elif slope_percent <= 15:
        return 3.5
    elif slope_percent <= 20:
        return 5.0
    else:
        return 7.0 + (slope_percent - 20) * 0.2

def _calculate_c_factor(vegetation_cover: float) -> float:
    """Calculate cover management (C) factor from vegetation cover."""
    # C factor decreases with more vegetation cover
    # 100% cover = 0.01, 0% cover = 1.0
    
    if vegetation_cover >= 80:
        return 0.01
    elif vegetation_cover >= 60:
        return 0.05
    elif vegetation_cover >= 40:
        return 0.15
    elif vegetation_cover >= 20:
        return 0.35
    else:
        return 0.8

def _calculate_slope_landslide_risk(slope_percent: float) -> float:
    """Calculate landslide risk based on slope percentage."""
    # Risk increases exponentially with slope
    if slope_percent <= 5:
        return 0.01
    elif slope_percent <= 10:
        return 0.05
    elif slope_percent <= 15:
        return 0.15
    elif slope_percent <= 25:
        return 0.35
    elif slope_percent <= 35:
        return 0.60
    else:
        return min(1.0, 0.60 + (slope_percent - 35) * 0.02)

def _estimate_rock_strength(geology_type: str) -> float:
    """Estimate rock strength from geology type."""
    strength_map = {
        "Igneous": 0.9,
        "Metamorphic": 0.85,
        "Sedimentary": 0.5,
        "Volcanic": 0.8,
        "Alluvial": 0.3,
        "Glacial": 0.6
    }
    
    for rock_type, strength in strength_map.items():
        if rock_type.lower() in geology_type.lower():
            return strength
    
    return 0.5  # Default medium strength

def _get_stability_label(suitability_score: float) -> str:
    """Get human-readable label for stability level."""
    if suitability_score >= 80:
        return "Very Stable (Excellent)"
    elif suitability_score >= 60:
        return "Stable (Good)"
    elif suitability_score >= 40:
        return "Moderately Stable (Fair)"
    elif suitability_score >= 20:
        return "Unstable (Poor)"
    else:
        return "Very Unstable (Very Poor)"

def _calculate_confidence(slope_data: Dict, soil_data: Dict, rainfall_data: Dict) -> float:
    """Calculate confidence based on data quality."""
    confidence = 50.0  # Base confidence
    
    # Add confidence for each data source
    if slope_data.get("source") != "Estimated":
        confidence += 15
    if soil_data.get("source") != "Regional Estimation":
        confidence += 20
    if rainfall_data.get("source") != "Climate Zone Estimation":
        confidence += 15
    
    return min(95, confidence)

def _generate_reasoning(erosion_risk: float, landslide_risk: float, stability_index: float, suitability: float) -> str:
    """Generate human-readable reasoning for stability assessment."""
    reasoning_parts = []
    
    # Erosion risk reasoning
    if erosion_risk > 70:
        reasoning_parts.append(f"High erosion risk ({erosion_risk:.0f}/100)")
    elif erosion_risk > 40:
        reasoning_parts.append(f"Moderate erosion risk ({erosion_risk:.0f}/100)")
    else:
        reasoning_parts.append(f"Low erosion risk ({erosion_risk:.0f}/100)")
    
    # Landslide risk reasoning
    if landslide_risk > 70:
        reasoning_parts.append(f"high landslide susceptibility")
    elif landslide_risk > 40:
        reasoning_parts.append(f"moderate landslide risk")
    else:
        reasoning_parts.append(f"low landslide risk")
    
    # Overall stability
    if stability_index > 70:
        reasoning_parts.append("overall stable terrain")
    elif stability_index > 40:
        reasoning_parts.append("moderately stable conditions")
    else:
        reasoning_parts.append("unstable terrain conditions")
    
    return ". ".join(reasoning_parts) + "."

def _get_fallback_stability(lat: float, lng: float) -> Dict[str, Any]:
    """Fallback stability estimation based on geographic context."""
    try:
        # Use geographic context for rough estimation
        is_mountainous = _is_mountainous_region(lat, lng)
        climate_zone = _get_climate_zone(lat)
        
        if is_mountainous:
            stability_index = 30.0  # Poor - mountainous terrain
            label = "Mountainous Terrain (Poor)"
        elif climate_zone == "tropical":
            stability_index = 50.0  # Fair - tropical erosion
            label = "Tropical Terrain (Fair)"
        else:
            stability_index = 70.0  # Good - moderate terrain
            label = "Moderate Terrain (Good)"
        
        suitability = _stability_to_suitability(stability_index)
        
        return {
            "value": suitability,
            "stability_index": stability_index,
            "erosion_risk": 50.0,
            "landslide_risk": 30.0,
            "slope_factor": 10.0,
            "soil_type": "Unknown",
            "erosivity": 400.0,
            "vegetation_cover": 50.0,
            "label": label,
            "source": "Geographic Estimation (Fallback)",
            "confidence": 25.0,
            "reasoning": f"Estimated based on geographic location ({'mountainous' if is_mountainous else climate_zone} region)."
        }
        
    except Exception:
        return {
            "value": 50.0,
            "stability_index": 50.0,
            "erosion_risk": 50.0,
            "landslide_risk": 50.0,
            "slope_factor": 10.0,
            "soil_type": "Unknown",
            "erosivity": 400.0,
            "vegetation_cover": 50.0,
            "label": "Unknown Stability (Moderate)",
            "source": "Default Fallback",
            "confidence": 10.0,
            "reasoning": "Unable to determine stability characteristics."
        }

# Helper functions
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

def _is_mountainous_region(lat: float, lng: float) -> bool:
    """Check if location is in known mountainous region."""
    # Major mountain ranges (simplified bounding boxes)
    mountain_ranges = [
        # Himalayas
        (27.0, 37.0, 67.0, 98.0),
        # Andes  
        (-55.0, -10.0, -80.0, -65.0),
        # Rockies
        (31.0, 60.0, -130.0, -105.0),
        # Alps
        (45.0, 48.0, 5.0, 17.0),
        # African mountains
        (-30.0, 15.0, 10.0, 40.0)
    ]
    
    for min_lat, max_lat, min_lng, max_lng in mountain_ranges:
        if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
            return True
    
    return False
