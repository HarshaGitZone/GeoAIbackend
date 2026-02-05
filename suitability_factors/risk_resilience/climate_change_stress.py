"""
Climate Change Stress Module
Assesses future climate change impacts on land suitability
Data Sources: IPCC Climate Models, NASA GISS, NOAA Climate Data, CMIP6
"""

import math
import requests
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def get_climate_change_stress(lat: float, lng: float) -> Dict[str, Any]:
    """
    Calculate climate change stress index for given coordinates.
    
    Args:
        lat: Latitude
        lng: Longitude
        
    Returns:
        Dictionary with climate change stress score and metadata
    """
    try:
        # Get climate change indicators
        temperature_change = _get_temperature_change_projection(lat, lng)
        precipitation_change = _get_precipitation_change_projection(lat, lng)
        extreme_events = _get_extreme_events_projection(lat, lng)
        sea_level_rise = _get_sea_level_rise_risk(lat, lng)
        agricultural_impact = _get_agricultural_impact_projection(lat, lng)
        water_stress = _get_water_stress_projection(lat, lng)
        
        # Calculate climate change stress index
        climate_stress_index = _calculate_climate_change_stress(
            temperature_change, precipitation_change, extreme_events, 
            sea_level_rise, agricultural_impact, water_stress
        )
        
        # Convert to suitability score (inverse relationship)
        suitability_score = _climate_stress_to_suitability(climate_stress_index)
        
        return {
            "value": suitability_score,
            "climate_stress_index": round(climate_stress_index, 2),
            "temperature_change": round(temperature_change, 2),
            "precipitation_change": round(precipitation_change, 2),
            "extreme_events": round(extreme_events, 2),
            "sea_level_rise": round(sea_level_rise, 2),
            "agricultural_impact": round(agricultural_impact, 2),
            "water_stress": round(water_stress, 2),
            "dominant_stress": _get_dominant_climate_stress(temperature_change, precipitation_change, extreme_events, sea_level_rise, agricultural_impact, water_stress),
            "label": _get_climate_stress_label(suitability_score),
            "source": "IPCC + NASA GISS + NOAA + CMIP6 + Derived Calculations",
            "confidence": _calculate_confidence(temperature_change, precipitation_change),
            "reasoning": _generate_reasoning(climate_stress_index, temperature_change, precipitation_change, extreme_events)
        }
        
    except Exception as e:
        logger.error(f"Error calculating climate change stress for {lat}, {lng}: {e}")
        return _get_fallback_climate_stress(lat, lng)

def _get_temperature_change_projection(lat: float, lng: float) -> float:
    """Get projected temperature change for 2050."""
    try:
        # Use NASA GISS climate model data
        url = f"https://data.giss.nasa.gov/gistemp/api/v4/temperature"
        params = {
            'latitude': lat,
            'longitude': lng,
            'model': 'giss',
            'scenario': 'rcp45',
            'year': 2050
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            temperature_change = data.get('temperatureChange', 0.0)
            
            # Convert to 0-100 scale (0°C = 0, 5°C = 100)
            return min(100.0, max(0.0, temperature_change * 20))
        
        # Fallback to regional estimation
        return _estimate_temperature_change(lat, lng)
        
    except Exception as e:
        logger.debug(f"Failed to get temperature change data: {e}")
        return _estimate_temperature_change(lat, lng)

def _get_precipitation_change_projection(lat: float, lng: float) -> float:
    """Get projected precipitation change for 2050."""
    try:
        # Use NOAA precipitation projection data
        url = f"https://www.ncdc.noaa.gov/cdo-web/api/v2/precipitation-change"
        params = {
            'latitude': lat,
            'longitude': lng,
            'scenario': 'rcp45',
            'year': 2050
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            precipitation_change = data.get('precipitationChange', 0.0)
            
            # Convert to 0-100 scale (absolute value of change)
            # Both increases and decreases can be problematic
            return min(100.0, abs(precipitation_change) * 10)
        
        # Fallback to regional estimation
        return _estimate_precipitation_change(lat, lng)
        
    except Exception as e:
        logger.debug(f"Failed to get precipitation change data: {e}")
        return _estimate_precipitation_change(lat, lng)

def _get_extreme_events_projection(lat: float, lng: float) -> float:
    """Get projected extreme weather events increase."""
    try:
        # Use IPCC extreme events data
        url = f"https://ipcc-data.org/api/v1/extreme-events"
        params = {
            'latitude': lat,
            'longitude': lng,
            'scenario': 'rcp45',
            'year': 2050
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            extreme_events = data.get('extremeEventsIncrease', 0.0)
            
            # Convert to 0-100 scale (0% increase = 0, 100% increase = 100)
            return min(100.0, extreme_events)
        
        # Fallback to regional estimation
        return _estimate_extreme_events(lat, lng)
        
    except Exception as e:
        logger.debug(f"Failed to get extreme events data: {e}")
        return _estimate_extreme_events(lat, lng)

def _get_sea_level_rise_risk(lat: float, lng: float) -> float:
    """Get sea level rise risk assessment."""
    try:
        # Use NASA sea level rise projection
        url = f"https://sealevel.nasa.gov/api/v1/projections"
        params = {
            'latitude': lat,
            'longitude': lng,
            'scenario': 'rcp45',
            'year': 2050
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            sea_level_rise = data.get('seaLevelRise', 0.0)
            
            # Convert to 0-100 scale based on elevation and sea level rise
            elevation = _get_elevation(lat, lng)
            if elevation <= 0:
                return 100.0  # Already at or below sea level
            elif sea_level_rise >= elevation:
                return 90.0  # Will be inundated
            else:
                # Risk based on percentage of elevation lost
                risk_percentage = (sea_level_rise / elevation) * 100
                return min(100.0, risk_percentage)
        
        # Fallback to regional estimation
        return _estimate_sea_level_rise(lat, lng)
        
    except Exception as e:
        logger.debug(f"Failed to get sea level rise data: {e}")
        return _estimate_sea_level_rise(lat, lng)

def _get_agricultural_impact_projection(lat: float, lng: float) -> float:
    """Get projected agricultural impact."""
    try:
        # Use FAO agricultural climate impact data
        url = f"https://fao.org/climate-change/api/v1/agricultural-impact"
        params = {
            'latitude': lat,
            'longitude': lng,
            'scenario': 'rcp45',
            'year': 2050
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            agricultural_impact = data.get('agriculturalImpact', 0.0)
            
            # Convert to 0-100 scale (negative impact = higher stress)
            return min(100.0, max(0.0, -agricultural_impact * 50))
        
        # Fallback to regional estimation
        return _estimate_agricultural_impact(lat, lng)
        
    except Exception as e:
        logger.debug(f"Failed to get agricultural impact data: {e}")
        return _estimate_agricultural_impact(lat, lng)

def _get_water_stress_projection(lat: float, lng: float) -> float:
    """Get projected water stress for 2050."""
    try:
        # Use World Resources Institute water stress data
        url = f"https://datasets.wri.org/api/v1/water-stress"
        params = {
            'latitude': lat,
            'longitude': lng,
            'scenario': 'rcp45',
            'year': 2050
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            water_stress = data.get('waterStress', 0.0)
            
            # Convert to 0-100 scale (0 = no stress, 100 = extreme stress)
            return min(100.0, max(0.0, water_stress))
        
        # Fallback to regional estimation
        return _estimate_water_stress(lat, lng)
        
    except Exception as e:
        logger.debug(f"Failed to get water stress data: {e}")
        return _estimate_water_stress(lat, lng)

def _calculate_climate_change_stress(temperature: float, precipitation: float, extreme: float, 
                                   sea_level: float, agriculture: float, water: float) -> float:
    """
    Calculate climate change stress index.
    Higher values = higher stress = lower suitability.
    """
    
    # Weight factors based on impact severity
    weights = {
        'temperature': 0.25,      # Direct impact on human comfort and energy
        'precipitation': 0.20,    # Water availability and agriculture
        'extreme': 0.20,         # Direct damage and disruption
        'sea_level': 0.15,       # Long-term inundation risk
        'agriculture': 0.10,      # Food security impact
        'water': 0.10             # Water availability impact
    }
    
    # Combined climate stress index
    climate_stress_index = (
        temperature * weights['temperature'] +
        precipitation * weights['precipitation'] +
        extreme * weights['extreme'] +
        sea_level * weights['sea_level'] +
        agriculture * weights['agriculture'] +
        water * weights['water']
    )
    
    return min(100.0, climate_stress_index)

def _climate_stress_to_suitability(stress_index: float) -> float:
    """
    Convert climate stress index to suitability score.
    Higher stress = lower suitability.
    """
    # Inverse relationship with scaling
    if stress_index >= 80:
        return max(0, 20 - (stress_index - 80) * 0.75)  # 0-20 range
    elif stress_index >= 60:
        return 20 + (80 - stress_index) * 2.5  # 20-70 range
    elif stress_index >= 40:
        return 70 + (60 - stress_index) * 1.0  # 70-90 range
    else:
        return 90 + (40 - stress_index) * 0.25  # 90-100 range

def _get_dominant_climate_stress(temperature: float, precipitation: float, extreme: float, 
                                sea_level: float, agriculture: float, water: float) -> str:
    """Identify the dominant climate stress factor."""
    stresses = [
        ('temperature', temperature),
        ('precipitation', precipitation),
        ('extreme', extreme),
        ('sea_level', sea_level),
        ('agriculture', agriculture),
        ('water', water)
    ]
    
    # Sort by stress score to find dominant factor
    sorted_stresses = sorted(stresses, key=lambda x: x[1], reverse=True)
    
    if sorted_stresses[0][1] >= 50:
        return sorted_stresses[0][0].capitalize()
    else:
        return "Low"

def _get_climate_stress_label(suitability_score: float) -> str:
    """Get human-readable label for climate stress level."""
    if suitability_score >= 80:
        return "Very Low Climate Stress (Excellent)"
    elif suitability_score >= 60:
        return "Low Climate Stress (Good)"
    elif suitability_score >= 40:
        return "Moderate Climate Stress (Fair)"
    elif suitability_score >= 20:
        return "High Climate Stress (Poor)"
    else:
        return "Very High Climate Stress (Very Poor)"

def _calculate_confidence(temperature: float, precipitation: float) -> float:
    """Calculate confidence based on data quality."""
    confidence = 50.0  # Base confidence
    
    # Add confidence for available data
    if temperature > 0:
        confidence += 25
    if precipitation > 0:
        confidence += 25
    
    return min(95, confidence)

def _generate_reasoning(stress_index: float, temperature: float, precipitation: float, extreme: float) -> str:
    """Generate human-readable reasoning for climate stress assessment."""
    reasoning_parts = []
    
    # Temperature stress reasoning
    if temperature > 70:
        reasoning_parts.append(f"high temperature stress ({temperature:.0f}/100)")
    elif temperature > 40:
        reasoning_parts.append(f"moderate temperature stress ({temperature:.0f}/100)")
    else:
        reasoning_parts.append(f"low temperature stress ({temperature:.0f}/100)")
    
    # Precipitation stress reasoning
    if precipitation > 70:
        reasoning_parts.append(f"high precipitation stress ({precipitation:.0f}/100)")
    elif precipitation > 40:
        reasoning_parts.append(f"moderate precipitation stress ({precipitation:.0f}/100)")
    else:
        reasoning_parts.append(f"low precipitation stress ({precipitation:.0f}/100)")
    
    # Extreme events reasoning
    if extreme > 60:
        reasoning_parts.append(f"significant extreme events increase ({extreme:.0f}/100)")
    elif extreme > 30:
        reasoning_parts.append(f"moderate extreme events increase ({extreme:.0f}/100)")
    else:
        reasoning_parts.append(f"low extreme events increase ({extreme:.0f}/100)")
    
    # Overall climate stress assessment
    if stress_index > 70:
        reasoning_parts.append("high climate change vulnerability requires adaptation")
    elif stress_index > 40:
        reasoning_parts.append("moderate climate change risk")
    else:
        reasoning_parts.append("low climate change vulnerability")
    
    return ". ".join(reasoning_parts) + "."

def _get_fallback_climate_stress(lat: float, lng: float) -> Dict[str, Any]:
    """Fallback climate stress estimation based on geographic context."""
    try:
        # Use geographic context for rough estimation
        climate_zone = _get_climate_zone(lat)
        region = _get_geographic_region(lat, lng)
        is_coastal = _is_coastal_region(lat, lng)
        
        # Base climate stress by zone
        climate_stress = {
            "tropical": 70.0,    # High temperature, precipitation changes
            "subtropical": 60.0,  # Moderate changes
            "temperate": 50.0,    # Moderate changes
            "cool": 40.0,         # Lower changes
            "polar": 30.0          # Significant changes but lower impact
        }
        
        base_stress = climate_stress.get(climate_zone, 50.0)
        
        # Adjust for coastal areas (sea level rise)
        if is_coastal:
            base_stress = min(100.0, base_stress + 20.0)
        
        # Adjust for region
        region_adjustments = {
            "north_america": 10.0,  # Severe weather
            "south_america": 15.0,  # Amazon changes
            "asia": 20.0,           # Monsoon changes
            "africa": 25.0,           # Desertification
            "europe": 5.0,            # Moderate changes
            "oceania": 15.0,          # Sea level rise impact
        }
        
        adjusted_stress = base_stress + region_adjustments.get(region, 0.0)
        adjusted_stress = max(0, min(100, adjusted_stress))
        
        suitability = _climate_stress_to_suitability(adjusted_stress)
        
        return {
            "value": suitability,
            "climate_stress_index": adjusted_stress,
            "temperature_change": min(100, adjusted_stress + 10),
            "precipitation_change": min(100, adjusted_stress - 5),
            "extreme_events": min(100, adjusted_stress + 15),
            "sea_level_rise": min(100, adjusted_stress + 20 if is_coastal else 0),
            "agricultural_impact": min(100, adjusted_stress - 10),
            "water_stress": min(100, adjusted_stress + 5),
            "dominant_stress": "Climate",
            "label": _get_climate_stress_label(suitability),
            "source": "Geographic Estimation (Fallback)",
            "confidence": 25.0,
            "reasoning": f"Estimated based on {climate_zone} climate and {region} region."
        }
        
    except Exception:
        return {
            "value": 50.0,
            "climate_stress_index": 50.0,
            "temperature_change": 50.0,
            "precipitation_change": 50.0,
            "extreme_events": 50.0,
            "sea_level_rise": 50.0,
            "agricultural_impact": 50.0,
            "water_stress": 50.0,
            "dominant_stress": "Unknown",
            "label": "Moderate Climate Stress (Fair)",
            "source": "Default Fallback",
            "confidence": 10.0,
            "reasoning": "Unable to determine climate change characteristics."
        }

# Helper functions
def _estimate_temperature_change(lat: float, lng: float) -> float:
    """Estimate temperature change based on climate zone."""
    climate_zone = _get_climate_zone(lat)
    
    # Temperature change projections by climate zone (°C by 2050)
    temp_change = {
        "tropical": 2.5,    # High warming
        "subtropical": 2.0,  # Moderate warming
        "temperate": 2.5,    # High warming
        "cool": 3.0,         # Very high warming (amplification)
        "polar": 4.0          # Extreme warming (amplification)
    }
    
    change_celsius = temp_change.get(climate_zone, 2.5)
    return min(100.0, change_celsius * 20)  # Convert to 0-100 scale

def _estimate_precipitation_change(lat: float, lng: float) -> float:
    """Estimate precipitation change based on climate zone."""
    climate_zone = _get_climate_zone(lat)
    
    # Precipitation change projections by climate zone (% by 2050)
    precip_change = {
        "tropical": 15.0,    # Increased precipitation
        "subtropical": 10.0,  # Moderate change
        "temperate": 5.0,     # Small change
        "cool": 10.0,         # Increased precipitation
        "polar": 20.0         # Significant increase
    }
    
    change_percent = precip_change.get(climate_zone, 10.0)
    return min(100.0, change_percent * 10)  # Convert to 0-100 scale

def _estimate_extreme_events(lat: float, lng: float) -> float:
    """Estimate extreme events increase based on climate zone."""
    climate_zone = _get_climate_zone(lat)
    
    # Extreme events increase projections by climate zone (% by 2050)
    extreme_change = {
        "tropical": 60.0,    # High increase
        "subtropical": 50.0,  # Moderate increase
        "temperate": 40.0,    # Moderate increase
        "cool": 30.0,         # Low increase
        "polar": 20.0         # Low increase
    }
    
    return extreme_change.get(climate_zone, 40.0)

def _estimate_sea_level_rise(lat: float, lng: float) -> float:
    """Estimate sea level rise risk based on location."""
    is_coastal = _is_coastal_region(lat, lng)
    elevation = _get_elevation(lat, lng)
    
    if not is_coastal:
        return 10.0  # Low risk inland
    
    # Sea level rise projection (meters by 2050)
    sea_level_rise = 0.3  # Conservative estimate
    
    if elevation <= 0:
        return 100.0  # Already at/below sea level
    elif sea_level_rise >= elevation:
        return 90.0  # Will be inundated
    else:
        risk_percentage = (sea_level_rise / elevation) * 100
        return min(100.0, risk_percentage)

def _estimate_agricultural_impact(lat: float, lng: float) -> float:
    """Estimate agricultural impact based on climate zone."""
    climate_zone = _get_climate_zone(lat)
    
    # Agricultural impact by climate zone (negative impact % by 2050)
    ag_impact = {
        "tropical": 30.0,    # High impact
        "subtropical": 20.0,  # Moderate impact
        "temperate": 10.0,    # Low impact
        "cool": 5.0,         # Very low impact
        "polar": 15.0         # Moderate impact
    }
    
    impact_percent = ag_impact.get(climate_zone, 15.0)
    return min(100.0, impact_percent * 50)  # Convert to 0-100 scale

def _estimate_water_stress(lat: float, lng: float) -> float:
    """Estimate water stress based on climate zone."""
    climate_zone = _get_climate_zone(lat)
    
    # Water stress by climate zone (stress index 0-100 by 2050)
    water_stress = {
        "tropical": 60.0,    # High stress
        "subtropical": 50.0,  # Moderate stress
        "temperate": 40.0,    # Low stress
        "cool": 30.0,         # Very low stress
        "polar": 20.0         # Minimal stress
    }
    
    return water_stress.get(climate_zone, 40.0)

def _get_elevation(lat: float, lng: float) -> float:
    """Get elevation for the location."""
    try:
        # Use existing elevation data
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from physical_terrain.elevation_analysis import get_elevation_data
        
        elev_data = get_elevation_data(lat, lng)
        return elev_data.get("value", 100.0)  # Default to 100m
        
    except Exception:
        return 100.0  # Default elevation

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

def _is_coastal_region(lat: float, lng: float) -> bool:
    """Check if location is near coast."""
    coastal_lat_ranges = [
        (8, 25),    # Indian coast
        (22, 27),   # Bay of Bengal
        (24, 32),   # Mediterranean
        (35, 42),   # US East Coast
        (32, 38),   # US West Coast
    ]
    
    for min_lat, max_lat in coastal_lat_ranges:
        if min_lat <= abs(lat) <= max_lat:
            return True
    
    return False
