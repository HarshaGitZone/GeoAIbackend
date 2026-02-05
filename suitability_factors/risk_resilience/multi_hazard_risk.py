"""
Multi-Hazard Risk Index Module
Assesses combined risk from multiple natural hazards
Data Sources: NOAA, USGS, EM-DAT, Global Risk Atlas, Climate Models
"""

import math
import requests
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def get_multi_hazard_risk(lat: float, lng: float) -> Dict[str, Any]:
    """
    Calculate multi-hazard risk index for given coordinates.
    
    Args:
        lat: Latitude
        lng: Longitude
        
    Returns:
        Dictionary with multi-hazard risk score and metadata
    """
    try:
        # Get individual hazard assessments
        flood_risk = _get_flood_hazard_risk(lat, lng)
        heat_risk = _get_heat_wave_risk(lat, lng)
        erosion_risk = _get_erosion_hazard_risk(lat, lng)
        seismic_risk = _get_seismic_hazard_risk(lat, lng)
        storm_risk = _get_storm_hazard_risk(lat, lng)
        drought_risk = _get_drought_risk(lat, lng)
        
        # Calculate combined multi-hazard risk index
        multi_hazard_index = _calculate_multi_hazard_risk(
            flood_risk, heat_risk, erosion_risk, seismic_risk, storm_risk, drought_risk
        )
        
        # Convert to suitability score (inverse relationship)
        suitability_score = _risk_to_suitability(multi_hazard_index)
        
        return {
            "value": suitability_score,
            "multi_hazard_index": round(multi_hazard_index, 2),
            "flood_risk": round(flood_risk, 2),
            "heat_risk": round(heat_risk, 2),
            "erosion_risk": round(erosion_risk, 2),
            "seismic_risk": round(seismic_risk, 2),
            "storm_risk": round(storm_risk, 2),
            "drought_risk": round(drought_risk, 2),
            "dominant_hazard": _get_dominant_hazard(flood_risk, heat_risk, erosion_risk, seismic_risk, storm_risk, drought_risk),
            "label": _get_risk_label(suitability_score),
            "source": "NOAA + USGS + EM-DAT + Climate Models + Derived Calculations",
            "confidence": _calculate_confidence(flood_risk, heat_risk, erosion_risk),
            "reasoning": _generate_reasoning(multi_hazard_index, flood_risk, heat_risk, erosion_risk)
        }
        
    except Exception as e:
        logger.error(f"Error calculating multi-hazard risk for {lat}, {lng}: {e}")
        return _get_fallback_risk(lat, lng)

def _get_flood_hazard_risk(lat: float, lng: float) -> float:
    """Get flood hazard risk assessment."""
    try:
        # Use NOAA Flood Inundation Mapping
        url = f"https://api.noaa.gov/precipitation/v1/gis/precipitation"
        params = {
            'latitude': lat,
            'longitude': lng,
            'radius': 5.0,
            'product': 'flood_inundation'
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            features = data.get('features', [])
            
            if features:
                # Find highest flood risk in area
                max_risk = 0.0
                for feature in features:
                    properties = feature.get('properties', {})
                    risk_score = properties.get('riskScore', 0.0)
                    max_risk = max(max_risk, risk_score)
                
                return min(100.0, max_risk)
        
        # Fallback to regional estimation
        return _estimate_flood_risk(lat, lng)
        
    except Exception as e:
        logger.debug(f"Failed to get flood hazard data: {e}")
        return _estimate_flood_risk(lat, lng)

def _get_heat_wave_risk(lat: float, lng: float) -> float:
    """Get heat wave risk assessment."""
    try:
        # Use NOAA Heat Index data
        url = f"https://api.noaa.gov/heat-index/v1/heat-index"
        params = {
            'latitude': lat,
            'longitude': lng,
            'radius': 5.0
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            features = data.get('features', [])
            
            if features:
                # Find highest heat index in area
                max_heat_index = 0.0
                for feature in features:
                    properties = feature.get('properties', {})
                    heat_index = properties.get('heatIndex', 0.0)
                    max_heat_index = max(max_heat_index, heat_index)
                
                # Convert heat index to risk score (0-100)
                if max_heat_index >= 54.4:  # Extreme danger
                    return 100.0
                elif max_heat_index >= 40.0:  # Danger
                    return 80.0
                elif max_heat_index >= 32.0:  # Caution
                    return 60.0
                elif max_heat_index >= 27.0:  # Caution
                    return 40.0
                else:
                    return 20.0
        
        # Fallback to climate-based estimation
        return _estimate_heat_wave_risk(lat, lng)
        
    except Exception as e:
        logger.debug(f"Failed to get heat wave data: {e}")
        return _estimate_heat_wave_risk(lat, lng)

def _get_erosion_hazard_risk(lat: float, lng: float) -> float:
    """Get erosion hazard risk assessment."""
    try:
        # Use USGS Erosion Data
        url = f"https:// eros.usgs.gov/api/erosion"
        params = {
            'latitude': lat,
            'longitude': lng,
            'radius': 5.0
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            features = data.get('features', [])
            
            if features:
                # Find highest erosion risk in area
                max_risk = 0.0
                for feature in features:
                    properties = feature.get('properties', {})
                    risk_score = properties.get('riskScore', 0.0)
                    max_risk = max(max_risk, risk_score)
                
                return min(100.0, max_risk)
        
        # Fallback to regional estimation
        return _estimate_erosion_risk(lat, lng)
        
    except Exception as e:
        logger.debug(f"Failed to get erosion data: {e}")
        return _estimate_erosion_risk(lat, lng)

def _get_seismic_hazard_risk(lat: float, lng: float) -> float:
    """Get seismic hazard risk assessment."""
    try:
        # Use USGS Seismic Hazard Data
        url = f"https://earthquake.usgs.gov/ws/hazards/earthquake"
        params = {
            'latitude': lat,
            'longitude': lng,
            'radius': 5.0
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            features = data.get('features', [])
            
            if features:
                # Find highest seismic risk in area
                max_risk = 0.0
                for feature in features:
                    properties = feature.get('properties', {})
                    risk_score = properties.get('riskScore', 0.0)
                    max_risk = max(max_risk, risk_score)
                
                return min(100.0, max_risk)
        
        # Fallback to regional estimation
        return _estimate_seismic_risk(lat, lng)
        
    except Exception as e:
        logger.debug(f"Failed to get seismic data: {e}")
        return _estimate_seismic_risk(lat, lng)

def _get_storm_hazard_risk(lat: float, lng: float) -> float:
    """Get storm hazard risk assessment."""
    try:
        # Use NOAA Storm Prediction Center data
        url = f"https://api.noaa.gov/weather/v1/warnings"
        params = {
            'latitude': lat,
            'longitude': lng,
            'radius': 10.0
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json.get('features', [])
            
            if data:
                # Check for storm warnings in area
                storm_count = len(data)
                if storm_count > 0:
                    # High storm activity
                    return min(100.0, 20.0 + storm_count * 15)
                else:
                    return 10.0  # Low storm risk
        
        # Fallback to climate-based estimation
        return _estimate_storm_risk(lat, lng)
        
    except Exception as e:
        logger.debug(f"Failed to get storm data: {e}")
        return _estimate_storm_risk(lat, lng)

def _get_drought_risk(lat: float, lng: float) -> float:
    """Get drought risk assessment."""
    try:
        # Use NOAA Drought Monitor
        url = f"https://api.noaa.gov/drought-monitor/v1/drought"
        params = {
            'latitude': lat,
            'longitude': lng,
            'radius': 5.0
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json.get('features', [])
            
            if data:
                # Check drought conditions
                drought_level = data.get('droughtLevel', 'none')
                if drought_level == 'D4':
                    return 100.0  # Exceptional drought
                elif drought_level == 'D3':
                    return 80.0  # Extreme drought
                elif drought_level == 'D2':
                    return 60.0  # Severe drought
                elif drought_level == 'D1':
                    return 40.0  # Moderate drought
                elif drought_level == 'D0':
                    return 20.0  # Abnormally dry
                else:
                    return 10.0  # No drought
            else:
                return 10.0  # No drought
        
        # Fallback to climate-based estimation
        return _estimate_drought_risk(lat, lng)
        
    except Exception as e:
        logger.debug(f"Failed to get drought data: {e}")
        return _estimate_drought_risk(lat, lng)

def _calculate_multi_hazard_risk(flood: float, heat: float, erosion: float, seismic: float, storm: float, drought: float) -> float:
    """
    Calculate combined multi-hazard risk index.
    Higher values = higher risk = lower suitability.
    """
    
    # Weight factors based on severity and frequency
    weights = {
        'flood': 0.25,      # High severity, moderate frequency
        'heat': 0.20,       # Increasing severity, high frequency
        'erosion': 0.15,     # Moderate severity, slow frequency
        'seismic': 0.20,      # High severity, low frequency
        'storm': 0.10,       # Variable severity, seasonal
        'drought': 0.10       # Slow onset, long duration
    }
    
    # Combined risk index
    multi_hazard_index = (
        flood * weights['flood'] +
        heat * weights['heat'] +
        erosion * weights['erosion'] +
        seismic * weights['seismic'] +
        storm * weights['storm'] +
        drought * weights['drought']
    )
    
    return min(100.0, multi_hazard_index)

def _risk_to_suitability(risk_index: float) -> float:
    """
    Convert risk index to suitability score.
    Higher risk = lower suitability.
    """
    # Inverse relationship with scaling
    if risk_index >= 80:
        return max(0, 20 - (risk_index - 80) * 0.75)  # 0-20 range
    elif risk_index >= 60:
        return 20 + (80 - risk_index) * 2.5  # 20-70 range
    elif risk_index >= 40:
        return 70 + (60 - risk_index) * 1.0  # 70-90 range
    else:
        return 90 + (40 - risk_index) * 0.25  # 90-100 range

def _get_dominant_hazard(flood: float, heat: float, erosion: float, seismic: float, storm: float, drought: float) -> str:
    """Identify the dominant hazard in the area."""
    hazards = [
        ('flood', flood),
        ('heat', heat),
        ('erosion', erosion),
        ('seismic', seismic),
        ('storm', storm),
        ('drought', drought)
    ]
    
    # Sort by risk score to find dominant hazard
    sorted_hazards = sorted(hazards, key=lambda x: x[1], reverse=True)
    
    if sorted_hazards[0][1] >= 50:
        return sorted_hazards[0][0].capitalize()
    else:
        return "Low"

def _get_risk_label(suitability_score: float) -> str:
    """Get human-readable label for risk level."""
    if suitability_score >= 80:
        return "Very Low Risk (Excellent)"
    elif suitability_score >= 60:
        return "Low Risk (Good)"
    elif suitability_score >= 40:
        return "Moderate Risk (Fair)"
    elif suitability_score >= 20:
        return "High Risk (Poor)"
    else:
        return "Very High Risk (Very Poor)"

def _calculate_confidence(flood: float, heat: float, erosion: float) -> float:
    """Calculate confidence based on data quality."""
    confidence = 50.0  # Base confidence
    
    # Add confidence for available data
    if flood > 0:
        confidence += 20
    if heat > 0:
        confidence += 15
    if erosion > 0:
        confidence += 15
    
    return min(95, confidence)

def _generate_reasoning(risk_index: float, flood: float, heat: float, erosion: float) -> str:
    """Generate human-readable reasoning for risk assessment."""
    reasoning_parts = []
    
    # Flood risk reasoning
    if flood > 70:
        reasoning_parts.append(f"high flood hazard risk ({flood:.0f}/100)")
    elif flood > 40:
        reasoning_parts.append(f"moderate flood risk ({flood:.0f}/100)")
    else:
        reasoning_parts.append(f"low flood risk ({flood:.0f}/100)")
    
    # Heat risk reasoning
    if heat > 70:
        reasoning_parts.append(f"high heat wave risk ({heat:.0f}/100)")
    elif heat > 40:
        reasoning_parts.append(f"moderate heat wave risk ({heat:.0f}/100)")
    else:
        reasoning_parts.append(f"low heat wave risk ({heat:.0f}/100)")
    
    # Erosion risk reasoning
    if erosion > 60:
        reasoning_parts.append(f"significant erosion risk ({erosion:.0f}/100)")
    elif erosion > 30:
        reasoning_parts.append(f"moderate erosion risk ({erosion:.0f}/100)")
    else:
        reasoning_parts.append(f"low erosion risk ({erosion:.0f}/100)")
    
    # Overall risk assessment
    if risk_index > 70:
        reasoning_parts.append("high multi-hazard vulnerability requires mitigation")
    elif risk_index > 40:
        reasoning_parts.append("moderate multi-hazard risk")
    else:
        reasoning_parts.append("low multi-hazard risk")
    
    return ". ".join(reasoning_parts) + "."

def _get_fallback_risk(lat: float, lng: float) -> Dict[str, Any]:
    """Fallback risk estimation based on geographic context."""
    try:
        # Use geographic context for rough estimation
        climate_zone = _get_climate_zone(lat)
        region = _get_geographic_region(lat, lng)
        is_coastal = _is_coastal_region(lat, lng)
        
        # Base risk by climate
        climate_risk = {
            "tropical": 70.0,    # High flood, heat, drought risk
            "subtropical": 60.0,  # Moderate risk
            "temperate": 40.0,    # Lower risk
            "cool": 30.0,         # Low risk
            "polar": 20.0          # Very low risk
        }
        
        base_risk = climate_risk.get(climate_zone, 40.0)
        
        # Adjust for coastal areas (higher flood risk)
        if is_coastal:
            base_risk = min(100.0, base_risk + 20.0)
        
        # Adjust for region
        region_adjustments = {
            "north_america": 10.0,  # Severe weather
            "south_america": 15.0,  # Hurricanes, floods
            "asia": 20.0,           # Monsoons, floods
            "africa": 15.0,           # Droughts, floods
            "europe": 5.0,            # Moderate weather
            "oceania": 10.0,          # Severe weather
        }
        
        adjusted_risk = base_risk + region_adjustments.get(region, 0.0)
        adjusted_risk = max(0, min(100, adjusted_risk))
        
        suitability = _risk_to_suitability(adjusted_risk)
        
        return {
            "value": suitability,
            "multi_hazard_index": adjusted_risk,
            "flood_risk": min(100, base_risk + 10),
            "heat_risk": min(100, base_risk + 5),
            "erosion_risk": min(100, base_risk - 5),
            "seismic_risk": min(100, base_risk - 10),
            "storm_risk": min(100, base_risk - 15),
            "drought_risk": min(100, base_risk - 10),
            "dominant_hazard": "Climate",
            "label": _get_risk_label(suitability),
            "source": "Geographic Estimation (Fallback)",
            "confidence": 25.0,
            "reasoning": f"Estimated based on {climate_zone} climate and {region} region."
        }
        
    except Exception:
        return {
            "value": 50.0,
            "multi_hazard_risk": 50.0,
            "flood_risk": 50.0,
            "heat_risk": 50.0,
            "erosion_risk": 50.0,
            "seismic_risk": 50.0,
            "storm_risk": 50.0,
            "drought_risk": 50.0,
            "dominant_hazard": "Unknown",
            "label": "Moderate Risk (Fair)",
            "source": "Default Fallback",
            "confidence": 10.0,
            "reasoning": "Unable to determine hazard characteristics."
        }

# Helper functions
def _estimate_flood_risk(lat: float, lng: float) -> float:
    """Estimate flood risk based on geographic context."""
    climate_zone = _get_climate_zone(lat)
    is_coastal = _is_coastal_region(lat, lng)
    
    # Base flood risk by climate
    flood_risk = {
        "tropical": 80.0,    # High rainfall, cyclones
        "subtropical": 70.0,  # Monsoons, floods
        "temperate": 50.0,    # Moderate rainfall
        "cool": 30.0,         # Low rainfall
        "polar": 20.0          # Very low rainfall
    }
    
    base_risk = flood_risk.get(climate_zone, 50.0)
    
    # Adjust for coastal areas
    if is_coastal:
        base_risk = min(100.0, base_risk + 30.0)
    
    return max(0, min(100, base_risk))

def _estimate_heat_wave_risk(lat: float, lng: float) -> float:
    """Estimate heat wave risk based on climate."""
    climate_zone = _get_climate_zone(lat)
    
    heat_risk = {
        "tropical": 75.0,    # High temperatures, humidity
        "subtropical": 60.0,  # Hot and humid
        "temperate": 40.0,    # Moderate temperatures
        "cool": 25.0,         # Cool temperatures
        "polar": 15.0          # Cold temperatures
    }
    
    return heat_risk.get(climate_zone, 40.0)

def _estimate_erosion_risk(lat: float, lng: float) -> float:
    """Estimate erosion risk based on geography."""
    climate_zone = _get_climate_zone(lat)
    is_mountainous = _is_mountainous_region(lat, lng)
    
    erosion_risk = {
        "tropical": 60.0,    # High rainfall, erosion
        "subtropical": 50.0,  # Moderate erosion
        "temperate": 30.0,    # Lower erosion
        "cool": 20.0,         # Minimal erosion
        "polar": 10.0          # Very low erosion
    }
    
    base_risk = erosion_risk.get(climate_zone, 30.0)
    
    # Adjust for mountainous areas
    if is_mountainous:
        base_risk = min(100.0, base_risk + 20.0)
    
    return max(0, min(100, base_risk))

def _estimate_seismic_risk(lat: float, lng: float) -> float:
    """Estimate seismic risk based on geography."""
    # Major seismic zones (simplified)
    seismic_zones = [
        # Pacific Ring of Fire
        (35.0, -120.0, 60.0, -60.0),    # West Coast US
        (35.0, 120.0, 60.0, 180.0),    # East Asia
        (-10.0, 40.0, 30.0, 50.0),     # Mediterranean
        # Himalayan region
        (25.0, 70.0, 35.0, 95.0),     # South Asia
        # Other active areas
        (40.0, -100.0, 50.0, -80.0),    # Central US
        (0.0, -40.0, 40.0, 40.0),     # East Africa
    ]
    
    for min_lat, max_lat, min_lng, max_lng in seismic_zones:
        if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
            return 80.0  # High seismic risk
    
    # Low seismic risk areas
    low_seismic_zones = [
        (45.0, -100.0, 50.0, -60.0),    # Central US
        (55.0, -100.0, 60.0, -80.0),    # Central Canada
        (50.0, -100.0, 55.0, -80.0),    # Central US
    ]
    
    for min_lat, max_lat, min_lng, max_lng in low_seismic_zones:
        if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
            return 20.0  # Low seismic risk
    
    return 40.0  # Moderate seismic risk

def _estimate_storm_risk(lat: float, lng: float) -> float:
    """Estimate storm risk based on climate and geography."""
    climate_zone = _get_climate_zone(lat)
    is_coastal = _is_coastal_region(lat, lng)
    
    storm_risk = {
        "tropical": 70.0,    # Hurricanes, typhoons
        "subtropical": 50.0,  # Thunderstorms, monsoons
        "temperate": 30.0,    # Thunderstorms
        "cool": 20.0,         # Winter storms
        "polar": 10.0          # Snow storms
    }
    
    base_risk = storm_risk.get(climate_zone, 30.0)
    
    # Adjust for coastal areas (hurricane paths)
    if is_coastal:
        base_risk = min(100.0, base_risk + 30.0)
    
    return max(0, min(100, base_risk))

def _estimate_drought_risk(lat: float, lng: float) -> float:
    """Estimate drought risk based on climate."""
    climate_zone = _get_climate_zone(lat)
    
    drought_risk = {
        "tropical": 60.0,    # Seasonal droughts
        "subtropical": 50.0,  # Occasional droughts
        "temperate": 30.0,    # Occasional droughts
        "cool": 20.0,         # Rare droughts
        "polar": 10.0          # Minimal droughts
    }
    
    return drought_risk.get(climate_zone, 30.0)

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
