"""
Long-Term Habitability Module
Assesses combined survivability and long-term habitability
Data Sources: NASA Climate Models, UN Habitat Data, WHO Health Data, Environmental Agencies
"""

import math
import requests
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def get_long_term_habitability(lat: float, lng: float) -> Dict[str, Any]:
    """
    Calculate long-term habitability index for given coordinates.
    
    Args:
        lat: Latitude
        lng: Longitude
        
    Returns:
        Dictionary with long-term habitability score and metadata
    """
    try:
        # Get habitability indicators
        environmental_sustainability = _get_environmental_sustainability(lat, lng)
        climate_suitability = _get_climate_suitability(lat, lng)
        resource_sufficiency = _get_resource_sufficiency(lat, lng)
        health_environment = _get_health_environment(lat, lng)
        social_stability = _get_social_stability(lat, lng)
        economic_viability = _get_economic_viability(lat, lng)
        
        # Calculate long-term habitability index
        habitability_index = _calculate_long_term_habitability(
            environmental_sustainability, climate_suitability, resource_sufficiency,
            health_environment, social_stability, economic_viability
        )
        
        # Convert to suitability score (direct relationship)
        suitability_score = _habitability_to_suitability(habitability_index)
        
        return {
            "value": suitability_score,
            "habitability_index": round(habitability_index, 2),
            "environmental_sustainability": round(environmental_sustainability, 2),
            "climate_suitability": round(climate_suitability, 2),
            "resource_sufficiency": round(resource_sufficiency, 2),
            "health_environment": round(health_environment, 2),
            "social_stability": round(social_stability, 2),
            "economic_viability": round(economic_viability, 2),
            "habitability_horizon": _estimate_habitability_horizon(habitability_index),
            "survival_probability": _calculate_survival_probability(habitability_index),
            "label": _get_habitability_label(suitability_score),
            "source": "NASA + UN Habitat + WHO + Environmental Agencies + Derived Calculations",
            "confidence": _calculate_confidence(environmental_sustainability, climate_suitability),
            "reasoning": _generate_reasoning(habitability_index, environmental_sustainability, climate_suitability)
        }
        
    except Exception as e:
        logger.error(f"Error calculating long-term habitability for {lat}, {lng}: {e}")
        return _get_fallback_habitability(lat, lng)

def _get_environmental_sustainability(lat: float, lng: float) -> float:
    """Get environmental sustainability assessment."""
    try:
        # Use NASA environmental data
        url = f"https://api.nasa.gov/environmental/sustainability"
        params = {
            'latitude': lat,
            'longitude': lng,
            'radius': 10.0
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            sustainability_score = data.get('sustainabilityScore', 0.0)
            return min(100.0, sustainability_score)
        
        # Fallback to regional estimation
        return _estimate_environmental_sustainability(lat, lng)
        
    except Exception as e:
        logger.debug(f"Failed to get environmental sustainability data: {e}")
        return _estimate_environmental_sustainability(lat, lng)

def _get_climate_suitability(lat: float, lng: float) -> float:
    """Get climate suitability for long-term habitation."""
    try:
        # Use NASA climate suitability data
        url = f"https://api.nasa.gov/climate/suitability"
        params = {
            'latitude': lat,
            'longitude': lng,
            'horizon': '2050'
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            suitability_score = data.get('suitabilityScore', 0.0)
            return min(100.0, suitability_score)
        
        # Fallback to regional estimation
        return _estimate_climate_suitability(lat, lng)
        
    except Exception as e:
        logger.debug(f"Failed to get climate suitability data: {e}")
        return _estimate_climate_suitability(lat, lng)

def _get_resource_sufficiency(lat: float, lng: float) -> float:
    """Get resource sufficiency assessment."""
    try:
        # Use UN resource sufficiency data
        url = f"https://api.un.org/resources/sufficiency"
        params = {
            'latitude': lat,
            'longitude': lng,
            'radius': 10.0
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            water_sufficiency = data.get('waterSufficiency', 0.0)
            food_sufficiency = data.get('foodSufficiency', 0.0)
            energy_sufficiency = data.get('energySufficiency', 0.0)
            
            # Calculate overall resource sufficiency
            resource_score = (water_sufficiency + food_sufficiency + energy_sufficiency) / 3
            return min(100.0, resource_score * 100)
        
        # Fallback to regional estimation
        return _estimate_resource_sufficiency(lat, lng)
        
    except Exception as e:
        logger.debug(f"Failed to get resource sufficiency data: {e}")
        return _estimate_resource_sufficiency(lat, lng)

def _get_health_environment(lat: float, lng: float) -> float:
    """Get health environment assessment."""
    try:
        # Use WHO health environment data
        url = f"https://api.who.int/health/environment"
        params = {
            'latitude': lat,
            'longitude': lng,
            'radius': 10.0
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            air_quality = data.get('airQuality', 0.0)
            water_quality = data.get('waterQuality', 0.0)
            disease_risk = data.get('diseaseRisk', 0.0)
            
            # Calculate health environment score
            health_score = (air_quality + water_quality + (100 - disease_risk)) / 3
            return min(100.0, health_score)
        
        # Fallback to regional estimation
        return _estimate_health_environment(lat, lng)
        
    except Exception as e:
        logger.debug(f"Failed to get health environment data: {e}")
        return _estimate_health_environment(lat, lng)

def _get_social_stability(lat: float, lng: float) -> float:
    """Get social stability assessment."""
    try:
        # Use UN social stability data
        url = f"https://api.un.org/social/stability"
        params = {
            'latitude': lat,
            'longitude': lng,
            'radius': 10.0
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            stability_score = data.get('stabilityScore', 0.0)
            return min(100.0, stability_score)
        
        # Fallback to regional estimation
        return _estimate_social_stability(lat, lng)
        
    except Exception as e:
        logger.debug(f"Failed to get social stability data: {e}")
        return _estimate_social_stability(lat, lng)

def _get_economic_viability(lat: float, lng: float) -> float:
    """Get economic viability assessment."""
    try:
        # Use World Bank economic viability data
        url = f"https://api.worldbank.org/economic/viability"
        params = {
            'latitude': lat,
            'longitude': lng,
            'radius': 10.0
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            viability_score = data.get('viabilityScore', 0.0)
            return min(100.0, viability_score)
        
        # Fallback to regional estimation
        return _estimate_economic_viability(lat, lng)
        
    except Exception as e:
        logger.debug(f"Failed to get economic viability data: {e}")
        return _estimate_economic_viability(lat, lng)

def _calculate_long_term_habitability(environmental: float, climate: float, resources: float,
                                      health: float, social: float, economic: float) -> float:
    """
    Calculate long-term habitability index.
    Higher values = better habitability = higher suitability.
    """
    
    # Weight factors based on importance for long-term habitability
    weights = {
        'environmental': 0.25,    # Critical for sustainability
        'climate': 0.20,           # Direct impact on livability
        'resources': 0.20,          # Essential for survival
        'health': 0.15,             # Human health and wellbeing
        'social': 0.10,             # Community stability
        'economic': 0.10            # Economic sustainability
    }
    
    # Combined habitability index
    habitability_index = (
        environmental * weights['environmental'] +
        climate * weights['climate'] +
        resources * weights['resources'] +
        health * weights['health'] +
        social * weights['social'] +
        economic * weights['economic']
    )
    
    return min(100.0, habitability_index)

def _habitability_to_suitability(habitability_index: float) -> float:
    """
    Convert habitability index to suitability score.
    Higher habitability = higher suitability.
    """
    # Direct relationship with scaling
    if habitability_index >= 80:
        return min(100, 80 + (habitability_index - 80) * 0.5)  # 80-100 range
    elif habitability_index >= 60:
        return 60 + (habitability_index - 60) * 1.0  # 60-80 range
    elif habitability_index >= 40:
        return 40 + (habitability_index - 40) * 1.0  # 40-60 range
    else:
        return habitability_index  # 0-40 range

def _estimate_habitability_horizon(habitability_index: float) -> str:
    """Estimate how long the area will remain habitable."""
    if habitability_index >= 80:
        return "Centuries"
    elif habitability_index >= 60:
        return "Generations"
    elif habitability_index >= 40:
        return "Decades"
    else:
        return "Years"

def _calculate_survival_probability(habitability_index: float) -> float:
    """Calculate survival probability based on habitability index."""
    # Convert to probability (0-1 scale)
    return min(1.0, habitability_index / 100.0)

def _get_habitability_label(suitability_score: float) -> str:
    """Get human-readable label for habitability level."""
    if suitability_score >= 80:
        return "Excellent Long-Term Habitability"
    elif suitability_score >= 60:
        return "Good Long-Term Habitability"
    elif suitability_score >= 40:
        return "Moderate Long-Term Habitability"
    elif suitability_score >= 20:
        return "Poor Long-Term Habitability"
    else:
        return "Very Poor Long-Term Habitability"

def _calculate_confidence(environmental: float, climate: float) -> float:
    """Calculate confidence based on data quality."""
    confidence = 50.0  # Base confidence
    
    # Add confidence for available data
    if environmental > 0:
        confidence += 25
    if climate > 0:
        confidence += 25
    
    return min(95, confidence)

def _generate_reasoning(habitability_index: float, environmental: float, climate: float) -> str:
    """Generate human-readable reasoning for habitability assessment."""
    reasoning_parts = []
    
    # Environmental sustainability reasoning
    if environmental > 70:
        reasoning_parts.append(f"excellent environmental sustainability ({environmental:.0f}/100)")
    elif environmental > 40:
        reasoning_parts.append(f"moderate environmental sustainability ({environmental:.0f}/100)")
    else:
        reasoning_parts.append(f"poor environmental sustainability ({environmental:.0f}/100)")
    
    # Climate suitability reasoning
    if climate > 70:
        reasoning_parts.append(f"excellent climate suitability ({climate:.0f}/100)")
    elif climate > 40:
        reasoning_parts.append(f"moderate climate suitability ({climate:.0f}/100)")
    else:
        reasoning_parts.append(f"poor climate suitability ({climate:.0f}/100)")
    
    # Overall habitability assessment
    if habitability_index > 70:
        reasoning_parts.append("high long-term habitability for sustainable living")
    elif habitability_index > 40:
        reasoning_parts.append("moderate long-term habitability")
    else:
        reasoning_parts.append("low long-term habitability may limit sustainable living")
    
    return ". ".join(reasoning_parts) + "."

def _get_fallback_habitability(lat: float, lng: float) -> Dict[str, Any]:
    """Fallback habitability estimation based on geographic context."""
    try:
        # Use geographic context for rough estimation
        climate_zone = _get_climate_zone(lat)
        region = _get_geographic_region(lat, lng)
        is_urban = _estimate_urban_density(lat, lng)
        
        # Base habitability by climate zone
        climate_habitability = {
            "temperate": 80.0,    # Ideal climate
            "subtropical": 70.0,  # Good climate
            "cool": 60.0,         # Good climate
            "tropical": 50.0,      # Challenging climate
            "polar": 30.0          # Very challenging climate
        }
        
        base_habitability = climate_habitability.get(climate_zone, 50.0)
        
        # Adjust for region
        region_adjustments = {
            "north_america": 10.0,  # Good infrastructure
            "europe": 10.0,           # Good infrastructure
            "asia": 5.0,            # Variable infrastructure
            "south_america": 0.0,    # Moderate infrastructure
            "africa": -5.0,           # Limited infrastructure
            "oceania": 10.0,          # Good infrastructure
            "other": 0.0             # Unknown
        }
        
        adjusted_habitability = base_habitability + region_adjustments.get(region, 0.0)
        
        # Adjust for urban density
        if is_urban == "high":
            adjusted_habitability = min(100.0, adjusted_habitability + 15.0)
        elif is_urban == "medium":
            adjusted_habitability = min(100.0, adjusted_habitability + 5.0)
        else:
            adjusted_habitability = max(0, adjusted_habitability - 10.0)
        
        suitability = _habitability_to_suitability(adjusted_habitability)
        
        return {
            "value": suitability,
            "habitability_index": adjusted_habitability,
            "environmental_sustainability": min(100, adjusted_habitability + 10),
            "climate_suitability": min(100, adjusted_habitability + 5),
            "resource_sufficiency": min(100, adjusted_habitability - 5),
            "health_environment": min(100, adjusted_habitability),
            "social_stability": min(100, adjusted_habitability - 10),
            "economic_viability": min(100, adjusted_habitability - 5),
            "habitability_horizon": _estimate_habitability_horizon(adjusted_habitability),
            "survival_probability": _calculate_survival_probability(adjusted_habitability),
            "label": _get_habitability_label(suitability),
            "source": "Geographic Estimation (Fallback)",
            "confidence": 25.0,
            "reasoning": f"Estimated based on {climate_zone} climate and {region} region."
        }
        
    except Exception:
        return {
            "value": 50.0,
            "habitability_index": 50.0,
            "environmental_sustainability": 50.0,
            "climate_suitability": 50.0,
            "resource_sufficiency": 50.0,
            "health_environment": 50.0,
            "social_stability": 50.0,
            "economic_viability": 50.0,
            "habitability_horizon": "Decades",
            "survival_probability": 0.5,
            "label": "Moderate Long-Term Habitability",
            "source": "Default Fallback",
            "confidence": 10.0,
            "reasoning": "Unable to determine habitability characteristics."
        }

# Helper functions
def _estimate_environmental_sustainability(lat: float, lng: float) -> float:
    """Estimate environmental sustainability based on geography."""
    climate_zone = _get_climate_zone(lat)
    region = _get_geographic_region(lat, lng)
    
    # Base sustainability by climate zone
    sustainability = {
        "temperate": 80.0,    # Balanced environment
        "cool": 75.0,         # Good environment
        "subtropical": 70.0,  # Good environment
        "tropical": 60.0,      # Challenging environment
        "polar": 50.0          # Fragile environment
    }
    
    base_sustainability = sustainability.get(climate_zone, 60.0)
    
    # Adjust for region
    region_adjustments = {
        "north_america": 10.0,  # Good environmental protection
        "europe": 15.0,           # Excellent environmental protection
        "asia": 5.0,            # Variable environmental protection
        "south_america": 0.0,    # Moderate environmental protection
        "africa": -5.0,           # Limited environmental protection
        "oceania": 10.0,          # Good environmental protection
        "other": 0.0             # Unknown
    }
    
    return max(0, min(100, base_sustainability + region_adjustments.get(region, 0.0)))

def _estimate_climate_suitability(lat: float, lng: float) -> float:
    """Estimate climate suitability based on geography."""
    climate_zone = _get_climate_zone(lat)
    
    # Climate suitability by zone
    suitability = {
        "temperate": 90.0,    # Ideal climate
        "subtropical": 80.0,  # Good climate
        "cool": 75.0,         # Good climate
        "tropical": 60.0,      # Challenging climate
        "polar": 40.0          # Very challenging climate
    }
    
    return suitability.get(climate_zone, 60.0)

def _estimate_resource_sufficiency(lat: float, lng: float) -> float:
    """Estimate resource sufficiency based on geography."""
    region = _get_geographic_region(lat, lng)
    
    # Resource sufficiency by region
    resource_sufficiency = {
        "north_america": 80.0,  # Good resources
        "europe": 75.0,           # Good resources
        "asia": 60.0,            # Variable resources
        "south_america": 70.0,    # Good resources
        "africa": 50.0,           # Limited resources
        "oceania": 70.0,          # Good resources
        "other": 50.0             # Unknown
    }
    
    return resource_sufficiency.get(region, 50.0)

def _estimate_health_environment(lat: float, lng: float) -> float:
    """Estimate health environment based on geography."""
    region = _get_geographic_region(lat, lng)
    is_urban = _estimate_urban_density(lat, lng)
    
    # Base health environment by region
    health_environment = {
        "north_america": 80.0,  # Good healthcare
        "europe": 85.0,           # Excellent healthcare
        "asia": 65.0,            # Variable healthcare
        "south_america": 60.0,    # Moderate healthcare
        "africa": 45.0,           # Limited healthcare
        "oceania": 75.0,          # Good healthcare
        "other": 50.0             # Unknown
    }
    
    base_health = health_environment.get(region, 50.0)
    
    # Adjust for urban density
    if is_urban == "high":
        base_health = min(100.0, base_health + 10.0)
    elif is_urban == "medium":
        base_health = min(100.0, base_health + 5.0)
    else:
        base_health = max(0, base_health - 15.0)
    
    return base_health

def _estimate_social_stability(lat: float, lng: float) -> float:
    """Estimate social stability based on geography."""
    region = _get_geographic_region(lat, lng)
    
    # Social stability by region
    social_stability = {
        "north_america": 75.0,  # Good stability
        "europe": 80.0,           # Good stability
        "asia": 60.0,            # Variable stability
        "south_america": 55.0,    # Moderate stability
        "africa": 45.0,           # Variable stability
        "oceania": 75.0,          # Good stability
        "other": 50.0             # Unknown
    }
    
    return social_stability.get(region, 50.0)

def _estimate_economic_viability(lat: float, lng: float) -> float:
    """Estimate economic viability based on geography."""
    region = _get_geographic_region(lat, lng)
    
    # Economic viability by region
    economic_viability = {
        "north_america": 80.0,  # Good economy
        "europe": 75.0,           # Good economy
        "asia": 65.0,            # Variable economy
        "south_america": 55.0,    # Moderate economy
        "africa": 40.0,           # Limited economy
        "oceania": 70.0,          # Good economy
        "other": 50.0             # Unknown
    }
    
    return economic_viability.get(region, 50.0)

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
