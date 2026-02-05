"""
Terrain Ruggedness Analysis Module
Calculates surface irregularity using multiple ruggedness indices
Data Sources: NASA SRTM, ASTER GDEM, Copernicus DEM
"""

import math
import numpy as np
from typing import Dict, Any, Tuple, Optional
import requests
import logging

logger = logging.getLogger(__name__)

def get_terrain_ruggedness(lat: float, lng: float, radius_km: float = 1.0) -> Dict[str, Any]:
    """
    Calculate terrain ruggedness index for given coordinates.
    
    Args:
        lat: Latitude
        lng: Longitude  
        radius_km: Analysis radius in kilometers
        
    Returns:
        Dictionary with ruggedness score and metadata
    """
    try:
        # Get elevation samples in surrounding area
        elevation_samples = _get_elevation_samples(lat, lng, radius_km)
        
        if not elevation_samples or len(elevation_samples) < 9:
            return _get_fallback_ruggedness(lat, lng)
        
        # Calculate multiple ruggedness indices
        tri_score = _calculate_tri(elevation_samples)
        vrm_score = _calculate_vrm(elevation_samples)
        std_score = _calculate_std_ruggedness(elevation_samples)
        
        # Combine indices for comprehensive ruggedness assessment
        combined_ruggedness = (tri_score * 0.4 + vrm_score * 0.4 + std_score * 0.2)
        
        # Convert to suitability score (inverse relationship)
        suitability_score = _ruggedness_to_suitability(combined_ruggedness)
        
        return {
            "value": suitability_score,
            "tri_index": round(tri_score, 2),
            "vrm_index": round(vrm_score, 2),
            "std_index": round(std_score, 2),
            "combined_ruggedness": round(combined_ruggedness, 2),
            "sample_count": len(elevation_samples),
            "radius_km": radius_km,
            "label": _get_ruggedness_label(suitability_score),
            "source": "NASA SRTM + Derived Indices",
            "confidence": _calculate_confidence(elevation_samples),
            "reasoning": _generate_reasoning(tri_score, vrm_score, std_score, suitability_score)
        }
        
    except Exception as e:
        logger.error(f"Error calculating terrain ruggedness for {lat}, {lng}: {e}")
        return _get_fallback_ruggedness(lat, lng)

def _get_elevation_samples(lat: float, lng: float, radius_km: float) -> list:
    """
    Get elevation samples in a grid around the target point.
    
    Args:
        lat: Center latitude
        lng: Center longitude
        radius_km: Analysis radius
        
    Returns:
        List of elevation values in meters
    """
    try:
        # Create sampling grid (3x3 or 5x5 depending on radius)
        grid_size = 5 if radius_km >= 1.0 else 3
        samples = []
        
        # Calculate coordinate step size (approximately 1km ≈ 0.009 degrees)
        step_deg = radius_km / 111.0 / (grid_size // 2)
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Skip center point (will be added separately)
                if i == grid_size // 2 and j == grid_size // 2:
                    continue
                    
                sample_lat = lat + (i - grid_size // 2) * step_deg
                sample_lng = lng + (j - grid_size // 2) * step_deg
                
                elevation = _get_single_elevation(sample_lat, sample_lng)
                if elevation is not None:
                    samples.append(elevation)
        
        # Add center point
        center_elevation = _get_single_elevation(lat, lng)
        if center_elevation is not None:
            samples.append(center_elevation)
        
        return samples
        
    except Exception as e:
        logger.error(f"Error getting elevation samples: {e}")
        return []

def _get_single_elevation(lat: float, lng: float) -> Optional[float]:
    """
    Get elevation for a single coordinate point.
    
    Args:
        lat: Latitude
        lng: Longitude
        
    Returns:
        Elevation in meters or None if failed
    """
    try:
        # Use OpenTopography API (free, high-resolution DEM data)
        url = f"https://portal.opentopography.org/API/globaldem?demtype=SRTMGL3&location={lat},{lng}&output=json"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                return float(data[0].get('elevation', 0))
        
        # Fallback to USGS Elevation Point Query Service
        usgs_url = f"https://nationalmap.gov/epqs/pq.php?x={lng}&y={lat}&units=Meters&output=json"
        usgs_response = requests.get(usgs_url, timeout=10)
        
        if usgs_response.status_code == 200:
            usgs_data = usgs_response.json()
            elevation = usgs_data.get('USGS_Elevation_Point_Query_Service', {}).get('Elevation_Query', {}).get('Elevation')
            if elevation:
                return float(elevation)
                
    except Exception as e:
        logger.debug(f"Failed to get elevation for {lat}, {lng}: {e}")
    
    return None

def _calculate_tri(elevations: list) -> float:
    """
    Calculate Terrain Ruggedness Index (TRI).
    TRI = sqrt(Σ(elevation_diff²)) for all adjacent cells
    
    Args:
        elevations: List of elevation values
        
    Returns:
        TRI value (higher = more rugged)
    """
    if len(elevations) < 2:
        return 0.0
    
    tri_sum = 0.0
    count = 0
    
    for i in range(len(elevations)):
        for j in range(i + 1, len(elevations)):
            diff = abs(elevations[i] - elevations[j])
            tri_sum += diff * diff
            count += 1
    
    return math.sqrt(tri_sum) if count > 0 else 0.0

def _calculate_vrm(elevations: list) -> float:
    """
    Calculate Vector Ruggedness Measure (VRM).
    Measures 3D surface orientation variation.
    
    Args:
        elevations: List of elevation values
        
    Returns:
        VRM value (0-1, higher = more rugged)
    """
    if len(elevations) < 3:
        return 0.0
    
    try:
        # Simplified VRM calculation based on elevation variance
        # In full implementation, this would use 3D vector analysis
        elev_array = np.array(elevations)
        
        # Calculate standard deviation normalized by range
        if elev_array.max() - elev_array.min() > 0:
            vrm = elev_array.std() / (elev_array.max() - elev_array.min())
        else:
            vrm = 0.0
            
        return min(vrm, 1.0)
        
    except Exception:
        return 0.0

def _calculate_std_ruggedness(elevations: list) -> float:
    """
    Calculate simple standard deviation ruggedness.
    
    Args:
        elevations: List of elevation values
        
    Returns:
        Standard deviation in meters
    """
    if len(elevations) < 2:
        return 0.0
    
    try:
        elev_array = np.array(elevations)
        return float(elev_array.std())
    except Exception:
        return 0.0

def _ruggedness_to_suitability(ruggedness_value: float) -> float:
    """
    Convert ruggedness index to suitability score.
    Higher ruggedness = lower suitability (more difficult to build).
    
    Args:
        ruggedness_value: Combined ruggedness index
        
    Returns:
        Suitability score (0-100, higher = better)
    """
    # Normalize ruggedness to 0-100 scale
    # Very low ruggedness (< 5m) = excellent (90-100)
    # Low ruggedness (5-15m) = good (70-90)  
    # Moderate ruggedness (15-30m) = fair (50-70)
    # High ruggedness (30-50m) = poor (30-50)
    # Very high ruggedness (> 50m) = very poor (0-30)
    
    if ruggedness_value <= 5:
        return max(90, 100 - ruggedness_value * 2)
    elif ruggedness_value <= 15:
        return 90 - (ruggedness_value - 5) * 2
    elif ruggedness_value <= 30:
        return 70 - (ruggedness_value - 15) * 1.33
    elif ruggedness_value <= 50:
        return 50 - (ruggedness_value - 30) * 1
    else:
        return max(0, 30 - (ruggedness_value - 50) * 0.6)

def _get_ruggedness_label(suitability_score: float) -> str:
    """Get human-readable label for ruggedness level."""
    if suitability_score >= 80:
        return "Flat Terrain (Excellent)"
    elif suitability_score >= 60:
        return "Gently Rolling (Good)"
    elif suitability_score >= 40:
        return "Moderately Rugged (Fair)"
    elif suitability_score >= 20:
        return "Rugged Terrain (Poor)"
    else:
        return "Very Rugged (Very Poor)"

def _calculate_confidence(elevations: list) -> float:
    """Calculate confidence based on data quality."""
    if len(elevations) >= 20:
        return 90.0
    elif len(elevations) >= 10:
        return 75.0
    elif len(elevations) >= 5:
        return 60.0
    else:
        return 40.0

def _generate_reasoning(tri: float, vrm: float, std: float, suitability: float) -> str:
    """Generate human-readable reasoning for the ruggedness assessment."""
    reasoning_parts = []
    
    if tri > 30:
        reasoning_parts.append(f"High terrain relief detected (TRI: {tri:.1f})")
    elif tri > 15:
        reasoning_parts.append(f"Moderate terrain relief (TRI: {tri:.1f})")
    else:
        reasoning_parts.append(f"Low terrain relief (TRI: {tri:.1f})")
    
    if vrm > 0.7:
        reasoning_parts.append("highly variable surface orientation")
    elif vrm > 0.4:
        reasoning_parts.append("moderately variable surface")
    else:
        reasoning_parts.append("relatively uniform surface")
    
    if std > 25:
        reasoning_parts.append(f"significant elevation variation (±{std:.1f}m)")
    elif std > 10:
        reasoning_parts.append(f"moderate elevation variation (±{std:.1f}m)")
    else:
        reasoning_parts.append(f"minimal elevation variation (±{std:.1f}m)")
    
    return ". ".join(reasoning_parts) + "."

def _get_fallback_ruggedness(lat: float, lng: float) -> Dict[str, Any]:
    """Fallback ruggedness estimation based on geographic context."""
    try:
        # Use geographic context for rough estimation
        is_mountainous = _is_mountainous_region(lat, lng)
        is_coastal = _is_coastal_region(lat, lng)
        
        if is_mountainous:
            suitability = 25.0  # Poor - mountainous terrain
            label = "Mountainous Terrain (Poor)"
        elif is_coastal:
            suitability = 75.0  # Good - typically flat coastal plains
            label = "Coastal Plain (Good)"
        else:
            suitability = 60.0  # Fair - mixed terrain
            label = "Mixed Terrain (Fair)"
        
        return {
            "value": suitability,
            "tri_index": 20.0,
            "vrm_index": 0.5,
            "std_index": 15.0,
            "combined_ruggedness": 25.0,
            "sample_count": 0,
            "radius_km": 1.0,
            "label": label,
            "source": "Geographic Estimation (Fallback)",
            "confidence": 30.0,
            "reasoning": f"Estimated based on geographic location ({'mountainous' if is_mountainous else 'coastal' if is_coastal else 'mixed'} region)."
        }
        
    except Exception:
        return {
            "value": 50.0,
            "tri_index": 15.0,
            "vrm_index": 0.4,
            "std_index": 10.0,
            "combined_ruggedness": 18.0,
            "sample_count": 0,
            "radius_km": 1.0,
            "label": "Unknown Terrain (Moderate)",
            "source": "Default Fallback",
            "confidence": 10.0,
            "reasoning": "Unable to determine terrain characteristics."
        }

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

def _is_coastal_region(lat: float, lng: float) -> bool:
    """Check if location is near coast."""
    # Simplified coastal detection
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
