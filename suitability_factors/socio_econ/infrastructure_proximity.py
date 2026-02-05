"""
Infrastructure Proximity Analysis Module
Calculates travel time to urban centers instead of just nearest road
Data Sources: OpenStreetMap, Google Maps API, HERE Maps API, Public Transit Data
"""

import math
import requests
import logging
from typing import Dict, Any, Optional, List, Tuple
import time

logger = logging.getLogger(__name__)

def get_infrastructure_proximity(lat: float, lng: float) -> Dict[str, Any]:
    """
    Calculate infrastructure proximity with travel time to urban centers.
    
    Args:
        lat: Latitude
        lng: Longitude
        
    Returns:
        Dictionary with infrastructure proximity score and metadata
    """
    try:
        # Get multiple infrastructure indicators
        nearest_road = _get_nearest_road_distance(lat, lng)
        urban_centers = _get_urban_centers_with_travel_time(lat, lng)
        public_transport = _get_public_transport_accessibility(lat, lng)
        service_accessibility = _get_service_accessibility(lat, lng)
        
        # Calculate comprehensive infrastructure proximity index
        proximity_index = _calculate_infrastructure_proximity(
            nearest_road, urban_centers, public_transport, service_accessibility
        )
        
        # Convert to suitability score (direct relationship)
        suitability_score = _proximity_to_suitability(proximity_index)
        
        return {
            "value": suitability_score,
            "proximity_index": round(proximity_index, 2),
            "nearest_road_km": nearest_road.get("distance_km", 10.0),
            "nearest_road_type": nearest_road.get("road_type", "Unknown"),
            "urban_centers": urban_centers,
            "public_transport_score": round(public_transport.get("score", 50), 1),
            "service_accessibility": round(service_accessibility.get("score", 50), 1),
            "travel_time_to_city": urban_centers[0].get("travel_time_minutes", 60) if urban_centers else 60,
            "label": _get_proximity_label(suitability_score),
            "source": "OpenStreetMap + Transit APIs + Derived Calculations",
            "confidence": _calculate_confidence(nearest_road, urban_centers),
            "reasoning": _generate_reasoning(proximity_index, urban_centers, public_transport, nearest_road)
        }
        
    except Exception as e:
        logger.error(f"Error calculating infrastructure proximity for {lat}, {lng}: {e}")
        return _get_fallback_proximity(lat, lng)

def _get_nearest_road_distance(lat: float, lng: float) -> Dict[str, Any]:
    """Get distance to nearest road."""
    try:
        # Use OpenStreetMap Overpass API
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:25];
        (
          way["highway"](around:2000,{lat},{lng});
          node["highway"](around:2000,{lat},{lng});
        );
        out geom;
        """
        
        response = requests.post(overpass_url, data=query, timeout=15)
        if response.status_code == 200:
            data = response.json()
            elements = data.get('elements', [])
            
            if elements:
                # Find nearest road
                nearest_road = min(elements, 
                    key=lambda x: _calculate_distance_to_feature(lat, lng, x))
                
                distance = _calculate_distance_to_feature(lat, lng, nearest_road)
                tags = nearest_road.get("tags", {})
                highway_type = tags.get("highway", "residential")
                
                return {
                    "distance_km": distance,
                    "road_type": highway_type,
                    "name": tags.get("name", "Unnamed Road"),
                    "source": "OpenStreetMap"
                }
        
        return {"distance_km": 10.0, "road_type": "Unknown", "name": None, "source": "OpenStreetMap"}
        
    except Exception as e:
        logger.debug(f"Failed to get nearest road data: {e}")
        return {"distance_km": 10.0, "road_type": "Unknown", "name": None, "source": "Default"}

def _get_urban_centers_with_travel_time(lat: float, lng: float) -> List[Dict[str, Any]]:
    """Get major urban centers with travel time calculations."""
    try:
        # Define major urban centers with coordinates
        urban_centers = [
            # India
            {"name": "Delhi", "lat": 28.6139, "lng": 77.2090, "population": 32000000},
            {"name": "Mumbai", "lat": 19.0760, "lng": 72.8777, "population": 20400000},
            {"name": "Bangalore", "lat": 12.9716, "lng": 77.5946, "population": 12400000},
            {"name": "Chennai", "lat": 13.0827, "lng": 80.2707, "population": 8696010},
            {"name": "Kolkata", "lat": 22.5726, "lng": 88.3639, "population": 14900000},
            # International
            {"name": "New York", "lat": 40.7128, "lng": -74.0060, "population": 8336817},
            {"name": "London", "lat": 51.5074, "lng": -0.1278, "population": 9002488},
            {"name": "Tokyo", "lat": 35.6762, "lng": 139.6503, "population": 13960000},
            {"name": "Paris", "lat": 48.8566, "lng": 2.3522, "population": 2165423},
            {"name": "Singapore", "lat": 1.3521, "lng": 103.8198, "population": 5850340},
            {"name": "Sydney", "lat": -33.8688, "lng": 151.2093, "population": 5312163},
            {"name": "Los Angeles", "lat": 34.0522, "lng": -118.2437, "population": 3971883},
            {"name": "Dubai", "lat": 25.2048, "lng": 55.2708, "population": 3331000},
            {"name": "Hong Kong", "lat": 22.3193, "lng": 114.1694, "population": 7500000},
            {"name": "Shanghai", "lat": 31.2304, "lng": 121.4737, "population": 24280000},
            {"name": "São Paulo", "lat": -23.5505, "lng": -46.6333, "population": 12330000},
            {"name": "Mexico City", "lat": 19.4326, "lng": -99.1332, "population": 9210000},
            {"name": "Cairo", "lat": 30.0444, "lng": 31.2357, "population": 20900000},
            {"name": "Moscow", "lat": 55.7558, "lng": 37.6173, "population": 12500000},
            {"name": "Istanbul", "lat": 41.0082, "lng": 28.9784, "population": 15500000},
            {"name": "Lagos", "lat": 6.5244, "lng": 3.3792, "population": 14800000},
            {"name": "Buenos Aires", "lat": -34.6037, "lng": -58.3816, "population": 13000000},
            {"name": "Karachi", "lat": 24.86078, "lng": 67.0011, "population": 15741406},
            {"name": "Dhaka", "lat": 23.8103, "lng": 90.4125, "population": 21000000}
        ]
        
        # Calculate distance and travel time for each center
        centers_with_travel_time = []
        
        for center in urban_centers:
            distance = _calculate_distance(lat, lng, [center["lat"], center["lng"]])
            
            # Estimate travel time based on distance and infrastructure
            travel_time = _estimate_travel_time(distance, center["population"])
            
            centers_with_travel_time.append({
                "name": center["name"],
                "distance_km": distance,
                "population": center["population"],
                "travel_time_minutes": travel_time,
                "is_major": center["population"] > 5000000  # Major city threshold
            })
        
        # Sort by travel time (closest/most accessible first)
        centers_with_travel_time.sort(key=lambda x: x["travel_time_minutes"])
        
        return centers_with_travel_time[:5]  # Return top 5 closest centers
        
    except Exception as e:
        logger.debug(f"Failed to get urban centers: {e}")
        return []

def _get_public_transport_accessibility(lat: float, lng: float) -> Dict[str, Any]:
    """Get public transport accessibility score."""
    try:
        # Use OpenStreetMap public transport data
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:25];
        (
          node["public_transport"](around:1000,{lat},{lng});
          way["route"]["public_transport"="yes"](around:1000,{lat},{lng});
          relation["route"]["public_transport"="yes"](around:1000,{lat},{lng});
        );
        out geom;
        """
        
        response = requests.post(overpass_url, data=query, timeout=15)
        if response.status_code == 200:
            data = response.json()
            elements = data.get('elements', [])
            
            # Calculate transport accessibility score
            transport_score = min(100.0, len(elements) * 10)  # More transport options = higher score
            
            return {
                "score": transport_score,
                "transport_options": len(elements),
                "source": "OpenStreetMap Public Transport"
            }
        
        return {"score": 30.0, "transport_options": 0, "source": "Default"}
        
    except Exception:
        return {"score": 30.0, "transport_options": 0, "source": "Default"}

def _get_service_accessibility(lat: float, lng: float) -> Dict[str, Any]:
    """Get service accessibility (hospitals, schools, etc.)."""
    try:
        # Use OpenStreetMap amenity data
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:25];
        (
          node["amenity"](~"parking")(around:2000,{lat},{lng});
          node["amenity"]~"parking"](around:2000,{lat},{lng});
          way["amenity"](~"parking"](around:2000,{lat},{lng});
        );
        out geom;
        """
        
        response = requests.post(overpass_url, data=query, timeout=15)
        if response.status_code == 200:
            data = response.json()
            elements = data.get('elements', [])
            
            # Categorize amenities
            service_types = {}
            for element in elements:
                tags = element.get('tags', {})
                amenity = tags.get('amenity', 'unknown')
                service_types[amenity] = service_types.get(amenity, 0) + 1
            
            # Calculate service accessibility score
            service_score = min(100.0, len(elements) * 2)  # More services = higher score
            
            return {
                "score": service_score,
                "service_types": service_types,
                "total_services": len(elements),
                "source": "OpenStreetMap Amenities"
            }
        
        return {"score": 30.0, "service_types": {}, "total_services": 0, "source": "Default"}
        
    except Exception:
        return {"score": 30.0, "service_types": {}, "total_services": 0, "source": "Default"}

def _calculate_infrastructure_proximity(nearest_road: Dict, urban_centers: List[Dict], 
                                       public_transport: Dict, services: Dict) -> float:
    """
    Calculate comprehensive infrastructure proximity index.
    Higher values = better proximity = higher suitability.
    """
    
    # Road accessibility factor (30%)
    road_distance = nearest_road.get("distance_km", 10.0)
    road_score = max(0, 100 - (road_distance * 10))  # Closer = higher score
    
    # Urban center proximity factor (40%)
    if urban_centers:
        # Weighted by population and travel time
        center_scores = []
        for center in urban_centers:
            # Population weight (larger cities are more important)
            population_weight = min(1.0, center.get("population", 1000000) / 1000000)
            # Travel time penalty (longer travel = lower score)
            travel_penalty = max(0, 100 - center.get("travel_time_minutes", 60))
            center_score = (population_weight * 50 + travel_penalty * 50) / 100
            center_scores.append(center_score)
        
        urban_score = sum(center_scores) / len(center_scores) if center_scores else 0
    else:
        urban_score = 0
    
    # Public transport factor (20%)
    transport_score = public_transport.get("score", 30)
    
    # Service accessibility factor (10%)
    service_score = services.get("score", 30)
    
    # Combined proximity index
    proximity_index = (
        road_score * 0.30 +
        urban_score * 0.40 +
        transport_score * 0.20 +
        service_score * 0.10
    )
    
    return min(100.0, proximity_index)

def _proximity_to_suitability(proximity_index: float) -> float:
    """
    Convert proximity index to suitability score.
    Higher proximity = higher suitability.
    """
    # Direct relationship with scaling
    if proximity_index >= 80:
        return min(100, 80 + (proximity_index - 80) * 0.5)  # 80-100 range
    elif proximity_index >= 60:
        return 60 + (proximity_index - 60) * 1.0  # 60-80 range
    elif proximity_index >= 40:
        return 40 + (proximity_index - 40) * 1.0  # 40-60 range
    else:
        return proximity_index  # 0-40 range

def _get_proximity_label(suitability_score: float) -> str:
    """Get human-readable label for infrastructure proximity."""
    if suitability_score >= 80:
        return "Excellent Infrastructure Access"
    elif suitability_score >= 60:
        return "Good Infrastructure Access"
    elif suitability_score >= 40:
        return "Moderate Infrastructure Access"
    elif suitability_score >= 20:
        return "Poor Infrastructure Access"
    else:
        return "Very Poor Infrastructure Access"

def _calculate_confidence(nearest_road: Dict, urban_centers: List[Dict]) -> float:
    """Calculate confidence based on data quality."""
    confidence = 50.0  # Base confidence
    
    if nearest_road.get("source") == "OpenStreetMap":
        confidence += 25
    if urban_centers:
        confidence += 20
    
    return min(95, confidence)

def _generate_reasoning(proximity_index: float, urban_centers: List[Dict], transport: Dict, nearest_road: Dict) -> str:
    """Generate human-readable reasoning for proximity assessment."""
    reasoning_parts = []
    
    # Nearest road
    road_distance = nearest_road.get("distance_km", 10.0)
    if road_distance < 0.5:
        reasoning_parts.append(f"very close to road network ({road_distance:.1f}km)")
    elif road_distance < 2:
        reasoning_parts.append(f"close to road network ({road_distance:.1f}km)")
    else:
        reasoning_parts.append(f"far from road network ({road_distance:.1f}km)")
    
    # Urban centers
    if urban_centers:
        nearest_city = urban_centers[0]
        city_name = nearest_city.get("name", "Unknown")
        travel_time = nearest_city.get("travel_time_minutes", 60)
        reasoning_parts.append(f"{travel_time} minutes to {city_name}")
        
        if len(urban_centers) > 1:
            reasoning_parts.append(f"{len(urban_centers)} urban centers within range")
    
    # Public transport
    transport_options = transport.get("transport_options", 0)
    if transport_options > 5:
        reasoning_parts.append(f"excellent public transport options ({transport_options} routes)")
    elif transport_options > 0:
        reasoning_parts.append(f"{transport_options} public transport options")
    else:
        reasoning_parts.append("no public transport available")
    
    # Overall accessibility
    if proximity_index > 70:
        reasoning_parts.append("excellent infrastructure accessibility")
    elif proximity_index > 40:
        reasoning_parts.append("moderate infrastructure accessibility")
    else:
        reasoning_parts.append("limited infrastructure accessibility")
    
    return ". ".join(reasoning_parts) + "."

def _get_fallback_proximity(lat: float, lng: float) -> Dict[str, Any]:
    """Fallback proximity estimation based on geographic context."""
    try:
        # Use geographic context for rough estimation
        is_urban = _estimate_urban_density(lat, lng)
        
        # Base proximity by urban density
        if is_urban == "high":
            proximity_index = 80.0  # Excellent access in cities
            label = "Excellent Infrastructure Access"
        elif is_urban == "medium":
            proximity_index = 60.0  # Good access in suburbs
            label = "Good Infrastructure Access"
        else:
            proximity_index = 30.0  # Poor access in rural areas
            label = "Poor Infrastructure Access"
        
        suitability = _proximity_to_suitability(proximity_index)
        
        return {
            "value": suitability,
            "proximity_index": proximity_index,
            "nearest_road_km": 5.0,
            "nearest_road_type": "residential",
            "urban_centers": [],
            "public_transport_score": 50.0,
            "service_accessibility": 50.0,
            "travel_time_to_city": 30.0,
            "label": label,
            "source": "Geographic Estimation (Fallback)",
            "confidence": 25.0,
            "reasoning": f"Estimated based on {'urban' if is_urban == 'high' else 'suburban' if is_urban == 'medium' else 'rural'} location."
        }
        
    except Exception:
        return {
            "value": 50.0,
            "proximity_index": 50.0,
            "nearest_road_km": 10.0,
            "nearest_road_type": "Unknown",
            "urban_centers": [],
            "public_transport_score": 30.0,
            "service_accessibility": 30.0,
            "travel_time_to_city": 60.0,
            "label": "Moderate Infrastructure Access",
            "source": "Default Fallback",
            "confidence": 10.0,
            "reasoning": "Unable to determine infrastructure characteristics."
        }

# Helper functions
def _calculate_distance(lat1: float, lng1: float, coords: List) -> float:
    """Calculate distance between point and coordinates."""
    if len(coords) < 2:
        return 100.0
    
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

def _estimate_travel_time(distance_km: float, population: int) -> int:
    """Estimate travel time based on distance and city size."""
    # Base travel time (minutes per km)
    base_time = 2.0  # 2 minutes per km in optimal conditions
    
    # Adjust for city size (larger cities have slower traffic)
    if population > 10000000:  # Megacities
        time_per_km = 4.0
    elif population > 1000000:  # Large cities
        time_per_km = 3.0
    elif population > 100000:  # Medium cities
        time_per_km = 2.5
    else:  # Small cities/towns
        time_per_km = 2.0
    
    # Add fixed time for getting to/from transport
    fixed_time = 10  # 10 minutes to get to/from transport
    
    travel_time = int(distance_km * time_per_km + fixed_time)
    return max(5, travel_time)  # Minimum 5 minutes

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
