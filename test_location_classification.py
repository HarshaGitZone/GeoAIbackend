#!/usr/bin/env python3
"""
Test script to verify location classification fix
"""

import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import the location classification function
    from app import _classify_location_type
    
    # Test with normal urban area (should not be classified as wetland)
    urban_factors = {
        "physical": {"elevation": 50, "ruggedness": 30, "slope": 20, "stability": 60},
        "environmental": {"vegetation": 25, "pollution": 65, "soil": 55, "biodiversity": 30, "heat_island": 70},
        "hydrology": {"flood": 45, "water": 60, "drainage": 55, "groundwater": 50},
        "climatic": {"intensity": 50, "rainfall": 55, "thermal": 60},
        "socio_econ": {"infrastructure": 75, "landuse": 70, "population": 80},
        "risk_resilience": {"multi_hazard": 50, "climate_change": 50, "recovery": 60, "habitability": 65}
    }
    
    print("Testing location classification with urban area...")
    urban_result = _classify_location_type(urban_factors)
    print(f"Urban Area Classification: {urban_result}")
    
    # Test with actual wetland area (should be classified as coastal wetland)
    wetland_factors = {
        "physical": {"elevation": 20, "ruggedness": 25, "slope": 10, "stability": 40},
        "environmental": {"vegetation": 30, "pollution": 25, "soil": 45, "biodiversity": 60, "heat_island": 35},
        "hydrology": {"flood": 85, "water": 90, "drainage": 80, "groundwater": 85},
        "climatic": {"intensity": 45, "rainfall": 70, "thermal": 50},
        "socio_econ": {"infrastructure": 20, "landuse": 15, "population": 25},
        "risk_resilience": {"multi_hazard": 70, "climate_change": 60, "recovery": 30, "habitability": 35}
    }
    
    print("\nTesting location classification with wetland area...")
    wetland_result = _classify_location_type(wetland_factors)
    print(f"Wetland Area Classification: {wetland_result}")
    
    # Test with normal area (moderate values)
    normal_factors = {
        "physical": {"elevation": 50, "ruggedness": 50, "slope": 50, "stability": 50},
        "environmental": {"vegetation": 50, "pollution": 50, "soil": 50, "biodiversity": 50, "heat_island": 50},
        "hydrology": {"flood": 50, "water": 50, "drainage": 50, "groundwater": 50},
        "climatic": {"intensity": 50, "rainfall": 50, "thermal": 50},
        "socio_econ": {"infrastructure": 50, "landuse": 50, "population": 50},
        "risk_resilience": {"multi_hazard": 50, "climate_change": 50, "recovery": 50, "habitability": 50}
    }
    
    print("\nTesting location classification with normal area...")
    normal_result = _classify_location_type(normal_factors)
    print(f"Normal Area Classification: {normal_result}")
    
    print("\n✅ Location Classification Test Complete!")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
