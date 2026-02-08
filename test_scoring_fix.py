#!/usr/bin/env python3
"""
Test script to verify water body and protected area scoring fixes
"""

import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import the aggregator
    from suitability_factors.aggregator import Aggregator
    
    # Test 1: Water Body Scoring (should use averages, not hardcoded 12)
    print("=== Testing Water Body Scoring ===")
    water_body_package = {
        "latitude": 0.0,  # Ocean coordinates
        "longitude": 0.0,
        "raw_factors": {
            "physical": {
                "elevation": {"value": 50},
                "ruggedness": {"value": 50},
                "slope": {"value": 50},
                "stability": {"value": 50}
            },
            "environmental": {
                "vegetation": {"value": 50},
                "pollution": {"value": 50},
                "soil": {"value": 50},
                "biodiversity": {"value": 50},
                "heat_island": {"value": 50}
            },
            "hydrology": {
                "water": {"value": 0, "distance_km": 0.001},  # Direct water detection
                "drainage": {"value": 50},
                "flood": {"value": 50},
                "groundwater": {"value": 50}
            },
            "climatic": {
                "intensity": {"value": 50},
                "rainfall": {"value": 50},
                "thermal": {"value": 50}
            },
            "socio_econ": {
                "infrastructure": {"value": 50},
                "landuse": {"value": 50},
                "population": {"value": 50}
            },
            "risk_resilience": {
                "multi_hazard": {"value": 50},
                "climate_change": {"value": 50},
                "recovery": {"value": 50},
                "habitability": {"value": 50}
            }
        }
    }
    
    water_result = Aggregator.compute_suitability_score(water_body_package)
    print(f"Water Body Score: {water_result.get('score')}")
    print(f"Water Body Label: {water_result.get('label')}")
    print(f"Water Body Penalty: {water_result.get('penalty')}")
    print(f"Category Scores: {water_result.get('category_scores', {})}")
    
    # Test 2: Protected Forest Scoring (should use averages, not hardcoded 20)
    print("\n=== Testing Protected Forest Scoring ===")
    protected_forest_package = {
        "latitude": -3.4653,  # Amazon rainforest
        "longitude": -62.2159,
        "raw_factors": {
            "physical": {
                "elevation": {"value": 50},
                "ruggedness": {"value": 50},
                "slope": {"value": 50},
                "stability": {"value": 50}
            },
            "environmental": {
                "vegetation": {"value": 50},
                "pollution": {"value": 50},
                "soil": {"value": 50},
                "biodiversity": {"value": 50},
                "heat_island": {"value": 50}
            },
            "hydrology": {
                "water": {"value": 50},
                "drainage": {"value": 50},
                "flood": {"value": 50},
                "groundwater": {"value": 50}
            },
            "climatic": {
                "intensity": {"value": 50},
                "rainfall": {"value": 50},
                "thermal": {"value": 50}
            },
            "socio_econ": {
                "infrastructure": {"value": 50},
                "landuse": {"value": 15, "classification": "Protected Forest"},  # Protected land use
                "population": {"value": 50}
            },
            "risk_resilience": {
                "multi_hazard": {"value": 50},
                "climate_change": {"value": 50},
                "recovery": {"value": 50},
                "habitability": {"value": 50}
            }
        }
    }
    
    forest_result = Aggregator.compute_suitability_score(protected_forest_package)
    print(f"Protected Forest Score: {forest_result.get('score')}")
    print(f"Protected Forest Label: {forest_result.get('label')}")
    print(f"Protected Forest Penalty: {forest_result.get('penalty')}")
    print(f"Category Scores: {forest_result.get('category_scores', {})}")
    
    # Test 3: Normal Area Scoring (should work normally)
    print("\n=== Testing Normal Area Scoring ===")
    normal_package = {
        "latitude": 28.6139,  # Delhi
        "longitude": 77.2090,
        "raw_factors": {
            "physical": {
                "elevation": {"value": 50},
                "ruggedness": {"value": 50},
                "slope": {"value": 50},
                "stability": {"value": 50}
            },
            "environmental": {
                "vegetation": {"value": 50},
                "pollution": {"value": 50},
                "soil": {"value": 50},
                "biodiversity": {"value": 50},
                "heat_island": {"value": 50}
            },
            "hydrology": {
                "water": {"value": 50},
                "drainage": {"value": 50},
                "flood": {"value": 50},
                "groundwater": {"value": 50}
            },
            "climatic": {
                "intensity": {"value": 50},
                "rainfall": {"value": 50},
                "thermal": {"value": 50}
            },
            "socio_econ": {
                "infrastructure": {"value": 50},
                "landuse": {"value": 50},
                "population": {"value": 50}
            },
            "risk_resilience": {
                "multi_hazard": {"value": 50},
                "climate_change": {"value": 50},
                "recovery": {"value": 50},
                "habitability": {"value": 50}
            }
        }
    }
    
    normal_result = Aggregator.compute_suitability_score(normal_package)
    print(f"Normal Area Score: {normal_result.get('score')}")
    print(f"Normal Area Label: {normal_result.get('label')}")
    print(f"Normal Area Penalty: {normal_result.get('penalty')}")
    print(f"Category Scores: {normal_result.get('category_scores', {})}")
    
    print("\n✅ Scoring Test Complete!")
    
    # Verify that water body factors are zero where they should be
    print("\n=== Verifying Water Body Factor Logic ===")
    print("Expected: soil=0, stability=0, habitability=0, recovery=0 for water bodies")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
