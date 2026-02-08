#!/usr/bin/env python3
"""
Test script to verify aggregator fix
"""

import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import the aggregator
    from suitability_factors.aggregator import Aggregator
    
    # Create test data package
    test_package = {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "raw_factors": {
            "physical": {
                "elevation": {"value": 216},
                "ruggedness": {"value": 45},
                "slope": {"value": 15},
                "stability": {"value": 70}
            },
            "environmental": {
                "vegetation": {"value": 35},
                "pollution": {"value": 65},
                "soil": {"value": 55},
                "biodiversity": {"value": 40},
                "heat_island": {"value": 60}
            },
            "hydrology": {
                "water": {"value": 75},
                "drainage": {"value": 60},
                "flood": {"value": 70},
                "groundwater": {"value": 65}
            },
            "climatic": {
                "rainfall": {"value": 55},
                "thermal": {"value": 65},
                "intensity": {"value": 50}
            },
            "socio_econ": {
                "infrastructure": {"value": 70},
                "landuse": {"value": 60},
                "population": {"value": 65}
            },
            "risk_resilience": {
                "multi_hazard": {"value": 55},
                "climate_change": {"value": 50},
                "recovery": {"value": 60},
                "habitability": {"value": 65}
            }
        }
    }
    
    print("Testing Aggregator with Delhi coordinates...")
    result = Aggregator.compute_suitability_score(test_package)
    
    if result:
        print("✅ Aggregator SUCCESS!")
        print(f"Score: {result.get('score')}")
        print(f"Label: {result.get('label')}")
        print(f"Category Scores: {result.get('category_scores', {})}")
    else:
        print("❌ Aggregator FAILED!")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
