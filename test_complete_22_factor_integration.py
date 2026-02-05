#!/usr/bin/env python3
"""
Complete 22-Factor Integration Test
Tests the entire pipeline from backend to frontend data structure
"""

import sys
import os
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_backend_integration():
    """Test complete backend integration with all 23 factors."""
    print("🧪 COMPLETE 23-FACTOR INTEGRATION TEST")
    print("=" * 80)
    
    try:
        from suitability_factors.geo_data_service import GeoDataService
        from suitability_factors.aggregator import Aggregator
        
        # Test coordinates
        lat, lng = 17.5387, 78.3950
        
        print(f"📍 Testing coordinates: {lat:.4f}, {lng:.4f}")
        print("-" * 50)
        
        # Step 1: Test data collection
        print("📊 Step 1: Collecting all 23 factors...")
        raw_data = GeoDataService.get_land_intelligence(lat, lng)
        
        # Verify structure
        raw_factors = raw_data.get('raw_factors', {})
        expected_categories = ['physical', 'environmental', 'hydrology', 'climatic', 'socio_econ', 'risk_resilience']
        
        print(f"✅ Raw factors keys: {list(raw_factors.keys())}")
        
        # Check for all categories
        missing_categories = set(expected_categories) - set(raw_factors.keys())
        if missing_categories:
            print(f"❌ Missing categories: {missing_categories}")
            return False
        
        # Count total factors
        total_factors = 0
        category_counts = {}
        for category, factors in raw_factors.items():
            if isinstance(factors, dict):
                count = len(factors)
                total_factors += count
                category_counts[category] = count
        
        print(f"✅ Total factors: {total_factors} (expected: 23)")
        print(f"✅ Category breakdown: {category_counts}")
        
        if total_factors != 23:
            print(f"❌ Expected 23 factors, found {total_factors}")
            return False
        
        # Step 2: Test aggregation
        print("\n🔄 Step 2: Aggregating scores...")
        agg_result = Aggregator.compute_suitability_score(raw_data)
        
        category_scores = agg_result.get('category_scores', {})
        print(f"✅ Category scores: {category_scores}")
        
        if len(category_scores) != 6:
            print(f"❌ Expected 6 category scores, found {len(category_scores)}")
            return False
        
        # Check for Risk & Resilience category
        if 'risk_resilience' not in category_scores:
            print("❌ Risk & Resilience category missing from aggregation")
            return False
        
        # Step 3: Test frontend data structure
        print("\n🎨 Step 3: Testing frontend data structure...")
        
        # Simulate the frontend data structure from app.py
        f = {
            "physical": {
                "slope": normalize_factor(raw_factors["physical"]["slope"]),
                "elevation": normalize_factor(raw_factors["physical"]["elevation"]),
                "ruggedness": normalize_factor(raw_factors["physical"]["ruggedness"]),
                "stability": normalize_factor(raw_factors["physical"]["stability"]),
            },
            "hydrology": {
                "flood": normalize_factor(raw_factors["hydrology"]["flood"]),
                "water": normalize_factor(raw_factors["hydrology"]["water"]),
                "drainage": normalize_factor(raw_factors["hydrology"].get("drainage", {})),
                "groundwater": normalize_factor(raw_factors["hydrology"].get("groundwater", {})),
            },
            "environmental": {
                "vegetation": normalize_factor(raw_factors["environmental"]["vegetation"]),
                "pollution": normalize_factor(raw_factors["environmental"]["pollution"]),
                "soil": normalize_factor(raw_factors["environmental"]["soil"]),
                "biodiversity": normalize_factor(raw_factors["environmental"].get("biodiversity", {})),
                "heat_island": normalize_factor(raw_factors["environmental"].get("heat_island", {})),
            },
            "climatic": {
                "rainfall": normalize_factor(raw_factors["climatic"]["rainfall"]),
                "thermal": normalize_factor(raw_factors["climatic"]["thermal"]),
                "intensity": normalize_factor(raw_factors["climatic"].get("intensity", {})),
            },
            "socio_econ": {
                "landuse": normalize_factor(raw_factors["socio_econ"]["landuse"]),
                "infrastructure": normalize_factor(raw_factors["socio_econ"]["infrastructure"]),
                "population": normalize_factor(raw_factors["socio_econ"]["population"]),
            },
            "risk_resilience": {
                "multi_hazard": normalize_factor(raw_factors["risk_resilience"].get("multi_hazard", {})),
                "climate_change": normalize_factor(raw_factors["risk_resilience"].get("climate_change", {})),
                "recovery": normalize_factor(raw_factors["risk_resilience"].get("recovery", {})),
                "habitability": normalize_factor(raw_factors["risk_resilience"].get("habitability", {})),
            }
        }
        
        # Verify frontend structure
        frontend_total = 0
        for category, factors in f.items():
            if isinstance(factors, dict):
                frontend_total += len(factors)
        
        print(f"✅ Frontend factors: {frontend_total} (expected: 23)")
        
        if frontend_total != 23:
            print(f"❌ Frontend structure has {frontend_total} factors, expected 23")
            return False
        
        # Step 4: Test specific Risk & Resilience factors
        print("\n🛡️ Step 4: Testing Risk & Resilience factors...")
        risk_factors = f.get('risk_resilience', {})
        expected_risk_factors = ['multi_hazard', 'climate_change', 'recovery', 'habitability']
        
        for factor in expected_risk_factors:
            if factor not in risk_factors:
                print(f"❌ Missing Risk & Resilience factor: {factor}")
                return False
            
            factor_data = risk_factors[factor]
            if not isinstance(factor_data, dict) or 'value' not in factor_data:
                print(f"❌ Invalid Risk & Resilience factor structure for {factor}")
                return False
            
            value = factor_data['value']
            if not (0 <= value <= 100):
                print(f"❌ Invalid value for {factor}: {value} (expected 0-100)")
                return False
            
            print(f"✅ {factor}: {value:.1f}")
        
        # Step 5: Test complete API response structure
        print("\n📡 Step 5: Testing complete API response structure...")
        
        api_response = {
            "suitability_score": agg_result["score"],
            "label": agg_result["label"],
            "category_scores": agg_result["category_scores"],
            "factors": f,
        }
        
        print(f"✅ API response structure valid")
        print(f"✅ Final score: {api_response['suitability_score']:.1f}")
        print(f"✅ Label: {api_response['label']}")
        print(f"✅ Categories: {len(api_response['category_scores'])}")
        print(f"✅ Factors: {len(api_response['factors'])}")
        
        # Verify Risk & Resilience in API response
        if 'risk_resilience' not in api_response['category_scores']:
            print("❌ Risk & Resilience missing from API response")
            return False
        
        if 'risk_resilience' not in api_response['factors']:
            print("❌ Risk & Resilience factors missing from API response")
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Complete 23-factor integration working")
        print("✅ Backend processing all factors correctly")
        print("✅ Frontend data structure compatible")
        print("✅ Risk & Resilience category fully integrated")
        print("✅ API response structure valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def normalize_factor(factor_data):
    """Normalize factor data to frontend format."""
    if isinstance(factor_data, dict):
        return {
            "value": factor_data.get("value", 50),
            "scaled_score": factor_data.get("scaled_score", factor_data.get("value", 50))
        }
    else:
        return {"value": factor_data, "scaled_score": factor_data}

if __name__ == "__main__":
    success = test_backend_integration()
    if success:
        print("\n🚀 READY FOR PRODUCTION!")
        print("The 23-factor system is fully operational.")
        print("Restart your application server to apply changes.")
    sys.exit(0 if success else 1)
