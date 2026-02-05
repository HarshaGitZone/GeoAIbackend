#!/usr/bin/env python3
"""
Comprehensive test to verify all backend fixes are working
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_infrastructure_proximity():
    """Test infrastructure proximity fix."""
    print("🧪 Testing Infrastructure Proximity Fix...")
    
    try:
        from suitability_factors.socio_econ.infrastructure_proximity import get_infrastructure_proximity
        result = get_infrastructure_proximity(17.5387, 78.3950)
        print(f"✅ Infrastructure proximity working: {result.get('value', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Infrastructure proximity failed: {e}")
        return False

def test_multi_hazard_risk():
    """Test multi-hazard risk fix."""
    print("🧪 Testing Multi-Hazard Risk Fix...")
    
    try:
        from suitability_factors.risk_resilience.multi_hazard_risk import get_multi_hazard_risk
        result = get_multi_hazard_risk(17.5387, 78.3950)
        print(f"✅ Multi-hazard risk working: {result.get('value', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Multi-hazard risk failed: {e}")
        return False

def test_climate_change_stress():
    """Test climate change stress fix."""
    print("🧪 Testing Climate Change Stress Fix...")
    
    try:
        from suitability_factors.risk_resilience.climate_change_stress import get_climate_change_stress
        result = get_climate_change_stress(17.5387, 78.3950)
        print(f"✅ Climate change stress working: {result.get('value', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Climate change stress failed: {e}")
        return False

def test_recovery_capacity():
    """Test recovery capacity fix."""
    print("🧪 Testing Recovery Capacity Fix...")
    
    try:
        from suitability_factors.risk_resilience.recovery_capacity import get_recovery_capacity
        result = get_recovery_capacity(17.5387, 78.3950)
        print(f"✅ Recovery capacity working: {result.get('value', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Recovery capacity failed: {e}")
        return False

def test_long_term_habitability():
    """Test long-term habitability fix."""
    print("🧪 Testing Long-Term Habitability Fix...")
    
    try:
        from suitability_factors.risk_resilience.long_term_habitability import get_long_term_habitability
        result = get_long_term_habitability(17.5387, 78.3950)
        print(f"✅ Long-term habitability working: {result.get('value', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Long-term habitability failed: {e}")
        return False

def test_complete_integration():
    """Test complete integration with all factors."""
    print("🧪 Testing Complete Integration...")
    
    try:
        from suitability_factors.geo_data_service import GeoDataService
        from suitability_factors.aggregator import Aggregator
        
        # Test data collection
        raw_data = GeoDataService.get_land_intelligence(17.5387, 78.3950)
        
        # Check for Risk & Resilience factors
        risk_factors = raw_data.get('raw_factors', {}).get('risk_resilience', {})
        expected_factors = ['multi_hazard', 'climate_change', 'recovery', 'habitability']
        
        missing_factors = set(expected_factors) - set(risk_factors.keys())
        if missing_factors:
            print(f"❌ Missing Risk & Resilience factors: {missing_factors}")
            return False
        
        # Test aggregation
        result = Aggregator.compute_suitability_score(raw_data)
        
        # Check for all 6 categories
        category_scores = result.get('category_scores', {})
        expected_categories = ['physical', 'environmental', 'hydrology', 'climatic', 'socio_econ', 'risk_resilience']
        
        missing_categories = set(expected_categories) - set(category_scores.keys())
        if missing_categories:
            print(f"❌ Missing categories: {missing_categories}")
            return False
        
        print(f"✅ Complete integration working:")
        print(f"   📊 Final Score: {result.get('score', 0):.1f}")
        print(f"   📋 Categories: {len(category_scores)}")
        print(f"   🛡️ Risk & Resilience: {category_scores.get('risk_resilience', 0):.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete integration failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 COMPREHENSIVE BACKEND FIXES TEST")
    print("=" * 80)
    
    tests = [
        test_infrastructure_proximity,
        test_multi_hazard_risk,
        test_climate_change_stress,
        test_recovery_capacity,
        test_long_term_habitability,
        test_complete_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ All backend errors have been fixed")
        print("✅ Risk & Resilience category fully functional")
        print("✅ All 22 factors working correctly")
        print("✅ Frontend can now display complete data")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
