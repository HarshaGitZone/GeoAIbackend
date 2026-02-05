#!/usr/bin/env python3
"""
Test imports for Risk & Resilience factors
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_imports():
    """Test all Risk & Resilience imports."""
    print("🧪 TESTING RISK & RESILIENCE IMPORTS")
    print("=" * 50)
    
    try:
        print("📦 Importing multi_hazard_risk...")
        from suitability_factors.risk_resilience.multi_hazard_risk import get_multi_hazard_risk
        print("✅ multi_hazard_risk imported successfully")
    except Exception as e:
        print(f"❌ multi_hazard_risk import failed: {e}")
        return False
    
    try:
        print("📦 Importing climate_change_stress...")
        from suitability_factors.risk_resilience.climate_change_stress import get_climate_change_stress
        print("✅ climate_change_stress imported successfully")
    except Exception as e:
        print(f"❌ climate_change_stress import failed: {e}")
        return False
    
    try:
        print("📦 Importing recovery_capacity...")
        from suitability_factors.risk_resilience.recovery_capacity import get_recovery_capacity
        print("✅ recovery_capacity imported successfully")
    except Exception as e:
        print(f"❌ recovery_capacity import failed: {e}")
        return False
    
    try:
        print("📦 Importing long_term_habitability...")
        from suitability_factors.risk_resilience.long_term_habitability import get_long_term_habitability
        print("✅ long_term_habitability imported successfully")
    except Exception as e:
        print(f"❌ long_term_habitability import failed: {e}")
        return False
    
    try:
        print("📦 Importing infrastructure_proximity...")
        from suitability_factors.socio_econ.infrastructure_proximity import get_infrastructure_proximity
        print("✅ infrastructure_proximity imported successfully")
    except Exception as e:
        print(f"❌ infrastructure_proximity import failed: {e}")
        return False
    
    print("\n🎉 ALL IMPORTS SUCCESSFUL!")
    return True

def test_function_calls():
    """Test calling the Risk & Resilience functions."""
    print("\n🧪 TESTING FUNCTION CALLS")
    print("=" * 50)
    
    # Test coordinates
    lat, lng = 17.5387, 78.3950
    
    try:
        from suitability_factors.risk_resilience.multi_hazard_risk import get_multi_hazard_risk
        print("📊 Testing multi_hazard_risk...")
        result = get_multi_hazard_risk(lat, lng)
        print(f"✅ multi_hazard_risk returned: {type(result)} with value: {result.get('value', 'N/A') if isinstance(result, dict) else result}")
    except Exception as e:
        print(f"❌ multi_hazard_risk call failed: {e}")
        return False
    
    try:
        from suitability_factors.risk_resilience.climate_change_stress import get_climate_change_stress
        print("📊 Testing climate_change_stress...")
        result = get_climate_change_stress(lat, lng)
        print(f"✅ climate_change_stress returned: {type(result)} with value: {result.get('value', 'N/A') if isinstance(result, dict) else result}")
    except Exception as e:
        print(f"❌ climate_change_stress call failed: {e}")
        return False
    
    try:
        from suitability_factors.risk_resilience.recovery_capacity import get_recovery_capacity
        print("📊 Testing recovery_capacity...")
        result = get_recovery_capacity(lat, lng)
        print(f"✅ recovery_capacity returned: {type(result)} with value: {result.get('value', 'N/A') if isinstance(result, dict) else result}")
    except Exception as e:
        print(f"❌ recovery_capacity call failed: {e}")
        return False
    
    try:
        from suitability_factors.risk_resilience.long_term_habitability import get_long_term_habitability
        print("📊 Testing long_term_habitability...")
        result = get_long_term_habitability(lat, lng)
        print(f"✅ long_term_habitability returned: {type(result)} with value: {result.get('value', 'N/A') if isinstance(result, dict) else result}")
    except Exception as e:
        print(f"❌ long_term_habitability call failed: {e}")
        return False
    
    try:
        from suitability_factors.socio_econ.infrastructure_proximity import get_infrastructure_proximity
        print("📊 Testing infrastructure_proximity...")
        result = get_infrastructure_proximity(lat, lng)
        print(f"✅ infrastructure_proximity returned: {type(result)} with value: {result.get('value', 'N/A') if isinstance(result, dict) else result}")
    except Exception as e:
        print(f"❌ infrastructure_proximity call failed: {e}")
        return False
    
    print("\n🎉 ALL FUNCTION CALLS SUCCESSFUL!")
    return True

if __name__ == "__main__":
    print("🚀 STARTING RISK & RESILIENCE INTEGRATION TEST")
    print("=" * 80)
    
    imports_ok = test_imports()
    if not imports_ok:
        print("❌ IMPORT TESTS FAILED")
        sys.exit(1)
    
    calls_ok = test_function_calls()
    if not calls_ok:
        print("❌ FUNCTION CALL TESTS FAILED")
        sys.exit(1)
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ Risk & Resilience factors are working correctly")
    sys.exit(0)
