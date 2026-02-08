#!/usr/bin/env python3
"""
Test script to verify CNN classification fix
"""

import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import the CNN classification function
    from app import get_cnn_classification
    
    # Test with sample coordinates
    lat, lng = 28.6139, 77.2090  # Delhi
    
    print("Testing CNN classification with Delhi coordinates...")
    result = get_cnn_classification(lat, lng)
    
    if result:
        print("✅ CNN Classification SUCCESS!")
        print(f"Class: {result.get('class')}")
        print(f"Confidence: {result.get('confidence_display')}")
        print(f"Location Type: {result.get('telemetry', {}).get('location_type')}")
        print(f"Total Factors: {result.get('telemetry', {}).get('analysis_summary', {}).get('total_factors')}")
        print(f"Categories: {result.get('telemetry', {}).get('analysis_summary', {}).get('categories')}")
    else:
        print("❌ CNN Classification FAILED!")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
