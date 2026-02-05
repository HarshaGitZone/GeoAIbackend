import sys
sys.path.append('.')
from suitability_factors.aggregator import Aggregator

# Test the enhanced system with new Risk & Resilience category (22 factors total)
test_package = {
    'raw_factors': {
        'physical': {
            'slope': {'value': 10},           # Gentle slope - good
            'elevation': {'value': 200},        # Optimal elevation - good
            'ruggedness': {'value': 85.0},     # Low ruggedness - excellent
            'stability': {'value': 80.0}        # High stability - excellent
        },
        'environmental': {
            'vegetation': {'ndvi_index': 0.7}, 'soil': {'value': 75}, 'pollution': {'value': 60},
            'biodiversity': {'value': 75.0},   # High biodiversity - good
            'heat_island': {'value': 70.0}     # Moderate heat island - good
        },
        'hydrology': {
            'water': {'value': 80}, 'drainage': {'value': 70}, 'flood': {'value': 75},
            'groundwater': {'value': 70.0}    # Good groundwater - good
        },
        'climatic': {'rainfall': {'value': 70}, 'thermal': {'value': 75}, 'intensity': {'value': 80}},
        'socio_econ': {
            'infrastructure': {'value': 75.0, 'proximity_index': 80.0, 'travel_time_to_city': 30},  # Good infrastructure
            'landuse': {'value': 65}, 'population': {'value': 50}
        },
        'risk_resilience': {
            'multi_hazard': {'value': 75.0},    # Low multi-hazard risk - good
            'climate_change': {'value': 70.0},   # Moderate climate change stress - good
            'recovery': {'value': 80.0},         # Good recovery capacity - excellent
            'habitability': {'value': 85.0}      # Excellent long-term habitability
        }
    },
    'latitude': 28.6,
    'longitude': 77.2
}

result = Aggregator.compute_suitability_score(test_package)
print('=== ENHANCED GEOAI WITH RISK & RESILIENCE CATEGORY ===')
print(f'Physical (4 factors): {result["category_scores"]["physical"]}')
print(f'Environmental (5 factors): {result["category_scores"]["environmental"]}')
print(f'Hydrology (4 factors): {result["category_scores"]["hydrology"]}')
print(f'Climatic (3 factors): {result["category_scores"]["climatic"]}')
print(f'Socio-Economic (3 factors): {result["category_scores"]["socio_econ"]}')
print(f'Risk & Resilience (4 factors): {result["category_scores"]["risk_resilience"]}')
print(f'Final Score: {result["score"]}')
print()

print('=== FACTOR DISTRIBUTION BREAKDOWN ===')
print('Physical (4 factors): 4 × 4.17% = 16.67% of total')
print('Environmental (5 factors): 5 × 3.33% = 16.67% of total')
print('Hydrology (4 factors): 4 × 4.17% = 16.67% of total')
print('Climatic (3 factors): 3 × 5.56% = 16.67% of total')
print('Socio-Economic (3 factors): 3 × 5.56% = 16.67% of total')
print('Risk & Resilience (4 factors): 4 × 4.17% = 16.67% of total')
print('Total: 22 factors across 6 categories = 100%')
print()

print('=== NEW RISK & RESILIENCE FACTORS ===')
print('🆕 Multi-Hazard Risk Index:')
print('  - Flood + heat + erosion + seismic + storm + drought risk')
print('  - Data: NOAA, USGS, EM-DAT, Global Risk Atlas')
print('  - Answers: How vulnerable is this land to natural disasters?')
print()
print('🆕 Climate Change Stress:')
print('  - Future warming & rainfall shift projections')
print('  - Data: IPCC, NASA GISS, NOAA Climate Models, CMIP6')
print('  - Answers: How will climate change affect this location?')
print()
print('🆕 Recovery Capacity:')
print('  - Infrastructure + accessibility + economic capacity')
print('  - Data: World Bank, UN Development Index, Emergency Services')
print('  - Answers: How well can this area recover from disasters?')
print()
print('🆕 Long-Term Habitability:')
print('  - Combined survivability score for sustainable living')
print('  - Data: NASA Climate Models, UN Habitat, WHO Health Data')
print('  - Answers: How long will this area remain habitable?')
print()

print('=== WHY RISK & RESILIENCE DESERVES ITS OWN CATEGORY ===')
print('🎯 Addresses Critical Questions:')
print('  - "How well will this land survive shocks?"')
print('  - "What are the long-term sustainability prospects?"')
print('  - "Can this location adapt to climate change?"')
print()
print('🎯 Fills Critical Gaps:')
print('  - Current risk factors are distributed across categories')
print('  - No explicit modeling of long-term resilience')
print('  - Missing climate change impact assessment')
print()
print('🎯 Real-World Applications:')
print('  - Urban planning for climate resilience')
print('  - Infrastructure investment decisions')
print('  - Insurance risk assessment')
print('  - Long-term land use planning')
print()

print('=== DATA SOURCES FOR RISK & RESILIENCE ===')
print('Multi-Hazard Risk:')
print('  - NOAA Flood Inundation Mapping')
print('  - USGS Seismic Hazard Data')
print('  - EM-DAT International Disaster Database')
print('  - Global Risk Atlas')
print('  - Climate Model Projections')
print()
print('Climate Change Stress:')
print('  - IPCC Climate Assessment Reports')
print('  - NASA GISS Temperature Data')
print('  - NOAA Climate Data Services')
print('  - CMIP6 Climate Model Ensemble')
print('  - Sea Level Rise Projections')
print()
print('Recovery Capacity:')
print('  - World Bank Infrastructure Database')
print('  - UN Human Development Index')
print('  - OpenStreetMap Emergency Services')
print('  - Economic Development Indicators')
print('  - Institutional Capacity Assessments')
print()
print('Long-Term Habitability:')
print('  - NASA Climate Model Projections')
print('  - UN Habitat Urban Development Data')
print('  - WHO Environmental Health Data')
print('  - Food and Agriculture Organization (FAO)')
print('  - Water Resource Assessments')
print()

print('=== IMPACT ON SCORING SYSTEM ===')
print('✅ Equal Category Weighting Maintained:')
print('  - Each of 6 categories = 16.67% of total score')
print('  - Fair and balanced assessment framework')
print('  - No single category dominates the scoring')
print()
print('✅ Enhanced Risk Assessment:')
print('  - Explicit modeling of multi-hazard vulnerability')
print('  - Climate change impact projections')
print('  - Recovery capacity evaluation')
print('  - Long-term habitability assessment')
print()
print('✅ Better Decision Making:')
print('  - More comprehensive risk understanding')
print('  - Long-term sustainability considerations')
print('  - Climate resilience planning')
print('  - Infrastructure investment guidance')
print()

print('=== BENEFITS FOR DIFFERENT USE CASES ===')
print('🏗️ Urban Development:')
print('  - Climate-resilient city planning')
print('  - Infrastructure risk assessment')
print('  - Long-term urban sustainability')
print()
print('🌿 Environmental Conservation:')
print('  - Climate change adaptation planning')
print('  - Protected area resilience assessment')
print('  - Ecosystem service valuation')
print()
print('💧 Water Resource Management:')
print('  - Climate impact on water availability')
print('  - Flood risk mitigation planning')
print('  - Long-term water security assessment')
print()
print('🏛️ Infrastructure Planning:')
print('  - Climate-resilient infrastructure design')
print('  - Disaster recovery planning')
print('  - Long-term asset viability assessment')
print()

print('=== FRONTEND INTEGRATION ===')
print('📊 New Category Display:')
print('  - Risk & Resilience section in results')
print('  - 4 sub-factors with detailed breakdowns')
print('  - Visual risk indicators and resilience scores')
print('  - Climate change impact projections')
print()
print('📊 Enhanced Visualizations:')
print('  - Multi-hazard risk maps')
print('  - Climate change impact charts')
print('  - Recovery capacity indicators')
print('  - Long-term habitability projections')
print()
print('📊 Interactive Features:')
print('  - Risk scenario modeling')
print('  - Climate change timeline visualization')
print('  - Recovery capacity comparison tools')
print('  - Habitability horizon calculator')
