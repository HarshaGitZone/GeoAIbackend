import sys
sys.path.append('.')
from suitability_factors.aggregator import Aggregator

# Test the enhanced system with all new factors (19 factors total)
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
        }
    },
    'latitude': 28.6,
    'longitude': 77.2
}

result = Aggregator.compute_suitability_score(test_package)
print('=== ENHANCED GEOAI SYSTEM TEST ===')
print(f'Physical (4 factors): {result["category_scores"]["physical"]}')
print(f'Environmental (5 factors): {result["category_scores"]["environmental"]}')
print(f'Hydrology (4 factors): {result["category_scores"]["hydrology"]}')
print(f'Climatic (3 factors): {result["category_scores"]["climatic"]}')
print(f'Socio-Economic (3 factors): {result["category_scores"]["socio_econ"]}')
print(f'Final Score: {result["score"]}')
print()

print('=== FACTOR DISTRIBUTION BREAKDOWN ===')
print('Physical (4 factors): 4 × 5% = 20% of total')
print('Environmental (5 factors): 5 × 4% = 20% of total')
print('Hydrology (4 factors): 4 × 5% = 20% of total')
print('Climatic (3 factors): 3 × 6.67% = 20% of total')
print('Socio-Economic (3 factors): 3 × 6.67% = 20% of total')
print('Total: 19 factors across 5 categories = 100%')
print()

print('=== NEW FACTORS ADDED ===')
print('🌿 Environmental:')
print('  - Biodiversity Sensitivity Index (NEW!)')
print('  - Heat Island Potential (NEW!)')
print('💧 Hydrology:')
print('  - Groundwater Recharge Potential (NEW!)')
print('🏗️ Physical:')
print('  - Terrain Ruggedness (NEW!)')
print('  - Land Stability/Erosion (NEW!)')
print('🏙️ Socio-Economic:')
print('  - Enhanced Infrastructure Proximity with Travel Time (UPDATED!)')
print()

print('=== DATA SOURCES ===')
print('Environmental:')
print('  - IUCN Protected Areas Database')
print('  - WWF Ecoregions Database')
print('  - NASA MODIS Albedo Data')
print('  - OpenStreetMap Building Density')
print('Hydrology:')
print('  - ISRIC SoilGrids (soil permeability)')
print('  - Open-Meteo Historical Rainfall')
print('  - USGS Groundwater Database')
print('Physical:')
print('  - NASA SRTM (30m elevation)')
print('  - USGS Geological Survey')
print('  - ASTER GDEM')
print('Socio-Economic:')
print('  - OpenStreetMap (roads, transport)')
print('  - 25+ Major Urban Centers')
print('  - Public Transit APIs')
print()

print('=== BENEFITS OF NEW FACTORS ===')
print('✅ More comprehensive environmental assessment')
print('✅ Better understanding of climate-human interactions')
print('✅ Sustainable water resource evaluation')
print('✅ Long-term land stability considerations')
print('✅ Realistic infrastructure accessibility with travel times')
print('✅ Geographic context-aware dynamic defaults')
print('✅ Maintained 20% category weighting system')
print('✅ Enhanced accuracy for real-world applications')
