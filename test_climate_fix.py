import sys
sys.path.append('.')
from suitability_factors.aggregator import Aggregator

# Test with your exact climate numbers
test_package = {
    'raw_factors': {
        'physical': {'slope': {'value': 50}, 'elevation': {'value': 50}},
        'environmental': {'vegetation': {'ndvi_index': 0.5}, 'soil': {'value': 50}, 'pollution': {'value': 50}},
        'hydrology': {'water': {'value': 50}, 'drainage': {'value': 50}, 'flood': {'value': 50}},
        'climatic': {
            'rainfall': {'value': 44.7},    # Your first number
            'thermal': {'value': 93.2},     # Your second number  
            'intensity': {'value': 100}     # Your third number
        },
        'socio_econ': {'infrastructure': {'value': 50}, 'landuse': {'value': 50}, 'population': {'value': 50}}
    },
    'latitude': 28.6,
    'longitude': 77.2
}

result = Aggregator.compute_suitability_score(test_package)
print('=== CLIMATE FIX TEST ===')
print(f'Climate Category: {result["category_scores"]["climatic"]}')
print(f'Final Score: {result["score"]}')
print()
print('=== MANUAL CALCULATION ===')
climate_avg = (44.7 + 93.2 + 100) / 3
climate_weighted = climate_avg * 0.2
print(f'Climate Average: {climate_avg}')
print(f'Climate Weighted (20%): {climate_weighted}')
print()
print('Other categories (assuming 50 each):')
other_weighted = 50 * 0.2 * 4  # 4 categories at 50 each, 20% weight
print(f'Other 4 categories weighted: {other_weighted}')
print(f'Total expected: {climate_weighted + other_weighted}')
