import sys
sys.path.append('.')
from suitability_factors.aggregator import Aggregator

# Test the new 4-factor physical category with terrain ruggedness and land stability
test_package = {
    'raw_factors': {
        'physical': {
            'slope': {'value': 10},           # Gentle slope - good
            'elevation': {'value': 200},        # Optimal elevation - good
            'ruggedness': {'value': 85.0},     # Low ruggedness - excellent
            'stability': {'value': 80.0}        # High stability - excellent
        },
        'environmental': {'vegetation': {'ndvi_index': 0.7}, 'soil': {'value': 75}, 'pollution': {'value': 60}},
        'hydrology': {'water': {'value': 80}, 'drainage': {'value': 70}, 'flood': {'value': 75}},
        'climatic': {'rainfall': {'value': 70}, 'thermal': {'value': 75}, 'intensity': {'value': 80}},
        'socio_econ': {'infrastructure': {'value': 70}, 'landuse': {'value': 65}, 'population': {'value': 50}}
    },
    'latitude': 28.6,
    'longitude': 77.2
}

result = Aggregator.compute_suitability_score(test_package)
print('=== NEW 4-FACTOR PHYSICAL CATEGORY TEST ===')
print(f'Physical Category: {result["category_scores"]["physical"]}')
print(f'Environmental: {result["category_scores"]["environmental"]}')
print(f'Hydrology: {result["category_scores"]["hydrology"]}')
print(f'Climatic: {result["category_scores"]["climatic"]}')
print(f'Socio-Economic: {result["category_scores"]["socio_econ"]}')
print(f'Final Score: {result["score"]}')
print()

print('=== PHYSICAL FACTOR BREAKDOWN ===')
print('Physical now has 4 factors instead of 2:')
print('- Slope Analysis: 10% of total score')
print('- Elevation Data: 10% of total score') 
print('- Terrain Ruggedness: 10% of total score')
print('- Land Stability: 10% of total score')
print('- Total Physical: 40% of total score (20% category weight)')
print()

print('=== FACTOR WEIGHT DISTRIBUTION ===')
print('Physical (4 factors): 4 × 5% = 20% of total')
print('Environmental (3 factors): 3 × 6.67% = 20% of total')
print('Hydrology (3 factors): 3 × 6.67% = 20% of total')
print('Climatic (3 factors): 3 × 6.67% = 20% of total')
print('Socio-Economic (3 factors): 3 × 6.67% = 20% of total')
print('Total: 16 factors across 5 categories = 100%')
print()

print('=== EXPECTED PHYSICAL CALCULATION ===')
physical_expected = (10.0 + 200.0 + 85.0 + 80.0) / 4  # Raw scores averaged
print(f'Physical Raw Average: {physical_expected}')
print(f'Physical Weighted (20%): {physical_expected * 0.2}')
print(f'Actual Physical: {result["category_scores"]["physical"]}')
