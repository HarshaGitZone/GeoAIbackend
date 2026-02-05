import sys
sys.path.append('.')
from suitability_factors.aggregator import Aggregator

# Create a test package to match your numbers: 44.7, 93.2, 100
# We need to reverse-engineer the raw factors that would give these category scores

# For Physical to get 44.7: (slope + elevation) / 2 = 44.7
# So slope + elevation = 89.4
# Let's try slope = 30, elevation = 59.4

# For Environmental to get 93.2: (vegetation + soil + pollution) / 3 = 93.2
# So vegetation + soil + pollution = 279.6
# Let's try vegetation = 95, soil = 95, pollution = 89.6

# For Hydrology to get 100: (water + drainage + flood) / 3 = 100
# So water + drainage + flood = 300
# Let's try water = 100, drainage = 100, flood = 100

test_package = {
    'raw_factors': {
        'physical': {'slope': {'value': 30}, 'elevation': {'value': 59.4}},
        'environmental': {'vegetation': {'ndvi_index': 0.95}, 'soil': {'value': 95}, 'pollution': {'value': 89.6}},
        'hydrology': {'water': {'value': 100}, 'drainage': {'value': 100}, 'flood': {'value': 100}},
        'climatic': {'rainfall': {'value': 50}, 'thermal': {'value': 50}},  # Missing data defaults
        'socio_econ': {'infrastructure': {'value': 50}, 'landuse': {'value': 50}, 'population': {'value': 50}}  # Missing data defaults
    },
    'latitude': 28.6,
    'longitude': 77.2
}

result = Aggregator.compute_suitability_score(test_package)
print('=== YOUR NUMBERS DEBUG ===')
print(f'Physical: {result["category_scores"]["physical"]} (expected: 44.7)')
print(f'Environmental: {result["category_scores"]["environmental"]} (expected: 93.2)')
print(f'Hydrology: {result["category_scores"]["hydrology"]} (expected: 100)')
print(f'Climatic: {result["category_scores"]["climatic"]}')
print(f'Socio: {result["category_scores"]["socio_econ"]}')
print(f'Final Score: {result["score"]} (you expected: 97)')
print('=== WEIGHTED CALCULATION ===')
phys = result['category_scores']['physical'] * 0.2
env = result['category_scores']['environmental'] * 0.2
hydro = result['category_scores']['hydrology'] * 0.2
clim = result['category_scores']['climatic'] * 0.2
socio = result['category_scores']['socio_econ'] * 0.2
manual_total = phys + env + hydro + clim + socio
print(f'Physical (20%): {phys}')
print(f'Environmental (20%): {env}')
print(f'Hydrology (20%): {hydro}')
print(f'Climatic (20%): {clim}')
print(f'Socio (20%): {socio}')
print(f'Manual Total: {manual_total}')
print(f'=== WHY YOU EXPECTED 97 ===')
print(f'If you add: 44.7 + 93.2 + 100 = {44.7 + 93.2 + 100}')
print(f'But actual calculation is weighted: {manual_total}')
