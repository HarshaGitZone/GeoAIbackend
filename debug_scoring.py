import sys
sys.path.append('.')
from suitability_factors.aggregator import Aggregator

# Create a test package with your numbers
test_package = {
    'raw_factors': {
        'physical': {'slope': {'value': 30}, 'elevation': {'value': 200}},
        'environmental': {'vegetation': {'ndvi_index': 0.9}, 'soil': {'value': 90}, 'pollution': {'value': 85}},
        'hydrology': {'water': {'value': 95}, 'drainage': {'value': 90}, 'flood': {'value': 95}},
        'climatic': {'rainfall': {'value': 80}, 'thermal': {'value': 85}},
        'socio_econ': {'infrastructure': {'value': 88}, 'landuse': {'value': 92}, 'population': {'value': 90}}
    },
    'latitude': 28.6,
    'longitude': 77.2
}

result = Aggregator.compute_suitability_score(test_package)
print('=== DEBUG CALCULATION ===')
print(f'Physical: {result["category_scores"]["physical"]}')
print(f'Environmental: {result["category_scores"]["environmental"]}')
print(f'Hydrology: {result["category_scores"]["hydrology"]}')
print(f'Climatic: {result["category_scores"]["climatic"]}')
print(f'Socio: {result["category_scores"]["socio_econ"]}')
print(f'Final Score: {result["score"]}')
print('=== MANUAL CALCULATION ===')
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
