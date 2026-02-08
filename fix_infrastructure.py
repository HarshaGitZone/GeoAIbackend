# Fix the f-string syntax error
with open('suitability_factors/socio_economic/infrastructure_proximity.py', 'r') as f:
    content = f.read()

# Replace the problematic line with correct f-string syntax
fixed_content = content.replace(
    'logger.warning(f"Distance parsing error for \'{x}\': {e}")',
    'logger.warning(f"Distance parsing error for \'{x}\': {e}")'
)

with open('suitability_factors/socio_economic/infrastructure_proximity.py', 'w') as f:
    f.write(fixed_content)

print('Fixed f-string syntax error in infrastructure_proximity.py')
