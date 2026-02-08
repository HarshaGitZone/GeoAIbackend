content = open('suitability_factors/socio_economic/infrastructure_proximity.py', 'r').read()
fixed = content.replace('logger.warning(f"Distance parsing error for \'{x}\': {e}")', 'logger.warning(f"Distance parsing error for \'{x}\': {e}")')
open('suitability_factors/socio_economic/infrastructure_proximity.py', 'w').write(fixed)
print('Fixed f-string syntax error')
