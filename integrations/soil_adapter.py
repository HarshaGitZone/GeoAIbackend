from typing import Optional
import random
# Import your existing water detection logic
from integrations.water_adapter import estimate_water_proximity_score

def estimate_soil_quality_score(latitude: float, longitude: float) -> Optional[float]:
    """
    Returns soil quality score.
    FORCES 0.0 if the location is a water body.
    """
    # 1. Check if the location is water first
    water_score, water_dist, _ = estimate_water_proximity_score(latitude, longitude)
    
    # 2. KILLER FILTER: If it is on water, soil quality is zero
    if water_score == 0.0 or (water_dist is not None and water_dist < 0.02):
        return 0.0

    # 3. For actual land, return the deterministic score
    seed = int(round(latitude * 1000)) ^ int(round(longitude * 1000))
    random.seed(seed)
    
    # Returns a consistent score between 40 and 100 for land
    return float(round(40 + random.random() * 60, 2))