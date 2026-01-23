"""Integration layer for the GeoAI unified platform.

Exposes lightweight adapters that reference sibling projects without
modifying or moving them. Adapters should gracefully degrade when
dependencies are unavailable, returning sensible defaults.
"""

from .paths import get_workspace_root, get_project_path
from .aggregator import compute_suitability_score
from .floodml_adapter import estimate_flood_risk_score
from .pylusat_adapter import compute_proximity_score
from .pylandslide_adapter import estimate_landslide_risk_score
from .water_adapter import estimate_water_proximity_score
from .pollution_adapter import estimate_pollution_score
from .landuse_adapter import infer_landuse_score
from .soil_adapter import estimate_soil_quality_score
from .rainfall_adapter import estimate_rainfall_score
from ..ml.train import load_or_create_sample_dataset
from .terrain_adapter import estimate_terrain_slope
__all__ = [
	"get_workspace_root",
	"get_project_path",
	"compute_suitability_score",
	"estimate_flood_risk_score",
	"compute_proximity_score",
	"estimate_landslide_risk_score",
	"estimate_water_proximity_score",
	"estimate_pollution_score",
	"infer_landuse_score",
	"estimate_soil_quality_score",
	"estimate_rainfall_score",
	"load_or_create_sample_dataset",
    "estimate_terrain_slope"
]


