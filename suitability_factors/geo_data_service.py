# # backend/suitability_factors/geo_data_service.py
# from datetime import datetime
# from typing import Dict, Any
# import requests
# # 📂 PHYSICAL TERRAIN
# from .physical_terrain.elevation_adapter import get_elevation_data

# from .physical_terrain.slope_analysis import get_slope_analysis

# # 📂 ENVIRONMENTAL
# from .environmental.pollution_index import get_pollution_metrics
# from .environmental.soil_health import get_soil_intelligence
# from .environmental.vegetation_ndvi import get_ndvi_data

# # 📂 HYDROLOGY
# from .hydrology.drainage_density import get_drainage_analysis
# from .hydrology.flood_hazard import estimate_flood_risk
# from .hydrology.water_utility import get_water_utility

# # 📂 CLIMATIC
# from .Climatic.rainfall_suitability import get_rainfall_analysis
# from .Climatic.thermal_comfort import get_thermal_comfort_analysis
# from .Climatic.thermal_intensity import get_thermal_intensity

# # 📂 SOCIO-ECONOMIC
# from .socio_economic.infrastructure_reach import get_infrastructure_score
# from .socio_economic.landuse_status import get_landuse_analysis
# from .socio_economic.population_density import get_population_data

# class GeoDataService:
#     @staticmethod
#     def get_land_intelligence(lat: float, lng: float) -> Dict[str, Any]:
#         """
#         Executes all 15 adapters and attaches real-world provenance data.
#         """
#         # 1. Category Recruitment
#         raw_results = {
#             "physical": {
#                 "slope": get_slope_analysis(lat, lng),
#                 "elevation": get_elevation_data(lat, lng),
#     
#             },
#             "environmental": {
#                 "vegetation": get_ndvi_data(lat, lng),
#                 "soil": get_soil_intelligence(lat, lng),
#                 "pollution": get_pollution_metrics(lat, lng)
#             },
#             "hydrology": {
#                 "flood": estimate_flood_risk(lat, lng),
#                 "water": get_water_utility(lat, lng),
#                 "drainage": get_drainage_analysis(lat, lng)
#             },
#             "climatic": {
#                 "rainfall": get_rainfall_analysis(lat, lng),
#                 "thermal": get_thermal_comfort_analysis(lat, lng),
#                 "intensity": get_thermal_intensity(lat, lng)
#             },
#             "socio_econ": {
#                 "infrastructure": get_infrastructure_score(lat, lng),
#                 "landuse": get_landuse_analysis(lat, lng),
#                 "population": get_population_data(lat, lng)
#             }
#         }

#         # 2. Attach Data Provenance (Metadata Proof)
#         provenance = {
#             "physical": {"source": "NASA SRTM v3.0", "link": "https://earthdata.nasa.gov/sensors/srtm"},
#             "environmental": {"source": "ESA Sentinel-2/5P", "link": "https://sentinel.esa.int/"},
#             "hydrology": {"source": "HydroSHEDS / OSM", "link": "https://www.hydrosheds.org/"},
#             "socio_econ": {"source": "WorldPop / OSM", "link": "https://www.worldpop.org/"},
#             "climatic": {"source": "Open-Meteo / Sentinel-3", "link": "https://open-meteo.com/"}
#         }

#         return {
#             "raw_factors": raw_results,
#             "metadata_proof": provenance,
#             "timestamp": datetime.now().isoformat()
#         }
# backend/suitability_factors/geo_data_service.py
from datetime import datetime
from typing import Dict, Any
import requests

# 📂 PHYSICAL TERRAIN
from .physical_terrain.elevation_adapter import get_elevation_data
from .physical_terrain.slope_analysis import get_slope_analysis

# 📂 ENVIRONMENTAL
from .environmental.pollution_index import estimate_pollution_score
from .environmental.soil_health import estimate_soil_quality_score
from .environmental.vegetation_ndvi import get_ndvi_data

# 📂 HYDROLOGY
from .hydrology.drainage_density import get_drainage_analysis
from .hydrology.flood_hazard import estimate_flood_risk
from .hydrology.water_utility import get_water_utility

# 📂 CLIMATIC
from .Climatic.rainfall_suitability import get_rainfall_analysis
from .Climatic.thermal_comfort import get_thermal_comfort_analysis
from .Climatic.thermal_intensity import get_thermal_intensity

# 📂 SOCIO-ECONOMIC
from .socio_economic.infrastructure_reach import get_infrastructure_score
from .socio_economic.landuse_status import infer_landuse_score
from .socio_economic.population_density import get_population_data


OPENAQ_URL = "https://api.openaq.org/v2/latest"


class GeoDataService:
    @staticmethod
    def _fetch_pollution_metrics(lat: float, lng: float) -> Dict[str, Any]:
        """
        Fetch raw pollution (PM2.5) data.
        NO suitability logic here.
        """
        try:
            params = {
                "coordinates": f"{lat},{lng}",
                "radius": 25000,
                "limit": 1,
            }
            resp = requests.get(OPENAQ_URL, params=params, timeout=8)
            resp.raise_for_status()
            js = resp.json()

            if not js.get("results"):
                return {
                    "pm25": None,
                    "source": "no_station"
                }

            res = js["results"][0]
            for m in res.get("measurements", []):
                if m.get("parameter") in ("pm25", "pm2.5", "pm_25"):
                    return {
                        "pm25": float(m.get("value")),
                        "unit": m.get("unit", "µg/m³"),
                        "location": res.get("location"),
                        "city": res.get("city"),
                        "last_updated": res.get("lastUpdated"),
                        "source": "OpenAQ"
                    }

            return {
                "pm25": None,
                "source": "station_found_no_pm25"
            }

        except Exception:
            return {
                "pm25": None,
                "source": "api_error_fallback"
            }

    # @staticmethod
    # def get_land_intelligence(lat: float, lng: float) -> Dict[str, Any]:
    #     """
    #     Executes all 15 adapters and attaches real-world provenance data.
    #     """

    #     # 🔹 Pollution (centralized fetch + pure scoring)
    #     pollution_ctx = GeoDataService._fetch_pollution_metrics(lat, lng)
    #     pollution_score, pm25, pollution_details = estimate_pollution_score(pollution_ctx)
    #     slope_data = get_slope_analysis(lat, lng)
    #     rainfall_data = get_rainfall_analysis(lat, lng)
    #     water_data = get_water_utility(lat, lng)
    #     soil_ctx = {
    #         "slope": slope_data.get("value"),
    #         "rain_mm_60d": rainfall_data.get("rain_mm_60d")
    #     }
    #     hydrology_ctx = {
    #         "rain_mm_60d": rainfall_data.get("rain_mm_60d"),
    #         "water_distance_km": water_data.get("distance_km")
    #     }

    #     soil_score = estimate_soil_quality_score(soil_ctx)
    #     flood_score = estimate_flood_risk(hydrology_ctx)
    #     # 1. Category Recruitment
    #     raw_results = {
    #         "physical": {
    #             "slope": slope_data,
    #             "elevation": get_elevation_data(lat, lng),
    #             "ruggedness": get_ruggedness_analysis(lat, lng)
    #         },
    #         "environmental": {
    #             "vegetation": get_ndvi_data(lat, lng),
    #             "soil": {
    #                 "value": soil_score,
    #                 "inputs": soil_ctx
    #             },
    #             "pollution": {
    #                 "value": pollution_score,
    #                 "pm25": pm25,
    #                 "details": pollution_details
    #             }
    #         },
    #         "hydrology": {
    #             # "flood": estimate_flood_risk(lat, lng),
    #             "flood": {
    #                 "value": flood_score,
    #                 "inputs": hydrology_ctx
    #             },
    #             "water": get_water_utility(lat, lng),
    #             "drainage": get_drainage_analysis(lat, lng)
    #         },
    #         "climatic": {
    #             "rainfall": get_rainfall_analysis(lat, lng),
    #             "thermal": get_thermal_comfort_analysis(lat, lng),
    #             "intensity": get_thermal_intensity(lat, lng)
    #         },
    #         "socio_econ": {
    #             "infrastructure": get_infrastructure_score(lat, lng),
    #             "landuse": get_landuse_analysis(lat, lng),
    #             "population": get_population_data(lat, lng)
    #         }
    #     }

    #     # 2. Attach Data Provenance (Metadata Proof)
    #     provenance = {
    #         "physical": {
    #             "source": "NASA SRTM v3.0",
    #             "link": "https://earthdata.nasa.gov/sensors/srtm"
    #         },
    #         "environmental": {
    #             "source": "ESA Sentinel-2/5P & OpenAQ",
    #             "link": "https://sentinel.esa.int/"
    #         },
    #         "hydrology": {
    #             "source": "HydroSHEDS / OSM",
    #             "link": "https://www.hydrosheds.org/"
    #         },
    #         "socio_econ": {
    #             "source": "WorldPop / OSM",
    #             "link": "https://www.worldpop.org/"
    #         },
    #         "climatic": {
    #             "source": "Open-Meteo / Sentinel-3",
    #             "link": "https://open-meteo.com/"
    #         }
    #     }

    #     return {
    #         "raw_factors": raw_results,
    #         "metadata_proof": provenance,
    #         "timestamp": datetime.now().isoformat()
    #     }
    @staticmethod
    def get_land_intelligence(lat: float, lng: float) -> Dict[str, Any]:
        """
        Executes all 15 adapters and attaches real-world provenance data.
        """

        # --------------------------------------------------
        # CENTRALIZED FETCHES (single source of truth)
        # --------------------------------------------------

        # Pollution (raw fetch + pure scoring)
        pollution_ctx = GeoDataService._fetch_pollution_metrics(lat, lng)
        # pollution_score, pm25, pollution_details = estimate_pollution_score(pollution_ctx)
        # pollution_data = estimate_pollution_score(pollution_ctx)
        # pollution_score = pollution_data.get("value")
        pollution_score, pm25, pollution_details = estimate_pollution_score(pollution_ctx)

        # pm25 = pollution_data.get("pm25")
        # pollution_details = pollution_data.get("details", {})


        # Hydrology (fetch first to detect water body before physical/others)
        water_data = get_water_utility(lat, lng)
        is_on_water = (
            water_data.get("value") == 0.0
            and water_data.get("distance_km") is not None
            and float(water_data.get("distance_km", 1)) < 0.02
        )

        # Physical terrain (override when on water — flat on water = flood prone, not suitable)
        if is_on_water:
            slope_data = {
                "value": 0.0,
                "label": "N/A (on water body)",
                "raw": None,
                "unit": "%",
                "source": "N/A — location on water body",
                "confidence": 99,
            }
            elevation_data_water = {
                "value": 0.0,
                "label": "N/A (on water body)",
                "raw": None,
                "unit": "m",
                "source": "N/A — location on water body",
                "confidence": 99,
            }
        else:
            slope_data = get_slope_analysis(lat, lng)
            elevation_data_water = None

        # Climatic
        rainfall_data = get_rainfall_analysis(lat, lng)

        # Hydrology (water already fetched)
        if is_on_water:
            drainage_data = {"value": 0.0, "label": "N/A (on water body)", "source": "N/A"}
        else:
            drainage_data = get_drainage_analysis(lat, lng)
        if isinstance(drainage_data, (int, float)):
            drainage_data = {"value": float(drainage_data)}
        elif drainage_data is None:
            drainage_data = {"value": 50.0}


        # --------------------------------------------------
        # CONTEXT CONSTRUCTION (NO LOGIC CHANGE)
        # --------------------------------------------------

        soil_ctx = {
            "slope": slope_data.get("value"),
            "rain_mm_60d": rainfall_data.get("rain_mm_60d")
        }

        hydrology_ctx = {
            "rain_mm_60d": rainfall_data.get("value"),
            "water_distance_km": water_data.get("distance_km")
        }

        # --------------------------------------------------
        # PURE SCORING (context-in, score-out)
        # --------------------------------------------------

        soil_score = estimate_soil_quality_score(soil_ctx)
        flood_score = 0.0 if is_on_water else estimate_flood_risk(hydrology_ctx)

        # --------------------------------------------------
        # CATEGORY RECRUITMENT (15 FACTORS)
        # --------------------------------------------------
        # --------------------------------------------------
        # LANDUSE NORMALIZATION (CRITICAL FIX)
        # --------------------------------------------------

        raw_landuse = infer_landuse_score(lat, lng)

        if isinstance(raw_landuse, tuple):
            landuse_score = raw_landuse[0]
            landuse_meta = raw_landuse[1] if len(raw_landuse) > 1 else {}
        elif isinstance(raw_landuse, dict):
            landuse_score = raw_landuse.get("value", 50)
            landuse_meta = raw_landuse
        else:
            landuse_score = raw_landuse
            landuse_meta = {}

        landuse_data = {
            "value": float(landuse_score),
            "classification": landuse_meta.get("classification", "Mixed"),
            "confidence": landuse_meta.get("confidence", 80),
            "source": landuse_meta.get("source", "Sentinel-2 + OSM")
        }
        # --------------------------------------------------
        # INFRASTRUCTURE NORMALIZATION (CRITICAL FIX)
        # --------------------------------------------------

        raw_infra = get_infrastructure_score(lat, lng)

        if isinstance(raw_infra, tuple):
            infra_score = raw_infra[0]
            infra_meta = raw_infra[1] if len(raw_infra) > 1 else {}
        elif isinstance(raw_infra, dict):
            infra_score = raw_infra.get("value", 50)
            infra_meta = raw_infra
        else:
            infra_score = raw_infra
            infra_meta = {}

        infrastructure_data = {
            "value": float(infra_score),
            "label": infra_meta.get("label", "Moderate Access"),
            "confidence": infra_meta.get("confidence", 80),
            "source": infra_meta.get("source", "OpenStreetMap")
        }
        # --------------------------------------------------
        # POPULATION NORMALIZATION (FINAL FIX)
        # --------------------------------------------------

        raw_population = get_population_data(lat, lng)

        if isinstance(raw_population, tuple):
            pop_score = raw_population[0]
            pop_meta = raw_population[1] if len(raw_population) > 1 else {}
        elif isinstance(raw_population, dict):
            pop_score = raw_population.get("value", 50)
            pop_meta = raw_population
        else:
            pop_score = raw_population
            pop_meta = {}

        population_data = {
            "value": float(pop_score),
            "density": pop_meta.get("density"),
            "label": pop_meta.get("label", "Moderate Density"),
            "source": pop_meta.get("source", "WorldPop"),
            "reasoning": pop_meta.get("reasoning"),
            "unit": pop_meta.get("unit", "people/km²")
        }
        # Override population when on water: no permanent population on open water
        if is_on_water:
            population_data = {
                "value": 0.0,
                "density": 0,
                "label": "N/A (on water body)",
                "source": "N/A",
                "reasoning": "No permanent population on open water. Density is 0 people/km².",
                "unit": "people/km²"
            }

        # Import the new physical terrain factors
        from .physical_terrain.terrain_ruggedness import get_terrain_ruggedness
        from .physical_terrain.land_stability import get_land_stability
        
        # Import the new environmental factors
        from .environmental.biodiversity_index import get_biodiversity_sensitivity
        from .environmental.heat_island_potential import get_heat_island_potential
        
        # Import the new hydrology factor
        from .hydrology.groundwater_recharge import get_groundwater_recharge_potential
        
        # Import the updated infrastructure factor
        from .socio_econ.infrastructure_proximity import get_infrastructure_proximity
        
        # Import the new Risk & Resilience factors
        from .risk_resilience.multi_hazard_risk import get_multi_hazard_risk
        from .risk_resilience.climate_change_stress import get_climate_change_stress
        from .risk_resilience.recovery_capacity import get_recovery_capacity
        from .risk_resilience.long_term_habitability import get_long_term_habitability
        
        # Get all the new factor data
        ruggedness_data = get_terrain_ruggedness(lat, lng)
        stability_data = get_land_stability(lat, lng)
        biodiversity_data = get_biodiversity_sensitivity(lat, lng)
        heat_island_data = get_heat_island_potential(lat, lng)
        groundwater_data = get_groundwater_recharge_potential(lat, lng)
        infrastructure_data = get_infrastructure_proximity(lat, lng)
        
        # Get Risk & Resilience data
        multi_hazard_data = get_multi_hazard_risk(lat, lng)
        climate_change_data = get_climate_change_stress(lat, lng)
        recovery_data = get_recovery_capacity(lat, lng)
        habitability_data = get_long_term_habitability(lat, lng)

        raw_results = {
            "physical": {
                "slope": slope_data,
                "elevation": elevation_data_water if is_on_water else get_elevation_data(lat, lng),
                "ruggedness": ruggedness_data,
                "stability": stability_data
            },
            "environmental": {
                "vegetation": get_ndvi_data(lat, lng),
                "soil": {
                    "value": soil_score,
                    "inputs": soil_ctx
                },
                "pollution": {
                    "value": pollution_score,
                    "pm25": pm25,
                    "details": pollution_details
                },
                "biodiversity": biodiversity_data,
                "heat_island": heat_island_data
            },
            "hydrology": {
                "flood": {
                    "value": flood_score,
                    "safety_score": flood_score,
                    "inputs": hydrology_ctx
                },
                "water": water_data,
                "drainage": drainage_data,
                "groundwater": groundwater_data
            },
            "climatic": {
                "rainfall": rainfall_data,
                "thermal": get_thermal_comfort_analysis(lat, lng),
                "intensity": get_thermal_intensity(lat, lng)
            },
            "socio_econ": {
                "infrastructure": infrastructure_data,
                "landuse": landuse_data,
                "population": population_data
            },
            "risk_resilience": {
                "multi_hazard": multi_hazard_data,
                "climate_change": climate_change_data,
                "recovery": recovery_data,
                "habitability": habitability_data
            }
        }

        # --------------------------------------------------
        # DATA PROVENANCE (METADATA PROOF)
        # --------------------------------------------------

        provenance = {
            "physical": {
                "source": "NASA SRTM v3.0",
                "link": "https://earthdata.nasa.gov/sensors/srtm"
            },
            "environmental": {
                "source": "ESA Sentinel-2/5P & OpenAQ",
                "link": "https://sentinel.esa.int/"
            },
            "hydrology": {
                "source": "HydroSHEDS / OSM",
                "link": "https://www.hydrosheds.org/"
            },
            "socio_econ": {
                "source": "WorldPop / OSM",
                "link": "https://www.worldpop.org/"
            },
            "climatic": {
                "source": "Open-Meteo / Sentinel-3",
                "link": "https://open-meteo.com/"
            }
        }

        return {
            "raw_factors": raw_results,
            "metadata_proof": provenance,
            "timestamp": datetime.now().isoformat()
        }
