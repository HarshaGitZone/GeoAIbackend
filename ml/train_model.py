"""
Train ensemble of regressors (RF, XGBoost, GBM, ExtraTrees, optional LightGBM) on Indian-location
suitability data using the same 14 factors as the app aggregator.
Prints train/test accuracy (MAE, RMSE, R²) per model and saves all for use in the app.

Run from project root (GeoAI):
  python backend/ml/train_model.py
  python backend/ml/train_model.py --samples 5000
  python backend/ml/train_model.py --report-accuracy
"""

import os
import sys
import random
import pickle
import argparse
import numpy as np

# Project root so backend imports work
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Same 23 factors and order as app ML_FACTOR_ORDER (complete factor system)
FACTOR_ORDER = [
    # Physical (4)
    "slope", "elevation", "ruggedness", "stability",
    # Hydrology (4) 
    "flood", "water", "drainage", "groundwater",
    # Environmental (5)
    "vegetation", "pollution", "soil", "biodiversity", "heat_island",
    # Climatic (3)
    "rainfall", "thermal", "intensity",
    # Socio-Economic (3)
    "landuse", "infrastructure", "population",
    # Risk & Resilience (4)
    "multi_hazard", "climate_change", "recovery", "habitability"
]

# Indian bounding box (lat, lng)
INDIA_LAT = (8.4, 37.6)
INDIA_LON = (68.1, 97.4)


def suitability_label_from_23_factors(features_dict_or_list):
    """
    Formula-derived suitability (0-100) matching aggregator: 6 categories with all 23 factors.
    Accepts dict keyed by factor name or list in FACTOR_ORDER order.
    """
    if isinstance(features_dict_or_list, dict):
        get = lambda k: float(features_dict_or_list.get(k, 50))
    else:
        arr = list(features_dict_or_list)
        get = lambda k: float(arr[FACTOR_ORDER.index(k)]) if k in FACTOR_ORDER else 50.0
    
    # Physical (4 factors)
    slope = get("slope")
    elevation = get("elevation")
    ruggedness = get("ruggedness")
    stability = get("stability")
    
    # Hydrology (4 factors)
    flood = get("flood")
    water = get("water")
    drainage = get("drainage")
    groundwater = get("groundwater")
    
    # Environmental (5 factors)
    vegetation = get("vegetation")
    pollution = get("pollution")
    soil = get("soil")
    biodiversity = get("biodiversity")
    heat_island = get("heat_island")
    
    # Climatic (3 factors)
    rainfall = get("rainfall")
    thermal = get("thermal")
    intensity = get("intensity")
    
    # Socio-Economic (3 factors)
    landuse = get("landuse")
    infrastructure = get("infrastructure")
    population = get("population")
    
    # Risk & Resilience (4 factors)
    multi_hazard = get("multi_hazard")
    climate_change = get("climate_change")
    recovery = get("recovery")
    habitability = get("habitability")

    # 6 categories (same weights as aggregator)
    cat_physical = (slope + elevation + ruggedness + stability) / 4
    cat_environmental = (vegetation + pollution + soil + biodiversity + heat_island) / 5
    cat_hydrology = (flood + water + drainage + groundwater) / 4
    cat_climatic = (rainfall + thermal + intensity) / 3
    cat_socio_econ = (landuse + infrastructure + population) / 3
    cat_risk_resilience = (multi_hazard + climate_change + recovery + habitability) / 4
    
    weights = {"physical": 0.18, "environmental": 0.20, "hydrology": 0.16, "climatic": 0.16, "socio_econ": 0.15, "risk_resilience": 0.15}
    base_score = (
        cat_physical * weights["physical"] + cat_environmental * weights["environmental"] +
        cat_hydrology * weights["hydrology"] + cat_climatic * weights["climatic"] +
        cat_socio_econ * weights["socio_econ"] + cat_risk_resilience * weights["risk_resilience"]
    )
    final_score = max(0.0, min(100.0, base_score))
    
    # Apply penalties (same as aggregator)
    if water <= 5:
        final_score = min(final_score, 12.0)
    if flood < 40:
        final_score *= 0.5
    if landuse <= 20:
        final_score = min(final_score, 20.0)
    return max(0.0, min(100.0, final_score))


def generate_indian_dataset_23(n_samples=10000, seed=42):
    """Generate synthetic 23-factor features for Indian-style locations; labels match aggregator formula."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_samples):
        on_water = rng.random() < 0.12
        row = {
            # Physical (4 factors)
            "slope": np_rng.uniform(30, 100),
            "elevation": np_rng.uniform(40, 95),
            "ruggedness": np_rng.uniform(20, 90),
            "stability": np_rng.uniform(35, 95),
            # Hydrology (4 factors)
            "flood": np_rng.uniform(25, 100),
            "water": 0.0 if on_water else np_rng.uniform(40, 100),
            "drainage": 0.0 if on_water else np_rng.uniform(50, 100),
            "groundwater": np_rng.uniform(30, 90),
            # Environmental (5 factors)
            "vegetation": np_rng.uniform(20, 95),
            "pollution": np_rng.uniform(30, 90),
            "soil": 0.0 if on_water else np_rng.uniform(35, 100),
            "biodiversity": np_rng.uniform(25, 85),
            "heat_island": np_rng.uniform(40, 90),
            # Climatic (3 factors)
            "rainfall": np_rng.uniform(25, 95),
            "thermal": np_rng.uniform(40, 90),
            "intensity": np_rng.uniform(30, 85),
            # Socio-Economic (3 factors)
            "landuse": 0.0 if on_water else np_rng.uniform(25, 100),
            "infrastructure": np_rng.uniform(35, 95),
            "population": np_rng.uniform(20, 90),
            # Risk & Resilience (4 factors)
            "multi_hazard": np_rng.uniform(30, 85),
            "climate_change": np_rng.uniform(35, 90),
            "recovery": np_rng.uniform(40, 95),
            "habitability": np_rng.uniform(30, 85),
        }
        label = suitability_label_from_23_factors(row)
        vec = [row[k] for k in FACTOR_ORDER]
        X.append(vec)
        y.append(label)
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)


# Model configs: (display_name, pkl_filename, builder_fn)
def _make_rf(seed):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=200, max_depth=12, random_state=seed, n_jobs=-1)


def _make_xgb(seed):
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=seed, n_jobs=-1)


def _make_gbm(seed):
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=seed)


def _make_et(seed):
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(n_estimators=200, max_depth=12, random_state=seed, n_jobs=-1)


def _make_lgbm(seed):
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=seed, n_jobs=-1, verbose=-1)
    except ImportError:
        return None


MODEL_CONFIGS = [
    ("Random Forest", "model_rf.pkl", _make_rf),
    ("XGBoost", "model_xgboost.pkl", _make_xgb),
    ("Gradient Boosting", "model_gbm.pkl", _make_gbm),
    ("Extra Trees", "model_et.pkl", _make_et),
    ("LightGBM", "model_lgbm.pkl", _make_lgbm),
]


def train_and_report(samples=10000, test_size=0.2, seed=42):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    print("Generating Indian-location synthetic dataset (23 factors, complete factor system)...")
    X, y = generate_indian_dataset_23(n_samples=samples, seed=seed)
    
    # Convert to numpy arrays to avoid feature name warnings
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    print(f"  Factors: {FACTOR_ORDER}")
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}\n")

    trained = []
    for display_name, fname, builder_fn in MODEL_CONFIGS:
        model = builder_fn(seed)
        if model is None:
            print(f"[Skip] {display_name}: not available (e.g. pip install lightgbm)")
            continue
        print(f"Training {display_name}...")
        try:
            model.fit(X_train, y_train)
            for name, Xt, yt in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
                pred = model.predict(Xt)
                mae = mean_absolute_error(yt, pred)
                rmse = np.sqrt(mean_squared_error(yt, pred))
                r2 = r2_score(yt, pred)
                print(f"  {display_name} {name}: MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}")
            path = os.path.join(MODELS_DIR, fname)
            with open(path, "wb") as f:
                pickle.dump(model, f)
            print(f"  Saved: {path}\n")
            trained.append((display_name, fname))
        except Exception as e:
            print(f"  {display_name} failed: {e}\n")

    print("[OK] Training complete. Models are used in History Analysis and main Suitability when available.")
    return trained


def report_accuracy_only(test_samples=2000, seed=123):
    """Load all saved models and print accuracy on a fresh 23-factor test set."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    print("Generating fresh 23-factor test set...")
    X_test, y_test = generate_indian_dataset_23(n_samples=test_samples, seed=seed)
    
    # Convert to numpy arrays to avoid feature name warnings
    X_test = np.asarray(X_test, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    
    print(f"  Test samples: {len(X_test)}\n")

    results = []
    for display_name, fname in [(c[0], c[1]) for c in MODEL_CONFIGS]:
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)
            results.append((display_name, mae, rmse, r2))
            print(f"  {display_name}: MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}")
        except Exception as e:
            print(f"  {display_name}: error - {e}")

    if results:
        print("\n--- Accuracy summary ---")
        for name, mae, rmse, r2 in results:
            print(f"  {name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train ensemble on 23-factor Indian suitability data; report accuracy.")
    parser.add_argument("--samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--report-accuracy", action="store_true", help="Only load saved models and print accuracy")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test set (0.2 = 20%%)")
    args = parser.parse_args()

    if args.report_accuracy:
        report_accuracy_only(test_samples=2000)
        return

    train_and_report(samples=args.samples, test_size=args.test_size)


if __name__ == "__main__":
    main()
