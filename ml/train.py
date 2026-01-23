"""Consolidated training CLI for GeoAI.

Supports modes:
- synthetic: train RandomForest water-detector + regressor on synthetic data
- xgboost: collect a few real points via adapters, augment, and train an XGBoost regressor
- all: run both flows

Run:
  python backend/ml/train.py --mode synthetic
  python backend/ml/train.py --mode xgboost
  python backend/ml/train.py --mode all

Use `--quick` for smaller/fast synthetic runs during development.
"""

import argparse
import os
import pickle
import numpy as np
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Ensure project root on sys.path so `backend.integrations` can be imported when running this script
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_or_create_sample_dataset(n=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    rainfall = rng.uniform(0, 100, size=n)
    flood = rng.uniform(0, 100, size=n)
    landslide = rng.uniform(0, 100, size=n)
    soil = rng.uniform(0, 100, size=n)
    proximity = rng.uniform(0, 100, size=n)
    water_dist_km = rng.exponential(scale=1.0, size=n)
    pollution = rng.uniform(0, 100, size=n)
    landuse = rng.uniform(0, 100, size=n)

    X = np.vstack([rainfall, flood, landslide, soil, proximity, water_dist_km, pollution, landuse]).T

    suitability = (
        0.1 * rainfall + 0.15 * flood + 0.1 * landslide + 0.15 * soil + 0.1 * proximity +
        0.2 * (100 - np.clip(water_dist_km * 30, 0, 100)) + 0.1 * pollution + 0.1 * landuse
    )
    suitability = np.clip(suitability, 0, 100)
    is_on_water = (water_dist_km < 0.02).astype(int)
    return X, suitability, is_on_water


def train_water_detector(X, y_cls, n_estimators=200):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    print("Training water detector (RandomForest)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_cls, test_size=0.2, random_state=1)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))
    path = os.path.join(MODELS_DIR, "water_detector.pkl")
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    print("Saved water detector to", path)
    return clf


def train_suitability_regressor(X, y_reg, n_estimators=200):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    print("Training suitability regressor (RandomForest)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=2)
    reg = RandomForestRegressor(n_estimators=n_estimators, random_state=2)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    print("MSE:", mse)
    path = os.path.join(MODELS_DIR, "suitability_regressor.pkl")
    with open(path, "wb") as f:
        pickle.dump(reg, f)
    print("Saved regressor to", path)
    return reg


def train_xgboost_regressor_from_adapters(base_coords=5, aug_per_coord=10, model_kwargs=None):
    """Collect a small set of coordinates using live adapters, augment, and train an XGBoost regressor.
    This function mirrors the logic previously in `train_model.py`.
    """
    try:
        import xgboost as xgb
    except Exception as e:
        raise RuntimeError("xgboost is required for this mode: pip install xgboost") from e

    from backend.integrations import (
        estimate_rainfall_score, estimate_landslide_risk_score, compute_proximity_score,
        estimate_water_proximity_score, estimate_pollution_score, infer_landuse_score,
        estimate_flood_risk_score, estimate_soil_quality_score, compute_suitability_score
    )

    import random
    random.seed(42)

    base_samples = []
    print("Collecting base coordinates via adapters...")
    for i in range(base_coords):
        lat = random.uniform(8.0, 37.0)
        lon = random.uniform(68.0, 97.0)

        rainfall_score, _ = estimate_rainfall_score(lat, lon)
        landslide_score = estimate_landslide_risk_score(lat, lon)
        proximity_score = compute_proximity_score(lat, lon)
        water_score, _ = estimate_water_proximity_score(lat, lon)
        pollution_score = estimate_pollution_score(lat, lon)
        landuse_score = infer_landuse_score(lat, lon)
        flood_score = estimate_flood_risk_score(lat, lon) or 50.0
        soil_score = estimate_soil_quality_score(lat, lon) or 60.0

        base_features = [
            rainfall_score or 70,
            flood_score,
            landslide_score or 70,
            soil_score,
            proximity_score or 60,
            water_score or 75,
            pollution_score or 65,
            landuse_score or 70
        ]
        base_samples.append(base_features)

    print(f"Collected {len(base_samples)} base locations (live adapters)")
    print("Augmenting samples...")

    X, y = [], []
    def jitter(v, sigma=5.0):
        if v is None:
            return None
        noisy = v + random.gauss(0, sigma)
        return max(0.0, min(100.0, noisy))

    for base in base_samples:
        for _ in range(aug_per_coord):
            vals = [jitter(v) for v in base]
            X.append(vals)
            agg = compute_suitability_score(
                rainfall_score=vals[0], flood_risk_score=vals[1], landslide_risk_score=vals[2],
                soil_quality_score=vals[3], proximity_score=vals[4], water_proximity_score=vals[5],
                pollution_score=vals[6], landuse_score=vals[7]
            )
            y.append(agg["score"])

    X, y = np.array(X, dtype=float), np.array(y, dtype=float)
    model_kwargs = model_kwargs or {}
    model = xgb.XGBRegressor(n_estimators=model_kwargs.get('n_estimators', 200), max_depth=model_kwargs.get('max_depth', 5), random_state=42, n_jobs=-1)
    print("Training XGBoost regressor...")
    model.fit(X, y)
    path = os.path.join(MODELS_DIR, "model_xgboost.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print("Saved XGBoost model to", path)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['synthetic','xgboost','all'], default='synthetic')
    parser.add_argument('--quick', action='store_true', help='Use smaller dataset / fewer estimators for fast dev runs')
    parser.add_argument('--evaluate', metavar='DATASET', help='Path to CSV/GeoJSON dataset for evaluation (columns: latitude,longitude,suitability[,is_on_water])')
    parser.add_argument('--max-sample', type=int, default=0, help='Limit number of samples to evaluate (0 = all)')
    args = parser.parse_args()

    # If evaluation is explicitly requested, run it and skip training to avoid accidental long trains
    if args.evaluate:
        quick = args.quick
        dataset_path = args.evaluate
        max_sample = int(args.max_sample or 0)
        try:
            from backend.integrations import (
                estimate_rainfall_score, estimate_landslide_risk_score, compute_proximity_score,
                estimate_water_proximity_score, estimate_pollution_score, infer_landuse_score,
                estimate_flood_risk_score, estimate_soil_quality_score
            )
        except Exception as e:
            print('Failed importing adapters for evaluation:', e)
            return

        import csv, json
        samples = []
        if dataset_path.lower().endswith('.csv'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                rd = csv.DictReader(f)
                for i, row in enumerate(rd):
                    if max_sample and i >= max_sample:
                        break
                    try:
                        lat = float(row.get('latitude') or row.get('lat'))
                        lon = float(row.get('longitude') or row.get('lon') or row.get('lng'))
                        y = float(row.get('suitability') or row.get('label') or row.get('score'))
                        is_on_water = row.get('is_on_water')
                        is_on_water = None if is_on_water is None else (str(is_on_water).strip() in ('1','true','True','yes','Y'))
                        samples.append({'lat': lat, 'lon': lon, 'y': y, 'is_on_water': is_on_water})
                    except Exception:
                        continue
        else:
            # try GeoJSON
            with open(dataset_path, 'r', encoding='utf-8') as f:
                gj = json.load(f)
                features = gj.get('features') or []
                for i, feat in enumerate(features):
                    if max_sample and i >= max_sample:
                        break
                    props = feat.get('properties') or {}
                    geom = feat.get('geometry') or {}
                    coords = geom.get('coordinates') or []
                    if len(coords) >= 2:
                        lon, lat = coords[0], coords[1]
                        y = props.get('suitability') or props.get('score')
                        is_on_water = props.get('is_on_water')
                        samples.append({'lat': lat, 'lon': lon, 'y': float(y) if y is not None else None, 'is_on_water': is_on_water})

        if not samples:
            print('No samples found in dataset for evaluation')
            return

        print(f'Preparing features for {len(samples)} samples (this may take time)')

        X = []
        ys = []
        water_labels = []
        metas = []
        for s in samples:
            lat, lon = s['lat'], s['lon']
            try:
                r_score, _ = estimate_rainfall_score(lat, lon)
            except Exception:
                r_score = None
            try:
                f_score = estimate_flood_risk_score(lat, lon)
            except Exception:
                f_score = None
            try:
                ls_score = estimate_landslide_risk_score(lat, lon)
            except Exception:
                ls_score = None
            try:
                s_score = estimate_soil_quality_score(lat, lon)
            except Exception:
                s_score = None
            try:
                prox = compute_proximity_score(lat, lon)
            except Exception:
                prox = None
            try:
                w_score, w_dist, w_det = estimate_water_proximity_score(lat, lon)
            except Exception:
                w_score, w_dist, w_det = None, None, None
            try:
                pol = estimate_pollution_score(lat, lon)
            except Exception:
                pol = None
            try:
                lu = infer_landuse_score(lat, lon)
            except Exception:
                lu = None

            vec = [
                r_score or 70,
                f_score or 50,
                ls_score or 70,
                s_score or 60,
                prox or 60,
                w_score or 75,
                pol or 65,
                lu or 70,
            ]
            X.append(vec)
            ys.append(s.get('y'))
            water_labels.append(s.get('is_on_water'))
            metas.append({'lat': lat, 'lon': lon, 'water_distance_km': w_dist, 'water_details': w_det})

        X = np.array(X, dtype=float)

        # Load models if present
        models = {}
        for name in ('suitability_regressor.pkl', 'model_xgboost.pkl', 'water_detector.pkl'):
            p = os.path.join(MODELS_DIR, name)
            if os.path.exists(p):
                try:
                    with open(p, 'rb') as f:
                        models[name] = pickle.load(f)
                    print('Loaded', name)
                except Exception as e:
                    print('Failed loading', name, e)

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report

        report = {'n_samples': len(X), 'models': {}}
        # Evaluate regressors
        for mname in ('suitability_regressor.pkl', 'model_xgboost.pkl'):
            mdl = models.get(mname)
            if mdl is None:
                continue
            try:
                preds = mdl.predict(X)
            except Exception:
                # try per-estimator predict for ensemble
                try:
                    preds = np.mean([est.predict(X) for est in getattr(mdl, 'estimators_', [mdl])], axis=0)
                except Exception as e:
                    print('Prediction failed for', mname, e)
                    continue
            ys_num = np.array([y if y is not None else np.nan for y in ys], dtype=float)
            mask = ~np.isnan(ys_num)
            if mask.sum() == 0:
                print('No ground-truth suitability labels present; skipping regressor evaluation')
                break
            mse = float(mean_squared_error(ys_num[mask], preds[mask]))
            mae = float(mean_absolute_error(ys_num[mask], preds[mask]))
            r2 = float(r2_score(ys_num[mask], preds[mask]))
            report['models'][mname] = {'mse': mse, 'mae': mae, 'r2': r2}
            print(f"Model {mname}: MSE={mse:.3f} MAE={mae:.3f} R2={r2:.3f}")

        # Evaluate water detector if labels present
        if any(v is not None for v in water_labels) and 'water_detector.pkl' in models:
            mdl = models['water_detector.pkl']
            # produce binary preds from model
            try:
                proba = mdl.predict_proba(X)[:, 1]
                preds_cls = (proba >= 0.5).astype(int)
            except Exception:
                try:
                    preds_cls = mdl.predict(X)
                except Exception as e:
                    print('Water detector prediction failed:', e)
                    preds_cls = None
            if preds_cls is not None:
                y_true = np.array([1 if v else 0 if v is not None else -1 for v in water_labels])
                mask = y_true >= 0
                if mask.sum() > 0:
                    cm = confusion_matrix(y_true[mask], preds_cls[mask])
                    crep = classification_report(y_true[mask], preds_cls[mask], output_dict=True)
                    report['models']['water_detector.pkl'] = {'confusion_matrix': cm.tolist(), 'classification_report': crep}
                    print('Water detector classification report:')
                    print(classification_report(y_true[mask], preds_cls[mask]))

        # Save report
        outp = os.path.join(MODELS_DIR, 'evaluation_report.json')
        try:
            with open(outp, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            print('Saved evaluation report to', outp)
        except Exception:
            pass
        return

    # Otherwise run training flows
    quick = args.quick
    if args.mode in ('synthetic','all'):
        n = 500 if quick else 2000
        X, y_reg, y_cls = load_or_create_sample_dataset(n=n)
        est = 100 if quick else 200
        train_water_detector(X, y_cls, n_estimators=est)
        train_suitability_regressor(X, y_reg, n_estimators=est)

    if args.mode in ('xgboost','all'):
        try:
            train_xgboost_regressor_from_adapters(base_coords=3 if quick else 10, aug_per_coord=3 if quick else 10)
        except Exception as e:
            print('XGBoost training failed:', e)
        dataset_path = args.evaluate
        max_sample = int(args.max_sample or 0)
        try:
            from backend.integrations import (
                estimate_rainfall_score, estimate_landslide_risk_score, compute_proximity_score,
                estimate_water_proximity_score, estimate_pollution_score, infer_landuse_score,
                estimate_flood_risk_score, estimate_soil_quality_score
            )
        except Exception as e:
            print('Failed importing adapters for evaluation:', e)
            return

        import csv, json
        samples = []
        if dataset_path.lower().endswith('.csv'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                rd = csv.DictReader(f)
                for i, row in enumerate(rd):
                    if max_sample and i >= max_sample:
                        break
                    try:
                        lat = float(row.get('latitude') or row.get('lat'))
                        lon = float(row.get('longitude') or row.get('lon') or row.get('lng'))
                        y = float(row.get('suitability') or row.get('label') or row.get('score'))
                        is_on_water = row.get('is_on_water')
                        is_on_water = None if is_on_water is None else (str(is_on_water).strip() in ('1','true','True','yes','Y'))
                        samples.append({'lat': lat, 'lon': lon, 'y': y, 'is_on_water': is_on_water})
                    except Exception:
                        continue
        else:
            # try GeoJSON
            with open(dataset_path, 'r', encoding='utf-8') as f:
                gj = json.load(f)
                features = gj.get('features') or []
                for i, feat in enumerate(features):
                    if max_sample and i >= max_sample:
                        break
                    props = feat.get('properties') or {}
                    geom = feat.get('geometry') or {}
                    coords = geom.get('coordinates') or []
                    if len(coords) >= 2:
                        lon, lat = coords[0], coords[1]
                        y = props.get('suitability') or props.get('score')
                        is_on_water = props.get('is_on_water')
                        samples.append({'lat': lat, 'lon': lon, 'y': float(y) if y is not None else None, 'is_on_water': is_on_water})

        if not samples:
            print('No samples found in dataset for evaluation')
            return

        print(f'Preparing features for {len(samples)} samples (this may take time)')

        X = []
        ys = []
        water_labels = []
        metas = []
        for s in samples:
            lat, lon = s['lat'], s['lon']
            try:
                r_score, _ = estimate_rainfall_score(lat, lon)
            except Exception:
                r_score = None
            try:
                f_score = estimate_flood_risk_score(lat, lon)
            except Exception:
                f_score = None
            try:
                ls_score = estimate_landslide_risk_score(lat, lon)
            except Exception:
                ls_score = None
            try:
                s_score = estimate_soil_quality_score(lat, lon)
            except Exception:
                s_score = None
            try:
                prox = compute_proximity_score(lat, lon)
            except Exception:
                prox = None
            try:
                w_score, w_dist, w_det = estimate_water_proximity_score(lat, lon)
            except Exception:
                w_score, w_dist, w_det = None, None, None
            try:
                pol = estimate_pollution_score(lat, lon)
            except Exception:
                pol = None
            try:
                lu = infer_landuse_score(lat, lon)
            except Exception:
                lu = None

            vec = [
                r_score or 70,
                f_score or 50,
                ls_score or 70,
                s_score or 60,
                prox or 60,
                w_score or 75,
                pol or 65,
                lu or 70,
            ]
            X.append(vec)
            ys.append(s.get('y'))
            water_labels.append(s.get('is_on_water'))
            metas.append({'lat': lat, 'lon': lon, 'water_distance_km': w_dist, 'water_details': w_det})

        X = np.array(X, dtype=float)

        # Load models if present
        models = {}
        for name in ('suitability_regressor.pkl', 'model_xgboost.pkl', 'water_detector.pkl'):
            p = os.path.join(MODELS_DIR, name)
            if os.path.exists(p):
                try:
                    with open(p, 'rb') as f:
                        models[name] = pickle.load(f)
                    print('Loaded', name)
                except Exception as e:
                    print('Failed loading', name, e)

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report

        report = {'n_samples': len(X), 'models': {}}
        # Evaluate regressors
        for mname in ('suitability_regressor.pkl', 'model_xgboost.pkl'):
            mdl = models.get(mname)
            if mdl is None:
                continue
            try:
                preds = mdl.predict(X)
            except Exception:
                # try per-estimator predict for ensemble
                try:
                    preds = np.mean([est.predict(X) for est in getattr(mdl, 'estimators_', [mdl])], axis=0)
                except Exception as e:
                    print('Prediction failed for', mname, e)
                    continue
            ys_num = np.array([y if y is not None else np.nan for y in ys], dtype=float)
            mask = ~np.isnan(ys_num)
            if mask.sum() == 0:
                print('No ground-truth suitability labels present; skipping regressor evaluation')
                break
            mse = float(mean_squared_error(ys_num[mask], preds[mask]))
            mae = float(mean_absolute_error(ys_num[mask], preds[mask]))
            r2 = float(r2_score(ys_num[mask], preds[mask]))
            report['models'][mname] = {'mse': mse, 'mae': mae, 'r2': r2}
            print(f"Model {mname}: MSE={mse:.3f} MAE={mae:.3f} R2={r2:.3f}")

        # Evaluate water detector if labels present
        if any(v is not None for v in water_labels) and 'water_detector.pkl' in models:
            mdl = models['water_detector.pkl']
            # produce binary preds from model
            try:
                proba = mdl.predict_proba(X)[:, 1]
                preds_cls = (proba >= 0.5).astype(int)
            except Exception:
                try:
                    preds_cls = mdl.predict(X)
                except Exception as e:
                    print('Water detector prediction failed:', e)
                    preds_cls = None
            if preds_cls is not None:
                y_true = np.array([1 if v else 0 if v is not None else -1 for v in water_labels])
                mask = y_true >= 0
                if mask.sum() > 0:
                    cm = confusion_matrix(y_true[mask], preds_cls[mask])
                    crep = classification_report(y_true[mask], preds_cls[mask], output_dict=True)
                    report['models']['water_detector.pkl'] = {'confusion_matrix': cm.tolist(), 'classification_report': crep}
                    print('Water detector classification report:')
                    print(classification_report(y_true[mask], preds_cls[mask]))

        # Save report
        outp = os.path.join(MODELS_DIR, 'evaluation_report.json')
        try:
            with open(outp, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            print('Saved evaluation report to', outp)
        except Exception:
            pass


if __name__ == '__main__':
    main()
