GeoAI ML training (optional)

The app works without these models (rule-based Aggregator + GeoDataService). When present, trained models are used in **main Suitability** (ml_score, score_source_ml) and **History Analysis** (past score per timeline).

**train_model.py** – 14-factor ensemble (same factors as app aggregator):
- Factors: slope, elevation, flood, water, drainage, vegetation, pollution, soil, rainfall, thermal, intensity, landuse, infrastructure, population
- Models: Random Forest, XGBoost, Gradient Boosting, Extra Trees, optional LightGBM
- Labels: formula-derived 0–100 matching aggregator (5 categories + water/flood/landuse penalties)
- Saves: `model_rf.pkl`, `model_xgboost.pkl`, `model_gbm.pkl`, `model_et.pkl`, `model_lgbm.pkl` (any that train successfully)

Usage:
```bash
python backend/ml/train_model.py
python backend/ml/train_model.py --samples 5000
python backend/ml/train_model.py --report-accuracy
```

**train.py** – legacy pipeline (water_detector, suitability_regressor, etc.).

Notes:
- Models live in `backend/ml/models/`. App loads all present `model_*.pkl` and ensembles them.
- Training uses synthetic Indian-location data; use real labeled data for production.
