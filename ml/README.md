GeoAI ML training helpers

Files:
- `train.py`: lightweight training pipeline that produces two models:
  - `water_detector.pkl` (RandomForestClassifier)
  - `suitability_regressor.pkl` (RandomForestRegressor)

Usage:
```bash
python backend/ml/train.py
```

Notes:
- Currently uses a synthetic dataset when no labeled data is provided. Replace `load_or_create_sample_dataset()` with real labeled datasets for production.
- Models are saved into `backend/ml/models/`.
