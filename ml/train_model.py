import sys, os
import random
import pickle
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
# Fix for ModuleNotFoundError:
# Adds the 'backend' directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from integrations import compute_suitability_score

# Geographic Bounding Box for India
LAT_RANGE, LON_RANGE = (8.4, 37.6), (68.1, 97.4)

print("Training Ensemble (XGBoost + RF) on Indian terrain samples...")

X, y = [], []
for i in range(10000):
    # Simulate a mix of 15% water and 85% land samples
    on_water = random.random() < 0.15 
    
    features = [
        random.uniform(30, 95), # rainfall
        random.uniform(20, 100), # flood
        random.uniform(50, 90), # landslide
        0.0 if on_water else random.uniform(30, 100), # soil
        random.uniform(40, 95), # proximity
        0.0 if on_water else random.uniform(40, 100), # water
        random.uniform(30, 90), # pollution
        0.0 if on_water else random.uniform(20, 100)  # landuse
    ]
    
    # Label using the Hard-Coded Logic
    agg = compute_suitability_score(
        rainfall_score=features[0], flood_risk_score=features[1],
        landslide_risk_score=features[2], soil_quality_score=features[3],
        proximity_score=features[4], water_proximity_score=features[5],
        pollution_score=features[6], landuse_score=features[7]
    )
    X.append(features)
    y.append(agg["score"])

X, y = np.array(X), np.array(y)

# Train Ensemble
model_xgb = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1)
model_rf = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1)

model_xgb.fit(X, y)
model_rf.fit(X, y)

# # Save both models
# os.makedirs("backend/ml", exist_ok=True)
# pickle.dump(model_xgb, open("backend/ml/model_xgboost.pkl", "wb"))
# pickle.dump(model_rf, open("backend/ml/model_rf.pkl", "wb"))

# print(f"Models saved successfully. XGB R2: {model_xgb.score(X, y):.4f}")
# Create the specific models path
# This goes from GeoAI/backend/ml/ -> GeoAI/backend/ml/models/
target_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(target_dir, exist_ok=True) 

model_path = os.path.join(target_dir, "model_xgboost.pkl")
rf_path = os.path.join(target_dir, "model_rf.pkl")

# Save the files directly into the models folder
pickle.dump(model_xgb, open(model_path, "wb"))
pickle.dump(model_rf, open(rf_path, "wb"))

print(f"✅ Models saved automatically to: {target_dir}")