# import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import random
# import pickle
# import numpy as np
# import xgboost as xgb
# from backend.integrations import *


# _cache = {}

# def safe_call(func, lat, lon, *args, fallback=None):
#     key = (func.__name__, round(lat, 3), round(lon, 3))
#     if key not in _cache:
#         try:
#             result = func(lat, lon, *args)
#             _cache[key] = result
#             print(f"Cached: {func.__name__} → {round(lat,3)},{round(lon,3)}")   
#         except Exception:
#             print(f"API failed ({func.__name__}), using fallback")             
#             result = fallback
#             _cache[key] = result
#     return _cache[key]

# print("Training XGBoost model with *minimal* API calls...")

# # STEP 1: FEW REAL COORDINATES
# BASE_COORDS = 2        # only 20 real locations
# AUG_PER_COORD = 5      # 10 synthetic variations per location
# # Total training samples = BASE_COORDS * AUG_PER_COORD = 200

# base_samples = []       

# for i in range(BASE_COORDS):
#     if i % 5 == 0:
#         print(f"   → Fetching real data for coord {i}/{BASE_COORDS}")

#     lat = random.uniform(8.0, 37.0)
#     lon = random.uniform(68.0, 97.0)


#     rainfall_score, _ = safe_call(estimate_rainfall_score, lat, lon, fallback=(70.0, 150))
#     landslide_score = safe_call(estimate_landslide_risk_score, lat, lon, fallback=70.0)
#     proximity_score = safe_call(compute_proximity_score, lat, lon, fallback=60.0)
#     water_score, _ = safe_call(estimate_water_proximity_score, lat, lon, fallback=(75.0, 2.0))
#     pollution_score = safe_call(estimate_pollution_score, lat, lon, fallback=65.0)
#     landuse_score = safe_call(infer_landuse_score, lat, lon, fallback=70.0)

#     flood_score = estimate_flood_risk_score(lat, lon) or 50.0
#     soil_score = estimate_soil_quality_score(lat, lon) or 60.0

#     base_features = [
#         rainfall_score or 70,
#         flood_score,
#         landslide_score or 70,
#         soil_score,
#         proximity_score or 60,
#         water_score or 75,
#         pollution_score or 65,
#         landuse_score or 70
#     ]

#     base_samples.append(base_features)

# print(f"Collected {len(base_samples)} base locations (real API calls done).")
# print("Now augmenting without more API calls...")

# # -----------------------------------
# # STEP 2: AUGMENT (NO API CALLS HERE)
# # -----------------------------------
# X, y = [], []

# def jitter(v, sigma=5.0):
#     """Add small noise, keep in [0,100]."""
#     if v is None:
#         return None
#     noisy = v + random.gauss(0, sigma)
#     return max(0.0, min(100.0, noisy))

# for idx, base in enumerate(base_samples):
#     if idx % 5 == 0:
#         print(f"   → Augmenting {idx+1}/{len(base_samples)}")

#     for _ in range(AUG_PER_COORD):
#         rain_n   = jitter(base[0])
#         flood_n  = jitter(base[1])
#         land_n   = jitter(base[2])
#         soil_n   = jitter(base[3])
#         prox_n   = jitter(base[4])
#         water_n  = jitter(base[5])
#         poll_n   = jitter(base[6])
#         lu_n     = jitter(base[7])

#         features = [
#             rain_n,
#             flood_n,
#             land_n,
#             soil_n,
#             prox_n,
#             water_n,
#             poll_n,
#             lu_n
#         ]
#         X.append(features)

#         agg = compute_suitability_score(
#             rainfall_score=rain_n,
#             flood_risk_score=flood_n,
#             landslide_risk_score=land_n,
#             soil_quality_score=soil_n,
#             proximity_score=prox_n,
#             water_proximity_score=water_n,
#             pollution_score=poll_n,
#             landuse_score=lu_n,
#         )
#         y.append(agg["score"])

# X, y = np.array(X, dtype=float), np.array(y, dtype=float)

# print(f"Total training samples: {len(X)}")  
# # STEP 3: TRAIN XGBOOST
# model = xgb.XGBRegressor(
#     n_estimators=200,
#     max_depth=5,
#     random_state=42,
#     n_jobs=-1
# )
# model.fit(X, y)

# # Save model
# # os.makedirs("backend/ml", exist_ok=True)
# # model_path = "backend/ml/model_xgboost.pkl"
# # pickle.dump(model, open(model_path, "wb"))
# # model_path = "model_xgboost.pkl"
# # pickle.dump(model, open(model_path, "wb"))
# # Save model in the same folder as this script
# model_path = os.path.join(os.path.dirname(__file__), "model_xgboost.pkl")
# pickle.dump(model, open(model_path, "wb"))



# print("\nMODEL TRAINED SUCCESSFULLY!")
# print(f"R² Score (train): {model.score(X, y):.4f}")
# print(f"Model saved: {model_path}")
# print(f"File size: {os.path.getsize(model_path)/1024:.1f} KB")





# # backend/ml/train_model.py   ← 15-SECOND VERSION (100% WORKS)
# # import sys, os
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # import random
# # import pickle
# # import numpy as np
# # import xgboost as xgb
# # from integrations import compute_suitability_score

# # print("Creating INSTANT XGBoost model (no API calls)...")

# # X, y = [], []
# # for i in range(5000):  
# #     if i % 1000 == 0: print(f"   → {i}/5000")
# #     rainfall = random.uniform(20, 90)
# #     flood = random.uniform(40, 100)
# #     landslide = random.uniform(50, 90)
# #     soil = random.uniform(50, 90)
# #     proximity = random.uniform(40, 95)
# #     water = random.uniform(10, 95)
# #     pollution = random.uniform(30, 90)
# #     landuse = random.uniform(30, 90)

# #     features = [rainfall, flood, landslide, soil, proximity, water, pollution, landuse]
# #     X.append(features)

# #     agg = compute_suitability_score(
# #         rainfall_score=rainfall,
# #         flood_risk_score=flood,
# #         landslide_risk_score=landslide,
# #         soil_quality_score=soil,
# #         proximity_score=proximity,
# #         water_proximity_score=water,
# #         pollution_score=pollution,
# #         landuse_score=landuse,
# #     )
# #     y.append(agg["score"])

# # X, y = np.array(X), np.array(y)

# # model = xgb.XGBRegressor(n_estimators=200, max_depth=5, random_state=42)
# # model.fit(X, y)

# # os.makedirs("backend/ml", exist_ok=True)
# # pickle.dump(model, open("backend/ml/model_xgboost.pkl", "wb"))

# # print("\nINSTANT MODEL CREATED IN 15 SECONDS!")
# # print(f"R²: {model.score(X, y):.4f}") 
# # print("File: backend/ml/model_xgboost.pkl")



import sys, os
import random
import pickle
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from backend.integrations import compute_suitability_score

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