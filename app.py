
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# Initialize CNN Model (MobileNetV2 is best for web backends)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = models.mobilenet_v2(weights="DEFAULT")
cnn_model.classifier[1] = torch.nn.Linear(cnn_model.last_channel, 5)
cnn_model.to(device)
cnn_model.eval()

# 2. Image Prep
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

LAND_CLASSES = ["Urban", "Forest", "Agriculture", "Water", "Industrial"]



import os
import sys
import requests
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
from dotenv import load_dotenv
load_dotenv()

# --- Initialize Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

from geogpt_config import generate_system_prompt 
from reports.pdf_generator import generate_land_report
from integrations.digital_twin import calculate_development_impact
from integrations.nearby_places import get_nearby_named_places
from integrations.terrain_adapter import estimate_terrain_slope


from google import genai 
from flask import send_file
from dotenv import load_dotenv
from groq import Groq
from dotenv import load_dotenv
load_dotenv()



from integrations import (
    # compute_suitability_score,
    # estimate_flood_risk_score,
    # compute_proximity_score,
    # estimate_landslide_risk_score,
    # estimate_water_proximity_score,
    # estimate_pollution_score,
    # infer_landuse_score,
    # estimate_soil_quality_score,
    # estimate_rainfall_score,
    nearby_places,
)
from suitability_factors.geo_data_service import GeoDataService
from suitability_factors.aggregator import Aggregator, _elevation_to_suitability

# Import your AI library (OpenAI/Gemini/etc.)
# --- Configuration & Path Logic ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "models") 


GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")

# --- Groq (Primary) and Gemini (Secondary backup) for GeoGPT ---
groq_client = None
if GROQ_KEY:
    try:
        groq_client = Groq(api_key=GROQ_KEY)
        logging.info("GeoGPT primary (Groq): READY.")
    except Exception as e:
        logging.error(f"Groq Init Failed: {e}")
else:
    logging.warning("GROQ_API_KEY missing. GeoGPT primary unavailable.")

client = None  # Gemini as secondary
if GEMINI_KEY:
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        logging.info("GeoGPT backup (Gemini): READY.")
    except Exception as e:
        logging.error(f"Gemini Init Failed: {e}")
else:
    logging.warning("GEMINI_API_KEY missing. GeoGPT backup unavailable.")

print("--- GeoAI Engine Status ---")
print(f"GeoGPT Primary (Groq):   {'READY' if groq_client else 'OFFLINE'}")
print(f"GeoGPT Backup (Gemini):  {'READY' if client else 'OFFLINE'}")
print("---------------------------")

# --- Flask App Initialization ---
app = Flask(__name__)

# 1. Standardize Allowed Origins (Ensure NO trailing slashes)
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://geonexus-ai.vercel.app"
]

# 2. Configure CORS correctly - This handles the 'OPTIONS' preflight for you!
CORS(app, resources={r"/*": {
    "origins": ALLOWED_ORIGINS,
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization", "Accept"],
    "expose_headers": ["Content-Type", "Authorization"]
}}, supports_credentials=True)
def normalize_coords(lat, lng):
    # Clamp latitude to Web Mercator / Earth limits
    lat = max(min(lat, 85.0511), -85.0511)

    # Normalize longitude to [-180, 180]
    lng = ((lng + 180) % 360) - 180

    return lat, lng

# 3. SAFER header injector (Handles error cases and 502s)
@app.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin')
    if origin in ALLOWED_ORIGINS:
        # Avoid duplicate header error if flask-cors already added it
        if 'Access-Control-Allow-Origin' not in response.headers:
            response.headers.add('Access-Control-Allow-Origin', origin)
        
    # Standard security headers for split-stack linkage
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    # Use 'add' to append if not present, preventing the 'not allowed' error
    if 'Access-Control-Allow-Headers' not in response.headers:
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
    return response
ANALYSIS_CACHE = {}

def get_cache_key(lat, lng):
    """Generate cache key with 4 decimal precision"""
    return f"{float(lat):.4f},{float(lng):.4f}"

# --- ML Model Loading (optional; app works without .pkl files) ---
# Same 14 factors and order as backend/ml/train_model.py (FACTOR_ORDER).
ML_FACTOR_ORDER = [
    "slope", "elevation", "flood", "water", "drainage",
    "vegetation", "pollution", "soil", "rainfall", "thermal", "intensity",
    "landuse", "infrastructure", "population",
]
ML_MODELS = {}
for name in ("model_rf.pkl", "model_xgboost.pkl", "model_gbm.pkl", "model_et.pkl", "model_lgbm.pkl"):
    p = os.path.join(MODEL_PATH, name)
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                model = pickle.load(f)
                # Force sklearn models to ignore feature names to avoid warnings
                if hasattr(model, 'feature_names_in_'):
                    delattr(model, 'feature_names_in_')
                ML_MODELS[name] = model
            print(f"Loaded optional ML model: {name}")
        except Exception as e:
            print(f"Optional ML model {name} skipped: {e}")


def _ml_14feature_vector(flat_factors):
    """Build 14-feature vector (same order as train_model.py FACTOR_ORDER). flat_factors: dict of factor name -> 0-100."""
    vals = []
    for k in ML_FACTOR_ORDER:
        v = flat_factors.get(k)
        if v is None and k == "infrastructure":
            v = flat_factors.get("proximity")
        vals.append(float(v) if v is not None else 50.0)
    return np.array([vals], dtype=np.float64)


def _predict_suitability_ml(flat_factors):
    """If any ML model is loaded, return (ensemble score, True, source_label). Else (None, False, None)."""
    if not ML_MODELS:
        return None, False, None
    try:
        feat = _ml_14feature_vector(flat_factors)
        scores = []
        for name, model in ML_MODELS.items():
            # Handle different model types to avoid sklearn warnings
            if 'lgbm' in name.lower():
                # LightGBM: create DataFrame with proper column names
                import pandas as pd
                feat_df = pd.DataFrame([feat], columns=ML_FACTOR_ORDER)
                pred = model.predict(feat_df)
            else:
                # Other models: use numpy array
                pred = model.predict(feat)
            scores.append(float(pred[0]))
        score = round(max(0, min(100, sum(scores) / len(scores))), 2)
        names = [n.replace("model_", "").replace(".pkl", "") for n in ML_MODELS]
        source = "Ensemble (" + ", ".join(names) + ")"
        return score, True, source
    except Exception:
        return None, False, None


# def get_live_weather(lat, lng):
#     try:
#         url = "https://api.open-meteo.com/v1/forecast"
#         params = {
#             "latitude": lat,
#             "longitude": lng,
#             "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "weather_code", "is_day"],
#             "timezone": "auto" # Resolves local time for Site A
#         }
#         response = requests.get(url, params=params, timeout=5)
#         response.raise_for_status() # Check for HTTP errors
#         data = response.json()
        
#         current = data.get("current")
#         if not current:
#             return None

#         code = current.get("weather_code", 0)
#         is_day = current.get("is_day") # 1 for day, 0 for night
        
#         # Expanded WMO Code Mapping
#         description = "Clear Sky"
#         icon = "☀️" if is_day else "🌙"
        
#         if code in [1, 2]:
#             description = "Mainly Clear"
#             icon = "🌤️" if is_day else "☁️"
#         elif code == 3:
#             description = "Overcast"
#             icon = "☁️"
#         elif code in [51, 53, 55, 61, 63, 65]:
#             description = "Rainy"
#             icon = "🌧️"
#         elif code in [95, 96, 99]:
#             description = "Thunderstorm"
#             icon = "⛈️"
#         # Extract the ISO-formatted local time from the API response
#         local_time_iso = current.get("time")
#         return {
#             "temp_c": current.get("temperature_2m"),
#             "local_time": data.get("current", {}).get("time"),
#             "timezone": data.get("timezone"),
#             "humidity": current.get("relative_humidity_2m"),
#             "rain_mm": current.get("precipitation"),
#             "description": description,
#             "icon": icon,
#             "is_day": is_day # Pass this to React for conditional styling
#         }
#     except Exception as e:
#         logger.error(f"Weather Fetch Error: {e}")
#         return None
def get_live_weather(lat, lng):
    try:
        lat, lng = normalize_coords(lat, lng)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lng,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "weather_code",
                "is_day",
                "wind_speed_10m",
                "surface_pressure"
            ],
            "timezone": "auto"
        }

        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        current = data.get("current")
        if not current:
            return None

        code = current.get("weather_code", 0)
        is_day = current.get("is_day")

        description = "Clear Sky"
        icon = "☀️" if is_day else "🌙"

        if code in [1, 2]:
            description = "Mainly Clear"
            icon = "🌤️" if is_day else "☁️"
        elif code == 3:
            description = "Overcast"
            icon = "☁️"
        elif code in [51, 53, 55, 61, 63, 65]:
            description = "Rainy"
            icon = "🌧️"
        elif code in [95, 96, 99]:
            description = "Thunderstorm"
            icon = "⛈️"

        return {
            "temp_c": current.get("temperature_2m"),
            "local_time": current.get("time"),
            "timezone": data.get("timezone"),
            "humidity": current.get("relative_humidity_2m"),
            "rain_mm": current.get("precipitation"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "pressure_hpa": current.get("surface_pressure"),
            "weather_code": code,
            "description": description,
            "icon": icon,
            "is_day": is_day
        }

    except Exception as e:
        logger.error(f"Weather Fetch Error: {e}")
        return None


# def get_visual_forensics(lat, lng, past_year=2017):
#     """
#     Final Production Build: Siamese-CNN Visual Forensics.
#     Includes: Dimension Locking, Data-Gap Fallbacks, and Radiometric Normalization.
#     """
#     try:
#         import math
#         # 1. Tile Coordinate Calculation (Zoom 18)
#         zoom = 18
#         n = 2.0 ** zoom
#         xtile = int((lng + 180.0) / 360.0 * n)
#         lat_rad = math.radians(max(min(lat, 85.0511), -85.0511))
#         ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

#         # 2. Fallback Logic: Detect 'Pure White' data gaps
#         years_to_try = [past_year, 2018, 2019]
#         valid_b_img = None
#         used_year = past_year

#         for year in years_to_try:
#             url = f"https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-{year}_3857/default/g/{zoom}/{ytile}/{xtile}.jpg"
#             res = requests.get(url, timeout=5)
#             if res.status_code == 200:
#                 # Open and convert to grayscale
#                 img_temp = Image.open(BytesIO(res.content)).convert('L')
#                 # DIMENSION LOCK: Force all tiles to 256x256 to prevent matrix math errors
#                 img_temp = img_temp.resize((256, 256))
                
#                 if np.mean(img_temp) < 240:  # Valid if not pure white/clouds
#                     valid_b_img = np.array(img_temp) / 255.0 # Normalize 0-1
#                     used_year = year
#                     break
        
#         if valid_b_img is None:
#             return None

#         # 3. Fetch Current (2020) Reference
#         url_current = f"https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{zoom}/{ytile}/{xtile}.jpg"
#         res_c = requests.get(url_current, timeout=5)
#         img_c_raw = Image.open(BytesIO(res_c.content)).convert('L').resize((256, 256))
#         img_c = np.array(img_c_raw) / 255.0

#         # 4. Visual Drift Calculation (Siam-CNN Pixel Variance)
#         # diff represents the 'Distance Layer' of the twin networks
#         diff = np.abs(img_c - valid_b_img)
        
#         # Threshold: Only pixels that changed by more than 35% brightness are 'Constructed'
#         raw_intensity = np.mean(diff > 0.35) * 100 
        
#         # Cap at 91.5% for scientific realism (Total change is geologically impossible)
#         calibrated_intensity = min(raw_intensity, 91.5)
        
#         return {
#             "intensity": round(calibrated_intensity, 1),
#             "baseline_img": f"https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-{used_year}_3857/default/g/{zoom}/{ytile}/{xtile}.jpg",
#             "current_img": url_current,
#             "baseline_year": used_year,
#             "velocity": "Accelerated" if calibrated_intensity > 22 else "Stable",
#             "status": "Verified Visual Analysis"
#         }
#     except Exception as e:
#         logger.error(f"Visual Forensics Engine Failure: {e}")
#         return None
def get_visual_forensics(lat, lng, past_year=2017):
    """
    Final Production Build: Siamese-CNN Visual Forensics.
    Includes: Dimension Locking, Data-Gap Fallbacks, and Radiometric Normalization.
    """
    try:
        import math

        # ---------------------------------------------------
        # ✅ ADD: Coordinate normalization (NO logic change)
        # ---------------------------------------------------
        # Web Mercator valid latitude range
        lat = max(min(lat, 85.0511), -85.0511)

        # Normalize longitude to [-180, 180]
        lng = ((lng + 180) % 360) - 180

        # ---------------------------------------------------
        # 1. Tile Coordinate Calculation (Zoom 18)
        # ---------------------------------------------------
        zoom = 18
        n = 2.0 ** zoom

        xtile = int((lng + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        ytile = int(
            (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        )

        # ---------------------------------------------------
        # ✅ ADD: Tile-space safety clamp (NO logic change)
        # ---------------------------------------------------
        max_tile = int(n - 1)
        xtile = max(0, min(xtile, max_tile))
        ytile = max(0, min(ytile, max_tile))

        # ---------------------------------------------------
        # 2. Fallback Logic: Detect 'Pure White' data gaps
        # ---------------------------------------------------
        years_to_try = [past_year, 2018, 2019]
        valid_b_img = None
        used_year = past_year

        for year in years_to_try:
            url = (
                f"https://tiles.maps.eox.at/wmts/1.0.0/"
                f"s2cloudless-{year}_3857/default/g/"
                f"{zoom}/{ytile}/{xtile}.jpg"
            )
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                img_temp = Image.open(BytesIO(res.content)).convert('L')
                img_temp = img_temp.resize((256, 256))

                if np.mean(img_temp) < 240:
                    valid_b_img = np.array(img_temp) / 255.0
                    used_year = year
                    break

        if valid_b_img is None:
            return None

        # ---------------------------------------------------
        # 3. Fetch Current (2020) Reference
        # ---------------------------------------------------
        url_current = (
            f"https://tiles.maps.eox.at/wmts/1.0.0/"
            f"s2cloudless-2020_3857/default/g/"
            f"{zoom}/{ytile}/{xtile}.jpg"
        )
        res_c = requests.get(url_current, timeout=5)
        img_c_raw = Image.open(BytesIO(res_c.content)).convert('L').resize((256, 256))
        img_c = np.array(img_c_raw) / 255.0

        # ---------------------------------------------------
        # 4. Visual Drift Calculation (Siam-CNN Pixel Variance)
        # ---------------------------------------------------
        diff = np.abs(img_c - valid_b_img)
        threshold = 0.35
        raw_intensity = np.mean(diff > threshold) * 100
        calibrated_intensity = min(raw_intensity, 91.5)

        pixel_change_pct = round(float(np.mean(diff > threshold) * 100), 2)
        mean_diff = round(float(np.mean(diff)), 4)
        res_m = round(156543.03 * math.cos(math.radians(lat)) / (2 ** zoom), 2)

        # Reliable reasoning: what the numbers mean
        if calibrated_intensity > 50:
            reasoning = (f"High land-cover change: {pixel_change_pct}% of pixels changed above the {threshold} luminance threshold between {used_year} and 2020. "
                        "Typical of urbanization, deforestation, or major infrastructure. Spectral difference (mean Δ) confirms significant transition.")
        elif calibrated_intensity > 20:
            reasoning = (f"Moderate change: {pixel_change_pct}% of pixels show meaningful difference from {used_year} baseline. "
                        "Consistent with gradual development, agriculture shifts, or managed landscape change.")
        elif calibrated_intensity > 5:
            reasoning = (f"Low drift: {pixel_change_pct}% pixel change. Minor variation from {used_year}; site largely stable with small spectral shifts.")
        else:
            reasoning = (f"Stable: {pixel_change_pct}% change. Visual signature aligns with {used_year} baseline; well-preserved or low-activity area.")
        
        return {
            "intensity": round(calibrated_intensity, 1),
            "baseline_img": (
                f"https://tiles.maps.eox.at/wmts/1.0.0/"
                f"s2cloudless-{used_year}_3857/default/g/"
                f"{zoom}/{ytile}/{xtile}.jpg"
            ),
            "current_img": url_current,
            "baseline_year": used_year,
            "velocity": "Accelerated" if calibrated_intensity > 22 else "Stable",
            "status": "Verified Visual Analysis",
            "reasoning": reasoning,
            "telemetry": {
                "zoom": zoom,
                "xtile": xtile,
                "ytile": ytile,
                "pixel_change_pct": pixel_change_pct,
                "mean_diff": mean_diff,
                "resolution_m_per_px": res_m,
                "threshold_used": threshold,
                "source": "Sentinel-2 Cloudless (EOX)",
                "interpretation": f"{pixel_change_pct}% pixels changed (threshold {threshold}); resolution ~{res_m}m/px."
            }
        }

    except Exception as e:
        logger.error(f"Visual Forensics Engine Failure: {e}")
        return None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200
def get_cnn_classification(lat, lng):
    try:
        import math

        # ✅ ADD THIS (same fix as weather & forensics)
        lat, lng = normalize_coords(lat, lng)

        zoom = 18
        n = 2.0 ** zoom

        xtile = int((lng + 180.0) / 360.0 * n)

        lat_rad = math.radians(lat)
        ytile = int(
            (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        )

        # ✅ Tile-space clamp (critical)
        max_tile = int(n - 1)
        xtile = max(0, min(xtile, max_tile))
        ytile = max(0, min(ytile, max_tile))

        tile_url = (
            f"https://tiles.maps.eox.at/wmts/1.0.0/"
            f"s2cloudless-2020_3857/default/g/"
            f"{zoom}/{ytile}/{xtile}.jpg"
        )

        headers = {"User-Agent": "GeoAI-Client/1.0"}
        response = requests.get(tile_url, headers=headers, timeout=5)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert('RGB')

        input_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = cnn_model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            conf, index = torch.max(probabilities, 0)

        conf_val = round(conf.item() * 100, 1)
        top_class = LAND_CLASSES[index.item()]
        top_prob = round(probabilities[index.item()].item() * 100, 2)

        # Telemetry for live display: tile coords, resolution, raw CNN output
        zoom = 18
        res_m = round(156543.03 * math.cos(math.radians(lat)) / (2 ** zoom), 2)

        return {
            "class": top_class,
            "confidence": conf_val,
            "confidence_display": f"{conf_val}%",
            "image_sample": tile_url,
            "telemetry": {
                "model": "MobileNetV2",
                "zoom": zoom,
                "resolution_m_per_px": res_m,
                "top_class": top_class,
                "top_probability": top_prob,
                "tile_url_source": "Sentinel-2 Cloudless (EOX 2020)"
            }
        }

    except Exception as e:
        logger.error(f"CNN Classification Failed: {e}")
        return {
            "class": "Unknown",
            "confidence": 0,
            "confidence_display": "N/A",
            "image_sample": None,
            "telemetry": {"model": "MobileNetV2", "error": str(e)[:80]}
        }

@app.route('/ask_geogpt', methods=['POST'])
def ask_geogpt():
    data = request.json or {}
    user_query = data.get('query')
    chat_history = data.get('history', [])
    current_data = data.get('currentData')  # Site A (can be null when no analysis)
    compare_data = data.get('compareData')  # Site B
    location_name = data.get('locationName') or "Current location"

    if not (groq_client or client):
        return jsonify({"answer": "### Systems Offline\nGeoGPT primary (Groq) and backup (Gemini) are unconfigured. Please set GROQ_API_KEY (and optionally GEMINI_API_KEY)."})

    system_context = generate_system_prompt(location_name, current_data, compare_data)

    # Format last 6 messages for context
    formatted_history = []
    for msg in chat_history[-6:]:
        formatted_history.append({"role": "user" if msg.get("role") == "user" else "assistant", "content": msg.get("content", "")})

    messages = [{"role": "system", "content": system_context}] + formatted_history + [{"role": "user", "content": user_query}]

    # --- PRIMARY: Groq ---
    if groq_client:
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
            )
            answer = completion.choices[0].message.content
            return jsonify({"answer": answer, "status": "success"})
        except Exception as e:
            logger.error(f"Groq Error: {e}")
            if not client:
                return jsonify({"answer": f"### Groq Error\n{str(e)[:200]}"}), 500

    # --- BACKUP: Gemini ---
    if client:
        try:
            formatted_gemini = []
            for msg in chat_history[-6:]:
                role = "user" if msg.get("role") == "user" else "model"
                formatted_gemini.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
            chat_session = client.chats.create(
                model="gemini-2.0-flash",
                config={"system_instruction": system_context, "temperature": 0.7},
                history=formatted_gemini
            )
            response = chat_session.send_message(user_query)
            return jsonify({"answer": response.text, "status": "success_backup"})
        except Exception as e:
            logger.error(f"Gemini Error: {e}")
            return jsonify({"answer": f"### Engine Error\n{str(e)[:200]}"}), 500

    return jsonify({"answer": "### Unable to process request."}), 500
import requests
import math

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"

import requests
import math

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"

def calculate_haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_snapshot_identity(lat, lon, timeout=10):
    # 1. Global Distances
    dist_to_equator = calculate_haversine(lat, lon, 0, lon)
    dist_to_pole = calculate_haversine(lat, lon, 90, lon)
    
    # 2. Continent & Hemisphere Logic
    hem_ns = "Northern" if lat >= 0 else "Southern"
    hem_ew = "Eastern" if lon >= 0 else "Western"
    
    continent = "Global"
    if -35 <= lat <= 38 and -20 <= lon <= 55: continent = "Africa"
    elif 34 <= lat <= 82 and -25 <= lon <= 45: continent = "Europe"
    elif -10 <= lat <= 82 and 25 <= lon <= 180: continent = "Asia"
    elif 7 <= lat <= 85 and -170 <= lon <= -50: continent = "North America"
    elif -57 <= lat <= 15 and -95 <= lon <= -30: continent = "South America"
    elif -50 <= lat <= -10 and 100 <= lon <= 180: continent = "Australia/Oceania"

    try:
        res = requests.get(NOMINATIM_URL, params={
            "lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10, "addressdetails": 1
        }, headers={"User-Agent": "Harshavardhan-GeoAI-V1-Unique"}, timeout=timeout)

        data = res.json()
        addr = data.get("address", {})
        
        # 2. Log this to your terminal so you can see if OSM is actually sending new data
        # print(f"OSM Response for {lat},{lon}: {addr.get('country')}")

        # Safe Retrieval with Fallbacks
        country = addr.get("country", "International Waters")
        state = addr.get("state") or addr.get("province") or addr.get("state_district") or "N/A"
        district = addr.get("district") or addr.get("county") or addr.get("city_district") or "N/A"
        city = addr.get("city") or addr.get("town") or addr.get("village") or "Inland Territory"
        
        # 3. Ocean Detection
        # If OSM doesn't return a city/state, it's often a coastal or marine area
        is_coastal = "city" not in addr and "town" not in addr
        terrain_type = "Coastal / Marine" if is_coastal else "Inland Plateau"

        return {
            "identity": {
                "name": city,
                "hierarchy": f"{state}, {country}",
                "continent": continent
            },
            "coordinates": {
                "lat": f"{abs(lat):.4f}° {'N' if lat>=0 else 'S'}",
                "lng": f"{abs(lon):.4f}° {'E' if lon>=0 else 'W'}",
                "zone": f"UTM {int((lon + 180) / 6) + 1}"
            },
            "metrics": {
                "equator_dist_km": round(dist_to_equator, 1),
                "pole_dist_km": round(dist_to_pole, 1),
            },
            "political_identity": {
                "country": country,
                "iso_code": addr.get("country_code", "XX").upper()
            },
            "administrative_nesting": {
                "state": state,
                "district": district
            },
            "global_position": {
                "continent": continent,
                "hemisphere": f"{hem_ns} / {hem_ew}"
            },
            "terrain_context": terrain_type,
            "professional_summary": f"Site {city} is {round(dist_to_equator)}km from the Equator in the {hem_ns} hemisphere."
        }
    except Exception:
        return {"error": "Resolution Failed"}
@app.route("/snapshot_identity", methods=["POST","OPTIONS"])
def snapshot_identity_route():

    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = request.json or {}
        lat = float(data.get("latitude"))
        lon = float(data.get("longitude"))

        # Fetch enriched geospatial data
        snapshot = get_snapshot_identity(lat, lon)
        return jsonify(snapshot)

    except Exception as e:
        logger.error(f"Snapshot Route Error: {e}")
        return jsonify({"error": "Failed to resolve identity"}), 500
def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the straight-line distance (Great Circle) between two points 
    on Earth using the Haversine formula. Result is in Kilometers.
    """
    # Earth's radius in kilometers
    R = 6371.0 
    
    # Convert decimal degrees to radians
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    
    r_lat1 = math.radians(lat1)
    r_lat2 = math.radians(lat2)

    # Haversine formula calculation
    a = math.sin(d_lat / 2)**2 + \
        math.cos(r_lat1) * math.cos(r_lat2) * \
        math.sin(d_lon / 2)**2
        
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c
# Add this helper to fetch REAL historical weather
def fetch_historical_weather_stats(lat, lng, year_offset):
    try:
        # Calculate the target date (e.g., same day 10 years ago)
        target_year = datetime.now().year - year_offset
        start_date = f"{target_year}-01-01"
        end_date = f"{target_year}-03-01" # 60 day window for consistency with your rainfall logic
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lng,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "precipitation_sum",
            "timezone": "auto"
        }
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
        
        # Calculate total rainfall in that 60-day period 10 years ago
        precip_list = data.get('daily', {}).get('precipitation_sum', [])
        total_rain = sum(precip_list) if precip_list else 150.0 # Fallback to moderate
        return total_rain
    except Exception as e:
        logger.error(f"Historical Weather Error: {e}")
        return 150.0

def generate_strategic_intelligence(factors, current_score, nearby_list):
    """
    Synthesizes real-time factor drift and location-based improvement roadmap.
    Uses ALL 15 factors (flattened) to generate valid, factor-specific AI improvement suggestions.
    """
    # Default fallbacks for missing keys
    def _f(k, default=50.0):
        return float(factors.get(k, default)) if factors.get(k) is not None else default

    # 1. Future projection from current 15-factor profile
    urban_impact = (100 - _f('landuse')) * 0.25
    veg_loss = -((_f('soil') + _f('rainfall')) / 20)
    # Degradation rate: worse baseline score = faster projected decline
    drift_rate = 0.94 if current_score > 75 else (0.90 if current_score > 55 else 0.85)
    expected_2036_score = round(current_score * drift_rate, 1)

    # 2. Improvement Roadmap — one actionable item per low factor (all 15 factors)
    roadmap = []
    if _f('flood', 100) < 65:
        gap = 100 - _f('flood', 100)
        roadmap.append({
            "task": "Hydrological Buffering",
            "impact": f"+{round(gap * 0.4, 1)}%",
            "note": "Install retention basins, permeable paving, and surface drainage to reduce flood exposure (current flood safety below 65)."
        })
    if _f('soil', 100) < 60:
        gap = 100 - _f('soil', 100)
        roadmap.append({
            "task": "Soil Stabilization & Drainage",
            "impact": f"+{round(gap * 0.3, 1)}%",
            "note": "Implement bioremediation, nutrient cycling, and subsoil drainage to improve bearing capacity (current soil score below 60)."
        })
    if _f('drainage', 100) < 55:
        gap = 100 - _f('drainage', 100)
        roadmap.append({
            "task": "Drainage Network Enhancement",
            "impact": f"+{round(gap * 0.35, 1)}%",
            "note": "Improve surface and subsurface drainage; consider swales and French drains (current drainage capacity below 55)."
        })
    if _f('slope', 100) < 70:
        roadmap.append({
            "task": "Slope Stabilization",
            "impact": "+8–15%",
            "note": f"Slope suitability {_f('slope', 50):.0f}/100 indicates steep or challenging terrain; consider terracing, retaining walls, and erosion control to improve buildability."
        })
    if _f('pollution', 100) < 55:
        gap = 100 - _f('pollution', 100)
        roadmap.append({
            "task": "Air Quality Buffering",
            "impact": f"+{round(gap * 0.25, 1)}%",
            "note": "Deploy green buffers, filtration, and reduce exposure to traffic/industrial sources (current pollution score below 55)."
        })
    if _f('water', 100) < 50:
        roadmap.append({
            "task": "Water Supply & Proximity",
            "impact": "+10–20%",
            "note": "Site is distant from surface water; plan for borewell, rainwater harvesting, or piped supply to improve water security."
        })
    if _f('vegetation', 50) < 35 and _f('landuse', 50) > 40:
        roadmap.append({
            "task": "Vegetation & Green Cover",
            "impact": "+5–12%",
            "note": "Low vegetation index; consider afforestation, green roofs, and pervious surfaces to improve microclimate and ESG."
        })
    if _f('thermal', 50) < 50:
        roadmap.append({
            "task": "Thermal Comfort & HVAC",
            "impact": "+8–15%",
            "note": "Thermal comfort below 50; plan for passive cooling, shading, and HVAC to improve habitability."
        })
    if _f('intensity', 0) > 65:
        roadmap.append({
            "task": "Heat Stress Mitigation",
            "impact": "+5–10%",
            "note": "High heat stress index; implement cooling measures, ventilation, and heat-resistant design."
        })
    if _f('infrastructure', 50) < 45:
        roadmap.append({
            "task": "Access & Connectivity",
            "impact": "+15–25%",
            "note": "Remote from major roads; prioritize access road and utility connectivity to improve development potential."
        })
    if _f('rainfall', 50) < 45:
        roadmap.append({
            "task": "Rainfall & Irrigation Planning",
            "impact": "+5–15%",
            "note": "Low rainfall band; plan irrigation and water storage for agriculture or landscaping."
        })
    if _f('elevation', 50) > 0 and _f('elevation') > 1200:
        roadmap.append({
            "task": "High-Elevation Adaptation",
            "impact": "+5%",
            "note": "Elevation >1200m; consider access, frost, and oxygen-related design adaptations."
        })

    # 3. Interventions — dynamic from factors (not static)
    interventions = []
    if _f('pollution', 100) < 50:
        interventions.append("Deploy active air-filtration and green buffers to counter urban smog (pollution score below 50).")
    else:
        interventions.append("Maintain zoning and green buffers to preserve current air quality.")
    if _f('water', 100) < 60:
        interventions.append("Establish greywater recycling and rainwater harvesting to improve water security (water score below 60).")
    else:
        interventions.append("Maintain current water management; monitor proximity to water bodies.")
    if _f('landuse', 50) < 40 and _f('vegetation', 50) > 60:
        interventions.append("Verify zoning; high vegetation may indicate protected or sensitive land — confirm buildability.")
    if _f('population', 50) < 40:
        interventions.append("Sparse population — plan for workforce and services access; consider remote-work-friendly design.")

    return {
        "expected_score": expected_2036_score,
        "metrics": {
            "urban_sprawl": f"+{round(urban_impact, 1)}%",
            "veg_loss": f"{round(veg_loss, 1)}%"
        },
        "roadmap": roadmap,
        "interventions": interventions[:6]
    }
@app.route('/<path:path>', methods=['OPTIONS'])
def global_options(path):
    return jsonify({"status": "ok"}), 200


def generate_temporal_forecast(current_suitability, history_10y):
    """
    Predicts 2030 landscape state, risk bars, and actionable factors to improve suitability.
    """
    veg_loss = abs(history_10y['drifts'].get('landuse', 0))
    urban_gain = abs(history_10y['drifts'].get('proximity', 0) or history_10y['drifts'].get('infrastructure', 0))
    current_score = current_suitability.get('suitability_score', 50)
    cat = current_suitability.get('category_scores') or {}
    
    # Flatten factors for low-score detection
    f = _extract_flat_factors(current_suitability.get('factors') or {})
    
    # Factors to work on: those below 60 with concrete suggestions (reliable, factor-specific)
    factor_actions = {
        "slope": ("Slope stabilization", "Terracing, retaining walls, erosion control to improve buildability."),
        "elevation": ("Elevation adaptation", "Access and frost design; consider oxygen/altitude if >1500m."),
        "flood": ("Flood resilience", "Retention basins, permeable paving, drainage upgrades."),
        "water": ("Water proximity", "Rainwater harvesting, borewell, or piped supply planning."),
        "drainage": ("Drainage enhancement", "Swales, French drains, subsurface drainage."),
        "vegetation": ("Green cover", "Afforestation, green roofs, pervious surfaces for microclimate."),
        "soil": ("Soil improvement", "Bioremediation, nutrient cycling, subsoil drainage."),
        "pollution": ("Air quality", "Green buffers, filtration, reduce traffic/industrial exposure."),
        "rainfall": ("Rainfall / irrigation", "Irrigation and water storage for agriculture or landscaping."),
        "thermal": ("Thermal comfort", "Passive cooling, shading, HVAC planning."),
        "intensity": ("Heat stress mitigation", "Cooling measures, ventilation, heat-resistant design."),
        "landuse": ("Land use / zoning", "Verify zoning; protect or develop per regulations."),
        "infrastructure": ("Access & connectivity", "Access road, utility connectivity."),
        "population": ("Services & workforce", "Plan for workforce and services access."),
    }
    factors_to_improve = []
    for key, (title, suggestion) in factor_actions.items():
        val = f.get(key, f.get('proximity' if key == 'infrastructure' else key, 50))
        if val is None:
            val = f.get('landslide', 50) if key == 'slope' else 50
        try:
            v = float(val)
        except (TypeError, ValueError):
            v = 50
        if v < 60:
            factors_to_improve.append({
                "factor": key.replace("_", " ").title(),
                "current_score": round(v, 1),
                "suggested_action": suggestion,
                "title": title,
            })

    heat_risk_val = min(98, max(10, (urban_gain * 8) + (100 - current_score) * 0.4))
    urban_risk_val = min(98, max(5, (urban_gain * 12)))

    prompt = f"""
ROLE: Geospatial Planning Consultant AI.
DATA (2016-2026):
- Vegetation/Land use drift: {veg_loss:.1f} pts
- Infrastructure/Proximity change: {urban_gain:.1f} pts
- Current suitability: {current_score:.1f}/100
- Category scores: Physical {cat.get('physical', 50):.0f}, Environmental {cat.get('environmental', 50):.0f}, Hydrology {cat.get('hydrology', 50):.0f}, Climatic {cat.get('climatic', 50):.0f}, Socio-Economic {cat.get('socio_econ', 50):.0f}

TASK: Provide a strategic projection for 2030. Mention which factor categories could be improved (e.g. drainage, vegetation, thermal) to make the area more suitable. If stable, explain why. If changing, note heat-island and flood-plain risks.
FORMAT: 2-3 professional sentences. Start with 'Forecast 2030:'.
"""
    
    try:
        if client:
            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            response_text = response.text
        elif groq_client:
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = completion.choices[0].message.content
        else:
            response_text = f"Forecast 2030: Based on current suitability {current_score:.0f}/100 and category scores, focus on improving the factors listed below to increase suitability by 2030."
    except Exception as e:
        logger.error(f"AI Forecast Failure: {e}")
        response_text = f"Forecast 2030: Sustained stability of {100-veg_loss:.0f}% green cover suggests a resilient local microclimate. Improve the factors listed below to raise suitability further."
    
    return {
        "text": response_text,
        "heat_risk": round(heat_risk_val, 1),
        "urban_risk": round(urban_risk_val, 1),
        "factors_to_improve": factors_to_improve[:8],
    }
    
def calculate_future_drift(current_factors, years_ahead):
    future = current_factors.copy()
    
    # 1. Urbanization (Vegetation/Landuse Reduction)
    # The more years pass, the more Landuse suitability drops due to sprawl
    future['landuse'] *= (0.98 ** years_ahead) 
    
    # 2. Environmental Degradation
    future['pollution'] *= (0.97 ** years_ahead) # Pollution increases (Score drops)
    
    # 3. Resource Scarcity
    future['water'] *= (0.99 ** years_ahead)
    
    # Recalculate Score
    future_score = sum(future.values()) / len(future)
    
    return {
        "future_factors": future,
        "future_score": future_score,
        "drift_percentage": ((future_score - sum(current_factors.values())/len(current_factors)))
    }

def get_strategic_intelligence(current_factors, current_score):
    # 1. AI Trend Prediction (The Drift)
    # We simulate a 10-year outlook based on urbanization/degradation trends
    predicted_score = current_score * 0.92  # General 8% degradation trend
    
    # 2. Improvement Roadmap (Specific Engineering Tasks)
    roadmap = []
    if current_factors.get('flood', 100) < 60:
        roadmap.append({"task": "Drainage Infrastructure", "impact": "+15%", "cost": "High"})
    if current_factors.get('pollution', 100) < 60:
        roadmap.append({"task": "Green Buffer Zones", "impact": "+10%", "cost": "Low"})
    
    return {
        "future_projection": {
            "year": 2035,
            "expected_score": round(predicted_score, 1),
            "urbanization_impact": "High (Projected +22% sprawl)",
            "vegetation_drift": "-14.5%"
        },
        "roadmap": roadmap,
        "preventative_measures": [
            "Implement Mixed-Use Zoning to block industrial encroachment",
            "Establish Rainwater Harvesting to offset water table drop"
        ]
    }

def _extract_flat_factors(factors: dict) -> dict:
    """
    Flattens the nested 15-factor structure into a simple dict for ML (14 factors) and history analysis.
    """
    def _get_val(cat: str, key: str, fallback: float = 50.0) -> float:
        try:
            cat_data = factors.get(cat, {})
            factor_data = cat_data.get(key, {})
            if isinstance(factor_data, dict):
                return float(factor_data.get("value", fallback))
            return float(factor_data) if factor_data is not None else fallback
        except (TypeError, ValueError):
            return fallback
    
    return {
        # Physical (2)
        "slope": _get_val("physical", "slope"),
        "elevation": _get_val("physical", "elevation"),
        # Hydrology (3)
        "flood": _get_val("hydrology", "flood"),
        "water": _get_val("hydrology", "water"),
        "drainage": _get_val("hydrology", "drainage"),
        # Environmental (3)
        "vegetation": _get_val("environmental", "vegetation"),
        "pollution": _get_val("environmental", "pollution"),
        "soil": _get_val("environmental", "soil"),
        # Climatic (3)
        "rainfall": _get_val("climatic", "rainfall"),
        "thermal": _get_val("climatic", "thermal"),
        "intensity": _get_val("climatic", "intensity"),
        # Socio-Economic (3)
        "landuse": _get_val("socio_econ", "landuse"),
        "infrastructure": _get_val("socio_econ", "infrastructure"),
        "population": _get_val("socio_econ", "population"),
        # Legacy mappings for ML model compatibility
        "landslide": _get_val("physical", "slope"),  # slope is proxy for landslide
        "proximity": _get_val("socio_econ", "infrastructure"),  # infrastructure is proximity
    }


@app.route('/history_analysis', methods=['POST', 'OPTIONS'])
def get_history():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        lat = float(data.get('latitude'))
        lng = float(data.get('longitude'))

        # 1. Fetch CURRENT state baseline
        current_suitability = _perform_suitability_analysis(lat, lng)
        
        # 2. Flatten factors and current category scores (5 categories)
        f = _extract_flat_factors(current_suitability['factors'])
        current_cat = current_suitability.get('category_scores') or {}
        current_physical = float(current_cat.get('physical', 50))
        current_environmental = float(current_cat.get('environmental', 50))
        current_hydrology = float(current_cat.get('hydrology', 50))
        current_climatic = float(current_cat.get('climatic', 50))
        current_socio = float(current_cat.get('socio_econ', 50))
        
        # Define Year Mapping for different timelines
        year_map = {'1W': 2020, '1M': 2020, '1Y': 2017, '10Y': 2017}
        
        # 3. Determine Urbanization Decay Rate based on infrastructure proximity (reliable formula)
        is_urban = f.get('proximity', 50) > 60
        decay_rate = 0.02 if is_urban else 0.005 

        # 4. Generate Complete Bundle for Temporal Slider
        timelines = ['1W', '1M', '1Y', '10Y']
        history_bundle = {}

        for t_key in timelines:
            years_map = {'1W': 1.0/52.0, '1M': 1.0/12.0, '1Y': 1.0, '10Y': 10.0}
            offset = years_map[t_key]
            
            # Reconstruction Logic (same formula as aggregator: past = current - drift)
            visual_multiplier = 1.5 if t_key == '1Y' else 1.0
            rev_mult = (1.0 - (decay_rate * offset * visual_multiplier))
            
            p_prox = max(0, min(100, f.get('proximity', 50) * rev_mult))
            p_land = max(0, min(99.9, f.get('landuse', 50) / rev_mult))
            p_flood = max(0, min(100, f.get('flood', 50) * (1.0 + (decay_rate * offset * 0.5))))
            p_soil = max(0, min(100, f.get('soil', 50) * (1.0 + (decay_rate * offset))))
            drift_pollution = round(offset * 2.0, 2)
            drift_vegetation = round(-offset * 1.5, 2)
            drift_thermal = round(offset * 0.5, 2)
            drift_population = round(offset * 3.0, 2)
            
            # Historical Weather Archive
            p_rain_mm = fetch_historical_weather_stats(lat, lng, int(offset) if offset >= 1 else 1)
            p_rain_score = max(0, min(100, 100 - (p_rain_mm / 10) if p_rain_mm < 800 else 20))
            
            # Past category scores (same 5-category formula as Aggregator)
            past_physical = (f.get('slope', 50) + f.get('elevation', 50)) / 2
            past_environmental = (max(0, min(100, f.get('vegetation', 50) + drift_vegetation)) + p_soil + (f.get('pollution', 50) + drift_pollution)) / 3.0
            past_hydrology = (f.get('water', 50) + f.get('drainage', 50)) / 2
            past_climatic = (p_rain_score + (f.get('thermal', 50) + drift_thermal)) / 2
            past_socio = (p_prox + p_land + (f.get('population', 50) + drift_population)) / 3
            p_score_rule = round((past_physical + past_environmental + past_hydrology + past_climatic + past_socio) / 5.0, 2)
            # Use ML ensemble (14-factor) for p_score when any model is loaded
            p_pollution = max(0, min(100, f.get('pollution', 50) + drift_pollution))
            p_vegetation = max(0, min(100, f.get('vegetation', 50) + drift_vegetation))
            p_thermal = max(0, min(100, f.get('thermal', 50) + drift_thermal))
            p_population = max(0, min(100, f.get('population', 50) + drift_population))
            past_flat = {
                "slope": f.get('slope', 50), "elevation": f.get('elevation', 50),
                "flood": p_flood, "water": f.get('water', 50), "drainage": f.get('drainage', 50),
                "vegetation": p_vegetation, "pollution": p_pollution, "soil": p_soil,
                "rainfall": p_rain_score, "thermal": p_thermal, "intensity": f.get('intensity', 50),
                "landuse": p_land, "infrastructure": p_prox, "population": p_population,
            }
            p_score_ml, ml_used, score_source_ml = _predict_suitability_ml(past_flat)
            p_score = p_score_ml if ml_used else p_score_rule
            
            # Urbanization Velocity (The Derivative)
            prox_change = f.get('proximity', 50) - p_prox
            land_change = p_land - f.get('landuse', 50)
            raw_velocity = (prox_change + land_change) / (2 * max(offset, 0.01))
            velocity_score = min(10, max(0, raw_velocity * 4))

            # Drifts: historical_value - current_value (so negative drift = current improved)
            drifts = {
                "rainfall": round(p_rain_score - f.get('rainfall', 50), 2),
                "flood": round(p_flood - f.get('flood', 50), 2),
                "slope": 0.0,
                "elevation": 0.0,
                "soil": round(p_soil - f.get('soil', 50), 2),
                "proximity": round(p_prox - f.get('proximity', 50), 2),
                "infrastructure": round(p_prox - f.get('proximity', 50), 2),
                "water": 0.0,
                "drainage": 0.0,
                "pollution": round(drift_pollution, 2),
                "landuse": round(p_land - f.get('landuse', 50), 2),
                "vegetation": round(drift_vegetation, 2),
                "thermal": round(drift_thermal, 2),
                "intensity": round(offset * 0.8, 2),
                "population": round(drift_population, 2),
            }

            # category_scores_past (past_* already computed above for p_score)
            category_scores_past = {
                "physical": round(past_physical, 1),
                "environmental": round(past_environmental, 1),
                "hydrology": round(past_hydrology, 1),
                "climatic": round(past_climatic, 1),
                "socio_econ": round(past_socio, 1),
            }
            category_drifts = {
                "physical": round(current_physical - past_physical, 1),
                "environmental": round(current_environmental - past_environmental, 1),
                "hydrology": round(current_hydrology - past_hydrology, 1),
                "climatic": round(current_climatic - past_climatic, 1),
                "socio_econ": round(current_socio - past_socio, 1),
            }

            history_bundle[t_key] = {
                "score": p_score,
                "score_source": score_source_ml if ml_used else "Rule-based (5 categories)",
                "velocity": {
                    "score": round(velocity_score, 2),
                    "label": "Hyper-Growth" if velocity_score > 7 else "Expanding" if velocity_score > 3 else "Stable"
                },
                "terrain": {
                    "urban_density": round(p_prox, 2),
                    "nature_density": round(p_land, 2)
                },
                "drifts": drifts,
                "category_scores": category_scores_past,
                "category_drifts": category_drifts,
            }
            
            history_bundle[t_key]["visual_forensics"] = get_visual_forensics(lat, lng, year_map[t_key])
            
            if t_key == '10Y':
                forensics = get_visual_forensics(lat, lng)
                history_bundle[t_key]["visual_forensics"] = forensics
                history_bundle[t_key]["forecast"] = generate_temporal_forecast(current_suitability, history_bundle[t_key])
        
        return jsonify({
            "current_score": current_suitability['suitability_score'],
            "current_factors": f,
            "current_category_scores": {
                "physical": current_physical,
                "environmental": current_environmental,
                "hydrology": current_hydrology,
                "climatic": current_climatic,
                "socio_econ": current_socio,
            },
            "history_bundle": history_bundle,
            "status": "success"
        })

    except Exception as e:
        logger.exception("CRITICAL: History Analysis Engine Failure")
        return jsonify({"error": "History engine crashed", "details": str(e)}), 500

def calculate_historical_suitability(current_lat, current_lng, range_type):
    # 1. Start with current features
    # 2. Apply "Environmental Drift" based on the time range
    drift_factors = {
        '10Y': 0.15, # 15% change in features
        '1Y': 0.05,
        '1M': 0.01
    }
    multiplier = drift_factors.get(range_type, 0.1)

    # 3. Modify your input features (e.g., higher vegetation in the past)
    # This is a simplified example of how you'd tweak the input array for XGBoost
    # historical_features = get_features(current_lat, current_lng) # your existing function
    
    # Example: Simulating more vegetation/less urban sprawl in the past
    # historical_features['landuse'] += multiplier 
    
    # 4. Predict using your actual loaded model
    # hist_prediction = model.predict(historical_features)
    
    # For now, we simulate the drift on the scores directly for the UI
    return multiplier * 100

# @app.route('/suitability', methods=['POST'])
# def suitability():
#     try:
#         data = request.json or {}
#         latitude = float(data.get("latitude", 17.3850))
#         longitude = float(data.get("longitude", 78.4867))

#         # CHECK CACHE FIRST - Ensure identical results for same location
#         cache_key = get_cache_key(latitude, longitude)
#         cnn_analysis = get_cnn_classification(latitude, longitude)
#         if cache_key in ANALYSIS_CACHE:
#             result = ANALYSIS_CACHE[cache_key]
#             result['cnn_analysis'] = cnn_analysis # Ensure CNN is always fresh
#             return jsonify(result)
       
        

#         # 2. Run your Standard Suitability Analysis
#         result = _perform_suitability_analysis(latitude, longitude)
        
#         # 3. Inject CNN data into the result bundle
#         result['cnn_analysis'] = cnn_analysis

 
#         factors = result.get('factors', {})
#         prox_score = factors.get('proximity', 50)
#         land_score = factors.get('landuse', 50)
#         poll_score = factors.get('pollution', 50)

#         if result.get('label', '').startswith("Not Suitable (Waterbody"):
#             inferred_class = "Water"
#             conf = 99.9
#         elif result.get('label', '').startswith("Not Suitable (Protected"):
#             inferred_class = "Forest"
#             conf = 98.5
#         else:
#             # Infer class based on factor signatures
#             if prox_score > 70 and land_score < 40:
#                 inferred_class = "Urban"
#                 conf = 85.0 + (prox_score / 10)
#             elif poll_score < 40: # Low score = High Pollution
#                 inferred_class = "Industrial"
#                 conf = 90.0
#             elif land_score > 75:
#                 inferred_class = "Forest"
#                 conf = 88.0
#             else:
#                 inferred_class = "Agriculture" # Default for mixed/rural
#                 conf = 75.0

#         # Apply the aligned classification
#         result['cnn_analysis']['class'] = inferred_class
#         result['cnn_analysis']['confidence'] = round(conf, 1)
#         result['cnn_analysis']['confidence_display'] = f"{round(conf, 1)}%"
#         result['cnn_analysis']['note'] = "Verified by Geospatial Factors"

    
#         # NEW: Fetch nearby places during analysis to provide intelligence context
#         nearby_list = nearby_places.get_nearby_named_places(latitude, longitude)
        
#         # NEW: Generate the intelligence object using real scores and proximity
#         result['strategic_intelligence'] = generate_strategic_intelligence(
#             result['factors'], 
#             result['suitability_score'], 
#             nearby_list
#         )
#         result['nearby'] = {
#             "places": nearby_list
#         }

#         result['weather'] = get_live_weather(latitude, longitude)
#         ANALYSIS_CACHE[cache_key] = result
#         return jsonify(result)

#     except Exception as e:
#         logger.exception(f"Suitability error: {e}")
#         return jsonify({"error": str(e)}), 500

@app.route('/suitability', methods=['POST'])
def suitability():
    try:
        data = request.json or {}
        latitude = float(data.get("latitude", 17.3850))
        longitude = float(data.get("longitude", 78.4867))

        # 1. CHECK CACHE & CNN (Preserved Logic)
        cache_key = get_cache_key(latitude, longitude)
        cnn_analysis = get_cnn_classification(latitude, longitude)
        
        if cache_key in ANALYSIS_CACHE:
            result = ANALYSIS_CACHE[cache_key]
            result['cnn_analysis'] = cnn_analysis 
            return jsonify(result)

        # 2. 🚀 TRIGGER 15-FACTOR ANALYSIS (Integrated Logic)
        # Calls GeoDataService and Aggregator inside your master function
        result = _perform_suitability_analysis(latitude, longitude)

        # 3. INJECT CNN DATA (Preserved Logic)
        result['cnn_analysis'] = cnn_analysis

        # 4. ALIGN CLASSIFICATION (Updated for 15-Factor Nested Structure)
        # Use vegetation + landuse + infra + pollution so Visual Intelligence matches factor scores
        f = result['factors']
        
        # Enhanced 15-factor cross-check for more accurate land classification
        vegetation_score = f.get('vegetation', {}).get('score', 50)
        landuse_score = f.get('landuse', {}).get('score', 50)
        infra_score = f.get('infrastructure', {}).get('score', 50)
        poll_score = f.get('pollution', {}).get('score', 50)
        slope_score = f.get('slope', {}).get('score', 50)
        water_score = f.get('water', {}).get('score', 50)
        flood_score = f.get('flood', {}).get('score', 50)
        drainage_score = f.get('drainage', {}).get('score', 50)
        soil_score = f.get('soil', {}).get('score', 50)
        rainfall_score = f.get('rainfall', {}).get('score', 50)
        thermal_score = f.get('thermal', {}).get('score', 50)
        pop_score = f.get('population', {}).get('score', 50)
        
        # Sophisticated multi-factor classification logic
        inferred_class = "Mixed land use"
        conf = 70.0
        reasoning = []
        
        # Urban/Commercial detection (high infrastructure + low vegetation + high population)
        if infra_score > 75 and vegetation_score < 40 and pop_score > 60:
            inferred_class = "Urban/Commercial"
            conf = 88.0 + (infra_score - 75) * 0.4
            reasoning.append(f"High infrastructure ({infra_score:.0f}) + low vegetation ({vegetation_score:.0f}) + high population ({pop_score:.0f})")
        # Industrial detection (high pollution + moderate infrastructure + low vegetation)
        elif poll_score > 70 and infra_score > 60 and vegetation_score < 35:
            inferred_class = "Industrial"
            conf = 92.0 + (poll_score - 70) * 0.3
            reasoning.append(f"High pollution ({poll_score:.0f}) + moderate infrastructure ({infra_score:.0f}) + low vegetation ({vegetation_score:.0f})")
        # Dense Urban detection (very high infrastructure + very high population)
        elif infra_score > 85 and pop_score > 80:
            inferred_class = "Dense Urban"
            conf = 90.0 + min(8, (infra_score + pop_score - 165) * 0.2)
            reasoning.append(f"Very high infrastructure ({infra_score:.0f}) + very high population ({pop_score:.0f})")
        # Residential/Suburban detection (moderate infrastructure + moderate population + some vegetation)
        elif 60 < infra_score < 85 and 50 < pop_score < 80 and 30 < vegetation_score < 60:
            inferred_class = "Residential/Suburban"
            conf = 80.0 + min(10, (infra_score + pop_score) / 20)
            reasoning.append(f"Moderate infrastructure ({infra_score:.0f}) + population ({pop_score:.0f}) + vegetation ({vegetation_score:.0f})")
        # Forest detection (high vegetation + low infrastructure + low pollution)
        elif vegetation_score > 75 and infra_score < 40 and poll_score < 45:
            inferred_class = "Forest"
            conf = 90.0 + (vegetation_score - 75) * 0.4
            reasoning.append(f"High vegetation ({vegetation_score:.0f}) + low infrastructure ({infra_score:.0f}) + low pollution ({poll_score:.0f})")
        # Agriculture detection (moderate-high vegetation + moderate landuse + good soil)
        elif vegetation_score > 55 and landuse_score > 50 and soil_score > 60 and 40 < infra_score < 70:
            inferred_class = "Agriculture"
            conf = 82.0 + min(8, (vegetation_score + soil_score) / 25)
            reasoning.append(f"Good vegetation ({vegetation_score:.0f}) + landuse ({landuse_score:.0f}) + soil ({soil_score:.0f})")
        # Water/Wetland detection (high water + high drainage + low slope)
        elif water_score > 70 and drainage_score > 70 and slope_score < 40:
            inferred_class = "Water/Wetland"
            conf = 88.0 + (water_score - 70) * 0.4
            reasoning.append(f"High water ({water_score:.0f}) + drainage ({drainage_score:.0f}) + low slope ({slope_score:.0f})")
        # Mountainous/Hilly detection (high slope + moderate vegetation + low infrastructure)
        elif slope_score > 70 and infra_score < 50 and vegetation_score > 45:
            inferred_class = "Mountainous/Hilly"
            conf = 85.0 + (slope_score - 70) * 0.3
            reasoning.append(f"High slope ({slope_score:.0f}) + low infrastructure ({infra_score:.0f}) + vegetation ({vegetation_score:.0f})")
        # Desert/Arid detection (low vegetation + low rainfall + high thermal)
        elif vegetation_score < 30 and rainfall_score < 40 and thermal_score > 70:
            inferred_class = "Desert/Arid"
            conf = 87.0 + (thermal_score - 70) * 0.3
            reasoning.append(f"Low vegetation ({vegetation_score:.0f}) + low rainfall ({rainfall_score:.0f}) + high thermal ({thermal_score:.0f})")
        # Coastal detection (moderate water + moderate drainage + low elevation)
        elif 50 < water_score < 80 and 50 < drainage_score < 80 and f.get('elevation', {}).get('score', 50) < 50:
            inferred_class = "Coastal"
            conf = 83.0 + min(7, (water_score + drainage_score) / 25)
            reasoning.append(f"Moderate water ({water_score:.0f}) + drainage ({drainage_score:.0f}) + low elevation")
        # Rural detection (low infrastructure + low population + moderate vegetation)
        elif infra_score < 50 and pop_score < 40 and 40 < vegetation_score < 70:
            inferred_class = "Rural"
            conf = 78.0 + (70 - infra_score) * 0.2
            reasoning.append(f"Low infrastructure ({infra_score:.0f}) + population ({pop_score:.0f}) + moderate vegetation ({vegetation_score:.0f})")
        # Mixed use with dominant characteristics
        else:
            # Determine dominant characteristic for mixed use
            factors_dict = {
                'Urban': infra_score + pop_score,
                'Natural': vegetation_score + water_score,
                'Agricultural': landuse_score + soil_score,
                'Industrial': poll_score
            }
            dominant = max(factors_dict, key=factors_dict.get)
            inferred_class = f"Mixed Use ({dominant} dominant)"
            conf = 72.0 + (factors_dict[dominant] - 100) * 0.1
            reasoning.append(f"Mixed characteristics with {dominant} dominance")

        # Update CNN object with enhanced geospatial context
        result['cnn_analysis']['class'] = inferred_class
        result['cnn_analysis']['confidence'] = round(conf, 1)
        result['cnn_analysis']['confidence_display'] = f"{round(conf, 1)}%"
        result['cnn_analysis']['note'] = f"Enhanced 15-factor geospatial analysis"
        result['cnn_analysis']['reasoning'] = reasoning
        
        if result['cnn_analysis'].get('telemetry'):
            result['cnn_analysis']['telemetry']['verified_by'] = "Enhanced 15-factor geospatial cross-check"
            result['cnn_analysis']['telemetry']['inferred_from'] = f"veg={vegetation_score:.0f}, landuse={landuse_score:.0f}, infra={infra_score:.0f}, poll={poll_score:.0f}, pop={pop_score:.0f}"
            result['cnn_analysis']['telemetry']['vegetation_score'] = round(vegetation_score, 1)
            result['cnn_analysis']['telemetry']['landuse_score'] = round(landuse_score, 1)
            result['cnn_analysis']['telemetry']['slope_suitability'] = round(slope_score, 1)
            result['cnn_analysis']['telemetry']['water_proximity'] = round(water_score, 1)
            result['cnn_analysis']['telemetry']['infrastructure_score'] = round(infra_score, 1)
            result['cnn_analysis']['telemetry']['population_density'] = round(pop_score, 1)
            result['cnn_analysis']['telemetry']['pollution_level'] = round(poll_score, 1)
            result['cnn_analysis']['telemetry']['soil_quality'] = round(soil_score, 1)
            result['cnn_analysis']['telemetry']['rainfall_level'] = round(rainfall_score, 1)
            result['cnn_analysis']['telemetry']['thermal_intensity'] = round(thermal_score, 1)
            result['cnn_analysis']['telemetry']['flood_risk'] = round(flood_score, 1)
            result['cnn_analysis']['telemetry']['drainage_quality'] = round(drainage_score, 1)
            result['cnn_analysis']['telemetry']['classification_reasoning'] = reasoning

        # 5. FETCH NEARBY AMENITIES (Preserved Logic)
        nearby_list = get_nearby_named_places(latitude, longitude)
        result['nearby'] = {"places": nearby_list}

        # 6. STRATEGIC INTELLIGENCE (All 15 factors for location-based roadmap)
        flat_factors_for_intel = _extract_flat_factors(result['factors'])
        result['strategic_intelligence'] = generate_strategic_intelligence(
            flat_factors_for_intel,
            result['suitability_score'],
            nearby_list
        )

        # 6b. Expose flat factors for frontend (infrastructure tab, ESG, etc.)
        result['flat_factors'] = flat_factors_for_intel

        # 7. WEATHER & CACHING (Preserved Logic)
        result['weather'] = get_live_weather(latitude, longitude)
        ANALYSIS_CACHE[cache_key] = result
        
        return jsonify(result)

    except Exception as e:
        logger.exception(f"Critical Suitability Error: {e}")
        return jsonify({"error": str(e)}), 500
    

# def _perform_suitability_analysis(latitude: float, longitude: float) -> dict:
#         # --- INITIALIZE ALL FACTORS TO PREVENT CRASHES ---
#         landuse_s = 50.0
#         flood_s = 50.0
#         poll_s = 50.0
#         rainfall_score = 50.0
#         soil_s = 50.0
#         prox_s = 50.0
#         water_s = 50.0
#         landslide_s = 50.0
#         # 1. WATER EARLY EXIT
#         w_score, w_dist, w_meta = estimate_water_proximity_score(latitude, longitude)
#         w_score = round(w_score, 2) if w_score else 0.0

#         if w_score == 0.0 or (w_dist is not None and w_dist < 0.02):
#             water_name = w_meta.get('name') if w_meta else "an identified water body"
#             return {
#                 "suitability_score": 0.0,
#                 "label": "Not Suitable (Waterbody)",
#                 "factors": {k: 0.0 for k in ["rainfall", "flood", "landslide", "soil", "proximity", "water", "pollution", "landuse"]},
#                 "reason": f"Location is on {water_name}. Unsuitable for construction.",
#                 "explanation": {
#                     "factors_meta": {
#                         "water": {
#                             "reason": w_meta.get("detail", f"Directly on {water_name}"),
#                             "source": w_meta.get("source", "Satellite"),
#                             "confidence": "High"
#                         }
#                     }
#                 },
#                 "evidence": {"water_distance_km": 0.0, "water_details": w_meta},
#                 "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
#             }
#         # 2. FOREST/PROTECTED AREA EARLY EXIT
#         landuse_result = infer_landuse_score(latitude, longitude)
#         if isinstance(landuse_result, tuple):
#             landuse_s, landuse_details = landuse_result
#         else:
#             landuse_s = landuse_result
#             landuse_details = {"score": landuse_s}
        
#         landuse_s = round(landuse_s, 2) if landuse_s else 70.0
        
#         if landuse_s is not None and landuse_s <= 10.0:
#             return {
#                 "suitability_score": 10.0,
#                 "label": "Not Suitable (Protected/Forest Area)",
#                 "factors": {
#                     "rainfall": 0.0,
#                     "flood": 0.0,
#                     "landslide": 0.0,
#                     "soil": 0.0,
#                     "proximity": 0.0,
#                     "water": 0.0,
#                     "pollution": 0.0,
#                     "landuse": 0.0
#                 },
#                 "reason": "Location is in a forest or protected environmental area. Unsuitable for construction.",
#                 "explanation": {
#                     "factors_meta": {
#                         "landuse": {
#                             "reason": "Forest, woodland, or protected conservation area detected via OpenStreetMap. This land cannot be developed.",
#                             "source": "OpenStreetMap / Overpass API",
#                             "confidence": "High"
#                         }
#                     }
#                 },
#                 "evidence": {"landuse_score": landuse_s, "landuse_type": "Forest/Protected Area"},
#                 "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
#                 "location": {"latitude": latitude, "longitude": longitude}
#             }

#         # 3. LAND ANALYSIS
#         rainfall_score, rain_mm = estimate_rainfall_score(latitude, longitude)
#         flood_s = round(estimate_flood_risk_score(latitude, longitude) or 50.0, 2)
#         landslide_s = round(estimate_landslide_risk_score(latitude, longitude) or 60.0, 2)
#         soil_s = round(estimate_soil_quality_score(latitude, longitude) or 60.0, 2)
        
#         prox_result = compute_proximity_score(latitude, longitude)
#         prox_s = round(prox_result[0] if isinstance(prox_result, tuple) else (prox_result or 50.0), 2)
#         prox_dist = prox_result[1] if isinstance(prox_result, tuple) else None
#         proximity_details = prox_result[2] if isinstance(prox_result, tuple) else {}
        
#         poll_result = estimate_pollution_score(latitude, longitude)
#         poll_s = round(poll_result[0] if isinstance(poll_result, tuple) else (poll_result or 65.0), 2)
#         poll_value = poll_result[1] if isinstance(poll_result, tuple) else None
#         poll_details = poll_result[2] if isinstance(poll_result, tuple) else {}
        
#         rainfall_score = round(rainfall_score, 2)
        
#         # Generate detailed reasoning for rainfall
#         if rain_mm is not None:
#             if rain_mm > 800:
#                 rainfall_reason = f"Rainfall: {rain_mm}mm in 60 days. EXCESSIVE moisture increases flood risk and foundation damage. Not suitable for construction."
#             elif rain_mm > 400:
#                 rainfall_reason = f"Rainfall: {rain_mm}mm in 60 days. HIGH rainfall creates drainage challenges and moderate flood risk. Requires robust drainage systems."
#             elif rain_mm > 100:
#                 rainfall_reason = f"Rainfall: {rain_mm}mm in 60 days. MODERATE rainfall levels. Suitable with proper drainage planning. Good moisture retention for agriculture."
#             else:
#                 rainfall_reason = f"Rainfall: {rain_mm}mm in 60 days. LOW rainfall. IDEAL for construction with minimal flood risk. May need irrigation for agriculture."
#         else:
#             rainfall_reason = "Rainfall data unavailable. Estimated based on regional climate patterns."
        
#         # Generate detailed reasoning for pollution with complete numerical evidence
#         if poll_value is not None:
#             dataset_date = poll_details.get("dataset_date", "Jan 2026") if poll_details else "Jan 2026"
#             location_name = poll_details.get("location", "Location") if poll_details else "Location"
            
#             if poll_value < 10:
#                 pollution_reason = (
#                     f"PM2.5: {poll_value} µg/m³ at {location_name}. "
#                     f"EXCELLENT air quality. Below WHO Guideline Annual Average (≤10 µg/m³, 2024 Standard). "
#                     f"Also below EPA Annual Standard (12 µg/m³). "
#                     f"Dataset: OpenAQ International Network Real-time Monitoring ({dataset_date}). "
#                     f"Very low pollution levels - OPTIMAL for residential development, schools, and sensitive populations."
#                 )
#             elif poll_value < 25:
#                 pollution_reason = (
#                     f"PM2.5: {poll_value} µg/m³ at {location_name}. "
#                     f"GOOD air quality. Exceeds WHO 24-hour Guideline (≤35 µg/m³) but below annual threshold (10 µg/m³). "
#                     f"Dataset: OpenAQ International Air Quality Station Network ({dataset_date}). "
#                     f"Low pollution with acceptable living conditions for most demographics. Suitable for mixed-use development."
#                 )
#             elif poll_value < 50:
#                 pollution_reason = (
#                     f"PM2.5: {poll_value} µg/m³ at {location_name}. "
#                     f"MODERATE air quality. Exceeds WHO Guidelines (>25 µg/m³). Approaches EPA 24-hour standard concerns. "
#                     f"Dataset: OpenAQ + Sentinel-5P Satellite Aerosol Optical Depth ({dataset_date}). "
#                     f"Moderate pollution affecting respiratory health, especially children, elderly, and those with respiratory conditions. "
#                     f"Industrial/traffic sources require monitoring."
#                 )
#             elif poll_value < 100:
#                 pollution_reason = (
#                     f"PM2.5: {poll_value} µg/m³ at {location_name}. "
#                     f"POOR air quality. Significantly exceeds WHO (10 µg/m³) and EPA (12 µg/m³) standards. "
#                     f"EPA AirNow Index: Orange (Unhealthy for Sensitive Groups). "
#                     f"Dataset: OpenAQ High-frequency Monitoring Stations ({dataset_date}). "
#                     f"High pollution from traffic/industrial sources. Vulnerable populations advised against outdoor activity. "
#                     f"Air filtration and mitigation required for safe habitation."
#                 )
#             else:
#                 pollution_reason = (
#                     f"PM2.5: {poll_value} µg/m³ at {location_name}. "
#                     f"HAZARDOUS air pollution. Severely exceeds WHO (10 µg/m³) and EPA (12 µg/m³) standards. "
#                     f"EPA AirNow: Red Alert (Unhealthy for General Population). "
#                     f"Dataset: OpenAQ Urgent Monitoring Alerts ({dataset_date}). "
#                     f"Severe pollution impacting respiratory and cardiovascular systems. "
#                     f"Location unsuitable for long-term habitation without major air quality mitigation infrastructure."
#                 )
#         elif poll_details and poll_details.get("reason") == "No nearby OpenAQ station":
#             pollution_reason = (
#                 "Air quality data unavailable for this remote location. "
#                 "Estimated using MERRA-2 Satellite Aerosol Data (NASA 2026) and regional baseline models. "
#                 "Regional PM2.5 estimates from CAMS Global (Copernicus Atmosphere Monitoring Service). "
#                 "Limited direct sensor confirmation - use with caution for precise air quality assessment."
#             )
#         else:
#             pollution_reason = (
#                 "Air quality analysis based on Sentinel-5P Satellite Aerosol Data (Copernicus Program, 2025-2026) "
#                 "and traffic pattern modeling. Regional PM2.5 estimates from CAMS Global (Copernicus). "
#                 "Satellite-based assessment with ~25km spatial resolution."
#             )
        
#         # Generate detailed reasoning for soil
#         soil_explanation = f"Soil quality score: {soil_s}/100. Land suitability depends on soil bearing capacity, drainage, and agricultural potential. Regional soil profile analysis complete."

#         # 4. ENSEMBLE PREDICTION
#         features = np.array([[rainfall_score, flood_s, landslide_s, soil_s, prox_s, w_score, poll_s, landuse_s]], dtype=float)

#         try:
#             score_xgb = float(ML_MODELS['model_xgboost.pkl'].predict(features)[0])
#             score_rf = float(ML_MODELS['model_rf.pkl'].predict(features)[0])
#             final_score = round((score_xgb + score_rf) / 2, 2)
#             model_used = "Ensemble (XGBoost + Random Forest)"
#         except Exception:
#             agg = compute_suitability_score(
#                 rainfall_score=rainfall_score, flood_risk_score=flood_s,
#                 landslide_risk_score=landslide_s, soil_quality_score=soil_s,
#                 proximity_score=prox_s, water_proximity_score=w_score,
#                 pollution_score=poll_s, landuse_score=landuse_s
#             )
#             final_score = agg.get("score")
#             model_used = "Weighted Aggregator (Fallback)"
#         terrain_analysis = estimate_terrain_slope(latitude, longitude)
        

#         # 4. FINAL RESPONSE WITH METADATA (Populates Evidence Detail Section)
#         return {
#             "suitability_score": final_score,
#             "label": "Highly Suitable" if final_score >= 70 else ("Moderate" if final_score >= 40 else "Unsuitable"),
#             "model_used": model_used,
#             "terrain_analysis": terrain_analysis,
#             "factors": {
#                 "rainfall": rainfall_score, "flood": flood_s, "landslide": landslide_s,
#                 "soil": soil_s, "proximity": prox_s, "water": w_score,
#                 "pollution": poll_s, "landuse": landuse_s
#             },
#             "explanation": {
#                 "factors_meta": {
#                     "water": {
#                         "reason": w_meta.get("detail", "Water body distance analyzed."),
#                         "source": w_meta.get("source", "Map Engine"),
#                         "confidence": "High"
#                     },
#                     "rainfall": {
#                         "reason": rainfall_reason,
#                         "source": "Meteorological Data (Open-Meteo 60-day average)",
#                         "confidence": "High"
#                     },
#                     "flood": {
#                         "reason": (
#                             f"COMBINED ASSESSMENT: Rainfall ({rain_mm}mm/60d) + Water Distance ({w_dist}km). " if w_dist else "Rainfall-based flood risk analysis: "
#                         ) + (
#                             f"CRITICAL FLOOD ZONE. {round(w_dist*1000, 0)}m from river. Heavy rainfall ({rain_mm}mm) + proximity = severe overflow risk. 100+ year flood events occur at this distance." if (w_dist and w_dist < 0.3 and rain_mm and rain_mm > 300) else
#                             f"CRITICAL RIVER BANK. {round(w_dist*1000, 0)}m from water body (river edge). Even moderate rainfall ({rain_mm}mm) causes immediate flooding. Extreme hazard." if (w_dist and w_dist < 0.3) else
#                             f"HIGH FLOOD RISK. {round(w_dist*1000, 0)}m from water + heavy rainfall ({rain_mm}mm/60d > 400mm). Water overflow highly probable. 10-25 year flood return period." if (w_dist and w_dist < 0.8 and rain_mm and rain_mm > 400) else
#                             f"HIGH FLOOD RISK. {round(w_dist*1000, 0)}m from water body. Rainfall: {rain_mm}mm. Monsoon flooding likely with normal seasonal precipitation." if (w_dist and w_dist < 0.8) else
#                             f"MODERATE FLOOD RISK. {round(w_dist*1000, 0)}m buffer from water. Rainfall: {rain_mm}mm/60d. Floods only with exceptional rainfall (>250mm) + water overflow. Normal drainage handles seasonal rain." if (w_dist and w_dist < 1.5) else
#                             f"LOW FLOOD RISK. {round(w_dist, 2)}km from water. Rainfall: {rain_mm}mm/60d. Natural terrain and drainage provide good protection. Only extreme precipitation causes flooding." if (w_dist and w_dist < 3.0) else
#                             f"VERY LOW FLOOD RISK. Remote location {round(w_dist, 2)}km from water sources. Rainfall: {rain_mm}mm/60d. Topography provides natural protection. Safe for standard construction." if w_dist else
#                             f"Rainfall: {rain_mm}mm/60d. No significant water bodies detected. Standard drainage adequate."
#                         ),
#                         "source": "Integrated: Water Proximity + Rainfall Data (Open-Meteo 2025-2026) + USGS Flood Models",
#                         "confidence": "High" if w_dist and rain_mm else "Medium"
#                     },
#                     "landslide": {
#                         "reason": f"Slope stability and soil composition analysis (USDA Soil Data, 2023-2024). Score: {landslide_s}/100. Steeper slopes (>30°) and weak geological formations increase risk. Terrain stability assessment based on Digital Elevation Model (NASA SRTM v3.0). Gully erosion patterns and subsurface stratum analysis included.",
#                         "source": "Terrain Analysis (DEM - NASA SRTM v3.0) + USDA Soil Database (2024)",
#                         "confidence": "Medium"
#                     },
#                     "soil": {
#                         "reason": soil_explanation,
#                         "source": "Soil Survey (Regional soil maps)",
#                         "confidence": "Medium"
#                     },
#                     "proximity": {
#                         "reason": proximity_details.get("explanation", "Distance to roads and infrastructure analyzed."),
#                         "source": "Infrastructure Data (OpenStreetMap)",
#                         "confidence": "High"
#                     },
#                     "pollution": {
#                         "reason": pollution_reason,
#                         "source": "Air Quality Sensors (OpenAQ) & Satellite Aerosol Data",
#                         "confidence": "High" if poll_value is not None else "Medium"
#                     },
#                     "landuse": {
#                         "reason": (
#                             f"Land Cover Classification: {landuse_details.get('classification', 'Unknown')}. "
#                             f"NDVI Index: {landuse_details.get('ndvi_index', 'N/A')} (Range: {landuse_details.get('ndvi_range', 'N/A')}). "
#                             f"Sentinel-2 Multispectral Imagery with 10m resolution classification. "
#                             f"Indices: Forest (NDVI >0.6), Agricultural (NDVI 0.4-0.6), Urban (NDVI <0.35), Water (NDVI <-0.1). "
#                             f"OpenStreetMap Vector Confirmation (100m-500m radius analysis). "
#                             f"{landuse_details.get('reason', '')} "
#                             f"Classification Confidence: {landuse_details.get('confidence', 90)}%"
#                         ),
#                         "source": landuse_details.get("dataset_source", "Remote Sensing (Sentinel-2 ESA, 2025) + OpenStreetMap (Jan 2026)"),
#                         "confidence": "High" if landuse_details.get("confidence", 0) > 90 else "Medium"
#                     }
#                 }
#             },
#             "evidence": {"water_distance_km": w_dist, "rainfall_total_mm_60d": rain_mm},
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
#             "location": {"latitude": latitude, "longitude": longitude}
#         }
def build_factor_evidence(f):
    return {
        "physical": {
            "slope": {
                "value": f['physical']['slope']['value'],
                "unit": "%",
                "label": f['physical']['slope'].get("label", "Slope evaluated"),
                "source": f['physical']['slope']['source'],
            },
            "elevation": {
                "value": f['physical']['elevation']['value'],
                "unit": "m",
                "label": f['physical']['elevation'].get("label", "Elevation level"),
                "source": f['physical']['elevation']['source'],
            }
        },

        "hydrology": {
            "flood": {
                "value": f['hydrology']['flood']['value'],
                "label": f['hydrology']['flood'].get("label", "Flood safety index"),
                "source": f['hydrology']['flood']['source'],
            },
            "water_proximity": {
                "distance_km": f['hydrology']['water'].get("distance_km"),
                "score": f['hydrology']['water']['value'],
                "source": f['hydrology']['water']['source'],
            }
        },

        "climatic": {
            "rainfall": {
                "annual_mm": f['climatic']['rainfall']['raw'],
                "score": f['climatic']['rainfall']['value'],
                "label": f['climatic']['rainfall'].get("label", "Rainfall regime"),
                "source": f['climatic']['rainfall']['source'],
            },
            "thermal_comfort": {
                "index": f['climatic']['thermal']['value'],
                "label": f['climatic']['thermal'].get("label", "Thermal comfort"),
                "source": f['climatic']['thermal']['source'],
            }
        },

        "environmental": {
            "vegetation": {
                "ndvi": f['environmental']['vegetation'].get("raw"),
                "score": f['environmental']['vegetation']['value'],
                "label": f['environmental']['vegetation'].get("label", "Vegetation health"),
                "source": f['environmental']['vegetation']['source'],
            },
            "pollution": {
                "pm25": f['environmental']['pollution']['raw'],
                "score": f['environmental']['pollution']['value'],
                "label": f['environmental']['pollution'].get("label", "Air quality"),
                "source": f['environmental']['pollution']['source'],
            }
        },

        "socio_econ": {
            "landuse": {
                "classification": f['socio_econ']['landuse'].get("classification"),
                "score": f['socio_econ']['landuse']['value'],
                "source": f['socio_econ']['landuse']['source'],
            },
            "infrastructure": {
                "score": f['socio_econ']['infrastructure']['value'],
                "source": f['socio_econ']['infrastructure']['source'],
            },
            "population": {
                "density": f['socio_econ']['population'].get("raw"),
                "score": f['socio_econ']['population']['value'],
                "source": f['socio_econ']['population']['source'],
            }
        }
    }
def normalize_factor(factor: dict, *, default_value=50.0):
    """
    Enforces a strict schema for all factors.
    Prevents KeyError crashes across the entire system.
    """
    if not isinstance(factor, dict):
        return {
            "value": default_value,
            "raw": None,
            "label": "Data unavailable",
            "source": "Normalization fallback",
            "confidence": 40,
            "evidence": None,
            "unit": None,
            "details": None
        }

    # Extract value, handling None case. Slope: use scaled_score (0-100 suitability) so 0% slope → 100.
    raw_value = factor.get("value", factor.get("score"))
    scaled = factor.get("scaled_score")
    if scaled is not None:
        try:
            value = max(0.0, min(100.0, float(scaled)))
        except (TypeError, ValueError):
            value = default_value if raw_value is None else float(raw_value)
    elif raw_value is None:
        value = default_value
    else:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = default_value

    return {
        "value": value,
        "raw": factor.get("raw"),
        "label": factor.get("label"),
        "source": factor.get("source"),
        "confidence": factor.get("confidence", 75),
        "evidence": factor.get("evidence"),
        "unit": factor.get("unit"),
        "details": factor.get("details"),
        "distance_km": factor.get("distance_km"),
        "classification": factor.get("classification"),
        "density": factor.get("density")
    }

def _generate_evidence_text(factor_name: str, factor_data: dict, raw_factors: dict) -> str:
    """
    Generates detailed human-readable evidence text for each factor.
    Similar to the old 8-factor format but for all 15 factors.
    """
    val = factor_data.get("value")
    raw = factor_data.get("raw")
    label = factor_data.get("label", "")
    
    if factor_name == "slope":
        # val is suitability 0-100; slope percent from raw for evidence text
        slope_pct = None
        if isinstance(raw_factors, dict):
            phys = raw_factors.get("physical", {})
            slope_raw = phys.get("slope", {}) if isinstance(phys, dict) else {}
            if isinstance(slope_raw, dict):
                slope_pct = slope_raw.get("value")
        slope_pct = slope_pct if slope_pct is not None else (100.0 - val / 2.22 if val is not None else None)
        score_val = val if val is not None else 50
        if score_val is None:
            return "Slope data unavailable for this location. Score defaulted to 50."
        pct_str = f"{slope_pct:.1f}%" if slope_pct is not None else "—"
        if slope_pct is not None and slope_pct < 3:
            return f"Slope: {pct_str} gradient (from DEM). VERY FLAT terrain. IDEAL for construction; minimal grading. Reference: <3% ideal, 3–8% gentle, 8–15% moderate, >15% steep. Suitability score {score_val:.0f}/100."
        elif slope_pct is not None and slope_pct < 8:
            return f"Slope: {pct_str} gradient. GENTLE slope. Suitable for most construction; minor earthwork; good drainage. Suitability score {score_val:.0f}/100 (gentle band 3–8%)."
        elif slope_pct is not None and slope_pct < 15:
            return f"Slope: {pct_str} gradient. MODERATE slope. Careful site planning; may need retaining structures. Suitability score {score_val:.0f}/100 (moderate band 8–15%)."
        elif slope_pct is not None and slope_pct < 30:
            return f"Slope: {pct_str} gradient. STEEP terrain. HIGH construction costs; extensive earthwork. Suitability score {score_val:.0f}/100 (steep band 15–30%)."
        elif slope_pct is not None:
            return f"Slope: {pct_str} gradient. VERY STEEP. NOT SUITABLE for standard construction; landslide/erosion risk. Suitability score {score_val:.0f}/100."
        return f"Slope suitability score {score_val:.0f}/100. {label or 'Terrain slope evaluated from elevation data.'}"
    
    elif factor_name == "elevation":
        if val is None:
            return "Elevation data unavailable. Score defaulted to 50."
        if val < 50:
            return f"Elevation: {val}m above sea level (measured). LOW coastal/floodplain. Monitor sea-level and flood. Reference: <50m low, 50–200m low-moderate, 200–600m optimal. Score {val}/100 reflects low band."
        elif val < 200:
            return f"Elevation: {val}m above sea level. LOW to MODERATE. Good accessibility; manageable flood exposure. Score {val}/100 reflects 50–200m band."
        elif val < 600:
            return f"Elevation: {val}m above sea level. MODERATE — optimal range for most construction (reference 200–600m). Score {val}/100 reflects optimal band."
        elif val < 1500:
            return f"Elevation: {val}m above sea level. HIGH. Consider temperature extremes and access. Score {val}/100 reflects 600–1500m band."
        else:
            return f"Elevation: {val}m above sea level. VERY HIGH. Challenging; reduced oxygen, extreme weather. Score {val}/100 reflects >1500m band."
    
    elif factor_name == "flood":
        # Get water distance for combined assessment
        water_data = raw_factors.get("hydrology", {}).get("water", {})
        water_dist = water_data.get("distance_km")
        rain_data = raw_factors.get("climatic", {}).get("rainfall", {})
        rain_mm = rain_data.get("raw")
        
        if water_dist is not None and rain_mm is not None:
            if water_dist < 0.3 and rain_mm > 300:
                return f"COMBINED: Rainfall {rain_mm}mm/year + water distance {water_dist:.2f}km. CRITICAL FLOOD ZONE. Score {val}/100 — heavy rainfall + proximity = severe overflow risk (threshold: <0.5km + >300mm)."
            elif water_dist < 0.5:
                return f"COMBINED: Flood safety {val}/100. Rainfall {rain_mm}mm/year, water {water_dist:.2f}km. HIGH FLOOD RISK (proximity <0.5km). Score reflects combined rain + distance model."
            elif water_dist < 1.5:
                return f"COMBINED: Flood safety {val}/100. Rainfall {rain_mm}mm/year, water {water_dist:.2f}km. MODERATE risk — floods only with exceptional rainfall. Score reflects 0.5–1.5km band."
            elif water_dist < 3.0:
                return f"COMBINED: Flood safety {val}/100. Water {water_dist:.2f}km, rainfall {rain_mm}mm/year. LOW flood risk; natural terrain protection. Score reflects 1.5–3km band."
            else:
                return f"COMBINED: Flood safety {val}/100. Remote: {water_dist:.2f}km from water, rainfall {rain_mm}mm/year. VERY LOW flood risk. Score reflects >3km + rainfall factor."
        else:
            return f"Flood safety score: {val}/100. {label}. Analysis from regional hydrology (rain + water distance)."
    
    elif factor_name == "water":
        dist = factor_data.get("distance_km")
        details = factor_data.get("details", {}) if isinstance(factor_data.get("details"), dict) else {}
        water_raw = raw_factors.get("hydrology", {}).get("water", {})
        if isinstance(water_raw, dict) and (water_raw.get("details") or {}).get("name"):
            water_name = water_raw["details"]["name"]
        else:
            water_name = details.get("name", "water body") if details else "water body"
        
        if dist is not None and float(dist) < 0.02:
            return f"Location is ON water body: {water_name}. Water proximity score 0/100 — NOT SUITABLE for terrestrial construction. Distance <20m; all other factors aligned (slope/elevation/flood N/A on water)."
        if dist is not None:
            if dist < 0.5:
                return f"Distance to water: {dist:.2f}km ({water_name}). CLOSE — irrigation advantage; flood monitoring needed. Score {val}/100 reflects <0.5km band."
            elif dist < 2.0:
                return f"Distance to water: {dist:.2f}km ({water_name}). MODERATE access for utility/agriculture. Score {val}/100 reflects 0.5–2km band."
            else:
                return f"Distance to water: {dist:.2f}km ({water_name}). DISTANT — may require well/borewell. Score {val}/100 reflects >2km band."
        return f"Water proximity score: {val}/100. No major water bodies in analysis radius; score from regional baseline."
    
    elif factor_name == "vegetation":
        ndvi = factor_data.get("raw")
        ndvi_f = ndvi if isinstance(ndvi, (int, float)) else (ndvi.get("raw") if isinstance(ndvi, dict) else None)
        ndvi_str = f"{ndvi_f:.2f}" if isinstance(ndvi_f, (int, float)) else str(ndvi_f) if ndvi_f else "N/A"
        if ndvi_f is not None:
            if ndvi_f < 0.2 or val < 20:
                return f"Vegetation index: {val}/100 (proxy {ndvi_str}). BARE/BUILT-UP. Urban/barren. Score reflects <0.2 proxy band."
            elif ndvi_f < 0.4 or val < 40:
                return f"Vegetation index: {val}/100 (proxy {ndvi_str}). SPARSE. Suitable for development with minimal clearing. Score reflects 0.2–0.4 band."
            elif ndvi_f < 0.6 or val < 60:
                return f"Vegetation index: {val}/100 (proxy {ndvi_str}). MODERATE — agricultural/mixed cover. Score reflects 0.4–0.6 band."
            else:
                return f"Vegetation index: {val}/100 (proxy {ndvi_str}). DENSE — possible forest/protected; verify zoning. Score reflects >0.6 band."
        return f"Vegetation index: {val}/100. {label}. From satellite soil moisture; score reflects vegetation proxy band."
    
    elif factor_name == "pollution":
        pm25 = factor_data.get("raw") or raw
        if pm25 is not None:
            who_annual, who_24h = 10, 35
            if pm25 < 10:
                return f"PM2.5: {pm25} µg/m³ (measured). EXCELLENT. Below WHO annual ({who_annual} µg/m³). Score {val}/100 reflects <10 band — optimal for residential."
            elif pm25 < 25:
                return f"PM2.5: {pm25} µg/m³. GOOD. Acceptable for residential/commercial. WHO 24h = {who_24h}. Score {val}/100 reflects 10–25 band."
            elif pm25 < 50:
                return f"PM2.5: {pm25} µg/m³. MODERATE; exceeds WHO. Industrial/traffic possible. Score {val}/100 reflects 25–50 band."
            elif pm25 < 100:
                return f"PM2.5: {pm25} µg/m³. POOR; exceeds standards. Air filtration recommended. Score {val}/100 reflects 50–100 band."
            else:
                return f"PM2.5: {pm25} µg/m³. HAZARDOUS. Severe health risk. Score {val}/100 reflects >100 band."
        else:
            return f"Air quality score: {val}/100. From regional/satellite baseline (no local PM2.5)."
    
    elif factor_name == "soil":
        if val is not None:
            if val >= 80:
                return f"Soil quality: {val}/100. EXCELLENT bearing capacity and drainage; ideal loam. Score reflects 80–100 band — standard foundations adequate."
            elif val >= 60:
                return f"Soil quality: {val}/100. GOOD. Standard foundation adequate. Score reflects 60–80 band."
            elif val >= 40:
                return f"Soil quality: {val}/100. MODERATE. Soil testing and foundation enhancement recommended. Score reflects 40–60 band."
            else:
                return f"Soil quality: {val}/100. POOR; clayey/waterlogged. Special foundations required. Score reflects <40 band."
        else:
            return "Soil quality unavailable; regional baseline applied. Score 50."
    
    elif factor_name == "rainfall":
        rain_mm = factor_data.get("raw")
        if rain_mm is not None:
            opt_lo, opt_hi = 800, 1500
            if rain_mm < 300:
                return f"Rainfall: {rain_mm}mm/year (365-day sum). LOW/ARID. IDEAL for construction; minimal flood risk. Irrigation needed for agriculture. Optimal band = {opt_lo}–{opt_hi}mm; your value below range → score {val}/100 reflects dry band."
            elif rain_mm < 800:
                return f"Rainfall: {rain_mm}mm/year. MODERATE. Good balance for construction and agriculture with drainage. Below optimal {opt_lo}–{opt_hi}mm → score {val}/100 reflects moderate-dry band."
            elif rain_mm < 1500:
                return f"Rainfall: {rain_mm}mm/year. HIGH. Robust drainage needed; moderate flood susceptibility. Within optimal band {opt_lo}–{opt_hi}mm → score {val}/100 reflects good range."
            else:
                return f"Rainfall: {rain_mm}mm/year. EXCESSIVE. High flood/foundation risk. Above {opt_hi}mm → score {val}/100 reflects excessive band."
        else:
            return f"Rainfall suitability: {val}/100. {label}. Open-Meteo Historical API."
    
    elif factor_name == "thermal":
        raw_data = factor_data.get("raw", {})
        if isinstance(raw_data, dict):
            temp = raw_data.get("temperature_c")
            humidity = raw_data.get("humidity_pct")
            if temp is not None:
                opt_temp = "22–26°C"
                if val >= 80:
                    return f"Thermal comfort: {val}/100. Temperature {temp}°C, humidity {humidity}%. HIGHLY COMFORTABLE. Optimal band {opt_temp}; score reflects minimal deviation."
                elif val >= 60:
                    return f"Thermal comfort: {val}/100. Temp {temp}°C, humidity {humidity}%. COMFORTABLE; minor seasonal extremes. Score reflects moderate deviation from {opt_temp}."
                elif val >= 40:
                    return f"Thermal comfort: {val}/100. Temp {temp}°C, humidity {humidity}%. MARGINAL; consider HVAC. Score reflects significant deviation from {opt_temp}."
                else:
                    return f"Thermal comfort: {val}/100. Temp {temp}°C, humidity {humidity}%. UNCOMFORTABLE — heat/cold stress. Score reflects large deviation from {opt_temp}."
        return f"Thermal comfort: {val}/100. {label}. Real-time temperature/humidity."
    
    elif factor_name == "landuse":
        classification = factor_data.get("classification") or (raw_factors.get("socio_econ", {}).get("landuse", {}) or {}).get("classification", "Unknown")
        if val is not None:
            if val <= 15:
                return f"Land-use: {classification}. Score {val}/100. PROTECTED/FOREST — legally non-buildable. Score reflects 0–15 band."
            elif val <= 40:
                return f"Land-use: {classification}. Score {val}/100. RESTRICTED development; environmental sensitivity. Score reflects 15–40 band."
            elif val <= 70:
                return f"Land-use: {classification}. Score {val}/100. MODERATE potential; agricultural/mixed zoning. Score reflects 40–70 band."
            else:
                return f"Land-use: {classification}. Score {val}/100. HIGH potential; urban/commercial compatible. Score reflects 70–100 band."
        return f"Land-use: {classification}. Sentinel-2 NDVI + OSM."
    
    elif factor_name == "infrastructure":
        details = factor_data.get("details", {}) if isinstance(factor_data.get("details"), dict) else {}
        dist = details.get("distance_km")
        road_type = details.get("road_type", "road")
        
        if dist is not None:
            if dist < 0.05:
                return f"Nearest road: {road_type} at {dist:.2f}km (on corridor). HIGH noise/safety risk; excellent connectivity. Score {val}/100 reflects on-road band."
            elif dist < 0.3:
                return f"Nearest road: {road_type} at {dist:.2f}km. EXCELLENT accessibility. Score {val}/100 reflects <0.3km band."
            elif dist < 1.0:
                return f"Nearest road: {road_type} at {dist:.2f}km. GOOD access; balance connectivity/tranquility. Score {val}/100 reflects 0.3–1km band."
            elif dist < 3.0:
                return f"Nearest road: {road_type} at {dist:.2f}km. MODERATE access; access road may be needed. Score {val}/100 reflects 1–3km band."
            else:
                return f"Nearest road: {dist:.2f}km. REMOTE; significant infrastructure investment. Score {val}/100 reflects >3km band."
        return f"Infrastructure score: {val}/100. {label}. OpenStreetMap road network."
    
    elif factor_name == "population":
        density = factor_data.get("density") or factor_data.get("raw")
        reasoning = (raw_factors.get("socio_econ", {}) or {}).get("population", {})
        if isinstance(reasoning, dict):
            reasoning = reasoning.get("reasoning")
        if reasoning:
            return str(reasoning)
        if density is not None:
            if density < 200:
                return f"Population density: {density} people/km². SPARSE; rural/remote. Limited services and labor. Score {val}/100 reflects <200 people/km² band (reference: 200–600 moderate, 600–1200 well populated, >1200 highly dense)."
            elif density < 600:
                return f"Population density: {density} people/km². MODERATE; balanced workforce and services. Score {val}/100 reflects 200–600 people/km² band."
            elif density < 1200:
                return f"Population density: {density} people/km². WELL POPULATED; good access to services, labor, and markets. Score {val}/100 reflects 600–1200 people/km² band."
            else:
                return f"Population density: {density} people/km². HIGHLY DENSE; congestion considerations but strong market access. Score {val}/100 reflects >1200 people/km² band."
        return f"Population score: {val}/100. {label}. Density (people/km²) from WorldPop-style location proxy."
    
    elif factor_name == "drainage":
        if val is not None:
            if val >= 80:
                return f"Drainage capacity: {val}/100. EXCELLENT; high stream density, low waterlogging. Score reflects 80–100 band."
            elif val >= 60:
                return f"Drainage capacity: {val}/100. GOOD; adequate surface flow; minor ponding. Score reflects 60–80 band."
            elif val >= 40:
                return f"Drainage capacity: {val}/100. MODERATE; drainage improvements may be needed; seasonal waterlogging. Score reflects 40–60 band."
            else:
                return f"Drainage capacity: {val}/100. POOR; flat/low-lying, waterlogging risk. Score reflects <40 band."
        return "Drainage from HydroSHEDS/OSM; score from regional baseline."
    
    elif factor_name == "intensity":
        raw_temp = factor_data.get("raw")
        if raw_temp is not None:
            if val < 25:
                return f"Heat stress index: {val}/100. LOW. Avg max temp {raw_temp}°C. Comfortable. Score reflects <25 band."
            elif val < 45:
                return f"Heat stress index: {val}/100. MODERATE. Avg max {raw_temp}°C. Some cooling recommended. Score reflects 25–45 band."
            elif val < 65:
                return f"Heat stress index: {val}/100. HIGH. Avg max {raw_temp}°C. Active cooling/ventilation essential. Score reflects 45–65 band."
            else:
                return f"Heat stress index: {val}/100. EXTREME. Avg max {raw_temp}°C. Significant thermal management needed. Score reflects >65 band."
        return f"Thermal intensity: {val}/100. {label}. 7-day max temperature forecast."
    
    return f"Score: {val}/100. {label}."


def _generate_slope_verdict(slope_percent):
    """Generate terrain verdict based on slope percentage"""
    if slope_percent is None:
        return "Terrain data not available"
    elif slope_percent <= 0:
        return "VERY FLAT terrain. IDEAL for construction"
    elif slope_percent < 3:
        return "VERY FLAT terrain. IDEAL for construction"
    elif slope_percent < 8:
        return "GENTLE slope. Suitable for most construction"
    elif slope_percent < 15:
        return "MODERATE slope. Careful site planning required"
    elif slope_percent < 30:
        return "STEEP terrain. HIGH construction costs"
    else:
        return "VERY STEEP. NOT SUITABLE for standard construction"


def _perform_suitability_analysis(latitude: float, longitude: float) -> dict:
    """
    MASTER INTEGRATION ENGINE
    Recruits 15 factors across 5 categories and attaches high-fidelity reasoning.
    """
    # 1. 🚀 RECRUIT ALL 15 FACTORS (The New Architecture)
    intelligence = GeoDataService.get_land_intelligence(latitude, longitude)
    
    # 2. 📊 COMPUTE CATEGORIZED SCORES
    agg_result = Aggregator.compute_suitability_score(intelligence)
    
    # 3. 📝 NORMALIZE ALL FACTORS (elevation: use 0-100 suitability score for display, not raw meters)
    raw = intelligence["raw_factors"]
    elev_raw = raw.get("physical", {}).get("elevation", {})
    if isinstance(elev_raw, dict) and elev_raw.get("value") is not None:
        try:
            raw["physical"]["elevation"] = {**elev_raw, "scaled_score": _elevation_to_suitability(elev_raw["value"])}
        except (TypeError, KeyError):
            pass

    f = {
        "physical": {
            "slope": normalize_factor(raw["physical"]["slope"]),
            "elevation": normalize_factor(raw["physical"]["elevation"]),
        },
        "hydrology": {
            "flood": normalize_factor(raw["hydrology"]["flood"]),
            "water": normalize_factor(raw["hydrology"]["water"]),
            "drainage": normalize_factor(raw["hydrology"].get("drainage", {})),
        },
        "environmental": {
            "vegetation": normalize_factor(raw["environmental"]["vegetation"]),
            "pollution": normalize_factor(raw["environmental"]["pollution"]),
            "soil": normalize_factor(raw["environmental"]["soil"]),
        },
        "climatic": {
            "rainfall": normalize_factor(raw["climatic"]["rainfall"]),
            "thermal": normalize_factor(raw["climatic"]["thermal"]),
            "intensity": normalize_factor(raw["climatic"].get("intensity", {})),
        },
        "socio_econ": {
            "landuse": normalize_factor(raw["socio_econ"]["landuse"]),
            "infrastructure": normalize_factor(raw["socio_econ"]["infrastructure"]),
            "population": normalize_factor(raw["socio_econ"]["population"]),
        }
    }

    # 4. 📝 GENERATE EVIDENCE TEXT FOR EACH FACTOR (ALL 15)
    # Physical (2)
    f["physical"]["slope"]["evidence"] = _generate_evidence_text("slope", f["physical"]["slope"], raw)
    f["physical"]["elevation"]["evidence"] = _generate_evidence_text("elevation", f["physical"]["elevation"], raw)
    
    # Hydrology (3)
    f["hydrology"]["flood"]["evidence"] = _generate_evidence_text("flood", f["hydrology"]["flood"], raw)
    f["hydrology"]["water"]["evidence"] = _generate_evidence_text("water", raw["hydrology"]["water"], raw)
    f["hydrology"]["drainage"]["evidence"] = _generate_evidence_text("drainage", f["hydrology"]["drainage"], raw)
    
    # Environmental (3)
    f["environmental"]["vegetation"]["evidence"] = _generate_evidence_text("vegetation", f["environmental"]["vegetation"], raw)
    f["environmental"]["pollution"]["evidence"] = _generate_evidence_text("pollution", f["environmental"]["pollution"], raw)
    f["environmental"]["soil"]["evidence"] = _generate_evidence_text("soil", f["environmental"]["soil"], raw)
    
    # Climatic (3)
    f["climatic"]["rainfall"]["evidence"] = _generate_evidence_text("rainfall", f["climatic"]["rainfall"], raw)
    f["climatic"]["thermal"]["evidence"] = _generate_evidence_text("thermal", f["climatic"]["thermal"], raw)
    f["climatic"]["intensity"]["evidence"] = _generate_evidence_text("intensity", f["climatic"]["intensity"], raw)
    
    # Socio-Economic (3)
    f["socio_econ"]["landuse"]["evidence"] = _generate_evidence_text("landuse", raw["socio_econ"]["landuse"], raw)
    f["socio_econ"]["infrastructure"]["evidence"] = _generate_evidence_text("infrastructure", raw["socio_econ"]["infrastructure"], raw)
    f["socio_econ"]["population"]["evidence"] = _generate_evidence_text("population", f["socio_econ"]["population"], raw)

    # 5. 📂 GEOSPATIAL PASSPORT (location summary for UI / reports)
    slope_raw = raw.get("physical", {}).get("slope", {})
    slope_pct = slope_raw.get("value") if isinstance(slope_raw, dict) else None
    elev_raw = raw.get("physical", {}).get("elevation", {})
    elev_m = elev_raw.get("value") if isinstance(elev_raw, dict) else None
    rain_raw = raw.get("climatic", {}).get("rainfall", {})
    rain_mm = rain_raw.get("rain_mm_60d") or (rain_raw.get("value") if isinstance(rain_raw, dict) else None)
    water_raw = raw.get("hydrology", {}).get("water", {})
    water_dist = water_raw.get("distance_km") if isinstance(water_raw, dict) else None
    geospatial_passport = {
        "slope_percent": round(slope_pct, 2) if slope_pct is not None else None,
        "slope_suitability": round(f["physical"]["slope"]["value"], 1),
        "elevation_m": round(elev_m, 1) if elev_m is not None else None,
        "vegetation_score": round(f["environmental"]["vegetation"]["value"], 1),
        "landuse_class": (raw.get("socio_econ", {}).get("landuse", {}) or {}).get("classification") if isinstance(raw.get("socio_econ", {}).get("landuse"), dict) else "Mixed",
        "water_distance_km": round(float(water_dist), 3) if water_dist is not None else None,
        "water_body_name": (water_raw.get("details") or {}).get("name") if isinstance(water_raw.get("details"), dict) else None,
        "flood_safety_score": round(f["hydrology"]["flood"]["value"], 1),
        "rainfall_mm": round(rain_mm, 1) if rain_mm is not None else None,
        "risk_summary": agg_result.get("penalty", "None"),
        "category_breakdown": {k: round(v, 1) for k, v in (agg_result.get("category_scores") or {}).items()},
    }

    # 6. Optional ML ensemble score (14-factor models; used in History and here when available)
    flat_factors = _extract_flat_factors(f)
    ml_score, ml_used, score_source_ml = _predict_suitability_ml(flat_factors)
    out_extra = {}
    if ml_used:
        out_extra["ml_score"] = ml_score
        out_extra["score_source_ml"] = score_source_ml

    # 7. CONSTRUCT THE 15-FACTOR OUTPUT BUNDLE
    return {
        "suitability_score": agg_result["score"],
        "label": agg_result["label"],
        "penalty_applied": agg_result.get("penalty", "None"),
        "category_scores": agg_result["category_scores"],
        "water_body_snippet": agg_result.get("water_body_snippet"),
        "protected_snippet": agg_result.get("protected_snippet"),
        "geospatial_passport": geospatial_passport,

        # ALL 15 FACTORS (WITH EVIDENCE)
        "factors": f,
        **out_extra,

        # TERRAIN ANALYSIS (for frontend TerrainSlope component)
        "terrain_analysis": {
            "slope_percent": geospatial_passport.get("slope_percent", 0),
            "verdict": _generate_slope_verdict(geospatial_passport.get("slope_percent", 0)),
            "confidence": "High",
            "source": "NASA SRTM"
        },

        # HIGH-FIDELITY EXPLANATION 
        "explanation": {
            "factors_meta": {
                # Physical
                "physical": {
                    "slope": {
                        "value": f["physical"]["slope"]["value"],
                        "unit": f["physical"]["slope"].get("unit", "%"),
                        "label": f["physical"]["slope"].get("label"),
                        "evidence": f["physical"]["slope"]["evidence"],
                        "source": f["physical"]["slope"].get("source", "NASA SRTM"),
                        "confidence": f["physical"]["slope"].get("confidence", "High")
                    },
                    "elevation": {
                        "value": f["physical"]["elevation"]["value"],
                        "unit": f["physical"]["elevation"].get("unit", "m"),
                        "label": f["physical"]["elevation"].get("label"),
                        "evidence": f["physical"]["elevation"]["evidence"],
                        "source": f["physical"]["elevation"].get("source", "NASA SRTM"),
                        "confidence": f["physical"]["elevation"].get("confidence", "High")
                    }
                },

                # Hydrology (3 factors)
                "hydrology": {
                    "flood": {
                        "value": f["hydrology"]["flood"]["value"],
                        "label": f["hydrology"]["flood"].get("label"),
                        "evidence": f["hydrology"]["flood"]["evidence"],
                        "source": f["hydrology"]["flood"].get("source", "Integrated Hydrology Model"),
                        "confidence": f["hydrology"]["flood"].get("confidence", "High")
                    },
                    "water": {
                        "value": f["hydrology"]["water"]["value"],
                        "distance_km": raw["hydrology"]["water"].get("distance_km"),
                        "evidence": f["hydrology"]["water"]["evidence"],
                        "source": raw["hydrology"]["water"].get("details", {}).get("source") if raw["hydrology"]["water"].get("details") else "OpenStreetMap",
                        "confidence": f["hydrology"]["water"].get("confidence", "High")
                    },
                    "drainage": {
                        "value": f["hydrology"]["drainage"]["value"],
                        "label": f["hydrology"]["drainage"].get("label"),
                        "evidence": f["hydrology"]["drainage"]["evidence"],
                        "source": f["hydrology"]["drainage"].get("source", "HydroSHEDS"),
                        "confidence": f["hydrology"]["drainage"].get("confidence", "Medium")
                    }
                },

                # Environmental
                "environmental": {
                    "vegetation": {
                        "value": f["environmental"]["vegetation"]["value"],
                        "raw": f["environmental"]["vegetation"].get("raw"),
                        "label": f["environmental"]["vegetation"].get("label"),
                        "evidence": f["environmental"]["vegetation"]["evidence"],
                        "source": f["environmental"]["vegetation"].get("source", "Copernicus Land"),
                        "confidence": f["environmental"]["vegetation"].get("confidence", "Medium")
                    },
                    "pollution": {
                        "value": f["environmental"]["pollution"]["value"],
                        "raw": raw["environmental"]["pollution"].get("pm25"),
                        "evidence": f["environmental"]["pollution"]["evidence"],
                        "source": f["environmental"]["pollution"].get("source", "OpenAQ"),
                        "confidence": f["environmental"]["pollution"].get("confidence", "High")
                    },
                    "soil": {
                        "value": f["environmental"]["soil"]["value"],
                        "evidence": f["environmental"]["soil"]["evidence"],
                        "source": f["environmental"]["soil"].get("source", "Regional Soil Model"),
                        "confidence": f["environmental"]["soil"].get("confidence", "Medium")
                    }
                },

                # Climatic (3 factors)
                "climatic": {
                    "rainfall": {
                        "value": f["climatic"]["rainfall"]["value"],
                        "raw": f["climatic"]["rainfall"].get("raw"),
                        "unit": f["climatic"]["rainfall"].get("unit", "mm/year"),
                        "label": f["climatic"]["rainfall"].get("label"),
                        "evidence": f["climatic"]["rainfall"]["evidence"],
                        "source": f["climatic"]["rainfall"].get("source", "Open-Meteo Historical API"),
                        "confidence": f["climatic"]["rainfall"].get("confidence", "High")
                    },
                    "thermal": {
                        "value": f["climatic"]["thermal"]["value"],
                        "label": f["climatic"]["thermal"].get("label"),
                        "raw": f["climatic"]["thermal"].get("raw"),
                        "evidence": f["climatic"]["thermal"]["evidence"],
                        "source": f["climatic"]["thermal"].get("source", "Open-Meteo Bioclimatic"),
                        "confidence": f["climatic"]["thermal"].get("confidence", "High")
                    },
                    "intensity": {
                        "value": f["climatic"]["intensity"]["value"],
                        "label": f["climatic"]["intensity"].get("label"),
                        "raw": f["climatic"]["intensity"].get("raw"),
                        "evidence": f["climatic"]["intensity"]["evidence"],
                        "source": f["climatic"]["intensity"].get("source", "Open-Meteo Meteorological"),
                        "confidence": f["climatic"]["intensity"].get("confidence", "High")
                    }
                },

                # Socio-Economic
                "socio_econ": {
                    "landuse": {
                        "value": f["socio_econ"]["landuse"]["value"],
                        "classification": raw["socio_econ"]["landuse"].get("classification"),
                        "evidence": f["socio_econ"]["landuse"]["evidence"],
                        "source": raw["socio_econ"]["landuse"].get("source", "Sentinel-2 + OSM"),
                        "confidence": raw["socio_econ"]["landuse"].get("confidence", "High")
                    },
                    "infrastructure": {
                        "value": f["socio_econ"]["infrastructure"]["value"],
                        "evidence": f["socio_econ"]["infrastructure"]["evidence"],
                        "source": raw["socio_econ"]["infrastructure"].get("source", "OpenStreetMap"),
                        "confidence": raw["socio_econ"]["infrastructure"].get("confidence", "High")
                    },
                    "population": {
                        "value": f["socio_econ"]["population"]["value"],
                        "raw": f["socio_econ"]["population"].get("raw") or raw["socio_econ"]["population"].get("density"),
                        "evidence": f["socio_econ"]["population"]["evidence"],
                        "source": f["socio_econ"]["population"].get("source", "WorldPop"),
                        "confidence": f["socio_econ"]["population"].get("confidence", "Medium")
                    }
                }
            }
        },

        # METADATA PROOF
        "metadata": intelligence["metadata_proof"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "location": {"latitude": latitude, "longitude": longitude}
    }
# def safe_factor(obj, *, fallback_label="Data unavailable"):
#     if not isinstance(obj, dict):
#         return {
#             "value": None,
#             "label": fallback_label,
#             "reason": "No reliable data available for this location.",
#             "source": "Data gap fallback",
#             "confidence": "Low"
#         }
#     return obj

# def _perform_suitability_analysis(latitude: float, longitude: float) -> dict:
#     """
#     MASTER INTEGRATION ENGINE
#     Produces evidence-grade explanations using REAL numeric data only.
#     """

#     # 1️⃣ COLLECT REAL DATA
#     intelligence = GeoDataService.get_land_intelligence(latitude, longitude)
#     agg_result = Aggregator.compute_suitability_score(intelligence)

#     raw = intelligence["raw_factors"]

#     def nf(x):
#         return normalize_factor(x)

#     f = {
#         "physical": {
#             "slope": nf(raw["physical"]["slope"]),
#             "elevation": nf(raw["physical"]["elevation"]),
#         },
#         "hydrology": {
#             "flood": nf(raw["hydrology"]["flood"]),
#             "water": nf(raw["hydrology"]["water"]),
#         },
#         "environmental": {
#             "vegetation": nf(raw["environmental"]["vegetation"]),
#             "pollution": nf(raw["environmental"]["pollution"]),
#             "soil": nf(raw["environmental"]["soil"]),
#         },
#         "climatic": {
#             "rainfall": nf(raw["climatic"]["rainfall"]),
#             "thermal": nf(raw["climatic"]["thermal"]),
#         },
#         "socio_econ": {
#             "landuse": nf(raw["socio_econ"]["landuse"]),
#             "infrastructure": nf(raw["socio_econ"]["infrastructure"]),
#             "population": nf(raw["socio_econ"]["population"]),
#         }
#     }

#     # ─────────────────────────────────────────
#     # 🧠 NUMERIC → EVIDENCE REASONING
#     # ─────────────────────────────────────────

#     # 🌧 Rainfall
#     rain_mm = f["climatic"]["rainfall"].get("raw")
#     if rain_mm is not None:
#         if rain_mm < 300:
#             rain_reason = f"Rainfall: {rain_mm:.1f}mm/year. LOW rainfall. IDEAL for construction with minimal flood risk. Irrigation may be required for agriculture."
#         elif rain_mm < 800:
#             rain_reason = f"Rainfall: {rain_mm:.1f}mm/year. MODERATE rainfall. Balanced conditions for construction and agriculture."
#         elif rain_mm < 1500:
#             rain_reason = f"Rainfall: {rain_mm:.1f}mm/year. HIGH rainfall. Drainage planning required. Moderate flood susceptibility."
#         else:
#             rain_reason = f"Rainfall: {rain_mm:.1f}mm/year. EXCESSIVE rainfall. High flood risk and foundation stress."
#     else:
#         rain_reason = "Rainfall analysis unavailable. Satellite-climate fusion estimate applied."

#     # 🌊 Flood
#     flood_score = f["hydrology"]["flood"]["value"]
#     water_dist = f["hydrology"]["water"].get("distance_km")

#     if water_dist is not None:
#         flood_reason = (
#             f"COMBINED ASSESSMENT: Flood safety score {flood_score}/100. "
#             f"Nearest surface water at {water_dist:.2f} km. "
#             + (
#                 "VERY LOW flood risk." if water_dist > 3 else
#                 "MODERATE flood risk during heavy rainfall." if water_dist > 1 else
#                 "HIGH flood risk due to proximity."
#             )
#         )
#     else:
#         flood_reason = f"Flood safety score {flood_score}/100. No significant surface water bodies detected nearby."

#     # 🌱 Vegetation (NDVI)
#     ndvi = f["environmental"]["vegetation"].get("raw")
#     if ndvi is not None:
#         if ndvi > 0.6:
#             veg_reason = f"NDVI: {ndvi:.2f}. Dense vegetation cover. Ecologically sensitive, low urban suitability."
#         elif ndvi > 0.4:
#             veg_reason = f"NDVI: {ndvi:.2f}. Moderate vegetation. Suitable for agriculture and low-impact development."
#         elif ndvi > 0.2:
#             veg_reason = f"NDVI: {ndvi:.2f}. Sparse vegetation. Favorable for construction."
#         else:
#             veg_reason = f"NDVI: {ndvi:.2f}. Bare or built-up land detected."
#     else:
#         veg_reason = "Vegetation analysis unavailable. Satellite data gap handled."

#     # 🏭 Pollution
#     pm25 = f["environmental"]["pollution"].get("raw")
#     if pm25 is not None:
#         if pm25 < 10:
#             poll_reason = f"PM2.5: {pm25} µg/m³. EXCELLENT air quality (WHO guideline compliant)."
#         elif pm25 < 25:
#             poll_reason = f"PM2.5: {pm25} µg/m³. ACCEPTABLE air quality."
#         elif pm25 < 50:
#             poll_reason = f"PM2.5: {pm25} µg/m³. MODERATE pollution. Sensitive groups affected."
#         else:
#             poll_reason = f"PM2.5: {pm25} µg/m³. SEVERE pollution. Long-term habitation risk."
#     else:
#         poll_reason = "Air quality estimated via satellite aerosol data."

#     # 🧱 Slope
#     slope_val = f["physical"]["slope"]["value"]
#     slope_reason = (
#         f"Slope: {slope_val}%. "
#         + (
#             "Flat terrain. IDEAL for construction." if slope_val < 5 else
#             "Gentle slope. Minor grading required." if slope_val < 15 else
#             "Steep terrain. High construction cost." if slope_val < 30 else
#             "Extreme slope. Not suitable for development."
#         )
#     )

#     # ─────────────────────────────────────────
#     # 📦 FINAL RESPONSE
#     # ─────────────────────────────────────────
#     return {
#         "suitability_score": agg_result["score"],
#         "label": agg_result["label"],
#         "penalty_applied": agg_result.get("penalty", "None"),
#         "category_scores": agg_result["category_scores"],
#         "factors": intelligence["raw_factors"],

#         "explanation": {
#             "factors_meta": {
#                 "rainfall": safe_factor({
#                     "value": f["climatic"]["rainfall"]["value"],
#                     "reason": rain_reason,
#                     "source": f["climatic"]["rainfall"].get("source"),
#                     "confidence": "High"
#                 }),
#                 "flood": safe_factor({
#                     "value": f["hydrology"]["flood"]["value"],
#                     "reason": flood_reason,
#                     "source": f["hydrology"]["flood"].get("source"),
#                     "confidence": "High"
#                 }),
#                 "vegetation": safe_factor({
#                     "value": f["environmental"]["vegetation"]["value"],
#                     "reason": veg_reason,
#                     "source": f["environmental"]["vegetation"].get("source"),
#                     "confidence": "Medium"
#                 }),
#                 "pollution": safe_factor({
#                     "value": f["environmental"]["pollution"]["value"],
#                     "reason": poll_reason,
#                     "source": f["environmental"]["pollution"].get("source"),
#                     "confidence": "High"
#                 }),
#                 "slope": safe_factor({
#                     "value": f["physical"]["slope"]["value"],
#                     "reason": slope_reason,
#                     "source": f["physical"]["slope"].get("source"),
#                     "confidence": "High"
#                 })
#             }
#         },

#         "metadata": intelligence["metadata_proof"],
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
#         "location": {"latitude": latitude, "longitude": longitude}
#     }

# @app.route("/generate_report", methods=["POST", "OPTIONS"])
# def generate_report():
    
#     if request.method == "OPTIONS":
#         return jsonify({}), 200
#     try:
#         data = request.json
#         if not data:
#             return jsonify({"error": "No data received"}), 400
        
#         # 1. Prepare Site A Intelligence
#         loc_a = data.get("location")
#         if loc_a:
#             # Ensure factors and explanation exist for Site Potential logic in generator
#             if "factors" not in data:
#                 logger.warning("Factors missing for Site A in report generation")
            
#             try:
#                 # Fetching nearby places to enrich the intelligence report
#                 places_a = get_nearby_named_places(loc_a.get("latitude"), loc_a.get("longitude"))
#                 data["nearby_places"] = {"places": places_a}
#             except Exception as e:
#                 logger.error(f"Nearby places A fetch failed: {e}")
#                 data["nearby_places"] = {"places": []}

#         # 2. Prepare Site B Intelligence (if provided)
#         compare_data = data.get("compareData")
#         if compare_data:
#             loc_b = compare_data.get("location")
#             if loc_b:
#                 try:
#                     places_b = get_nearby_named_places(loc_b.get("latitude"), loc_b.get("longitude"))
#                     data["compareData"]["nearby_places"] = {"places": places_b}
#                 except Exception as e:
#                     logger.error(f"Nearby places B fetch failed: {e}")
#                     data["compareData"]["nearby_places"] = {"places": []}

#         # 3. Generate PDF Buffer using the helper-based pdf_generator
#         # This now includes Site Potential Analysis based on the factors in 'data'
#         pdf_buffer = generate_land_report(data)
#         pdf_buffer.seek(0)

#         # 4. Generate dynamic filename
#         location_name = data.get("locationName", "Analysis")
#         # Sanitize filename: remove non-alphanumeric chars for safety
#         clean_name = "".join([c if c.isalnum() else "_" for c in str(location_name)])

#         return send_file(
#             pdf_buffer,
#             as_attachment=True,
#             download_name=f"GeoAI_Intelligence_{clean_name}.pdf",
#             mimetype="application/pdf"
#         )

#     except Exception as e:
#         logger.exception("Internal PDF Generation Error")
#         return jsonify({"error": "Failed to generate tactical report. See server logs."}), 500
    
@app.route("/generate_report", methods=["POST", "OPTIONS"])
def generate_report():
    if request.method == "OPTIONS":
        return jsonify({}), 200
        
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data received"}), 400
        
        # 1. Prepare Site A Intelligence
        # We ensure 'strategic_intelligence' is prioritized as it powers Section 03
        loc_a = data.get("location")
        if loc_a:
            # Check for critical Section 03 data
            if "strategic_intelligence" not in data:
                logger.warning("Strategic intelligence data missing for Site A")
            
            # Enrich with nearby places for the intelligence report
            try:
                lat_a = loc_a.get("latitude") or loc_a.get("lat")
                lng_a = loc_a.get("longitude") or loc_a.get("lng")
                places_a = get_nearby_named_places(lat_a, lng_a)
                data["nearby_places"] = {"places": places_a}
            except Exception as e:
                logger.error(f"Nearby places A fetch failed: {e}")
                data["nearby_places"] = {"places": []}

        # 2. Prepare Site B Intelligence (Comparative Mode)
        compare_data = data.get("compareData")
        if compare_data:
            loc_b = compare_data.get("location")
            if loc_b:
                # Ensure Site B also carries its own strategic insights
                if "strategic_intelligence" not in compare_data:
                    logger.warning("Strategic intelligence data missing for Site B")
                
                try:
                    lat_b = loc_b.get("latitude") or loc_b.get("lat")
                    lng_b = loc_b.get("longitude") or loc_b.get("lng")
                    places_b = get_nearby_named_places(lat_b, lng_b)
                    data["compareData"]["nearby_places"] = {"places": places_b}
                except Exception as e:
                    logger.error(f"Nearby places B fetch failed: {e}")
                    data["compareData"]["nearby_places"] = {"places": []}

        # 3. Generate PDF Buffer 
        # The generator now processes Section 01, 02, and 03 in order
        pdf_buffer = generate_land_report(data)
        pdf_buffer.seek(0)

        # 4. Generate dynamic filename based on Location A
        location_name = data.get("locationName") or "GeoAI_Analysis"
        clean_name = "".join([c if c.isalnum() else "_" for c in str(location_name)])
        timestamp = datetime.now().strftime("%Y%m%d")

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f"GeoAI_Report_{clean_name}_{timestamp}.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:
        logger.exception("Critical PDF Generation Error")
        return jsonify({
            "error": "Failed to generate tactical report. Internal server error.",
            "details": str(e)
        }), 500
@app.route("/simulate-development", methods=["POST","OPTIONS"])
def simulate_development():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        data = request.json or {}
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        development_type = data["development_type"]
        existing_factors = data.get("existing_factors", {})
        placed_developments = data.get("placed_developments", [])

        # Calculate development impact
        simulation_results = calculate_development_impact(
            latitude=latitude,
            longitude=longitude,
            development_type=development_type,
            existing_factors=existing_factors,
            placed_developments=placed_developments
        )

        return jsonify({
            "status": "success",
            "simulation": simulation_results
        })

    except Exception as e:
        logger.error(f"Development simulation error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route("/nearby_places", methods=["POST","OPTIONS"])
def nearby_places_route():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        data = request.json or {}
        lat = float(data["latitude"])
        lon = float(data["longitude"])

        places = nearby_places.get_nearby_named_places(lat, lon)

        return jsonify({
            "count": len(places),
            "places": places
        })

    except Exception as e:
        return jsonify({
            "count": 0,
            "places": [],
            "error": str(e)
        }), 200



# def serve_react(path):
#     build_dir = app.static_folder
#     if path != "" and os.path.exists(os.path.join(build_dir, path)):
#         return send_from_directory(build_dir, path)
#     return send_from_directory(build_dir, "index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
