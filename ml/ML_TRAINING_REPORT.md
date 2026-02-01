# GeoAI ML & Training Report

This document explains **how training works**, **where ML is used** in the app, **what models are used and why**, and **what the terminal accuracies mean**.

---

## 1. Purpose of Training

**What we predict:** A single **land suitability score** (0–100) for a location, given 14 geo factors (slope, elevation, flood, water, drainage, vegetation, pollution, soil, rainfall, thermal, intensity, landuse, infrastructure, population).

**Why we train:** The main app already computes suitability with a **rule-based engine** (Aggregator + GeoDataService). The ML models learn to **reproduce that same logic** from synthetic data, so we can:

- Use an **ensemble of models** for a second opinion (ML score) alongside the rule-based score.
- Use ML in **History Analysis** to estimate past suitability from reconstructed past factors, without re-running the full rule engine for every timeline.

**Important:** The app works **without** any trained models. If no `.pkl` files are present, it uses only the rule-based score everywhere.

---

## 2. Where ML Is Used (Not Only History)

| Place in app | What happens when models are loaded |
|--------------|-------------------------------------|
| **Main Suitability** | Response gets extra fields: `ml_score` (0–100) and `score_source_ml` (e.g. `"Ensemble (rf, xgboost, gbm, et)"`). The primary score is still rule-based `suitability_score`; `ml_score` is an ML view. |
| **History Analysis** | For each timeline (1W, 1M, 1Y, 10Y), the **past suitability score** (`p_score`) is computed with the **ML ensemble** when any model is loaded. Each timeline also gets `score_source` (e.g. `"Ensemble (rf, xgboost, gbm, et)"` or `"Rule-based (5 categories)"`). |

So: **training is used in both main suitability (as an extra ML score) and in history (for past scores)**. It is not only for history.

---

## 3. The 14 Factors (Same as the App)

Training uses the **exact same 14 factors** as the app’s aggregator, in this order:

| Category | Factors |
|----------|---------|
| Physical | slope, elevation |
| Environmental | vegetation, pollution, soil |
| Hydrology | flood, water, drainage |
| Climatic | rainfall, thermal, intensity |
| Socio-economic | landuse, infrastructure, population |

Each value is a 0–100 suitability-style score. The app builds this 14-dimensional vector from the nested factor structure (`_extract_flat_factors` in `app.py`) and passes it to the ML predictor.

---

## 4. How the Label (Target) Is Computed

We do **not** have real labeled data from the field. Labels are **formula-derived** so they match the app’s rule-based score:

1. **Five category scores** (same formula as the Aggregator):
   - Physical = (slope + elevation) / 2  
   - Environmental = (vegetation + soil + pollution) / 3  
   - Hydrology = (water + drainage) / 2  
   - Climatic = (rainfall + thermal) / 2  
   - Socio-economic = (infrastructure + landuse + population) / 3  

2. **Base score** = 0.2 × (sum of the five category scores).

3. **Penalties** (same logic as the app):
   - If water ≤ 5 → cap score at 12 (water body).
   - If flood &lt; 40 → multiply score by 0.5 (flood hazard).
   - If landuse ≤ 20 → cap score at 20 (protected/forest).

So the model is trained to **approximate the rule-based engine** on synthetic data. That keeps the ML score aligned with the app’s behaviour.

---

## 5. Data: Synthetic Indian-Location Data

- **Source:** `generate_indian_dataset_14()` in `train_model.py`.
- **No real lat/lng or real labels:** We generate random 14-factor vectors with distributions that look “Indian-style” (e.g. ~12% on-water, realistic ranges for slope, flood, soil, etc.).
- **Size:** Default 10,000 samples; override with `--samples` (e.g. `--samples 5000`).
- **Split:** 80% train, 20% test (configurable with `--test-size`).
- **Reproducibility:** Fixed seeds (e.g. 42 for train/test split, 123 for report-accuracy) so runs are reproducible.

---

## 6. Models Used and Why Each Is Used

We train **several regressors** and use their **average** as the ML score. Each model type brings different strengths; averaging reduces variance and often improves robustness.

| Model | File | Why we use it |
|-------|------|----------------|
| **Random Forest** | `model_rf.pkl` | Robust, handles non-linearity and interactions well, less overfitting than a single tree. Good baseline. |
| **XGBoost** | `model_xgboost.pkl` | Strong performance on tabular data, fast, handles missing values and regularization. Industry standard for this kind of task. |
| **Gradient Boosting** | `model_gbm.pkl` | sklearn’s boosting; builds trees sequentially to correct errors. Often very accurate on formula-like targets. |
| **Extra Trees** | `model_et.pkl` | Like RF but with random splits; more randomness can reduce overfitting and complement RF. |
| **LightGBM** | `model_lgbm.pkl` | Optional (`pip install lightgbm`). Fast, memory-efficient boosting; good for larger data. Skipped if not installed. |

**Ensemble:** The app loads every `model_*.pkl` it finds, runs the same 14-factor vector through each, and uses the **mean** of their predictions as `ml_score` (and for history `p_score` when ML is used). So “all models we used” = all that were trained and saved; each is used in that single ensemble.

---

## 7. How Training Works (Step by Step)

1. **Generate data**  
   `generate_indian_dataset_14(n_samples, seed)` → feature matrix `X` (14 columns), label vector `y` (formula-derived 0–100).

2. **Split**  
   `train_test_split(X, y, test_size=0.2, random_state=seed)` → `X_train`, `X_test`, `y_train`, `y_test`.

3. **Train each model**  
   For each of RF, XGBoost, GBM, Extra Trees, (and LightGBM if available):
   - `model.fit(X_train, y_train)`
   - Predict on **Train** and **Test**
   - Compute **MAE**, **RMSE**, **R²** for Train and Test
   - **Print** those metrics in the terminal (see below)
   - **Save** the model to `backend/ml/models/<model_*.pkl>`

4. **Terminal output**  
   You see lines like:
   - `Random Forest Train: MAE=... RMSE=... R2=...`
   - `Random Forest Test: MAE=... RMSE=... R2=...`
   - Same for XGBoost, Gradient Boosting, Extra Trees, (LightGBM if trained).
   - `Saved: .../models/model_rf.pkl` (and similarly for others).

So: **training and testing accuracies printed in the terminal are for each model, on the same 80/20 split**: Train = performance on the training set, Test = performance on the held-out 20% to check generalization.

---

## 8. What the Accuracy Metrics Mean

- **MAE (Mean Absolute Error):** Average absolute difference between predicted and formula-derived score. Lower is better (e.g. MAE=1.5 → predictions are off by ~1.5 points on average).
- **RMSE (Root Mean Squared Error):** Penalizes large errors more. Lower is better.
- **R² (R-squared):** Fraction of variance in the target explained by the model. Closer to 1.0 is better (e.g. R2=0.99 → model explains 99% of the variance).

High R² and low MAE/RMSE on **Test** mean the model generalizes well to unseen synthetic samples (which follow the same formula). They do **not** measure performance on real-world labels, because we don’t have those yet.

---

## 9. Report-Accuracy Mode (No Retraining)

- **Command:** `python backend/ml/train_model.py --report-accuracy`
- **What it does:**  
  - Does **not** train.  
  - Loads all saved `model_*.pkl` from `backend/ml/models/`.  
  - Generates a **new** synthetic test set (default 2000 samples, different seed 123).  
  - Runs each loaded model on this set and prints **MAE**, **RMSE**, **R²** per model, then an “Accuracy summary” block.

So the accuracies you see in report-accuracy mode are **testing accuracies on a holdout synthetic set**, for whatever models are currently saved. Useful to check models after loading or after copying `backend/ml/models/` to another machine.

---

## 10. How the App Uses the Models

- **On startup:** `app.py` scans `backend/ml/models/` for `model_rf.pkl`, `model_xgboost.pkl`, `model_gbm.pkl`, `model_et.pkl`, `model_lgbm.pkl`. Every file present is loaded into `ML_MODELS` (dict: filename → model). Missing or broken files are skipped; the app does not crash.
- **When predicting:**  
  - Build 14-factor vector from current (or past) factors in the **same order** as `FACTOR_ORDER` in `train_model.py`.  
  - For each loaded model, `model.predict(feature_vector)`.  
  - **Ensemble score** = mean of those predictions, clamped to 0–100.  
  - `score_source_ml` / `score_source` = string like `"Ensemble (rf, xgboost, gbm, et)"` listing which models contributed.

So: **all models we used** = all that were trained, saved, and successfully loaded; each is used in that single ensemble for both main suitability (`ml_score`) and history (`p_score` when ML is used).

---

## 11. Summary Table

| Question | Answer |
|----------|--------|
| Is training only for history? | No. ML is used in **main suitability** (extra `ml_score`) and in **history** (past score per timeline). |
| What are the terminal accuracies? | **Train** and **Test** MAE, RMSE, R² for each model on the 80/20 split during training; in report-accuracy mode, Test MAE/RMSE/R² on a separate holdout set. |
| What models are used? | Random Forest, XGBoost, Gradient Boosting, Extra Trees, and optionally LightGBM. |
| Why each model? | RF: robust baseline. XGBoost/GBM: strong tabular performance. Extra Trees: complementary to RF. LightGBM: optional, fast. Together they form one ensemble. |
| Where is the training script? | `backend/ml/train_model.py`. Models are saved under `backend/ml/models/`. |

---

## 12. Commands Quick Reference

```bash
# From project root (GeoAI)
python backend/ml/train_model.py                    # Train all models (default 10k samples), print Train/Test accuracies
python backend/ml/train_model.py --samples 5000    # Train with 5k samples
python backend/ml/train_model.py --report-accuracy  # Load saved models, print accuracies on fresh test set (no training)
```

Training prints **training and testing metrics** (MAE, RMSE, R²) in the terminal for every model. Report-accuracy prints **test-set metrics only** for every model it loads.

---

## 13. All commands (in order)

Run these from the **project root** (folder `GeoAI`):

| Step | Command | What it does |
|------|---------|--------------|
| 1. Train | `python backend/ml/train_model.py` | Generates synthetic data, trains all models (RF, XGBoost, GBM, Extra Trees, LightGBM if installed), prints **Train** and **Test** MAE/RMSE/R² per model, saves `.pkl` files to `backend/ml/models/`. |
| 2. (Optional) Train with fewer/more samples | `python backend/ml/train_model.py --samples 5000` | Same as step 1 but with 5000 samples. Use `--samples 10000` (default) or any number. |
| 3. Check accuracy later (no retraining) | `python backend/ml/train_model.py --report-accuracy` | Loads all saved models, generates a **new** test set (2000 samples), prints **Test** MAE/RMSE/R² per model and an "Accuracy summary" block. Use this to verify models after copying or without retraining. |

**Word list (copy-paste):**
- `python backend/ml/train_model.py`
- `python backend/ml/train_model.py --samples 5000`
- `python backend/ml/train_model.py --report-accuracy`

---

## 14. Are we using “accuracy” or something else? Why?

We **do not** use the metric **accuracy**. We use **MAE**, **RMSE**, and **R²**.

- **Accuracy** = fraction of correct **class** predictions (e.g. “suitable” vs “not suitable”). It is for **classification** (discrete labels).
- Our task is **regression**: we predict a **continuous score** (0–100). So we use **regression metrics**, not accuracy.

| Metric | What it is | Why we use it |
|--------|------------|----------------|
| **MAE** (Mean Absolute Error) | Average of \|predicted − actual\| over all samples. | Easy to read: “average error in points” (e.g. MAE=1.5 → off by ~1.5 on average). Same unit as the score (0–100). |
| **RMSE** (Root Mean Squared Error) | Square root of average of (predicted − actual)². | Penalizes large errors more than MAE. Good for spotting outliers and model stability. |
| **R²** (R-squared) | Fraction of variance in the target explained by the model (0–1, can go negative if worse than mean). | “How much of the variation in scores does the model explain?” High R² (e.g. 0.99) = model fits the (synthetic) data very well. |

**Why only these (and not accuracy)?**

- Suitability is a **number** (0–100), not a category. So we measure **how close** predictions are to the true score (MAE, RMSE) and **how much variance** we explain (R²).
- Accuracy would require turning the score into classes (e.g. “suitable if &gt; 60”) and then counting correct classes; that loses information and is not what we optimize or report.
