import json
import logging
import pathlib
import pandas as pd
import numpy as np
import joblib
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Geospatial Site Selection API")

# Allow frontend to consume the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global memory state
APP_STATE = {}

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "backend" / "app" / "models"
GEOJSON_PATH = PROCESSED_DIR / "features.geojson"

@app.on_event("startup")
def load_models_and_data():
    log.info("Starting up API, loading models and computing scores...")
    
    # Load Models
    try:
        model = joblib.load(MODEL_DIR / "xgb_model.pkl")
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        explainer = shap.TreeExplainer(model)
        APP_STATE["model"] = model
        APP_STATE["scaler"] = scaler
        APP_STATE["explainer"] = explainer
    except Exception as e:
        log.error("Failed to load models: %s", e)
        raise e

    # Load Data
    try:
        features_df = pd.read_csv(PROCESSED_DIR / "features.csv")
        baseline_df = pd.read_csv(PROCESSED_DIR / "baseline_scores.csv")
    except Exception as e:
        log.error("Failed to load feature tables: %s", e)
        raise e

    feature_cols = [c for c in features_df.columns if c not in ["site_id", "synthetic_revenue"]]
    APP_STATE["feature_cols"] = feature_cols
    APP_STATE["features_df"] = features_df
    
    # Pre-compute predictions for all 50 sites
    X_raw = features_df[feature_cols]
    X_scaled = scaler.transform(X_raw)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    APP_STATE["X_scaled_df"] = X_scaled_df # Cache scaled DF for SHAP later
    
    predicted_scores = model.predict(X_scaled_df)
    
    # Merge and Rank
    scores_list = []
    for i, row in features_df.iterrows():
        site_id = row["site_id"]
        # Find baseline score (in case order is mismatched, match on site_id)
        baseline = baseline_df.loc[baseline_df["site_id"] == site_id, "baseline_score"].values[0]
        
        scores_list.append({
            "site_id": site_id,
            "predicted_score": float(predicted_scores[i]),
            "baseline_score": float(baseline)
        })
        
    # Sort descending by predicted_score
    scores_list.sort(key=lambda x: x["predicted_score"], reverse=True)
    
    # Assign Rank
    for rank, item in enumerate(scores_list, start=1):
        item["rank"] = rank
        
    APP_STATE["scores"] = scores_list
    APP_STATE["top_site"] = scores_list[0]
    
    log.info("✓ Startup complete. Ranked %d sites.", len(scores_list))


@app.get("/api/candidates")
def get_candidates():
    """
    Returns the full GeoJSON FeatureCollection of candidate sites, 
    pre-projected to EPSG:4326 (WGS84 lat/lon) for map rendering.
    """
    if not GEOJSON_PATH.exists():
        raise HTTPException(status_code=404, detail="features.geojson not found")
    return FileResponse(GEOJSON_PATH, media_type="application/json")


@app.get("/api/scores")
def get_scores():
    """
    Returns the ranked list of candidate sites. 
    Primary sorting and ranking is based on the ML predicted_score.
    baseline_score is included for fallback/comparison.
    """
    if "scores" not in APP_STATE:
        raise HTTPException(status_code=500, detail="API not initialized correctly.")
    return APP_STATE["scores"]


@app.get("/api/shap/{site_id}")
def get_shap_values(site_id: str):
    """
    Returns the local SHAP feature contributions for a single site.
    Values are generated on-the-fly using the globally loaded TreeExplainer.
    """
    features_df = APP_STATE.get("features_df")
    if features_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded.")
        
    idx = features_df.index[features_df["site_id"] == site_id].tolist()
    if not idx:
        raise HTTPException(status_code=404, detail=f"Site {site_id} not found.")
        
    row_idx = idx[0]
    X_scaled_df = APP_STATE["X_scaled_df"]
    
    # Get the single scaled row
    single_row = X_scaled_df.iloc[[row_idx]]
    
    explainer = APP_STATE["explainer"]
    # TreeExplainer on a single row returns a single set of SHAP values
    shap_vals = explainer.shap_values(single_row)[0]
    
    feature_cols = APP_STATE["feature_cols"]
    
    # Zip together and sort by absolute magnitude
    contributions = {feat: float(val) for feat, val in zip(feature_cols, shap_vals)}
    sorted_contributions = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))
    
    return sorted_contributions


@app.get("/api/summary")
def get_summary():
    """
    Returns an auto-generated executive summary text based on the #1 ranked site.
    Uses its local SHAP values to explain the top driver and detractor.
    """
    top_site = APP_STATE.get("top_site")
    if not top_site:
        raise HTTPException(status_code=500, detail="Rankings not loaded.")
        
    site_id = top_site["site_id"]
    predicted_score = top_site["predicted_score"]
    
    # Call the SHAP logic internally for the top site
    shap_dict = get_shap_values(site_id)
    
    # Find top positive (driver) and top negative (detractor)
    positives = {k: v for k, v in shap_dict.items() if v > 0}
    negatives = {k: v for k, v in shap_dict.items() if v < 0}
    
    top_driver = max(positives.items(), key=lambda x: x[1])[0] if positives else "overall spatial viability"
    top_detractor = min(negatives.items(), key=lambda x: x[1])[0] if negatives else "any significant spatial penalties"
    
    # Format the numbers nicely
    score_formatted = f"${predicted_score:,.0f}"
    
    summary_text = (
        f"The top recommended location is {site_id} with a projected monthly revenue of {score_formatted}. "
        f"Its primary advantage is strong {top_driver}, though it faces minor cannibalization/penalties from {top_detractor}."
    )
    
    return {"summary": summary_text}
