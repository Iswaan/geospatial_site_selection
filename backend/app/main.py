import json
import re
import time
import logging
import pathlib
import bisect
import pandas as pd
import numpy as np
import joblib
import shap
import geopandas as gpd
import osmnx as ox
from pyproj import Transformer
from shapely.geometry import Point
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.app.features.feature_engine import compute_features_for_point

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Geospatial Site Selection API")

# Allow frontend to consume the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global memory state
APP_STATE = {}

def format_inr(value: float) -> str:
    """Format a number in the Indian numbering system with Rs symbol.
    Negative values render as -Rs3,867 (minus before symbol, matching
    Intl.NumberFormat('en-IN') browser behavior).
    """
    is_negative = value < 0
    val_str = str(int(round(abs(value))))
    if len(val_str) > 3:
        last_3 = val_str[-3:]
        rest = val_str[:-3]
        rest = re.sub(r'(\d)(?=(\d{2})+(?!\d))', r'\1,', rest)
        formatted = f"{rest},{last_3}"
    else:
        formatted = val_str
    return f"-\u20b9{formatted}" if is_negative else f"\u20b9{formatted}"

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "backend" / "app" / "models"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
GEOJSON_PATH = PROCESSED_DIR / "features.geojson"

WGS84_CRS = "EPSG:4326"
PROJECT_CRS = "EPSG:32643"
# WGS84 → UTM 43N transformer (always_xy=True means lon,lat input order)
_WGS84_TO_UTM = Transformer.from_crs(WGS84_CRS, PROJECT_CRS, always_xy=True)

# Core features used in the baseline revenue formula
_CORE_FEATURES = ["pop_1000m", "wealth_poi_count_1000m", "poi_diversity_1000m", "competitor_count_1000m"]
_FORMULA_WEIGHTS = np.array([0.3, 0.3, 0.2, -0.2])


@app.on_event("startup")
def load_models_and_data():
    log.info("Starting up API, loading models and computing scores...")
    
    # Load Models
    try:
        model = joblib.load(MODEL_DIR / "xgb_model.pkl")
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        z_scaler = joblib.load(MODEL_DIR / "z_scaler.pkl")
        explainer = shap.TreeExplainer(model)
        APP_STATE["model"] = model
        APP_STATE["scaler"] = scaler
        APP_STATE["z_scaler"] = z_scaler
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
    APP_STATE["X_scaled_df"] = X_scaled_df
    
    predicted_scores = model.predict(X_scaled_df)
    
    # Merge and Rank
    scores_list = []
    for i, row in features_df.iterrows():
        site_id = row["site_id"]
        baseline = baseline_df.loc[baseline_df["site_id"] == site_id, "baseline_score"].values[0]
        scores_list.append({
            "site_id": site_id,
            "predicted_score": float(predicted_scores[i]),
            "baseline_score": float(baseline)
        })
        
    scores_list.sort(key=lambda x: x["predicted_score"], reverse=True)
    for rank, item in enumerate(scores_list, start=1):
        item["rank"] = rank
        
    APP_STATE["scores"] = scores_list
    APP_STATE["top_site"] = scores_list[0]
    # Sorted predicted scores for rank-insertion of custom points
    APP_STATE["sorted_scores"] = sorted(s["predicted_score"] for s in scores_list)

    # Training feature ranges — used at request time for OOB detection
    feature_ranges = {
        col: {"min": float(features_df[col].min()), "max": float(features_df[col].max())}
        for col in feature_cols
    }
    APP_STATE["feature_ranges"] = feature_ranges
    log.info("\u2713 Feature ranges cached for %d features (OOB detection)", len(feature_ranges))

    
    # Load spatial datasets for custom-point scoring (projected, cached in RAM)
    def _load_gdf(filename: str) -> gpd.GeoDataFrame:
        path = RAW_DIR / filename
        if not path.exists():
            log.warning("Spatial file not found: %s — custom scoring will be limited", filename)
            return gpd.GeoDataFrame(geometry=[], crs=PROJECT_CRS)
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(WGS84_CRS)
        return gdf.to_crs(PROJECT_CRS)

    APP_STATE["competitors_gdf"] = _load_gdf("osm_competitors.geojson")
    APP_STATE["transit_gdf"]     = _load_gdf("osm_transit.geojson")
    APP_STATE["wealth_gdf"]      = _load_gdf("osm_wealth_pois.geojson")
    APP_STATE["diversity_gdf"]   = _load_gdf("osm_diversity_pois.geojson")
    APP_STATE["pop_raster_path"] = RAW_DIR / "bengaluru_pop_100m.tif"

    # Load road edges
    roads_path = RAW_DIR / "osm_roads.graphml"
    if roads_path.exists():
        G = ox.load_graphml(roads_path)
        edges = ox.graph_to_gdfs(G, nodes=False)
        if edges.crs is None:
            edges = edges.set_crs(WGS84_CRS)
        APP_STATE["edges_gdf"] = edges.to_crs(PROJECT_CRS)
        log.info("✓ Road edges loaded (%d)", len(APP_STATE["edges_gdf"]))
    else:
        log.warning("osm_roads.graphml not found — road density will be 0 for custom points")
        APP_STATE["edges_gdf"] = gpd.GeoDataFrame(geometry=[], crs=PROJECT_CRS)

    # Load validation polygons
    boundary_path = RAW_DIR / "bengaluru_boundary.geojson"
    if boundary_path.exists():
        bdf = gpd.read_file(boundary_path)
        if bdf.crs is None:
            bdf = bdf.set_crs(WGS84_CRS)
        APP_STATE["boundary_gdf"] = bdf.to_crs(PROJECT_CRS)
        log.info("✓ Bengaluru boundary loaded")
    else:
        log.warning("bengaluru_boundary.geojson not found — run save_validation_data.py")
        APP_STATE["boundary_gdf"] = None

    exclusion_path = RAW_DIR / "exclusion_zones.geojson"
    if exclusion_path.exists():
        edf = gpd.read_file(exclusion_path)
        if edf.crs is None:
            edf = edf.set_crs(WGS84_CRS)
        edf = edf.to_crs(PROJECT_CRS)
        # Make geometries valid before union (OSM ways can have self-intersections)
        from shapely.validation import make_valid
        edf["geometry"] = edf["geometry"].apply(lambda g: make_valid(g) if g is not None else g)
        edf = edf[edf.geometry.notna() & ~edf.geometry.is_empty]
        APP_STATE["exclusion_gdf"] = edf
        # Pre-compute the union once at startup — avoids per-request cost.
        # grid_size=0.01 (1 cm in metres) resolves GEOS TopologyException from
        # near-coincident edges in projected OSM polygon geometries.
        import shapely
        APP_STATE["exclusion_union"] = shapely.union_all(
            edf.geometry.values, grid_size=0.01
        )
        log.info("✓ Exclusion zones loaded and union computed (%d polygons)", len(edf))
    else:
        log.warning("exclusion_zones.geojson not found — run save_validation_data.py")
        APP_STATE["exclusion_gdf"] = None
        APP_STATE["exclusion_union"] = None
    
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


class CustomPointRequest(BaseModel):
    lat: float
    lon: float


@app.post("/api/score-custom")
def score_custom_point(req: CustomPointRequest):
    """
    Score an arbitrary lat/lon point using the same model and feature
    pipeline as the 50 pre-computed candidates.

    Validates that the point:
    - Falls within the Bengaluru administrative boundary
    - Does not fall inside an exclusion zone (park/water/military)

    Returns predicted_score, baseline_score, rank-relative-to-50,
    and the full SHAP local breakdown — same shape as /api/shap/{site_id}.
    """
    t0 = time.time()

    # 1. Reproject WGS84 → EPSG:32643
    x_utm, y_utm = _WGS84_TO_UTM.transform(req.lon, req.lat)
    point_utm = Point(x_utm, y_utm)

    # 2. Boundary check
    boundary_gdf = APP_STATE.get("boundary_gdf")
    if boundary_gdf is not None:
        if not boundary_gdf.geometry.iloc[0].contains(point_utm):
            raise HTTPException(
                status_code=422,
                detail="Location is outside the Bengaluru administrative boundary."
            )

    # 3. Exclusion zone check — use pre-computed union from startup
    exclusion_union = APP_STATE.get("exclusion_union")
    if exclusion_union is not None:
        if exclusion_union.contains(point_utm):
            raise HTTPException(
                status_code=422,
                detail="Location falls within an excluded zone (park, water body, or military area)."
            )

    # 4. Compute features
    feat_dict = compute_features_for_point(
        point_utm=point_utm,
        competitors=APP_STATE["competitors_gdf"],
        transit=APP_STATE["transit_gdf"],
        wealth=APP_STATE["wealth_gdf"],
        diversity=APP_STATE["diversity_gdf"],
        edges=APP_STATE["edges_gdf"],
        pop_raster_path=APP_STATE["pop_raster_path"],
    )
    log.info("Feature computation took %.2fs", time.time() - t0)

    # 5a. Out-of-bounds (OOB) detection — flag features outside training range
    feature_ranges = APP_STATE.get("feature_ranges", {})
    oob_features = []
    for feat_name, val in feat_dict.items():
        rng = feature_ranges.get(feat_name)
        if rng is None:
            continue
        if val < rng["min"] or val > rng["max"]:
            side = "high" if val > rng["max"] else "low"
            rng_width = rng["max"] - rng["min"] if rng["max"] != rng["min"] else 1.0
            delta_pct = (
                (val - rng["max"]) / rng_width * 100 if side == "high"
                else (rng["min"] - val) / rng_width * 100
            )
            oob_features.append({
                "feature": feat_name,
                "value": float(val),
                "train_min": rng["min"],
                "train_max": rng["max"],
                "side": side,
                "delta_pct": round(delta_pct, 1),
            })
    if oob_features:
        log.warning(
            "Custom point has %d OOB features: %s",
            len(oob_features),
            ", ".join(f["feature"] for f in oob_features),
        )

    # 5. Align features to trained model column order
    feature_cols = APP_STATE["feature_cols"]
    feat_row = pd.DataFrame([[feat_dict.get(c, 0.0) for c in feature_cols]], columns=feature_cols)

    # 6. Scale and predict
    scaler = APP_STATE["scaler"]
    model = APP_STATE["model"]
    feat_scaled = scaler.transform(feat_row)
    feat_scaled_df = pd.DataFrame(feat_scaled, columns=feature_cols)
    predicted_score = float(model.predict(feat_scaled_df)[0])

    # 7. Baseline score — Z-score core features against the SAME 50-site
    #    distribution the model was trained on (z_scaler fit in build_model.py)
    z_scaler = APP_STATE["z_scaler"]
    core_vals = np.array([[feat_dict.get(c, 0.0) for c in _CORE_FEATURES]])
    z_scores = z_scaler.transform(core_vals)[0]
    baseline_score = float(
        4_000_000 + 800_000 * (
            0.3 * z_scores[0] +   # pop_1000m
            0.3 * z_scores[1] +   # wealth_poi_count_1000m
            0.2 * z_scores[2] -   # poi_diversity_1000m
            0.2 * z_scores[3]     # competitor_count_1000m
        )
    )

    # 8. SHAP
    explainer = APP_STATE["explainer"]
    shap_vals = explainer.shap_values(feat_scaled_df)[0]
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(expected_value[0])
    else:
        expected_value = float(expected_value)

    shap_sum = expected_value + float(sum(shap_vals))
    assert abs(shap_sum - predicted_score) < 100.0, \
        f"SHAP math error for custom point: {shap_sum} != {predicted_score}"

    contributions = {feat: float(val) for feat, val in zip(feature_cols, shap_vals)}
    sorted_contributions = dict(
        sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    # 9. Rank — bisect into the sorted 50-score list (ascending)
    sorted_scores = APP_STATE["sorted_scores"]  # ascending list of 50 scores
    # Number of existing sites with score >= predicted_score gives rank-1
    rank = len(sorted_scores) - bisect.bisect_left(sorted_scores, predicted_score) + 1
    rank = max(1, min(rank, 51))  # clamp to valid range

    log.info(
        "Custom point (%.4f, %.4f): predicted=%.0f baseline=%.0f rank=%d/51 (%.2fs total)",
        req.lat, req.lon, predicted_score, baseline_score, rank, time.time() - t0
    )

    return {
        "site_id": "custom",
        "lat": req.lat,
        "lon": req.lon,
        "predicted_score": predicted_score,
        "baseline_score": baseline_score,
        "rank": rank,
        "total_candidates": 51,
        "oob_features": oob_features,  # empty list = all features in training range
        "shap": {
            "base_value": expected_value,
            "features": sorted_contributions,
        },
        "features": feat_dict,
    }


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
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(expected_value[0])
    else:
        expected_value = float(expected_value)
        
    # Brief Section 5 requirement: Assert SHAP sum consistency
    site_score_info = next((item for item in APP_STATE["scores"] if item["site_id"] == site_id), None)
    if site_score_info:
        predicted_score = site_score_info["predicted_score"]
        shap_sum = expected_value + sum(shap_vals)
        assert abs(shap_sum - predicted_score) < 100.0, f"SHAP math error: {shap_sum} != {predicted_score}"
    
    feature_cols = APP_STATE["feature_cols"]
    
    # Zip together and sort by absolute magnitude
    contributions = {feat: float(val) for feat, val in zip(feature_cols, shap_vals)}
    sorted_contributions = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))
    
    return {
        "base_value": expected_value,
        "features": sorted_contributions
    }


@app.get("/api/summary")
def get_summary(site_id: str = None):
    """
    Returns an auto-generated executive summary for a given site.
    Defaults to the #1 ranked site if site_id is not provided.
    """
    if site_id:
        # Find the requested site in scores
        site_info = next((s for s in APP_STATE.get("scores", []) if s["site_id"] == site_id), None)
        if not site_info:
            raise HTTPException(status_code=404, detail=f"Site '{site_id}' not found.")
        target_site = site_info
    else:
        target_site = APP_STATE.get("top_site")
    
    if not target_site:
        raise HTTPException(status_code=500, detail="Rankings not loaded.")
        
    site_id = target_site["site_id"]
    predicted_score = target_site["predicted_score"]
    rank = target_site["rank"]
    
    # Call the SHAP logic internally for the top site
    shap_response = get_shap_values(site_id)
    shap_features = shap_response["features"]
    
    # Find top positive (driver) and top negative (detractor)
    positives = {k: v for k, v in shap_features.items() if v > 0}
    negatives = {k: v for k, v in shap_features.items() if v < 0}
    
    top_driver = max(positives.items(), key=lambda x: x[1])[0] if positives else "overall_positive"
    
    display_labels = {
        "pop_1000m": "population density within 1km",
        "wealth_poi_count_1000m": "premium amenities and wealth proxies within 1km",
        "poi_diversity_1000m": "retail and mixed-use diversity within 1km",
        "competitor_count_1000m": "direct competitor density within 1km",
        "nearest_competitor_dist_m": "proximity to the nearest competitor",
        "nearest_transit_dist_m": "proximity to transit nodes",
        "transit_stop_count_1000m": "local transit stops within 1km",
        "road_density_1000m": "drivable road network density within 1km",
        "pop_3000m": "broader population catchment (3km)",
        "competitor_count_3000m": "broader competitor cannibalization (3km)",
        "transit_stop_count_3000m": "broader transit connectivity (3km)",
        "wealth_poi_count_3000m": "broader premium amenity presence (3km)",
        "poi_diversity_3000m": "broader commercial diversity (3km)",
        "road_density_3000m": "broader road accessibility (3km)",
        "overall_positive": "overall spatial viability",
        "overall_negative": "any significant spatial penalties"
    }
    
    driver_text = display_labels.get(top_driver, top_driver)
    
    # Materiality threshold: 3% of total absolute impact
    total_abs_impact = sum(abs(v) for v in shap_features.values())
    
    if negatives:
        top_detractor, top_detractor_val = min(negatives.items(), key=lambda x: x[1])
        if abs(top_detractor_val) > 0.03 * total_abs_impact:
            detractor_text = display_labels.get(top_detractor, top_detractor)
            detractor_clause = f", though it sees a modest offset from {detractor_text}."
        else:
            detractor_clause = " with no significant spatial detractors identified."
    else:
        detractor_clause = " with no significant spatial detractors identified."
    
    # Format the numbers nicely
    score_formatted = format_inr(predicted_score)
    
    # Tailor the opening sentence based on rank
    if rank == 1:
        opening = f"{site_id} is the top recommended location"
    else:
        opening = f"{site_id} ranks #{rank} among all candidates"
    
    summary_text = (
        f"{opening} with a projected monthly revenue of {score_formatted}. "
        f"Its primary advantage is strong {driver_text}{detractor_clause}"
    )
    
    return {"summary": summary_text}

@app.get("/api/model-comparison")
def get_model_comparison():
    comp_csv_path = PROCESSED_DIR / "model_comparison.csv"
    note_json_path = PROCESSED_DIR / "model_comparison_note.json"
    
    if not comp_csv_path.exists() or not note_json_path.exists():
        raise HTTPException(status_code=404, detail="Model comparison data not found.")
        
    import pandas as pd
    comp_df = pd.read_csv(comp_csv_path)
    models = comp_df.to_dict(orient="records")
    
    import json
    with open(note_json_path, "r") as f:
        note_data = json.load(f)
        
    return {
        "models": models,
        "note": note_data
    }
