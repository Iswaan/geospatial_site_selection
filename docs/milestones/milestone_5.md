# Milestone 5: The Backend API

## What Was Accomplished
We built a FastAPI backend that serves our spatial models and dataset directly to the frontend. Rather than just wrapping the pre-computed CSV files, the API genuinely operationalizes the ML pipeline by loading the pickled XGBoost model, Scaler, and SHAP Explainer into memory.

### Endpoint Architecture & Engineering Decisions

#### 1. Global In-Memory State
To maximize request throughput and guarantee consistency, all ML artifacts (`xgb_model.pkl`, `scaler.pkl`, `shap.TreeExplainer`) are loaded precisely once during the `@app.on_event("startup")` phase. 
- The `predicted_score` for all 50 candidate sites is computed at startup, ranked, and cached. This prevents identical requests from wastefully re-running the ML predict step on static geometry.

#### 2. `GET /api/candidates`
Directly serves the `features.geojson` file. 
- **Coordinate Reference System:** We verified this GeoJSON is correctly projected in **EPSG:4326** (WGS84 lat/lon degrees), allowing the frontend mapping library (React-Leaflet) to render the sites natively without manual client-side reprojection.

#### 3. `GET /api/scores`
Returns the ranked candidate list. 
- **Ranking Basis:** The app "eats its own dogfood" by using the ML `predicted_score` to determine the 1-to-50 rank, not the noiseless baseline formula. 
- The `baseline_score` is also provided in the payload for transparency and business-logic comparison.

#### 4. `GET /api/shap/{site_id}`
Returns local, site-specific explainability.
- Instead of returning pre-computed global averages, this endpoint grabs the exact feature row for the requested `site_id`, runs it through the loaded `StandardScaler`, and calls `.shap_values()` on the `TreeExplainer` on-the-fly. This guarantees precision local explainability.

#### 5. `GET /api/summary`
Dynamically generates an executive summary.
- The API identifies the #1 ranked site based on the ML predictions. It then internally calls the SHAP logic to find that specific site's top positive feature (driver) and top negative feature (detractor), formatting them into a human-readable string.
- *Example output:* "The top recommended location is site_031 with a projected monthly revenue of $63,542. Its primary advantage is strong poi_diversity_1000m, though it faces minor cannibalization/penalties from road_density_3000m."
