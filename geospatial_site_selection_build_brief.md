# Geospatial Store Site Selection Optimization — Antigravity Build Brief

**Owner:** Ishaan
**Goal:** Explainable ML model that ranks candidate retail locations, validated with leave-one-store-out CV, delivered as an interactive map + executive summary. Built to read as both a Data Science and Business Analyst project.

**Status: FINALIZED — all decisions locked, $0 cost.**

| Decision | Choice | Why |
|---|---|---|
| City | **Bengaluru** | Highest OSM POI density in India, local relevance for interviews, easy to sanity-check results yourself |
| Demographics source | **WorldPop gridded population** (worldpop.org) — free, no key, global coverage, ~100m resolution | US Census API only covers US geography, ruled out |
| Income proxy | **OSM POI category mix** (density of premium retail/cafe/etc. tags) as a wealth proxy, since granular income data isn't freely available for Indian cities | Documented as a proxy, not real income — stated explicitly in the exec summary to keep the claim honest |
| Competitors / transit / roads | **OSM Overpass + OSMnx** | Free, no key |
| Target variable (for the regression) | **Synthetic, formula-derived** — documented in `data/synthetic_target_rationale.md` | No free real store-revenue-by-location dataset exists for Bengaluru; synthetic is stated openly rather than disguised as real |
| Frontend | **Static Folium HTML export for v1**, optional Next.js dashboard as v2 stretch goal | Faster to a shippable deliverable; reuses your Next.js portfolio stack later if you want to extend it |
| All APIs/tools | **100% free tier, no card required** | Census key (free, US-only, unused here), Overpass (free), OSMnx (free), WorldPop (free download), scikit-learn/SHAP/geopandas/folium (open source) |

---

## 0. Prompt block for Antigravity (paste this as the initial agent instruction)

```
Build a full-stack geospatial site-selection analytics project in this repo.
Backend: Python (FastAPI) exposing data pipeline + model endpoints.
Frontend: Next.js dashboard (interactive Folium/Leaflet-style map via react-leaflet)
showing ranked candidate sites, SHAP feature importances, and an executive
summary panel. Follow the folder structure, data contracts, and milestones
in geospatial_site_selection_build_brief.md exactly. Implement in the
milestone order given. After each milestone, run the validation checks
listed before moving to the next. Do not fabricate data — if an API key
or dataset is missing, stub it clearly and flag it in a TODO, don't
silently mock realistic-looking numbers into the final deliverable.
```

---

## 1. Objective & Framing

**Business question:** "Given N candidate locations in a target city, which should we open next, and why?"

**Technical question:** Can we predict a location-performance score from demographic + competitive + accessibility features, explain which factors drive it, and validate the model doesn't just memorize existing stores?

**North star deliverable:** One command (`make run` or `npm run dev` + `uvicorn app:app`) that produces:
1. An interactive map with candidate sites color-coded by predicted score
2. A SHAP feature-importance chart
3. A one-page executive recommendation (auto-generated markdown/PDF)

---

## 2. Data Sources (all free, no API keys requiring payment)

| Source | Purpose | Access | Cost |
|---|---|---|---|
| **WorldPop** (worldpop.org, "India 100m population density") | Population per grid cell, aggregated into candidate-site buffers | Direct GeoTIFF download, no signup | Free |
| **OpenStreetMap Overpass API** | Competitor locations (`shop=*`), transit stops (`public_transport=stop`), amenity mix | Free, no key, rate-limited — cache every response locally, don't re-query | Free |
| **OSMnx** | Street network for road density / accessibility | pip install, pulls from OSM | Free |
| **Synthetic target variable** | Ground truth for the regression, since no free real Bengaluru store-revenue dataset exists | Generated from a documented formula combining population density, competitor scarcity, and transit access + noise | Free — but must be labeled synthetic everywhere it appears (README, exec summary, CV bullet) |

No paid API, no credit card, no rate-limit tier purchases anywhere in this pipeline.

---

## 3. Repo Structure

```
geo-site-selection/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI entrypoint
│   │   ├── data/
│   │   │   ├── census_client.py    # demographic pull
│   │   │   ├── osm_client.py       # Overpass queries (competitors, transit, roads)
│   │   │   └── candidate_sites.py  # generates/loads candidate lat-long grid
│   │   ├── features/
│   │   │   └── build_features.py   # radius-buffer feature engineering (geopandas)
│   │   ├── models/
│   │   │   ├── train.py            # GBM/RF regression + LOSO CV
│   │   │   ├── explain.py          # SHAP values
│   │   │   └── artifacts/          # saved model.pkl, feature_list.json
│   │   └── api/
│   │       ├── routes_sites.py     # GET /candidates, /scores
│   │       └── routes_explain.py   # GET /shap/{site_id}
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── page.tsx                 # map + ranking table
│   │   └── components/
│   │       ├── SiteMap.tsx          # react-leaflet map
│   │       ├── ShapChart.tsx        # recharts bar chart
│   │       └── ExecSummary.tsx
│   └── package.json
├── notebooks/
│   └── 01_eda_and_baseline.ipynb   # exploratory + weighted-score baseline
├── reports/
│   └── executive_summary_template.md
├── data/
│   └── synthetic_target_rationale.md  # only if synthetic target used
└── geospatial_site_selection_build_brief.md
```

---

## 4. Feature Engineering Spec

For each candidate site (lat, lon), compute within configurable radii (e.g. 1km, 3km, 5km):

| Feature | Source | Computation |
|---|---|---|
| `pop_within_r` | Census | Sum population of tracts intersecting buffer |
| `median_income_within_r` | Census | Population-weighted average |
| `competitor_count_within_r` | OSM (shop=* tags matching category) | Point-in-buffer count |
| `nearest_competitor_dist_m` | OSM | Min haversine distance |
| `transit_stop_count_within_r` | OSM (public_transport=stop) | Point-in-buffer count |
| `nearest_transit_dist_m` | OSM | Min distance |
| `road_density_within_r` | OSMnx | Total edge length / area |
| `poi_diversity_within_r` | OSM | Count of distinct amenity/shop categories (proxy for foot traffic mix) |

Implementation: `geopandas` with projected CRS (not raw lat/lon) for accurate distance buffers — use a local UTM zone, not EPSG:32643, for all buffer math.

---

## 5. Modeling Approach

**Baseline (BA-credibility, ship first):** Weighted linear score — document the weights as a business assumption, not a fitted model. This is your fallback deliverable if the ML target turns out weak.

**Primary model (DS-credibility):**
- Gradient Boosting Regressor (or RF) predicting the target variable from the feature table
- **Leave-One-Store-Out CV** across existing/known stores — same rigor pattern as your causal discovery LOSO work, different domain
- Report CV R² / RMSE per fold, not just mean — show the spread honestly
- **SHAP** (TreeExplainer) for global feature importance + per-candidate-site local explanation ("Site A scored high mainly due to low competitor density and high transit access")

**Validation guardrail:** if LOSO R² is weak (<0.2 or so, depending on n), don't force a misleading regression narrative — pivot the framing to "explainable ranking model" rather than "revenue predictor," and say so explicitly in the executive summary. This is a decision point, flag it in the report either way.

---

## 6. API Contract (backend)

```
GET /api/candidates          -> list of candidate sites with lat/lon
GET /api/scores              -> candidate sites + predicted score + rank
GET /api/shap/{site_id}      -> local SHAP contributions for one site
GET /api/summary             -> auto-generated exec summary text (top site, rationale, trade-offs)
```

---

## 7. Milestones (build in this order)

1. **Data layer** — Census client + OSM client working standalone, cached to `data/raw/`. Validate: row counts, no nulls in key fields.
2. **Candidate grid** — generate/import N candidate sites for the chosen city. Validate: all sites within city bounds.
3. **Feature table** — geopandas buffer features for all candidates + existing stores (if using real target). Validate: spot-check 2–3 sites manually against a real map.
4. **Baseline weighted score** — ship this first, it's a safe fallback. Validate: sanity-check top 3 vs bottom 3 sites make intuitive sense.
5. **ML model + LOSO CV** — train, cross-validate, save artifacts. Validate: CV metrics logged, no data leakage (features must not use future/target-derived info).
6. **SHAP explainability** — global + local. Validate: SHAP values sum consistently with model output (sanity check).
7. **FastAPI endpoints** — wire model + data to API. Validate: manual curl/Postman check of all 4 routes.
8. **Next.js dashboard** — map, ranking table, SHAP chart, exec summary panel.
9. **Executive summary generator** — template-driven markdown pulling live numbers, not hardcoded.
10. **Polish pass** — README, one CV-ready screenshot of the map, final metrics table.

---

## 8. Deliverables Checklist (for CV/portfolio)

- [ ] Interactive map screenshot (candidate sites, color-coded by score)
- [ ] SHAP summary plot
- [ ] LOSO CV metrics table
- [ ] One-page executive recommendation (PDF or rendered markdown)
- [ ] CV bullet (draft once results are in — don't template it, use real numbers per your existing CV convention)

---

## 9. Step-by-Step: What To Actually Do

**Step 1 — Set up the repo (15 min)**
- Create the folder structure in Section 3.
- `pip install geopandas osmnx overpy shap scikit-learn folium requests rasterio --break-system-packages` (locally, or let Antigravity's environment handle it).
- No signups needed except optionally a Census key — skip it, unused in this pipeline.

**Step 2 — Get demographic data (30–45 min)**
- Download the WorldPop India 100m population raster (GeoTIFF) from worldpop.org — pick the most recent constrained population count file.
- Write `census_client.py`-equivalent (rename to `worldpop_client.py`) that clips the raster to Bengaluru's bounding box using `rasterio`.

**Step 3 — Get competitor/transit/road data (45–60 min)**
- Define your retail category (e.g. `shop=supermarket` or whatever matches your hypothetical brand) in `osm_client.py`.
- Query Overpass once for the whole Bengaluru bounding box, cache to `data/raw/osm_competitors.geojson`, `osm_transit.geojson`, `osm_roads.graphml` (via OSMnx). Never re-query per-candidate — that's what burns Overpass rate limits.

**Step 4 — Generate candidate sites (15 min)**
- Build a grid (e.g. every 1.5km) or sample N=40–60 realistic points across Bengaluru, filtered to exclude obviously non-commercial areas (parks, water bodies via OSM landuse tags).

**Step 5 — Feature engineering (1–2 hrs)**
- Reproject everything to a Bengaluru-appropriate UTM CRS (EPSG:32643) before any buffer math.
- Compute the feature table from Section 4 for every candidate site, using WorldPop for population and OSM for the rest.

**Step 6 — Build and document the synthetic target (30 min, important)**
- Write `data/synthetic_target_rationale.md`: state the exact formula (e.g. `score = 0.4*pop_density + 0.3*(1/competitor_count) + 0.3*transit_access + noise`), and why each weight is defensible.
- This file is what keeps the project honest — reference it in your exec summary and be ready to explain it if asked in an interview.

**Step 7 — Baseline weighted score (30 min)**
- Ship this first as your safety-net deliverable — a transparent weighted sum, no ML.

**Step 8 — ML model + LOSO CV (2–3 hrs)**
- Train Gradient Boosting/RF on the synthetic target.
- Run leave-one-candidate-out (or leave-one-cluster-out if you group nearby sites) CV, log R²/RMSE per fold honestly.
- If CV is weak, pivot framing to "explainable ranking model" per Section 5's guardrail — don't force a revenue-prediction narrative that doesn't hold up.

**Step 9 — SHAP explainability (1 hr)**
- Global summary plot + local explanation for your top-3 recommended sites.

**Step 10 — Map + exec summary (1–2 hrs)**
- Folium map, color-coded by score, with popups showing top SHAP drivers per site.
- Auto-generate the one-page markdown exec summary pulling live numbers from the model output — not hand-written.

**Step 11 — Polish (1 hr)**
- README with the synthetic-data disclosure up front, screenshot of the map for your portfolio, final metrics table, CV bullet drafted from real numbers.

**Total estimated time: ~10–14 hours across a few sessions.**
