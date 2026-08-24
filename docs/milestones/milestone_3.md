# Milestone 3: Feature Engineering

## What Was Accomplished
We transformed the raw OSINT data into a structured 14-feature spatial dataset for each of our 50 candidate sites. Every calculation was performed exactly in meters utilizing the Bengaluru UTM Zone 43N projection (EPSG:32643).

### Advanced Data Fetching
- **Wealth Proxy Layer:** Fetched 6,337 premium POIs (cafes, fine dining, boutiques, fitness centres).
- **Diversity Proxy Layer:** Fetched 39,373 general shops and amenities.
- **Overpass Optimization:** To prevent downloading 150MB+ of complex building polygon geometries for these 45k POIs, we utilized Overpass QL's `out center qt;` directive. This returned only the center point coordinates, shrinking the payload to ~7MB while retaining perfect accuracy for point-in-polygon buffer math.

### Feature Computation (14 Core Features)
For each candidate site, we computed:
1. **Global Distances:** `nearest_competitor_dist_m` and `nearest_transit_dist_m`. (Because we used an unbounded `gpd.sjoin_nearest`, this gracefully handled sites with zero competitors in the immediate radius by finding the true distance to the nearest competitor citywide).
2. **1km & 3km Radii Buffers:**
   - `pop_{r}m`: Zonal population sum across the WorldPop 100m raster via `rasterstats`.
   - `competitor_count_{r}m`: Storefront counts.
   - `transit_stop_count_{r}m`: Node counts.
   - `wealth_poi_count_{r}m`: Premium POI proxy.
   - `poi_diversity_{r}m`: Count of unique categories/tags across the buffer.
   - `road_density_{r}m`: Total length of all intersecting drivable road segments divided by the buffer area.

### Deliberate Constraints
- We explicitly dropped the `5km` radius option. Given our 50-sample CV size, injecting an additional 7 features representing highly smoothed 5km areas would only invite massive overfitting without adding granular local signal.

### Data Exported to `data/processed/`
- **features.csv & features.geojson**: 50 rows, 15 columns (14 features + ID).
