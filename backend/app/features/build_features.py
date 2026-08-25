"""
build_features.py
-----------------
Computes radius-based and distance-based features for all candidate sites.
Uses EPSG:32643 for all accurate spatial buffering and distance math.

Feature math is delegated to feature_engine.compute_features_for_point()
so the batch pipeline and the live /api/score-custom endpoint share a
single implementation.
"""

import pathlib
import sys
import logging
import geopandas as gpd
import pandas as pd
import osmnx as ox
import warnings

from backend.app.features.feature_engine import compute_features_for_point

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_CRS = "EPSG:32643"  # UTM Zone 43N
WGS84_CRS = "EPSG:4326"

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_and_project(filename: str) -> gpd.GeoDataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        log.error("Missing required input file: %s", path)
        sys.exit(1)
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs(WGS84_CRS, inplace=True)
    return gdf.to_crs(PROJECT_CRS)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("1. Loading datasets...")
    candidates = load_and_project("candidate_sites.geojson")
    competitors = load_and_project("osm_competitors.geojson")
    transit = load_and_project("osm_transit.geojson")
    wealth = load_and_project("osm_wealth_pois.geojson")
    diversity = load_and_project("osm_diversity_pois.geojson")

    log.info("2. Loading road network graph...")
    roads_path = RAW_DIR / "osm_roads.graphml"
    if not roads_path.exists():
        log.error("Missing osm_roads.graphml")
        sys.exit(1)
    G = ox.load_graphml(roads_path)
    edges = ox.graph_to_gdfs(G, nodes=False)
    if edges.crs is None:
        edges.set_crs(WGS84_CRS, inplace=True)
    edges = edges.to_crs(PROJECT_CRS)

    pop_raster_path = RAW_DIR / "bengaluru_pop_100m.tif"
    if not pop_raster_path.exists():
        log.error("Missing bengaluru_pop_100m.tif")
        sys.exit(1)

    log.info("3. Computing features for %d candidate sites...", len(candidates))
    all_features = []
    for i, row in candidates.iterrows():
        if i % 10 == 0:
            log.info("   -> Site %d / %d", i, len(candidates))
        feat = compute_features_for_point(
            point_utm=row.geometry,
            competitors=competitors,
            transit=transit,
            wealth=wealth,
            diversity=diversity,
            edges=edges,
            pop_raster_path=pop_raster_path,
        )
        feat["site_id"] = candidates.loc[i, "site_id"] if "site_id" in candidates.columns else f"site_{i:03d}"
        all_features.append(feat)

    log.info("4. Assembling feature table...")
    features_df = pd.DataFrame(all_features)
    # Reorder: site_id first, then features in canonical order
    feat_cols = [c for c in features_df.columns if c != "site_id"]
    features_df = features_df[["site_id"] + feat_cols]

    # Merge back geometry for GeoJSON export
    candidates_reset = candidates.reset_index(drop=True)
    features_gdf = gpd.GeoDataFrame(
        features_df,
        geometry=candidates_reset.geometry,
        crs=PROJECT_CRS,
    )

    log.info("5. Exporting feature table...")
    out_csv = PROCESSED_DIR / "features.csv"
    out_geojson = PROCESSED_DIR / "features.geojson"

    features_df.to_csv(out_csv, index=False)

    candidates_wgs84 = features_gdf.to_crs(WGS84_CRS)
    candidates_wgs84.to_file(out_geojson, driver="GeoJSON")

    log.info("✓ Feature table: %s (Rows: %d, Cols: %d)", out_csv.name, features_df.shape[0], features_df.shape[1])
    log.info("✓ GeoJSON exported: %s", out_geojson.name)


if __name__ == "__main__":
    main()
