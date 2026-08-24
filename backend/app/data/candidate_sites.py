"""
candidate_sites.py
------------------
Generates a grid of candidate retail sites for Bengaluru.
Filters candidates strictly to the city limits and excludes non-commercial
areas (parks, water bodies, military zones) fetched from OSM.

Usage:
  python -m backend.app.data.candidate_sites
"""

import json
import logging
import pathlib
import sys
import numpy as np
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, Polygon, MultiPolygon

from backend.app.data.osm_client import (
    overpass_post,
    BBOX_S, BBOX_W, BBOX_N, BBOX_E,
    INTER_CALL_SLEEP
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_CRS = "EPSG:32643"  # UTM Zone 43N for Bengaluru (meters)
WGS84_CRS = "EPSG:4326"

RAW_DIR = pathlib.Path(__file__).resolve().parents[3] / "data" / "raw"
CANDIDATES_OUT = RAW_DIR / "candidate_sites.geojson"
GRID_SPACING_M = 1500  # 1.5 km grid spacing

TARGET_SITES_MIN = 40
TARGET_SITES_MAX = 60
SAMPLE_SIZE = 50
RANDOM_SEED = 42

ox.settings.log_console = False
ox.settings.use_cache = True


# ---------------------------------------------------------------------------
# HELPER: Parse Overpass Ways (with geom) to Polygons
# ---------------------------------------------------------------------------
def parse_exclusion_zones(data: dict) -> gpd.GeoDataFrame:
    """
    Parses an Overpass JSON response (using `out geom;`) into a GeoDataFrame
    of polygons for exclusion zones.
    """
    polygons = []
    
    for el in data.get("elements", []):
        if el.get("type") == "way" and "geometry" in el:
            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
            # Needs at least 3 points to form a polygon (Overpass closed ways have first == last)
            if len(coords) >= 3:
                try:
                    polygons.append(Polygon(coords))
                except Exception as e:
                    continue
        elif el.get("type") == "relation" and "members" in el:
            # For relations (multipolygons), we simplify by taking the outer ways
            for member in el.get("members", []):
                if member.get("type") == "way" and member.get("role") == "outer" and "geometry" in member:
                    coords = [(pt["lon"], pt["lat"]) for pt in member["geometry"]]
                    if len(coords) >= 3:
                        try:
                            polygons.append(Polygon(coords))
                        except Exception:
                            continue

    if not polygons:
        return gpd.GeoDataFrame(columns=["geometry"], crs=WGS84_CRS)

    return gpd.GeoDataFrame({"geometry": polygons}, crs=WGS84_CRS)


# ---------------------------------------------------------------------------
# CORE: Grid Generation & Filtering
# ---------------------------------------------------------------------------
def run():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    log.info("1. Fetching Bengaluru administrative boundary...")
    city_gdf = ox.geocode_to_gdf("Bengaluru, Karnataka, India")
    city_poly = city_gdf.geometry.iloc[0]
    
    # Reproject boundary to UTM for metric grid generation
    city_gdf_proj = city_gdf.to_crs(PROJECT_CRS)
    city_poly_proj = city_gdf_proj.geometry.iloc[0]
    
    log.info("2. Generating raw grid in projected CRS (spacing: %dm)...", GRID_SPACING_M)
    minx, miny, maxx, maxy = city_poly_proj.bounds
    
    x_coords = np.arange(minx, maxx, GRID_SPACING_M)
    y_coords = np.arange(miny, maxy, GRID_SPACING_M)
    
    grid_points = []
    for x in x_coords:
        for y in y_coords:
            grid_points.append(Point(x, y))
            
    grid_gdf = gpd.GeoDataFrame({"geometry": grid_points}, crs=PROJECT_CRS)
    log.info("   -> Raw grid count: %d points", len(grid_gdf))
    
    log.info("3. Filtering points strictly inside city boundary...")
    # Point-in-polygon filter
    inside_mask = grid_gdf.intersects(city_poly_proj)
    grid_gdf = grid_gdf[inside_mask].copy()
    log.info("   -> Surviving count: %d points", len(grid_gdf))
    
    if len(grid_gdf) == 0:
        log.error("Zero points survived boundary filter. Check grid spacing or bounds.")
        sys.exit(1)
        
    log.info("4. Fetching exclusion zones (parks, water, military) via Overpass...")
    # Query using out geom so we get the exact shape of ways
    query = f"""
[out:json][timeout:120];
(
  way["leisure"="park"]({BBOX_S},{BBOX_W},{BBOX_N},{BBOX_E});
  relation["leisure"="park"]({BBOX_S},{BBOX_W},{BBOX_N},{BBOX_E});
  
  way["natural"="water"]({BBOX_S},{BBOX_W},{BBOX_N},{BBOX_E});
  relation["natural"="water"]({BBOX_S},{BBOX_W},{BBOX_N},{BBOX_E});
  
  way["waterway"]({BBOX_S},{BBOX_W},{BBOX_N},{BBOX_E});
  relation["waterway"]({BBOX_S},{BBOX_W},{BBOX_N},{BBOX_E});
  
  way["landuse"="military"]({BBOX_S},{BBOX_W},{BBOX_N},{BBOX_E});
  relation["landuse"="military"]({BBOX_S},{BBOX_W},{BBOX_N},{BBOX_E});
);
out body geom;
"""
    data = overpass_post(query, "exclusion_zones")
    excl_gdf = parse_exclusion_zones(data)
    log.info("   -> Parsed %d exclusion polygons", len(excl_gdf))
    
    if len(excl_gdf) > 0:
        log.info("5. Filtering points intersecting exclusion zones...")
        # Reproject exclusions to projected CRS
        excl_gdf_proj = excl_gdf.to_crs(PROJECT_CRS)
        
        # We drop any candidate that intersects ANY exclusion polygon
        joined = gpd.sjoin(grid_gdf, excl_gdf_proj, how="left", predicate="intersects")
        # Keep points that did NOT match an exclusion zone (index_right is NaN)
        valid_joined = joined[joined["index_right"].isna()].copy()
        
        # Drop the join column and duplicates (if any)
        grid_gdf = valid_joined.drop(columns=["index_right"]).drop_duplicates(subset=["geometry"])
        
    log.info("   -> Surviving count: %d points", len(grid_gdf))
    
    if len(grid_gdf) < TARGET_SITES_MIN:
        log.warning("Final pool (%d) is below target minimum (%d). Using all.", len(grid_gdf), TARGET_SITES_MIN)
        final_gdf = grid_gdf
    else:
        log.info("6. Sampling %d points with seed %d...", SAMPLE_SIZE, RANDOM_SEED)
        final_gdf = grid_gdf.sample(n=min(SAMPLE_SIZE, len(grid_gdf)), random_state=RANDOM_SEED)
        
    final_count = len(final_gdf)
    log.info("   -> Final candidate count: %d", final_count)
    
    # Final Validation
    if not (TARGET_SITES_MIN <= final_count <= TARGET_SITES_MAX):
        log.error("VALIDATION FAILED: Final count %d is outside the acceptable range (%d-%d).", 
                  final_count, TARGET_SITES_MIN, TARGET_SITES_MAX)
        sys.exit(1)

    # Reproject back to WGS84 for GeoJSON export and road snapping
    final_gdf_wgs84 = final_gdf.to_crs(WGS84_CRS)
    
    log.info("7. Snapping candidates to the nearest drivable road node...")
    roads_path = RAW_DIR / "osm_roads.graphml"
    if roads_path.exists():
        G = ox.load_graphml(roads_path)
        # Extract lat/lon for all candidate points
        X = final_gdf_wgs84.geometry.x
        Y = final_gdf_wgs84.geometry.y
        # Find nearest node in the graph for each point
        nearest_nodes = ox.distance.nearest_nodes(G, X, Y)
        # Replace geometry with the actual node coordinates
        snapped_points = []
        for n in nearest_nodes:
            snapped_points.append(Point(G.nodes[n]['x'], G.nodes[n]['y']))
        final_gdf_wgs84["geometry"] = snapped_points
        log.info("   -> Successfully snapped %d candidates to road network", len(final_gdf_wgs84))
    else:
        log.warning("   -> osm_roads.graphml not found, skipping road snap step.")
    
    # Add an ID column
    final_gdf_wgs84 = final_gdf_wgs84.reset_index(drop=True)
    final_gdf_wgs84["site_id"] = [f"site_{i:03d}" for i in range(len(final_gdf_wgs84))]
    
    final_gdf_wgs84.to_file(CANDIDATES_OUT, driver="GeoJSON")
    log.info("✓ Saved candidate sites to %s", CANDIDATES_OUT)

if __name__ == "__main__":
    run()
