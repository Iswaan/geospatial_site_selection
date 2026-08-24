"""
build_features.py
-----------------
Computes radius-based and distance-based features for all candidate sites.
Uses EPSG:32643 for all accurate spatial buffering and distance math.
"""

import pathlib
import sys
import logging
import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
from rasterstats import zonal_stats
import warnings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Suppress GeoPandas warnings about missing spatial index on empty joins
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_CRS = "EPSG:32643"  # UTM Zone 43N
WGS84_CRS = "EPSG:4326"

RADII_M = [1000, 3000]

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def load_and_project(filename: str) -> gpd.GeoDataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        log.error("Missing required input file: %s", path)
        sys.exit(1)
    gdf = gpd.read_file(path)
    # Ensure it has a CRS before projecting
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

    # For raster stats, we need the raster CRS
    pop_raster_path = RAW_DIR / "bengaluru_pop_100m.tif"
    if not pop_raster_path.exists():
        log.error("Missing bengaluru_pop_100m.tif")
        sys.exit(1)

    log.info("3. Computing distance-based features...")
    # Nearest competitor
    # gpd.sjoin_nearest can return multiple matches if equidistant.
    # We drop duplicates on the candidate index.
    merged_comp = gpd.sjoin_nearest(candidates, competitors, how="left", distance_col="nearest_competitor_dist_m")
    merged_comp = merged_comp[~merged_comp.index.duplicated(keep="first")]
    candidates["nearest_competitor_dist_m"] = merged_comp["nearest_competitor_dist_m"]

    # Nearest transit
    merged_tran = gpd.sjoin_nearest(candidates, transit, how="left", distance_col="nearest_transit_dist_m")
    merged_tran = merged_tran[~merged_tran.index.duplicated(keep="first")]
    candidates["nearest_transit_dist_m"] = merged_tran["nearest_transit_dist_m"]

    # Fill NaNs if any (e.g. if the dataset was empty, though it shouldn't be)
    candidates["nearest_competitor_dist_m"] = candidates["nearest_competitor_dist_m"].fillna(-1)
    candidates["nearest_transit_dist_m"] = candidates["nearest_transit_dist_m"].fillna(-1)

    log.info("4. Computing radius-based features...")
    # For population, zonal_stats requires geometries in the same CRS as the raster.
    import rasterio
    with rasterio.open(pop_raster_path) as src:
        raster_crs = src.crs

    for r in RADII_M:
        log.info("   -> Processing radius: %dm", r)
        # Create buffers
        buffers = candidates.geometry.buffer(r)
        buffer_gdf = gpd.GeoDataFrame(geometry=buffers, crs=PROJECT_CRS)
        
        buffer_area = np.pi * (r ** 2)

        # Spatial joins
        # Competitors
        joined = gpd.sjoin(competitors, buffer_gdf, how="inner", predicate="intersects")
        comp_counts = joined.groupby("index_right").size()
        candidates[f"competitor_count_{r}m"] = candidates.index.map(comp_counts).fillna(0).astype(int)

        # Transit
        joined = gpd.sjoin(transit, buffer_gdf, how="inner", predicate="intersects")
        tran_counts = joined.groupby("index_right").size()
        candidates[f"transit_stop_count_{r}m"] = candidates.index.map(tran_counts).fillna(0).astype(int)

        # Wealth POIs
        joined = gpd.sjoin(wealth, buffer_gdf, how="inner", predicate="intersects")
        wealth_counts = joined.groupby("index_right").size()
        candidates[f"wealth_poi_count_{r}m"] = candidates.index.map(wealth_counts).fillna(0).astype(int)

        # Diversity
        joined = gpd.sjoin(diversity, buffer_gdf, how="inner", predicate="intersects")
        # For each buffer, count unique values in amenity + shop
        diversity_counts = {}
        for idx, group in joined.groupby("index_right"):
            amenities = set(group["amenity"].dropna().unique())
            shops = set(group["shop"].dropna().unique())
            diversity_counts[idx] = len(amenities) + len(shops)
        candidates[f"poi_diversity_{r}m"] = candidates.index.map(diversity_counts).fillna(0).astype(int)

        # Road density
        # For performance, use sjoin to find intersecting edges, then exact clip
        joined_edges = gpd.sjoin(edges, buffer_gdf, how="inner", predicate="intersects")
        densities = {}
        # This loop could be slow if there are thousands of candidates, but for 50 it's instant
        for idx in range(len(candidates)):
            buf = buffer_gdf.geometry.iloc[idx]
            # Get edges that intersect this buffer
            intersecting_edges = joined_edges[joined_edges["index_right"] == idx]
            if intersecting_edges.empty:
                densities[idx] = 0.0
                continue
            # Clip edges to exact buffer shape to get accurate lengths
            clipped = intersecting_edges.clip(buf)
            total_length = clipped.geometry.length.sum()
            densities[idx] = total_length / buffer_area
            
        candidates[f"road_density_{r}m"] = candidates.index.map(densities).fillna(0.0)

        # Population via zonal_stats
        buffers_raster_crs = buffer_gdf.to_crs(raster_crs)
        # We assume the raster contains population counts per pixel (or density). WorldPop is usually count per pixel.
        stats = zonal_stats(buffers_raster_crs, str(pop_raster_path), stats=["sum"], nodata=-99999)
        pop_sums = [s["sum"] if s["sum"] is not None else 0 for s in stats]
        candidates[f"pop_{r}m"] = pop_sums

    log.info("5. Exporting feature table...")
    # Clean up and export
    # Keep final features
    out_csv = PROCESSED_DIR / "features.csv"
    out_geojson = PROCESSED_DIR / "features.geojson"
    
    # Save CSV (without geometry)
    df = pd.DataFrame(candidates.drop(columns=["geometry"]))
    df.to_csv(out_csv, index=False)
    
    # Save GeoJSON (reproject back to WGS84)
    candidates_wgs84 = candidates.to_crs(WGS84_CRS)
    candidates_wgs84.to_file(out_geojson, driver="GeoJSON")

    log.info("✓ Feature table generated: %s (Rows: %d, Cols: %d)", out_csv.name, df.shape[0], df.shape[1])
    log.info("✓ GeoJSON exported: %s", out_geojson.name)


if __name__ == "__main__":
    main()
