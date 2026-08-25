"""
feature_engine.py
-----------------
Pure spatial feature computation for a single point.
No I/O, no file paths, no logging setup — only math.

Both the batch pipeline (build_features.py over 50 candidates) and the
live /api/score-custom endpoint call compute_features_for_point() so
there is exactly one implementation of the buffer/distance logic.

CRS contract: ALL inputs must already be projected to EPSG:32643 (UTM 43N).
              The raster path is the only file reference; zonal_stats handles
              its own CRS internally via the raster's embedded metadata.
"""

import pathlib
import warnings

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, mapping
from rasterstats import zonal_stats

warnings.filterwarnings("ignore", category=UserWarning)

RADII_M = [1000, 3000]
PROJECT_CRS = "EPSG:32643"


def compute_features_for_point(
    point_utm: Point,
    competitors: gpd.GeoDataFrame,
    transit: gpd.GeoDataFrame,
    wealth: gpd.GeoDataFrame,
    diversity: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    pop_raster_path: pathlib.Path,
) -> dict:
    """
    Compute all 14 model features for a single point in EPSG:32643.

    Parameters
    ----------
    point_utm : shapely.geometry.Point
        The candidate location in EPSG:32643 (metres).
    competitors, transit, wealth, diversity, edges : GeoDataFrame
        Pre-loaded and pre-projected to EPSG:32643.
    pop_raster_path : Path
        Path to the WorldPop raster (any CRS — zonal_stats reads it directly).

    Returns
    -------
    dict with keys matching the trained model's feature_cols order:
        nearest_competitor_dist_m, nearest_transit_dist_m,
        competitor_count_{1000,3000}m, transit_stop_count_{1000,3000}m,
        wealth_poi_count_{1000,3000}m, poi_diversity_{1000,3000}m,
        road_density_{1000,3000}m, pop_{1000,3000}m
    """
    # Wrap the single point in a 1-row GeoDataFrame for sjoin operations
    point_gdf = gpd.GeoDataFrame(geometry=[point_utm], crs=PROJECT_CRS)

    features: dict = {}

    # ------------------------------------------------------------------
    # Distance features
    # ------------------------------------------------------------------
    merged_comp = gpd.sjoin_nearest(point_gdf, competitors, how="left", distance_col="_d")
    features["nearest_competitor_dist_m"] = float(merged_comp["_d"].iloc[0]) if not merged_comp["_d"].isna().all() else -1.0

    merged_tran = gpd.sjoin_nearest(point_gdf, transit, how="left", distance_col="_d")
    features["nearest_transit_dist_m"] = float(merged_tran["_d"].iloc[0]) if not merged_tran["_d"].isna().all() else -1.0

    # ------------------------------------------------------------------
    # Radius features
    # ------------------------------------------------------------------
    for r in RADII_M:
        buf = point_utm.buffer(r)
        buffer_area = np.pi * (r ** 2)
        buf_gdf = gpd.GeoDataFrame(geometry=[buf], crs=PROJECT_CRS)

        # Competitor count
        joined = gpd.sjoin(competitors, buf_gdf, how="inner", predicate="intersects")
        features[f"competitor_count_{r}m"] = int(len(joined))

        # Transit count
        joined = gpd.sjoin(transit, buf_gdf, how="inner", predicate="intersects")
        features[f"transit_stop_count_{r}m"] = int(len(joined))

        # Wealth POI count
        joined = gpd.sjoin(wealth, buf_gdf, how="inner", predicate="intersects")
        features[f"wealth_poi_count_{r}m"] = int(len(joined))

        # POI diversity (unique amenity + shop type count)
        joined = gpd.sjoin(diversity, buf_gdf, how="inner", predicate="intersects")
        if not joined.empty:
            amenities = set(joined["amenity"].dropna().unique()) if "amenity" in joined.columns else set()
            shops = set(joined["shop"].dropna().unique()) if "shop" in joined.columns else set()
            features[f"poi_diversity_{r}m"] = int(len(amenities) + len(shops))
        else:
            features[f"poi_diversity_{r}m"] = 0

        # Road density — clip edges to exact buffer, sum lengths
        joined_edges = gpd.sjoin(edges, buf_gdf, how="inner", predicate="intersects")
        if not joined_edges.empty:
            clipped = joined_edges.clip(buf)
            total_length = clipped.geometry.length.sum()
            features[f"road_density_{r}m"] = float(total_length / buffer_area)
        else:
            features[f"road_density_{r}m"] = 0.0

        # Population (WorldPop raster zonal sum)
        buf_raster_crs = buf_gdf.to_crs("EPSG:4326")  # zonal_stats in raster CRS
        stats = zonal_stats(buf_raster_crs, str(pop_raster_path), stats=["sum"], nodata=-99999)
        features[f"pop_{r}m"] = float(stats[0]["sum"]) if stats[0]["sum"] is not None else 0.0

    return features
