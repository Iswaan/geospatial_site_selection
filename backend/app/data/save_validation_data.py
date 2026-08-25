"""
save_validation_data.py
-----------------------
One-time setup script. Generates and saves two static GeoJSON files needed
by the live /api/score-custom endpoint for point validation at runtime:

  data/raw/bengaluru_boundary.geojson   — city admin boundary polygon
  data/raw/exclusion_zones.geojson      — parks, water bodies, military areas

These files are committed to data/raw/ and loaded from disk at API startup.
No live Overpass or Nominatim calls are made at request time.

IMPORTANT: The exclusion-zone query reuses the existing robust POST-based
overpass_post() helper from osm_client.py (requests.post with 3-mirror
fallback) — NOT overpy or ox.features_from_bbox which hit 504s during M2.

Usage:
  python -m backend.app.data.save_validation_data
"""

import logging
import pathlib
import sys
import time

import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon

from backend.app.data.osm_client import (
    overpass_post,
    BBOX_S, BBOX_W, BBOX_N, BBOX_E,
    INTER_CALL_SLEEP,
)
from backend.app.data.candidate_sites import parse_exclusion_zones

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

BOUNDARY_OUT = RAW_DIR / "bengaluru_boundary.geojson"
EXCLUSION_OUT = RAW_DIR / "exclusion_zones.geojson"

WGS84_CRS = "EPSG:4326"


def save_boundary() -> None:
    """Fetch and save the Bengaluru administrative boundary polygon."""
    if BOUNDARY_OUT.exists():
        log.info("Boundary cache found — skipping (%s)", BOUNDARY_OUT.name)
        return

    log.info("Fetching Bengaluru admin boundary via ox.geocode_to_gdf()...")
    city_gdf = ox.geocode_to_gdf("Bengaluru, Karnataka, India")
    city_gdf = city_gdf[["geometry"]].copy()
    city_gdf.crs = WGS84_CRS  # geocode_to_gdf already returns WGS84

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    city_gdf.to_file(BOUNDARY_OUT, driver="GeoJSON")
    log.info("✓ Saved bengaluru_boundary.geojson (%d polygon(s))", len(city_gdf))


def save_exclusion_zones() -> None:
    """
    Fetch and save parks, water bodies, and military zones.
    Uses the identical Overpass query and parse_exclusion_zones() function
    from candidate_sites.py — same data, no reimplementation.
    """
    if EXCLUSION_OUT.exists():
        log.info("Exclusion zones cache found — skipping (%s)", EXCLUSION_OUT.name)
        return

    log.info("Fetching exclusion zones via Overpass POST (reusing osm_client helper)...")
    time.sleep(INTER_CALL_SLEEP)

    # Exact same query used in candidate_sites.py
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
    # overpass_post is the hardened requests.post helper with 3-mirror fallback
    data = overpass_post(query, "exclusion_zones")

    # parse_exclusion_zones is the exact same parser used in candidate_sites.py
    excl_gdf = parse_exclusion_zones(data)
    log.info("Parsed %d exclusion polygons", len(excl_gdf))

    if len(excl_gdf) == 0:
        log.warning("No exclusion polygons found — saving empty GeoJSON. "
                    "Validation will skip exclusion check (not block all points).")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    excl_gdf.to_file(EXCLUSION_OUT, driver="GeoJSON")
    log.info("✓ Saved exclusion_zones.geojson (%d features)", len(excl_gdf))


if __name__ == "__main__":
    errors = []

    try:
        save_boundary()
    except Exception as e:
        log.error("Boundary fetch failed: %s", e)
        errors.append(e)

    try:
        save_exclusion_zones()
    except Exception as e:
        log.error("Exclusion zones fetch failed: %s", e)
        errors.append(e)

    if errors:
        log.error("%d error(s) — check logs above.", len(errors))
        sys.exit(1)

    log.info("✓ save_validation_data complete.")
    log.info("  %s", BOUNDARY_OUT)
    log.info("  %s", EXCLUSION_OUT)
