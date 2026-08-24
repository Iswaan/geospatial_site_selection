"""
osm_client.py
-------------
Fetches OpenStreetMap data for Bengaluru and caches results locally.

Design rules (per build brief §2 & §9):
  - Competitors use a SINGLE Overpass query with regex alternation:
        shop~"supermarket|convenience"
    One round-trip, not two — per the brief's explicit instruction to
    minimise Overpass hits on the free rate-limited tier.
  - All queries use HTTP POST (not GET) — POST is recommended for large
    bbox queries; it avoids URL length limits and is treated differently
    by Overpass's request queuing.
  - Competitor and transit queries go directly via requests.post() to
    Overpass, bypassing OSMnx's internal sub-query splitting logic which
    inflates a simple bbox into thousands of sub-requests.
  - Road network uses ox.graph_from_place() (Nominatim → admin boundary
    polygon → Overpass), which is OSMnx's most robust fetch path.
  - Every query response is cached to data/raw/. If the cache file
    already exists, the query is skipped entirely.
  - Geometry null-checks are performed after every GeoJSON write to catch
    silent API failures.

Outputs:
  data/raw/osm_competitors.geojson   — supermarket + convenience shops
  data/raw/osm_transit.geojson       — transit stops (bus + rail)
  data/raw/osm_roads.graphml         — drivable street network (OSMnx)

Usage:
  python -m backend.app.data.osm_client
"""

import json
import logging
import pathlib
import sys
import time
from typing import Optional

import geopandas as gpd
import osmnx as ox
import requests
from shapely.geometry import Point, shape

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Bengaluru bounding box: (south, west, north, east) for Overpass queries
BBOX_S, BBOX_W, BBOX_N, BBOX_E = 12.75, 77.35, 13.20, 77.85

# parents[3] = geospatial_site_selection/ (project root)
# backend/app/data/osm_client.py → [0]=data [1]=app [2]=backend [3]=project_root
RAW_DIR = pathlib.Path(__file__).resolve().parents[3] / "data" / "raw"
COMPETITORS_OUT = RAW_DIR / "osm_competitors.geojson"
TRANSIT_OUT = RAW_DIR / "osm_transit.geojson"
ROADS_OUT = RAW_DIR / "osm_roads.graphml"

# Overpass mirrors to try in sequence (POST requests).
# lz4 and z are generally more responsive for large city-scale queries.
OVERPASS_MIRRORS = [
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
]

# Seconds to wait between successive Overpass calls
INTER_CALL_SLEEP = 10

# Configure OSMnx (used only for road graph)
ox.settings.log_console = False
ox.settings.use_cache = True


# ---------------------------------------------------------------------------
# CORE: Overpass POST request with mirror fallback
# ---------------------------------------------------------------------------

def _overpass_post(query: str, label: str, timeout: int = 120) -> dict:
    """
    Send an Overpass QL query via HTTP POST to each mirror in sequence.
    Returns the parsed JSON response dict on first success.
    Raises RuntimeError if all mirrors fail.

    Using POST instead of GET:
      - Avoids URL length limits on large queries
      - Overpass servers handle POST queries in a separate queue that is
        less aggressively rate-limited than GET requests
    """
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "geospatial-site-selection/1.0 (portfolio project)",
    }
    last_exc: Optional[Exception] = None

    for mirror in OVERPASS_MIRRORS:
        try:
            log.info("  [%s] POST → %s", label, mirror)
            resp = requests.post(
                mirror,
                data={"data": query},
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            n_elements = len(data.get("elements", []))
            log.info("  [%s] ✓ %d elements from %s", label, n_elements, mirror)
            return data
        except Exception as exc:
            log.warning("  [%s] Mirror %s failed: %s", label, mirror, exc)
            last_exc = exc
            time.sleep(3)

    raise RuntimeError(
        f"All Overpass mirrors failed for [{label}].\n"
        f"Last error: {last_exc}\n"
        "Overpass may be under heavy load. Wait a few minutes and re-run.\n"
        "The script will skip already-cached outputs on re-run."
    )


# ---------------------------------------------------------------------------
# HELPER: Parse Overpass JSON → GeoDataFrame of point geometries
# ---------------------------------------------------------------------------

def _overpass_json_to_gdf(data: dict) -> gpd.GeoDataFrame:
    """
    Convert Overpass JSON response to a point GeoDataFrame.
    - Node elements → direct Point from lat/lon
    - Way elements → centroid of member nodes (sufficient for buffer math)
    Skips elements with no usable geometry.
    """
    records = []
    node_coords: dict = {}  # osm_id → (lon, lat) for way centroid computation

    # First pass: index all nodes by id
    for el in data.get("elements", []):
        if el.get("type") == "node" and "lat" in el and "lon" in el:
            node_coords[el["id"]] = (el["lon"], el["lat"])

    # Second pass: build records
    for el in data.get("elements", []):
        t = el.get("type")
        tags = el.get("tags", {})

        if t == "node" and "lat" in el and "lon" in el:
            records.append({"geometry": Point(el["lon"], el["lat"]), **tags})

        elif t == "way":
            nd_refs = el.get("nodes", [])
            coords = [node_coords[nid] for nid in nd_refs if nid in node_coords]
            if coords:
                cx = sum(c[0] for c in coords) / len(coords)
                cy = sum(c[1] for c in coords) / len(coords)
                records.append({"geometry": Point(cx, cy), **tags})

    if not records:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return gdf


# ---------------------------------------------------------------------------
# VALIDATION: Geometry null-check
# ---------------------------------------------------------------------------

def validate_geojson(path: pathlib.Path, label: str) -> None:
    """
    Validate that a GeoJSON file is non-empty and has valid geometries.
    Raises RuntimeError on failure — a silent empty result would cause
    silently-zero buffer counts in Milestone 3, corrupting the feature table.
    """
    if not path.exists():
        raise RuntimeError(f"VALIDATION FAILED [{label}]: file not found at {path}")

    gdf = gpd.read_file(path)
    total = len(gdf)

    if total == 0:
        raise RuntimeError(
            f"VALIDATION FAILED [{label}]: {path} contains 0 features.\n"
            "Possible causes: Overpass returned an empty result, query timed out,\n"
            "or the bounding box doesn't intersect any matching OSM elements."
        )

    null_geom = int(gdf.geometry.isna().sum()) + int(gdf.geometry.is_empty.sum())
    valid = total - null_geom

    if valid == 0:
        raise RuntimeError(
            f"VALIDATION FAILED [{label}]: all {total} features have null/empty geometry.\n"
            f"Delete {path} and re-run."
        )

    if null_geom > 0:
        log.warning(
            "  [%s] %d / %d features have null/empty geometry — excluded from analysis.",
            label, null_geom, total,
        )

    log.info(
        "  ✓ [%s] %d features total, %d valid geometries, %d null/empty — PASSED",
        label, total, valid, null_geom,
    )


# ---------------------------------------------------------------------------
# FETCH: Competitors (supermarket + convenience — single query, POST)
# ---------------------------------------------------------------------------

def fetch_competitors() -> gpd.GeoDataFrame:
    """
    Fetch supermarket + convenience shops from OSM in a single Overpass POST
    request using regex alternation: shop~"supermarket|convenience"

    This is one round-trip (not two), satisfying the brief's requirement to
    minimise Overpass API calls.
    """
    if COMPETITORS_OUT.exists():
        log.info("Competitors cache found — skipping Overpass query.")
        return gpd.read_file(COMPETITORS_OUT)

    # Single query, regex alternation — one API call for both shop types
    query = f"""
[out:json][timeout:120];
(
  node[shop~"supermarket|convenience"]({BBOX_S},{BBOX_W},{BBOX_N},{BBOX_E});
  way[shop~"supermarket|convenience"]({BBOX_S},{BBOX_W},{BBOX_N},{BBOX_E});
);
out body;
>;
out skel qt;
"""
    log.info("Fetching competitors (shop~'supermarket|convenience') via Overpass POST…")
    data = _overpass_post(query, "competitors")

    gdf = _overpass_json_to_gdf(data)
    log.info("  Parsed %d competitor features", len(gdf))

    # Keep only relevant columns
    keep = ["geometry", "shop", "name"]
    keep = [c for c in keep if c in gdf.columns]
    gdf = gdf[keep]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    gdf.to_file(COMPETITORS_OUT, driver="GeoJSON")
    log.info("Competitors saved to %s", COMPETITORS_OUT)
    return gdf


# ---------------------------------------------------------------------------
# FETCH: Transit Stops (POST)
# ---------------------------------------------------------------------------

def fetch_transit() -> gpd.GeoDataFrame:
    """
    Fetch bus stops and rail stations from OSM via Overpass POST.
    Covers: highway=bus_stop, public_transport=stop, railway=station|halt
    """
    if TRANSIT_OUT.exists():
        log.info("Transit cache found — skipping Overpass query.")
        return gpd.read_file(TRANSIT_OUT)

    query = f"""
[out:json][timeout:120];
(
  node[highway=bus_stop]({BBOX_S},{BBOX_W},{BBOX_N},{BBOX_E});
  node[public_transport=stop]({BBOX_S},{BBOX_W},{BBOX_N},{BBOX_E});
  node[railway=station]({BBOX_S},{BBOX_W},{BBOX_N},{BBOX_E});
  node[railway=halt]({BBOX_S},{BBOX_W},{BBOX_N},{BBOX_E});
);
out body;
"""
    log.info("Fetching transit stops via Overpass POST…")
    time.sleep(INTER_CALL_SLEEP)

    data = _overpass_post(query, "transit")
    gdf = _overpass_json_to_gdf(data)
    log.info("  Parsed %d transit features", len(gdf))

    keep = ["geometry", "name", "highway", "railway", "public_transport"]
    keep = [c for c in keep if c in gdf.columns]
    gdf = gdf[keep]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    gdf.to_file(TRANSIT_OUT, driver="GeoJSON")
    log.info("Transit stops saved to %s", TRANSIT_OUT)
    return gdf


# ---------------------------------------------------------------------------
# FETCH: Road Network (OSMnx graph_from_place)
# ---------------------------------------------------------------------------

def fetch_roads():
    """
    Download the Bengaluru drive network via OSMnx using place-name geocoding.
    ox.graph_from_place() uses Nominatim to resolve the admin boundary polygon
    and then queries Overpass for that exact shape — OSMnx's most robust path.
    OSMnx 2.x API: graph_from_bbox takes a positional bbox tuple if needed,
    but graph_from_place is preferred here.
    """
    if ROADS_OUT.exists():
        log.info("Roads cache found — skipping OSMnx download.")
        return ox.load_graphml(ROADS_OUT)

    log.info("Downloading Bengaluru drive network via OSMnx graph_from_place…")
    time.sleep(INTER_CALL_SLEEP)

    # Point OSMnx at lz4 mirror — lz4 successfully served both feature queries
    # above; overpass-api.de is currently under heavy load for this session.
    # OSMnx 2.x: overpass_url is the base path without /interpreter.
    ox.settings.overpass_url = "https://lz4.overpass-api.de/api"
    log.info("  Using Overpass mirror: %s", ox.settings.overpass_url)

    G = ox.graph_from_place(
        "Bengaluru, Karnataka, India",
        network_type="drive",
        retain_all=False,
        simplify=True,
    )

    log.info("  Road network: %d nodes, %d edges", len(G.nodes), len(G.edges))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, filepath=str(ROADS_OUT))
    log.info("Road network saved to %s", ROADS_OUT)
    return G


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run():
    """Orchestrate all three fetches with validation. Non-zero exit on any failure."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    errors = []

    # --- Competitors ---
    try:
        fetch_competitors()
        validate_geojson(COMPETITORS_OUT, "competitors")
    except Exception as exc:
        log.error("Competitor fetch/validation failed: %s", exc)
        errors.append(str(exc))

    # --- Transit ---
    try:
        fetch_transit()
        validate_geojson(TRANSIT_OUT, "transit")
    except Exception as exc:
        log.error("Transit fetch/validation failed: %s", exc)
        errors.append(str(exc))

    # --- Roads ---
    try:
        fetch_roads()
        if not ROADS_OUT.exists():
            raise RuntimeError(f"Road network GraphML not found at {ROADS_OUT}")
        size_mb = ROADS_OUT.stat().st_size / 1e6
        log.info("  ✓ [roads] GraphML exists (%.2f MB)", size_mb)
    except Exception as exc:
        log.error("Road network fetch failed: %s", exc)
        errors.append(str(exc))

    # --- Summary ---
    if errors:
        log.error("osm_client: %d error(s) encountered:", len(errors))
        for e in errors:
            log.error("  • %s", e)
        sys.exit(1)

    log.info("osm_client: all fetches complete and validated ✓")
    for p in [COMPETITORS_OUT, TRANSIT_OUT, ROADS_OUT]:
        log.info("  %s (%.1f KB)", p.name, p.stat().st_size / 1024)


if __name__ == "__main__":
    run()
