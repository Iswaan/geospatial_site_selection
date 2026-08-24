"""
worldpop_client.py
------------------
Downloads the WorldPop India 2020 constrained population count raster (~100m)
and clips it to Bengaluru's bounding box.

Source: WorldPop Hub (https://hub.worldpop.org)
Dataset: India constrained individual countries, 2020, 100m resolution
License: CC BY 4.0

Output: data/raw/bengaluru_pop_100m.tif

IMPORTANT: The synthetic target variable in this project is derived in part from
this population raster. See data/synthetic_target_rationale.md.
"""

import os
import sys
import logging
import pathlib
import urllib.request
from typing import Tuple

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.crs import CRS
from shapely.geometry import box
import json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Bengaluru bounding box (EPSG:4326): [min_lon, min_lat, max_lon, max_lat]
BENGALURU_BBOX = (77.35, 12.75, 77.85, 13.20)

# WorldPop direct-download URL for India 2020 constrained 100m
# NOTE: WorldPop sometimes rotates file paths; if this 404s, fall back to
# manual download — see MANUAL_DOWNLOAD_INSTRUCTIONS below.
WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained"
    "/2020/BSGM/IND/ind_ppp_2020_constrained.tif"
)

# parents[3] = geospatial_site_selection/ (project root)
# backend/app/data/worldpop_client.py → [0]=data [1]=app [2]=backend [3]=project_root
RAW_DIR = pathlib.Path(__file__).resolve().parents[3] / "data" / "raw"
RAW_TIF = RAW_DIR / "ind_ppp_2020_constrained.tif"
CLIPPED_TIF = RAW_DIR / "bengaluru_pop_100m.tif"

MANUAL_DOWNLOAD_INSTRUCTIONS = """
================================================================================
MANUAL DOWNLOAD REQUIRED
================================================================================
The automatic download of the WorldPop India population raster failed.
This is expected — WorldPop rotates its download URLs periodically.

Please follow these steps:
  1. Go to: https://hub.worldpop.org/geodata/listing?id=29
  2. Download the India 2020 constrained population count GeoTIFF.
  3. Save the file as:
       {raw_tif}
  4. Re-run this script: python -m backend.app.data.worldpop_client

The project WILL NOT proceed to Milestone 3 (feature engineering) until this
file exists and passes the non-all-nodata validation below.
================================================================================
"""


# ---------------------------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------------------------

def download_worldpop(dest: pathlib.Path) -> bool:
    """
    Attempt to download the WorldPop India GeoTIFF.
    Returns True on success, False on failure.
    Does NOT raise — failure is handled by the caller with a clear message.
    """
    if dest.exists():
        log.info("WorldPop source raster already exists at %s — skipping download.", dest)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading WorldPop India 2020 constrained raster…")
    log.info("URL: %s", WORLDPOP_URL)
    log.info("Destination: %s", dest)

    try:
        # Stream download with progress reporting
        def _reporthook(block_num, block_size, total_size):
            if total_size > 0:
                pct = block_num * block_size / total_size * 100
                print(f"\r  {min(pct, 100):.1f}%", end="", flush=True)

        urllib.request.urlretrieve(WORLDPOP_URL, dest, reporthook=_reporthook)
        print()  # newline after progress bar
        log.info("Download complete: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
        return True
    except Exception as exc:
        log.error("Download failed: %s", exc)
        if dest.exists():
            dest.unlink()  # remove partial file
        return False


# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------

def clip_to_bbox(
    src_path: pathlib.Path,
    dst_path: pathlib.Path,
    bbox: Tuple[float, float, float, float],
) -> None:
    """
    Clip a raster to a bounding box (given as min_lon, min_lat, max_lon, max_lat
    in EPSG:4326) and write the result to dst_path.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    geom = [box(min_lon, min_lat, max_lon, max_lat).__geo_interface__]

    log.info("Clipping raster to Bengaluru bbox %s…", bbox)

    with rasterio.open(src_path) as src:
        # Reproject bbox geometry to source CRS if necessary
        clipped, transform = rio_mask(src, geom, crop=True)
        meta = src.meta.copy()

    meta.update(
        {
            "driver": "GTiff",
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "transform": transform,
            "compress": "lzw",
        }
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(clipped)

    log.info("Clipped raster written to %s", dst_path)


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def validate_clipped_raster(path: pathlib.Path) -> None:
    """
    Validate that the clipped GeoTIFF is not all-nodata.

    A bad bounding box or a failed-but-partially-written download can silently
    produce a raster where every pixel is the nodata value. This does NOT raise
    a file-not-found error — the file exists — but it will silently break
    Milestone 3's population buffer feature.

    Raises RuntimeError with a descriptive message if validation fails.
    """
    log.info("Validating clipped raster: %s", path)

    with rasterio.open(path) as src:
        data = src.read(1, masked=True)  # read band 1 as masked array
        nodata = src.nodata
        bounds = src.bounds

    log.info(
        "  Bounds: lon=[%.4f, %.4f], lat=[%.4f, %.4f]",
        bounds.left, bounds.right, bounds.bottom, bounds.top,
    )
    log.info("  Shape: %s, nodata=%s", data.shape, nodata)

    valid_pixels = int(np.sum(~data.mask)) if hasattr(data, "mask") else int(np.sum(np.isfinite(data)))
    total_pixels = data.size

    log.info("  Valid pixels: %d / %d (%.1f%%)", valid_pixels, total_pixels, valid_pixels / total_pixels * 100)

    if valid_pixels == 0:
        raise RuntimeError(
            f"VALIDATION FAILED: {path} contains zero valid (non-nodata) pixels.\n"
            "Possible causes:\n"
            "  1. The bounding box does not overlap the downloaded raster.\n"
            "  2. The download was incomplete or corrupted.\n"
            "  3. The WorldPop file uses a CRS that needs reprojection before clipping.\n"
            "Resolution: delete both TIF files in data/raw/ and re-run this script."
        )

    pop_sum = float(np.nansum(data.filled(0)))
    log.info("  Total population in clipped area: ~{:,.0f}".format(pop_sum))

    if pop_sum < 1_000_000:
        log.warning(
            "  Population sum (%,.0f) seems very low for Bengaluru (expected >8M). "
            "Check the bounding box or WorldPop file version.",
            pop_sum,
        )
    else:
        log.info("  ✓ Population sum looks reasonable for Bengaluru.")

    log.info("  ✓ Clipped raster validation passed.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run() -> pathlib.Path:
    """
    Orchestrate download → clip → validate.
    Returns the path to the validated clipped raster.
    Exits with code 1 on failure so CI/orchestration pipelines can catch it.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # --- Download ---
    success = download_worldpop(RAW_TIF)
    if not success:
        print(MANUAL_DOWNLOAD_INSTRUCTIONS.format(raw_tif=RAW_TIF))
        sys.exit(1)

    # --- Clip ---
    clip_to_bbox(RAW_TIF, CLIPPED_TIF, BENGALURU_BBOX)

    # --- Validate ---
    try:
        validate_clipped_raster(CLIPPED_TIF)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(1)

    log.info("worldpop_client: all steps complete. Output: %s", CLIPPED_TIF)
    return CLIPPED_TIF


if __name__ == "__main__":
    run()
