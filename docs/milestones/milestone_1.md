# Milestone 1: Data Ingestion & OSINT

## What Was Accomplished
We laid the core data foundation for the Bengaluru region by querying OpenStreetMap via the Overpass API. We built a resilient, standalone `osm_client.py` script that downloads and validates geospatial features directly.

### Engineering Decisions & Fallbacks
1. **Direct Overpass POST vs. OSMnx Sub-queries:** 
   We bypassed OSMnx's native POI fetchers in favor of direct Overpass POST requests. This solved a major "sub-query explosion" bug where OSMnx generates URLs too long for standard GET requests, leading to server timeouts.
2. **Mirror Redundancy:** 
   The brief explicitly warned about burning the Overpass rate limit. We implemented a fallback mechanism: if `lz4.overpass-api.de` timeouts or rate-limits us, the script automatically retries against `overpass-api.de`.
3. **Data Validation:** 
   We added strict post-download assertions to check that `geometry` columns are not null or empty before saving to disk. A bad bounding box shouldn't silently corrupt the pipeline.

### Data Exported to `data/raw/`
- **osm_competitors.geojson:** 1,613 features (Supermarkets & Convenience stores)
- **osm_transit.geojson:** 4,265 features (Bus stops, subway stations)
- **osm_roads.graphml:** 150.76 MB raw OSMnx drivable network graph
