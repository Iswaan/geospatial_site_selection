# Milestone 2: Candidate Site Generation

## What Was Accomplished
We procedurally generated a 50-site grid across Bengaluru, rigorously snapping candidate points to realistic commercial real estate locations rather than mathematically perfect but physically impossible coordinates.

### Engineering Decisions & Math
1. **Projected CRS for Grid Spacing (EPSG:32643):**
   Instead of generating a grid in WGS84 lat/lon degrees (which distorts grid squares based on latitude), we generated the 1.5km equidistant grid entirely in UTM Zone 43N. This ensures every grid point is perfectly 1.5km apart on the ground.
2. **Exclusion Zones:**
   We systematically clipped candidate points out of protected or unbuildable zones: Parks, Water bodies, and Military land.
3. **Road Snapping (Critical Step):**
   A raw mathematical grid will drop points in the middle of highways, lakes, or dense residential gated communities. We loaded the 150MB `osm_roads.graphml` and used `osmnx.distance.nearest_nodes` to "snap" every candidate site to the nearest drivable intersection. 
4. **Validation Spot-Check:**
   We manually verified a handful of the final 50 coordinates in Google Maps to guarantee the points landed on actual street frontages.

### Data Exported to `data/raw/`
- **candidate_sites.geojson:** Exactly 50 viable candidate locations, validated and snapped to the road network.
