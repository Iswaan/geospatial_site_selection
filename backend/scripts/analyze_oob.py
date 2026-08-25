"""
analyze_oob.py
--------------
Investigates the ML-vs-baseline gap for the Koramangala custom point
by checking which features fall outside the 50-candidate training range.
"""
import requests
import json
import pandas as pd
import numpy as np
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Pull Koramangala custom-point features from the live API
print("Fetching Koramangala custom point from API...")
r = requests.post(
    "http://localhost:8000/api/score-custom",
    json={"lat": 12.9352, "lon": 77.6245},
    timeout=60,
)
data = r.json()
features = data["features"]
predicted = data["predicted_score"]
baseline  = data["baseline_score"]
rank      = data["rank"]

print()
print("=== Koramangala custom point ===")
print(f"  predicted : Rs{predicted:,.0f}")
print(f"  baseline  : Rs{baseline:,.0f}")
gap_pct = (baseline - predicted) / baseline * 100
print(f"  gap       : {gap_pct:.1f}% (baseline - predicted) / baseline")
print()

# Load training feature ranges
df = pd.read_csv(PROCESSED_DIR / "features.csv")
feat_cols = [c for c in df.columns if c not in ["site_id", "synthetic_revenue"]]

# Pretty table
header = f"{'Feature':<40} {'Point':>14}  {'Train Min':>12}  {'Train Max':>12}  {'Status'}"
print(header)
print("-" * len(header))

oor = []
for col in feat_cols:
    pval = features.get(col, None)
    if pval is None:
        print(f"  {col:<38}  MISSING")
        continue
    lo = df[col].min()
    hi = df[col].max()
    oob = pval < lo or pval > hi
    if oob:
        side = "HIGH" if pval > hi else "LOW"
        rng = hi - lo if hi != lo else 1
        delta_pct = (pval - hi) / rng * 100 if pval > hi else (lo - pval) / rng * 100
        flag = f"<< OOB {side} +{delta_pct:.0f}% of range >>"
        oor.append((col, pval, lo, hi, side, delta_pct))
    else:
        flag = "in range"
    print(f"  {col:<38}  {pval:>14.2f}  {lo:>12.2f}  {hi:>12.2f}  {flag}")

print()
print(f"Out-of-range features: {len(oor)} / {len(feat_cols)}")
if oor:
    print()
    print("=== OOB detail ===")
    for col, pval, lo, hi, side, delta in oor:
        print(f"  {col}")
        print(f"    point={pval:.2f}  train=[{lo:.2f}, {hi:.2f}]  {side} by {delta:.0f}% of range width")

# Also show the 4 formula features specifically
print()
print("=== Core formula features (used in baseline Z-scoring) ===")
core = ["pop_1000m", "wealth_poi_count_1000m", "poi_diversity_1000m", "competitor_count_1000m"]
for col in core:
    pval = features.get(col, None)
    lo = df[col].min()
    hi = df[col].max()
    mu = df[col].mean()
    sd = df[col].std()
    z  = (pval - mu) / sd if sd > 0 else 0
    oob = pval < lo or pval > hi
    print(f"  {col:<40}  val={pval:.1f}  z={z:.2f}  range=[{lo:.1f},{hi:.1f}]  {'OOB' if oob else 'ok'}")

print()
print("=== Extrapolation hypothesis ===")
n_oob = len(oor)
if n_oob > 0:
    print(f"  CONFIRMED: {n_oob} feature(s) are outside the 50-site training range.")
    print("  XGBoost cannot extrapolate beyond leaf boundaries — its prediction")
    print("  flattens at the outermost training value for those dimensions.")
    print("  The baseline linear formula continues extrapolating, explaining the gap.")
else:
    print("  NOT CONFIRMED: all features are within the training range.")
    print("  The gap has a different cause (non-linear interactions, noise).")
