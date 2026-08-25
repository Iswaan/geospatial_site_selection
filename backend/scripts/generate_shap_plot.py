import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import pathlib
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "backend" / "app" / "models"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the features
df = pd.read_csv(PROCESSED_DIR / "features.csv")
feature_cols = [c for c in df.columns if c not in ["site_id", "synthetic_revenue"]]
X = df[feature_cols]

# Load model and scaler
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
model = joblib.load(MODEL_DIR / "xgb_model.pkl")

# Scale features
X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)

# Human readable labels
display_labels = {
    "pop_1000m": "Pop (1km)",
    "pop_3000m": "Pop (3km)",
    "wealth_poi_count_1000m": "Wealth POIs (1km)",
    "wealth_poi_count_3000m": "Wealth POIs (3km)",
    "poi_diversity_1000m": "Poi Diversity (1km)",
    "poi_diversity_3000m": "Poi Diversity (3km)",
    "competitor_count_1000m": "Competitor Count (1km)",
    "competitor_count_3000m": "Competitor Count (3km)",
    "nearest_competitor_dist_m": "Nearest Competitor Dist (m)",
    "transit_stop_count_1000m": "Transit Stop Count (1km)",
    "transit_stop_count_3000m": "Transit Stop Count (3km)",
    "nearest_transit_dist_m": "Nearest Transit Dist (m)",
    "road_density_1000m": "Road Density (1km)",
    "road_density_3000m": "Road Density (3km)"
}
X_scaled = X_scaled.rename(columns=display_labels)

# Explain
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

# Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_scaled, show=False)
plt.title("Global Feature Importances (SHAP Summary Plot)\nAcross all 50 Candidate Sites", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'shap_summary_plot.png', dpi=300, bbox_inches='tight')
print(f"Saved SHAP plot to {OUTPUT_DIR / 'shap_summary_plot.png'}")
