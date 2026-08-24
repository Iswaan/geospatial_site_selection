"""
build_model.py
--------------
Trains an explainable ML model to predict synthetic site revenue.
Uses Leave-One-Out Cross Validation to simulate site-selection generalization.
Includes non-target features (distance, road density, 3000m metrics) as a 
stress-test for the SHAP explainability layer.
"""

import sys
import json
import logging
import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import shap
import joblib

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "backend" / "app" / "models"

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    features_path = PROCESSED_DIR / "features.csv"
    
    if not features_path.exists():
        log.error("Missing features.csv")
        sys.exit(1)
        
    df = pd.read_csv(features_path)
    
    # 1. GENERATE SYNTHETIC TARGET
    # We Z-score the core 4 features to apply weights commensurately
    z_scaler = StandardScaler()
    core_features = ["pop_1000m", "wealth_poi_count_1000m", "poi_diversity_1000m", "competitor_count_1000m"]
    z_scored = pd.DataFrame(z_scaler.fit_transform(df[core_features]), columns=core_features)
    
    # Base = 50k, Multiplier = 10k, Weights = +0.3, +0.3, +0.2, -0.2
    baseline_score = 50000 + 10000 * (
        0.3 * z_scored["pop_1000m"] + 
        0.3 * z_scored["wealth_poi_count_1000m"] + 
        0.2 * z_scored["poi_diversity_1000m"] - 
        0.2 * z_scored["competitor_count_1000m"]
    )
    
    # Add Gaussian noise (mean 0, std 2500)
    np.random.seed(42) # Fixed seed for reproducibility
    noise = np.random.normal(0, 2500, size=len(df))
    df["synthetic_revenue"] = baseline_score + noise
    
    # Save baseline score (noiseless) as the primary business deliverable
    baseline_df = df[["site_id"]].copy()
    baseline_df["baseline_score"] = baseline_score
    baseline_df["synthetic_revenue"] = df["synthetic_revenue"]
    baseline_df.to_csv(PROCESSED_DIR / "baseline_scores.csv", index=False)
    log.info("✓ Saved baseline_scores.csv")

    # 2. MODEL SETUP
    # We intentionally include features NOT in the formula to stress-test SHAP
    # (e.g. 3000m metrics, distance metrics, road densities)
    feature_cols = [c for c in df.columns if c not in ["site_id", "synthetic_revenue"]]
    X = df[feature_cols]
    y = df["synthetic_revenue"]
    
    log.info("Features used for training: %s", feature_cols)
    
    # 3. LEAVE-ONE-OUT CV
    loo = LeaveOneOut()
    predictions = np.zeros(len(df))
    absolute_errors = []
    
    log.info("Starting Leave-One-Out CV across %d sites...", len(df))
    
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Scale data (fit on train only to prevent leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        pred = model.predict(X_test_scaled)[0]
        predictions[test_index] = pred
        
        # R^2 is undefined for a single point, so we just log Absolute Error
        ae = abs(pred - y_test.iloc[0])
        absolute_errors.append(ae)
        
    # Summarize CV Errors
    overall_r2 = r2_score(y, predictions)
    log.info("--- CV RESULTS ---")
    log.info("Overall R^2 across all LOO folds: %.4f", overall_r2)
    log.info("Absolute Error Distribution (n=%d):", len(absolute_errors))
    log.info("  Min:    $%.2f", np.min(absolute_errors))
    log.info("  Median: $%.2f", np.median(absolute_errors))
    log.info("  Max:    $%.2f", np.max(absolute_errors))
    log.info("  StdDev: $%.2f", np.std(absolute_errors))
    
    # 4. FINAL FULL-DATA MODEL & SHAP
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    final_model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    final_model.fit(X_scaled_df, y)
    
    # Global SHAP
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_scaled_df)
    
    # Save SHAP feature importances to JSON
    # Mean absolute SHAP value per feature
    global_importances = np.abs(shap_values).mean(axis=0)
    importance_dict = {feat: float(imp) for feat, imp in zip(feature_cols, global_importances)}
    
    # Sort for logging
    sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    log.info("--- GLOBAL SHAP IMPORTANCES ---")
    for feat, imp in sorted_imp:
        log.info("  %s: %.2f", feat, imp)
        
    with open(PROCESSED_DIR / "shap_importances.json", "w") as f:
        json.dump(importance_dict, f, indent=2)
        
    # Save the actual model and scaler for the Milestone 5 API
    joblib.dump(final_model, MODEL_DIR / "xgb_model.pkl")
    joblib.dump(final_scaler, MODEL_DIR / "scaler.pkl")
    log.info("✓ Saved model and scaler to %s", MODEL_DIR.name)
        
    log.info("✓ Model pipeline complete.")

if __name__ == "__main__":
    main()
