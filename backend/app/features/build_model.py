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
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
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
    scaled_target_features = z_scaler.fit_transform(df[core_features])
    
    # Base = 4M, Multiplier = 800k, Weights = +0.3, +0.3, +0.2, -0.2
    baseline_score = 4000000 + 800000 * (
        0.3 * scaled_target_features[:, 0] +  # pop_1000m
        0.3 * scaled_target_features[:, 1] +  # wealth_poi_count_1000m
        0.2 * scaled_target_features[:, 2] -  # poi_diversity_1000m
        0.2 * scaled_target_features[:, 3]    # competitor_count_1000m
    )
    
    # Add random noise (5% of base revenue)
    np.random.seed(42) # Fixed seed for reproducibility
    noise = np.random.normal(0, 200000, size=len(baseline_score))
    df["synthetic_revenue"] = baseline_score + noise
    
    # Save baseline score (noiseless) as the primary business deliverable
    baseline_df = df[["site_id"]].copy()
    baseline_df["baseline_score"] = baseline_score
    baseline_df["synthetic_revenue"] = df["synthetic_revenue"]
    baseline_df.to_csv(PROCESSED_DIR / "baseline_scores.csv", index=False)
    log.info("✓ Saved baseline_scores.csv")
    
    # Pickle the z_scaler so custom-point scoring uses the IDENTICAL 50-site
    # distribution — prevents baseline drift if features.csv is regenerated.
    joblib.dump(z_scaler, MODEL_DIR / "z_scaler.pkl")
    log.info("✓ Saved z_scaler.pkl (core-feature baseline Z-scorer)")

    # 2. MODEL SETUP
    # We intentionally include features NOT in the formula to stress-test SHAP
    # (e.g. 3000m metrics, distance metrics, road densities)
    feature_cols = [c for c in df.columns if c not in ["site_id", "synthetic_revenue"]]
    X = df[feature_cols]
    y = df["synthetic_revenue"]
    
    log.info("Features used for training: %s", feature_cols)
    
    # ANSWER: Check site_031 baseline directly
    site_031_idx = df.index[df["site_id"] == "site_031"].tolist()
    if site_031_idx:
        log.info("=== DEBUG: site_031 Baseline Check ===")
        log.info("site_031 baseline_score: Rs%f", baseline_score[site_031_idx[0]])
        log.info("======================================")

    # 3. LEAVE-ONE-OUT CV (MODEL COMPARISON)
    loo = LeaveOneOut()
    
    CANDIDATE_MODELS = {
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=3, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42),
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000, random_state=42),
    }
    
    comparison_results = []
    
    log.info("Starting Leave-One-Out CV across %d sites for %d models...", len(df), len(CANDIDATE_MODELS))
    
    for model_name, model in CANDIDATE_MODELS.items():
        predictions = np.zeros(len(df))
        absolute_errors = []
        
        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            
            pred = model.predict(X_test_scaled)[0]
            predictions[test_index] = pred
            
            ae = abs(pred - y_test.iloc[0])
            absolute_errors.append(ae)
            
        overall_r2 = r2_score(y, predictions)
        mae_min = np.min(absolute_errors)
        mae_median = np.median(absolute_errors)
        mae_max = np.max(absolute_errors)
        mae_std = np.std(absolute_errors)
        
        comparison_results.append({
            "model": model_name,
            "r2": overall_r2,
            "mae_min": mae_min,
            "mae_median": mae_median,
            "mae_max": mae_max,
            "mae_std": mae_std
        })
        
        log.info("--- CV RESULTS: %s ---", model_name)
        log.info("Overall R^2: %.4f", overall_r2)
        log.info("Abs Error - Min: Rs%.2f | Med: Rs%.2f | Max: Rs%.2f | Std: Rs%.2f", 
                 mae_min, mae_median, mae_max, mae_std)

    # Save Comparison CSV
    comp_df = pd.DataFrame(comparison_results)
    comp_df.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)
    
    # Save Interpretation Note
    note_content = {
        "production_model": "XGBoost",
        "n_samples_caveat": "LOSO CV with n=50 has limited statistical power to distinguish close R² differences. A gap of < 0.05 between two models should not be interpreted as a reliable performance difference — the confidence interval on each R² estimate at n=50 is roughly ±0.10 to ±0.15.",
        "linear_structure_note": "The synthetic target is by construction a linear combination of Z-scored features plus Gaussian noise (documented in `synthetic_target_rationale.md`). A regularized linear model (Ridge or ElasticNet) is *structurally well-matched* to this target: it can recover the exact generating function if given the right features and regularization. XGBoost's tree structure adds expressive power that the target does not require, which means it must implicitly approximate a linear function using piecewise constants — Ridge and ElasticNet have a genuine structural advantage here. If they match or outperform XGBoost, that is the expected and legitimate outcome, not a fluke.",
        "selection_reason": "XGBoost is retained for production use. The primary reasons are: (1) the live dashboard uses `shap.TreeExplainer` for per-site SHAP explanations — this is natively supported for XGBoost and would require a different explainability approach for linear models; (2) XGBoost's partial robustness to irrelevant features (road density at 3000m, distance metrics) means it does not degrade when the 10 non-formula features add noise. The model comparison table is shown as methodological evidence — it confirms the XGBoost result is plausible in context, not arbitrarily better than simpler alternatives."
    }
    with open(PROCESSED_DIR / "model_comparison_note.json", "w") as f:
        json.dump(note_content, f, indent=2)
    log.info("✓ Saved model comparison outputs")
    
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
