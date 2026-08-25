import pandas as pd, joblib, numpy as np
df = pd.read_csv("data/processed/features.csv")
idx = df.index[df["site_id"] == "site_031"][0]
feature_cols = [c for c in df.columns if c not in ["site_id", "synthetic_revenue"]]
X = df[feature_cols]
scaler = joblib.load("backend/app/models/scaler.pkl")
model = joblib.load("backend/app/models/xgb_model.pkl")
X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)
print(f"INR ML Prediction for site_031: {preds[idx]}")
print(f"INR True Label for site_031: {df['synthetic_revenue'].iloc[idx]}")
