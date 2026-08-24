# Synthetic Target Rationale

Because we lack proprietary store revenue data for the candidate sites in Bengaluru, we must generate a synthetic target variable (`synthetic_revenue`) to serve as the ground truth for our machine learning model. This allows us to demonstrate the end-to-end site selection pipeline (feature engineering, cross-validation, and SHAP explainability) without relying on sensitive data.

## Formula Evolution

**Original Milestone 1 Formula:**
`0.4 × pop_density + 0.3 × (1 / competitor_count) + 0.3 × transit_access + noise`

**Updated Milestone 4 Formula:**
During Milestone 3, we engineered stronger local economic proxies (`wealth_poi_count_1000m` and `poi_diversity_1000m`) and recognized that a straight penalty for competitors handles 0-counts better than an inverse term. The formula was updated to reflect these richer features.

## The Final Synthetic Formula

To ensure weights are applied commensurately regardless of feature magnitude, all input features are first Z-scored (StandardScaled).

**Base Revenue:** $50,000 / month
**Variance Multiplier:** $10,000

**Feature Weights:**
- `pop_1000m`: +0.3 (High population drives base footfall)
- `wealth_poi_count_1000m`: +0.3 (Premium POIs drive higher ticket sizes)
- `poi_diversity_1000m`: +0.2 (Mixed-use areas generate consistent all-day traffic)
- `competitor_count_1000m`: -0.2 (Direct competition cannibalizes sales)

**Noise Distribution:**
- $\mathcal{N}(0, 2500)$ — Additive Gaussian noise with mean 0 and standard deviation 2,500 (representing ~5% random variance on the base revenue).

**Calculation:**
```python
Z(x) = (x - mean(x)) / std(x)

synthetic_revenue = 50000 + 10000 * (
    0.3 * Z(pop_1000m) +
    0.3 * Z(wealth_poi_count_1000m) +
    0.2 * Z(poi_diversity_1000m) -
    0.2 * Z(competitor_count_1000m)
) + numpy.random.normal(0, 2500)
```

**Guardrail:** Because the ML model will train on the exact features used to generate this target, the resulting $R^2$ will artificially be very high. This is explicitly a validation of pipeline mechanics, not a discovery of novel predictive signal.
