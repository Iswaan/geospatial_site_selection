# Synthetic Target Rationale

Because we lack proprietary store revenue data for the candidate sites in Bengaluru, we must generate a synthetic target variable (`synthetic_revenue`) to serve as the ground truth for our machine learning model. This allows us to demonstrate the end-to-end site selection pipeline (feature engineering, cross-validation, and SHAP explainability) without relying on sensitive data.

## Formula Evolution

**Original Milestone 1 Formula:**
`0.4 × pop_density + 0.3 × (1 / competitor_count) + 0.3 × transit_access + noise`

**Updated Milestone 4 Formula:**
During Milestone 3, we engineered stronger local economic proxies (`wealth_poi_count_1000m` and `poi_diversity_1000m`). We revised the original formula for two specific business and mathematical reasons:

1. **Dropping Transit in favor of Wealth/Diversity:** While transit stops indicate foot traffic volume, they do not qualify *purchasing power* or *trip purpose*. By swapping transit for `wealth_poi_count` (which proxies discretionary income via cafes/boutiques) and `poi_diversity` (which proxies mixed-use vibrancy), we capture a more direct signal of retail viability than simple bus stop counts.
2. **Competitor Term Revision (Inverse vs. Linear):** The original `1 / competitor_count` term creates a mathematical singularity when a site has zero competitors (division by zero). While we could add a smoothing term (e.g., `1 / (1 + count)`), an inverse curve causes the penalty to flatten out rapidly as competitor counts increase (e.g., the penalty drop from 1 to 2 is massive, but 10 to 11 is negligible). For dense urban retail, cannibalization remains severe even at higher counts. We therefore switched to a straight linear subtraction on the Z-scored feature, which applies a consistent, non-diminishing penalty per competitor and elegantly handles zero-counts.

## The Final Synthetic Formula

To ensure weights are applied commensurately regardless of raw feature magnitude, all input features are first Z-scored (`StandardScaler` with $\mu=0, \sigma=1$).

- **Base Revenue:** ₹40,00,000 / month
- **Variance Multiplier:** ₹8,00,000

**Feature Weights:**
- `pop_1000m`: +0.3 (High population drives base footfall)
- `wealth_poi_count_1000m`: +0.3 (Premium POIs drive higher average ticket sizes)
- `poi_diversity_1000m`: +0.2 (Mixed-use areas generate consistent all-day traffic)
- `competitor_count_1000m`: -0.2 (Direct competition cannibalizes sales linearly)

**Noise Distribution:**
- $\mathcal{N}(\mu=0, \sigma=200,000)$ — Additive Gaussian noise with a standard deviation of 2,00,000 representing ~5% baseline random variance, preventing the ML model from achieving an artificial $R^2 = 1.0$.

**Calculation (Python):**
```python
Z(x) = (x - mean(x)) / std(x)

synthetic_revenue = 4000000 + 800000 * (
    0.3 * Z(pop_1000m) +
    0.3 * Z(wealth_poi_count_1000m) +
    0.2 * Z(poi_diversity_1000m) -
    0.2 * Z(competitor_count_1000m)
) + numpy.random.normal(0, 200000)
```

**Guardrail:** Because the ML model will train on the exact features used to generate this target, the resulting $R^2$ will artificially be very high. This is explicitly a validation of pipeline mechanics, not a discovery of novel predictive signal.

## Noise Features & SHAP Validation

Several engineered features (e.g., 
earest_competitor_dist_m, 
earest_transit_dist_m, and 
oad_density_1000m) are deliberately **excluded** from the synthetic target formula but will be **included** in the model's training feature set. 

**Rationale:** This serves as a deliberate stress-test for our SHAP explainability layer. By feeding the model features that we know *a priori* have zero true relationship to the target, we can empirically validate the model's robustness and SHAP's accuracy. In a successful validation, the SHAP summary plot should assign near-zero importance to these noise features while heavily weighting the four core formula features, proving the model learned the true data generating process rather than simply overfitting to the 50-sample set.


**Note on SHAP & Spatial Collinearity:**
During testing, non-target features like `transit_stop_count_3000m` and `nearest_competitor_dist_m` received non-trivial SHAP importance. We mathematically verified this is driven by real **spatial collinearity**, not just small-n instability:
1. **Measurement Twins**: `competitor_count_1000m` has a **-0.65** correlation with `nearest_competitor_dist_m` (they describe the exact same underlying reality measured two different ways, so the noise feature is just a twin of the target formula feature).
2. **Genuine Spatial Correlation**: `pop_1000m` has a **+0.59** correlation with `transit_stop_count_3000m` (two distinct metrics that map to the same urban density).
3. **Geometric Tautology**: `poi_diversity_1000m` has a **+0.72** correlation with `poi_diversity_3000m` (expected by construction, since a 3km buffer fully contains the 1km buffer).

Because tree-based models arbitrarily split importance between highly collinear proxies, the model borrows signal from these non-target features. This serves as a powerful demonstration of why domain-aware feature selection remains necessary even with explainable AI.