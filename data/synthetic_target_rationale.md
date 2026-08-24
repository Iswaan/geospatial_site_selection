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

- **Base Revenue:** $50,000 / month
- **Variance Multiplier:** $10,000

**Feature Weights:**
- `pop_1000m`: +0.3 (High population drives base footfall)
- `wealth_poi_count_1000m`: +0.3 (Premium POIs drive higher average ticket sizes)
- `poi_diversity_1000m`: +0.2 (Mixed-use areas generate consistent all-day traffic)
- `competitor_count_1000m`: -0.2 (Direct competition cannibalizes sales linearly)

**Noise Distribution:**
- $\mathcal{N}(\mu=0, \sigma=2500)$ — Additive Gaussian noise with a standard deviation of 2,500 representing ~5% baseline random variance, preventing the ML model from achieving an artificial $R^2 = 1.0$.

**Calculation (Python):**
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
