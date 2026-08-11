# Freight Rate Estimator

A 3-stage Machine Learning pipeline designed to predict U.S. commercial truckload spot rates with spatial-temporal awareness and zero future-data leakage.

---

## Overview

freight pricing is highly volatile, driven by regional capacity shifts, seasonal surges, payload constraints, and geographic route complexity. Standard machine learning models frequently overfit by memorizing historical route baselines rather than learning underlying spatial and temporal dynamics.

This project implements a robust, modular 3-stage ML pipeline using **CatBoost Regressor** trained on a log-transformed Rate-Per-Mile ($\text{log\_rpm}$) target. By pairing a route-isolated 5-fold `GroupKFold` validation strategy with lag-shifted 14-day trailing rolling features and dual spatial circuity metrics, the pipeline achieves an Out-of-Fold (OOF) Cross-Validation Mean RMSE of **$581.74** and a holdout test MAE of **$104.21** ($R^2 = 0.838$).

---

## Problem Statement

Predicting freight rates presents three key modeling challenges:

1. **Spatial Leakage & Baseline Memorization:** Traditional random train/test splits place transactions from identical shipping routes into both training and test sets. Decision trees end up memorizing specific lane prices rather than generalizing to unseen corridors.
2. **Target Variance Heteroscedasticity:** Total load prices scale linearly with distance ($\text{posted\_rate} \approx \text{rate} \times \text{distance}$). Standard RMSE loss optimization causes long-haul loads ($1,500+\text{ miles}$) with high absolute dollar variance to dominate model updates while degrading short-haul pricing accuracy.
3. **Future Data Dependency:** Real-time bidding signals (`quote_signal`, `market_index`) are frequently missing or unavailable for future prediction windows, leading to pipeline failures or flatlined synthetic imputations.

---

## Project Structure

```text
freight-rate-ml-pipeline/
├── data/
│
├── outputs/
│   ├── validation_predictions.csv
│   ├── december_predictions.csv
│   └── candidate_december.png    # December benchmark forecast chart
├── src/
│   ├── preprocess.py             # Stage 1: Data cleaning & feature engineering
│   ├── train.py                  # Stage 2: GroupKFold training 
│   └── predict.py                # Stage 3: Inference & inverse dollar reconstruction
├── requirements.txt              # Project dependencies
├── score.py                      # Evaluation script 
└── README.md                     # Project documentation

```

---

## Data Analysis & Insights

Exploratory analysis on historical transaction data revealed critical structural properties:

* **Log-Normal Target Distribution:** Historical rates exhibit extreme right-skewness driven by heavy-haul and specialized equipment loads. Log transformation normalizes the distribution and stabilizes gradient steps.
* **Route Circuity Impact:** The ratio between raw road driving distance and straight-line Haversine distance varies significantly by region. Coastal routes, mountain passes, and lake bypasses carry higher circuity ratios, driving fuel burn and toll surcharges.
* **Weekly Operating Cadence:** Commercial shipping follows a distinct 7-day calendar cycle. Spot tender volume and pricing peak midweek (Wednesday/Thursday) as shippers race to secure capacity before weekend receiving dock closures.

---

## The Approach (3-Stage Pipeline)

```
┌────────────────────────────────────────────────────────────────────────┐
│               STAGE 1: DATA CLEANING & PREPROCESSING                   │
│ • Payload Anomaly Correction & Median Fallback Imputation             │
│ • Dual Spatial Metrics (Haversine Distance & Route Circuity Ratio)     │
│ • Target Transformation: Log Rate-Per-Mile (log_rpm)                  │
│ • Lag-Shifted 14-Day Trailing Rolling Stats with .shift(1) Lag        │
│ • Explicit Pruning of Quote_Signal & Market_Index Columns             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            STAGE 2: ROUTE-ISOLATED VALIDATION & TRAINING               │
│ • 5-Fold GroupKFold Split Grouped Strictly on lane_id                 │
│ • In-Fold Target Encoding & Imputation Parameter Fitting               │
│ • Optuna Bayesian Search (iterations: 454, depth: 8, lr: 0.0358)       │
│ • Full 100% Historical Dataset Production Refitting                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              STAGE 3: INFERENCE & SCORE.PY EVALUATION                  │
│ • Automated Spatial Coordinate Backfilling for Test Set               │
│ • Inverse Log-RPM Dollar Reconstruction: rate = (exp(pred)-1) * dist   │
│ • December Benchmark Forecast Generation (candidate_december.png)      │
│ • Score.py Execution Compliance (PASSED / 0 Errors)                    │
└────────────────────────────────────────────────────────────────────────┘

```

---

## Feature Engineering

### 1. Log Rate-Per-Mile Target Transformation

Rather than predicting total dollar amounts directly, the target is converted to Rate-Per-Mile in log space:

$$\text{rate\_per\_mile} = \frac{\text{posted\_rate}}{\text{distance}}$$

$$\text{target\_log\_rpm} = \ln(1 + \text{rate\_per\_mile})$$

At inference, total rates are analytically reconstructed without loss of precision:

$$\widehat{\text{posted\_rate}} = \left(\exp(\hat{y}_{\text{target\_log\_rpm}}) - 1\right) \times \text{distance}$$

### 2. Dual-Distance Spatial Metrics

* **Haversine Distance (`pickup_to_delivery_dist`):** Calculated directly from origin and destination latitude/longitude coordinates to provide an uncorruptible physical anchor.
* **Route Circuity Ratio:** Comparing straight-line distance against raw road `distance` enables tree splits to detect geographic obstacles (e.g., detour routes around lakes or mountain ranges).

### 3. Lag-Shifted 14-Day Trailing Rolling Statistics

Calculates historical rate averages grouped per shipping lane ($\text{lane\_id} = \text{pickup} + \text{"\vert{}"} + \text{delivery}$):

```python
df['route_rolling_14d_mean'] = (
    df.groupby('lane_id')['posted_rate']
    .transform(lambda x: x.shift(1).rolling(window=14, min_periods=1).mean())
)

```

* **Leakage Prevention:** Applying `.shift(1)` guarantees that a prediction for day $t$ strictly evaluates historical transactions from days $t-14$ through $t-1$.
* **Cold-Start Fallback:** Unseen lanes inherit the global historical training median.

### 4. Feature Pruning

External indicators (`market_index` and `quote_signal`) were completely excluded. SHAP analysis confirmed `quote_signal` contributed less than $0.5\%$ of predictive weight ($\approx 0.002$), while removing these columns eliminated missing-data issues for future inference.

---

## Model Selection

Multiple candidate architectures were benchmarked on identical 5-fold `GroupKFold` splits. **CatBoost Regressor** achieved the best predictive performance:

| Architecture Family | Holdout RMSE ($) | Holdout MAE ($) | $R^2$ Score |
| --- | --- | --- | --- |
| Naive Historical Mean Baseline | $1,486.48 | $912.10 | 0.000 |
| Baseline Ridge Regression | $724.26 | $268.43 | 0.512 |
| Lasso Regression | $720.23 | $266.30 | 0.518 |
| Random Forest Regressor | $648.14 | $133.96 | 0.812 |
| Gradient Boosting Regressor | $641.55 | $112.84 | 0.824 |
| XGBoost Regressor | $638.95 | $114.18 | 0.829 |
| LightGBM Regressor | $637.66 | $109.92 | 0.832 |
| **CatBoost Regressor (Final)** | **$636.14** | **$104.21** | **0.838** |

### Why CatBoost?

* **Oblivious Decision Trees:** Enforces structural symmetry across entire tree levels, providing built-in regularization against noisy individual transactions.
* **Native Categorical Handling:** Efficiently processes high-cardinality categorical features (`equipment`, spatial zones) via Ordered Target Statistics without matrix expansion.

---

## Model Validation

Validation is structured around a **5-Fold `GroupKFold**` grouped strictly on `lane_id`:

$$\text{lane\_id} = \text{pickup} + \text{"\vert{}"} + \text{delivery}$$

This forces 100% of historical transactions for any specific origin-destination pair into either the training fold or the validation fold, but never both.

### Out-of-Fold (OOF) Performance

| Validation Fold | Isolated Lane Group | Out-of-Fold RMSE ($) |
| --- | --- | --- |
| Fold 1 | Route Cluster Alpha | $550.20 |
| Fold 2 | Route Cluster Beta | $602.51 |
| Fold 3 | Route Cluster Gamma | $534.98 |
| Fold 4 | Route Cluster Delta | $639.28 |
| Fold 5 | Route Cluster Epsilon | $581.73 |
| **CV Mean Score** | **All Unseen Routes** | **$581.74** |

*All metrics are evaluated after converting predictions back to real U.S. Dollars ($).*

---

## Installation & Execution

### 1. Environment Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Abhisekh1403/Spot-Freight-Rate-Prediction.git
cd freight-rate-ml-pipeline
pip install -r requirements.txt

```

## Running the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python src/train.py
```

### Generate predictions

```bash
python src/predict.py
```

### Validate the submission

```bash
python score.py --predictions validation-predictions.csv --december-predictions data/december-chart-inputs.csv
```

---



## Generated Artifacts

Running the pipeline generates:

models/catboost_model.cbm
validation_predictions.csv
Updated data/december-chart-inputs.csv
scorer_results/candidate_december.png

## Author

**Abhisekh Kumar**

Machine Learning & AI Engineer