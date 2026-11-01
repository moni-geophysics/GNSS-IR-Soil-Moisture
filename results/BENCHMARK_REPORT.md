# Benchmark Evaluation Report: All Four Implemented Methods

**Station:** P041 (Marshall, Colorado) | **Latitude:** 39.9495°N | **Longitude:** 105.1943°W | **Elevation:** 1728.8 m  
**Year:** 2016 | **Target In-Situ Ground Truth:** ISMN COSMOS Cosmic-Ray Soil Moisture Probe (`cosmos_sm_median`, 0–25 cm depth)  
**Partition Protocol:** Strict Chronological 80/20 Holdout  
- **Training Period:** 2016-01-04 to 2016-10-06 ($N = 252$ valid dates)  
- **Official Held-Out Test Period:** 2016-10-07 to 2016-12-31 ($N = 63$ valid dates)  
- **Zero Test Leakage:** All feature scalers, baselines, regression weights, and hyperparameters were trained/tuned strictly on the 252 training days.

---

## 1. Executive Performance Summary

Across the 63 held-out test dates, the four implemented methods yielded the following primary evaluation metrics:

| Method / Model | Paradigm | Benchmark Tier | Test RMSE (m³/m³) | Test MAE (m³/m³) | Mean Bias (m³/m³) | ubRMSE (m³/m³) | Pearson r | Sequential Δr |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Climatology** | Empirical (Constant) | **Primary Benchmark** | **0.049958** | 0.046461 | +0.046387 | 0.018550 | NaN | NaN |
| **Precipitation OLS** | Empirical (API 1d, 3d, 7d) | **Primary Benchmark** | **0.051139** | 0.047687 | +0.047658 | 0.018545 | 0.036767 | +0.007242 |
| **Random Forest (Primary Median)** | ML (Bagging) | **Primary Benchmark** | **0.033697** | 0.025343 | +0.022714 | 0.024891 | **0.424853** | **+0.159010** |
| **XGBoost (Optimal Tuned Grid_092)** | ML (Gradient Boosting) | **Primary Benchmark** | **0.034783** | 0.030787 | +0.028665 | **0.019703** | 0.225266 | +0.061097 |
| *RF (Sensitivity Mean)* | ML (Bagging) | Sensitivity Analysis | 0.026531 | 0.019857 | +0.016216 | 0.020998 | 0.512156 | +0.170328 |
| *XGBoost (Default Untuned)* | ML (Gradient Boosting) | Sensitivity Analysis | 0.030437 | 0.024945 | +0.021180 | 0.021858 | 0.381512 | +0.063056 |

*Diagnostic Event Sensitivity Analysis (Excluding December 17 Anomaly, N = 62 Dates):*
- **Random Forest Median (Excl. Dec 17):** $\text{RMSE} = 0.028480\text{ m}^3/\text{m}^3$ ($-15.48\%$), $\text{ubRMSE} = 0.019529\text{ m}^3/\text{m}^3$, $r = 0.413505$
- **XGBoost Tuned (Excl. Dec 17):** $\text{RMSE} = 0.034435\text{ m}^3/\text{m}^3$ ($-1.00\%$), $\text{ubRMSE} = 0.019635\text{ m}^3/\text{m}^3$, $r = 0.134200$
- **Precipitation Baseline (Excl. Dec 17):** $\text{RMSE} = 0.051500\text{ m}^3/\text{m}^3$, $\text{Bias} = +0.048139\text{ m}^3/\text{m}^3$
- **Climatology Baseline (Excl. Dec 17):** $\text{RMSE} = 0.050318\text{ m}^3/\text{m}^3$, $\text{Bias} = +0.046874\text{ m}^3/\text{m}^3$

---

## 2. Metric Definitions & Scientific Significance

1. **Root Mean Square Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2}$$
   Quantifies overall retrieval accuracy in physical units (m³/m³), penalizing larger errors more heavily.
2. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|$$
   Measures average absolute magnitude of errors, providing a linear penalty that is less sensitive to isolated outliers than RMSE.
3. **Mean Bias Error (Bias):**
   $$\text{Bias} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$
   Measures systematic over- or under-estimation. Both empirical baselines have large positive biases (+0.046 to +0.048 m³/m³) because the training period had higher mean moisture (0.1257 m³/m³) than the dry late-autumn test period (0.0793 m³/m³).
4. **Unbiased RMSE (ubRMSE):**
   $$\text{ubRMSE} = \sqrt{\text{RMSE}^2 - \text{Bias}^2} = \sqrt{\frac{1}{N} \sum_{i=1}^N [(\hat{y}_i - y_i) - \text{Bias}]^2}$$
   Measures random error after removing the mean systematic bias. It isolates the model's ability to track dynamic fluctuations.
5. **Pearson Correlation Coefficient ($r$):**
   Measures linear co-variation between predictions and observations. Climatology has $r = \text{NaN}$ due to zero variance.
6. **Sequential Rate-of-Change Correlation (Seq $\Delta r$):**
   $$\Delta r = \text{corr}(\hat{y}_i - \hat{y}_{i-1}, y_i - y_{i-1})$$
   Tests whether day-to-day rate and direction of wetting/drying match ground observations.

---

## 3. Scientific Questions & Analysis

### Q1: Why can Random Forest have a lower RMSE than Baselines but a higher ubRMSE than XGBoost?
This behavior is fully explained by the **MSE Error Decomposition Theorem**:
$$\text{MSE} = \text{Bias}^2 + \text{ubRMSE}^2$$

- **Against Baselines:** The Climatology and Precipitation baselines have massive systematic biases ($+0.0464$ and $+0.0477\text{ m}^3/\text{m}^3$), which account for **$86.4\%$ and $86.8\%$ of their total MSE**. Random Forest dramatically reduces systematic bias to $+0.0227\text{ m}^3/\text{m}^3$ ($45.4\%$ of MSE), thereby reducing total RMSE from $0.0511$ down to $0.0337\text{ m}^3/\text{m}^3$ (a **$34.1\%$ error reduction**).
- **Against Tuned XGBoost:** Tuned XGBoost uses shallow depth-2 trees that regularize predictions into tighter bounds. On the extreme December 17 excursion, RF predicted $0.2553\text{ m}^3/\text{m}^3$ whereas XGBoost predicted $0.1615\text{ m}^3/\text{m}^3$. This single day inflated RF's variance around its mean bias, yielding $\text{ubRMSE} = 0.0249\text{ m}^3/\text{m}^3$ vs. XGBoost's $\text{ubRMSE} = 0.0197\text{ m}^3/\text{m}^3$. Excluding Dec 17, RF's ubRMSE drops to $0.0195\text{ m}^3/\text{m}^3$.

### Q2: Why did the Tuned XGBoost Model perform slightly worse on the Held-Out Test Set than the Default Model?
- **Model Selection Protocol:** The optimal tuned model (`Grid_092`: depth=2, lr=0.05, n_est=150) was selected strictly on the basis of **5-fold expanding-window temporal cross-validation on the training set** (OOF RMSE = $0.0384\text{ m}^3/\text{m}^3$ vs. Default OOF RMSE = $0.0416\text{ m}^3/\text{m}^3$).
- **Distribution Shift:** The training period (Jan 4 – Oct 6) spans spring wetting and summer precipitation dynamics where soil moisture ranged up to $0.28\text{ m}^3/\text{m}^3$ (mean $0.1257\text{ m}^3/\text{m}^3$). In contrast, the held-out test period (Oct 7 – Dec 31) was an unseasonal prolonged dry-down regime with mean soil moisture of only $0.0793\text{ m}^3/\text{m}^3$ and a narrow dynamic range ($0.05$ to $0.10\text{ m}^3/\text{m}^3$).
- **Depth Regularization vs. Default:** The shallow depth-2 model applied heavy conservative regularization that performed best across the broad dynamic range of the cross-validation folds, while the deeper default model (depth=6, lr=0.30) happened to split more aggressively on low-amplitude dry-season features in late autumn.
- **Scientific Integrity:** Selecting a model based on held-out test performance constitutes **test leakage**. Tuned `Grid_092` remains the designated primary configuration because it followed proper cross-validation selection on training data only.

### Q3: Why is the December 17 Sensitivity Analysis NOT the Primary Result?
- **Neutral Multi-Sensor Evidence:**
  1. *Meteorological Context:* On December 17 (DOY 352), a severe arctic cold surge entered the Front Range of Colorado, dropping surface temperatures to $-24^\circ\text{C}$.
  2. *COSMOS Probe Observation:* The COSMOS cosmic-ray sensor reported 6 valid hourly measurements between 00:00 and 06:00 UTC with quality flag `'G'` (mean $= 0.1095\text{ m}^3/\text{m}^3$). After 06:00 UTC, the probe triggered sensor quality flag `'D03'` (associated with soil freezing/snow moderation effects), resulting in no valid `'G'` data for the remaining 17 hours of the day (and zero valid data for Dec 18–20).
  3. *GNSS-IR Observable:* Quality-passed GPS tracks recorded prior to 05:00 UTC showed normal baseline reflector height ($H \approx 1.956\text{ m}$). Tracks starting after 05:00 UTC across all azimuth quadrants exhibited a sudden drop to $H \approx 1.83\text{ m}$ (a $\sim 12.7\text{ cm}$ drop in apparent vertical height).
  4. *Inversion Mapping:* The empirical decision trees (trained during non-freezing conditions where lower reflector height signifies shallower reflection from wetter soil) mapped $H = 1.83\text{ m}$ to higher moisture values (RF: $0.2553\text{ m}^3/\text{m}^3$, XGBoost: $0.1615\text{ m}^3/\text{m}^3$).
- **Benchmark Governance:** In standard scientific practice, held-out test sets must remain complete and unedited. Excluding DOY 352 is reported strictly as a **diagnostic sensitivity analysis** to demonstrate that operational deployment under sub-zero conditions requires thermal/frost gating ($T < 0^\circ\text{C}$).

---

## 4. Organization of Output Files

All outputs have been placed in standard, persistent project locations:

1. **Prediction Tables (`data/processed/p041/`):**
   - `p041_2016_climatology_predictions.csv` (Dedicated Climatology predictions)
   - `p041_2016_precipitation_predictions.csv` (Dedicated Precipitation OLS predictions)
   - `p041_2016_rf_predictions.csv` (Dedicated Random Forest predictions)
   - `p041_2016_xgboost_predictions.csv` (Dedicated XGBoost predictions)
   - `p041_2016_all_methods_predictions_aligned.csv` (Consolidated 63-date master prediction table)
2. **Metrics Table (`data/processed/p041/`):**
   - `p041_2016_benchmark_metrics_all_methods.csv` (Master comparison metrics table)
3. **Machine-Readable Configurations (`results/configs/`):**
   - `climatology_config.json`
   - `precipitation_ols_config.json`
   - `random_forest_config.json`
   - `xgboost_config.json`
   - `all_models_config.json`
4. **Publication Figures (`results/figures/`):**
   - `benchmark_timeseries_all_methods.png`
   - `benchmark_performance_all_methods.png`
5. **Execution Scripts (`scripts/`):**
   - `run_climatology_baseline.py`
   - `run_precipitation_baseline.py`
   - `evaluate_rf_test.py`
   - `evaluate_xgboost_test.py`
   - `generate_consolidated_results.py`
