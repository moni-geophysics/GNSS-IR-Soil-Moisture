# Machine Learning vs. Physical and Empirical Benchmarks for GNSS-IR Soil Moisture Retrieval

[![Tests](https://img.shields.io/badge/tests-40%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

This repository contains the leakage-safe experimental framework, baseline implementations, Random Forest models, and benchmark audit for daily soil moisture estimation using Global Navigation Satellite System Interferometric Reflectometry (GNSS-IR) at station **P041** (Marshall, Colorado) across the full calendar year 2016.

---

## 1. Research Question

> **"Does machine learning improve the estimation of soil-moisture variations from GNSS-IR observables compared with a physically based retrieval and a precipitation-based baseline, when evaluated against independent COSMOS measurements under leakage-safe temporal validation?"**

---

## 2. Experimental Setup & Data

### 2.1 Target Station and Sensors
- **GNSS Station:** `P041` (Marshall, Colorado; UNAVCO Plate Boundary Observatory / EarthScope).
  - Receiver: Trimble NETR9; Antenna: TRM59800.00 with SCIT radome.
  - Observation Period: Full year 2016 (DOY 001 to DOY 366; 2016-01-01 to 2016-12-31).
  - Raw Data: GPS L1 C/A SNR observations across ascending and descending satellite arcs.
- **Independent In Situ Target:** Cosmic-Ray Neutron Probe (COSMOS) at ISMN station `Marshall`.
  - Representative footprint: ~200 m radius, integrating soil moisture across 0–30 cm effective depth.
  - Target variable: Daily median volumetric soil moisture (`target_cosmos_sm_median`, $\text{m}^3/\text{m}^3$).

### 2.2 Chronological 80/20 Partition
To prevent temporal data leakage and autocorrelation contamination, the dataset is partitioned chronologically across the 315 valid observation days:
- **Training Period:** 252 dates (2016-01-04 to 2016-10-06, 80.0%)
- **Held-Out Test Period:** 63 dates (2016-10-07 to 2016-12-31, 20.0%)

```
2016-01-04                                     2016-10-06   2016-10-07                     2016-12-31
|---------------------------------------------------------|   |---------------------------------------|
                  TRAINING SET (N = 252)                                HELD-OUT TEST SET (N = 63)
             5-Fold Expanding Temporal CV                             Frozen Final Evaluation
```

### 2.3 Environmental Distribution Shift
The continuous chronological split exposes the models to a realistic seasonal dry-down distribution shift:
- **Train Ground Truth ($N=252$):** Mean = $0.1257\text{ m}^3/\text{m}^3$, Std = $0.0550$, Range = $[0.0600, 0.4210]$
- **Test Ground Truth ($N=63$):** Mean = $0.0793\text{ m}^3/\text{m}^3$, Std = $0.0187$, Range = $[0.0570, 0.1280]$
- **Distribution Shift ($\Delta\mu$):** $-0.0464\text{ m}^3/\text{m}^3$ ($-36.91\%$), representing late-autumn drying and freezing conditions.

---

## 3. Feature Engineering & Daily Aggregation

For every raw GNSS multipath arc, Lomb-Scargle periodograms and non-linear least-squares fitting extract three fundamental observables:
1. **Reflector Height ($h_r$):** Distance between antenna phase center and reflecting surface (meters).
2. **Normalized Multipath Amplitude ($A$):** Amplitude of the interferometric oscillation relative to direct signal.
3. **Interferometric Phase ($\phi$):** Relative phase offset (radians).

Raw arcs are filtered using strict quality control criteria (`qc_flagged == False`). Daily aggregation is performed using two independent schemes:
- **Median Aggregation (Primary):** Daily median of $h_r$, daily median of $A$, and intrinsic circular median of $\phi$ (minimizing angular geodesic dispersion on $[-\pi, \pi]$).
- **Mean Aggregation (Secondary Sensitivity):** Daily arithmetic mean of $h_r$, daily arithmetic mean of $A$, and circular vector mean of $\phi$ via resultant vector $(\bar{C}, \bar{S})$.

Phase angles are projected into circular components to avoid boundary discontinuity:
$$\text{phase\_cos} = \cos(\phi), \quad \text{phase\_sin} = \sin(\phi)$$

Final feature dimension: 4 features (`rh`, `amp`, `phase_cos`, `phase_sin`).

---

## 4. Model Architectures & Selection

### 4.1 Primary Model: Random Forest Regressor
- **Architecture:** Scikit-Learn `RandomForestRegressor`.
- **Hyperparameter Tuning:** 5-fold expanding-window temporal cross-validation on the 252 training dates (evaluating 24 hyperparameter configurations; zero test access).
- **Locked Hyperparameters:**
  - `n_estimators`: 300
  - `min_samples_leaf`: 3
  - `max_features`: 1.0 (all 4 features evaluated at each split)
  - `max_depth`: None (unconstrained tree depth)
  - `random_state`: 42
- **Models:**
  - **RF Median (Primary):** Trained on median-aggregated daily features.
  - **RF Mean (Secondary Sensitivity):** Trained on mean-aggregated daily features.

### 4.2 Benchmark Baselines
1. **Climatology Baseline:**
   - Predicts a flat constant equal to the training-period mean ground truth ($\hat{y}_t = 0.125673\text{ m}^3/\text{m}^3$).
   - Serves as the zero-information baseline.
2. **Precipitation Baseline:**
   - Multiple Linear Regression (OLS) fitted strictly on the 252 training dates using backward-looking cumulative precipitation windows ($1\text{d}, 3\text{d}, 7\text{d}$ antecedent sums) from ISMN records.
   - Evaluates whether GNSS-IR provides information beyond meteorological precipitation forcing.
3. **Physical GNSS-IR Retrieval (Pending):**
   - Single-layer Fresnel amplitude-ratio inversion (Jing et al. 2019 / Chew et al. 2014).
   - **Status:** **PENDING**. Initial DOY 001 feasibility tests revealed severe mathematical instability (13.4% complex-valued phasor radicand violations and 2nd-order detrending boundary collapse at $0.50\text{ m}$). Proper full-year inversion requires antenna choke-ring gain pattern de-embedding and site-specific dielectric mixing models.

---

## 5. Audited Benchmark Results

Evaluation on the **63 held-out test dates (2016-10-07 to 2016-12-31)**. All metrics computed with double-precision arithmetic.

### 5.1 Official Frozen Benchmark ($N = 63$ Dates)

| Model | Paradigm | RMSE ($\text{m}^3/\text{m}^3$) | MAE ($\text{m}^3/\text{m}^3$) | Mean Bias ($\text{m}^3/\text{m}^3$) | ubRMSE ($\text{m}^3/\text{m}^3$) | Pearson $r$ | Sequential $\Delta r$ | Calendar $\Delta r$ | Causal $r_{\text{anom}}$ | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **RF Median (Primary)** | Machine Learning | **0.0337** | **0.0253** | +0.0227 | 0.0249 | **0.4249** | **+0.1590** | **+0.1578** | **0.3148** | **Frozen Benchmark** |
| **RF Mean (Sensitivity)** | Machine Learning | **0.0265** | **0.0199** | +0.0162 | 0.0210 | **0.5122** | **+0.1703** | **+0.1691** | **0.3935** | **Frozen Benchmark** |
| **Precipitation Baseline** | Empirical (OLS) | 0.0511 | 0.0477 | +0.0477 | 0.0185 | 0.0368 | +0.0072 | +0.0071 | 0.0481 | **Frozen Benchmark** |
| **Climatology Baseline** | Empirical (Mean) | 0.0500 | 0.0465 | +0.0464 | 0.0185 | N/A | N/A | N/A | 0.0307 | **Frozen Benchmark** |
| **Physical GNSS-IR Inversion** | Physics-Based | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending Calibration* |

*Notes:*
- **Sequential $\Delta r$:** Correlation of first differences across sequential available dates ($N = 62$).
- **Calendar $\Delta r$:** Correlation of first differences strictly across consecutive calendar days ($\Delta t = 1$ day, $N = 57$).
- **Causal $r_{\text{anom}}$:** Correlation of daily anomalies computed against a causal, strictly backward-looking 30-day trailing baseline ($\text{shift}(1)$ ensures zero lookahead).

### 5.2 Resolution of the ubRMSE "Paradox"
A superficial inspection shows that Climatology achieves a lower nominal ubRMSE ($0.0185\text{ m}^3/\text{m}^3$) than RF Median ($0.0249\text{ m}^3/\text{m}^3$). This is a well-known mathematical artifact of constant predictions:
- For any constant prediction $\hat{y} = C$, $\text{ubRMSE} \equiv \mathrm{std}(y_{\text{test}})$. Here, $\mathrm{std}(y_{\text{test}}) = 0.0187\text{ m}^3/\text{m}^3$.
- Climatology outputs a flat line: it explains **0% of dynamic temporal variance** ($r$ undefined, $\Delta r$ undefined).
- In contrast, RF Median reduces total RMSE by **32.6%** relative to Climatology ($0.0337$ vs. $0.0500$) and tracks dynamic temporal variations ($r = 0.4249$, $r_{\text{anom}} = 0.3148$, $\Delta r = +0.1590$).
- When decomposing Mean Squared Error ($\text{MSE} = \text{Bias}^2 + \text{ubRMSE}^2$):
  - Climatology error is **86.2% systematic bias** due to seasonal shift.
  - RF Median cuts mean bias by more than half (+0.0227 vs. +0.0464).

---

## 6. December 17 Anomaly Audit & Diagnostic Sensitivity

### 6.1 Anomaly Decomposition (2016-12-17 / DOY 352)
On 2016-12-17, an extreme prediction error occurred:
- Ground Truth COSMOS SM: $0.1095\text{ m}^3/\text{m}^3$
- RF Median Prediction: $0.2553\text{ m}^3/\text{m}^3$ (Error: $+0.1458\text{ m}^3/\text{m}^3$)
- RF Mean Prediction: $0.2138\text{ m}^3/\text{m}^3$ (Error: $+0.1043\text{ m}^3/\text{m}^3$)
- **Impact:** This single day contributes **$18.8\%$ of the total MSE** for RF Median across the entire 63-day test set.

### 6.2 Physical Mechanism Audit
- **Reflector Height Drop:** Mean reflector height abruptly dropped by $>11\text{ cm}$ (from $1.9608\text{ m}$ on DOY 351 to $1.8488\text{ m}$ on DOY 352).
- **Environmental Context:** Collocated ISMN meteorological data shows air temperature dropped into extreme freezing range ($-17.2^\circ\text{C}$ to $-24.0^\circ\text{C}$) accompanied by high relative humidity ($>85\%$).
- **Hydrological Interpretation:** The surface reflector height drop is consistent with shallow snow cover or a frozen surface boundary altering the electromagnetic reflection phase center, whereas the cosmic-ray probe (integrating down to 30 cm) measured liquid/effective dielectric response at depth.
- Under strict scientific safeguards, **December 17 is retained in the official frozen benchmark table**.

### 6.3 Diagnostic Sensitivity Analysis (Excluding Dec 17, $N = 62$)

| Model | RMSE ($\text{m}^3/\text{m}^3$) | MAE ($\text{m}^3/\text{m}^3$) | Bias ($\text{m}^3/\text{m}^3$) | ubRMSE ($\text{m}^3/\text{m}^3$) | Pearson $r$ | Sequential $\Delta r$ | Causal $r_{\text{anom}}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RF Median (Excl. Dec 17)** | 0.0285 | 0.0234 | +0.0207 | 0.0195 | 0.4135 | +0.2110 | 0.3810 |
| **RF Mean (Excl. Dec 17)** | 0.0232 | 0.0185 | +0.0148 | 0.0179 | 0.5074 | +0.1913 | 0.4463 |
| **Precipitation Baseline** | 0.0515 | 0.0482 | +0.0481 | 0.0183 | 0.0298 | +0.0083 | 0.0674 |
| **Climatology Baseline** | 0.0503 | 0.0469 | +0.0469 | 0.0183 | N/A | N/A | 0.0501 |

Excluding the Dec 17 surface anomaly drops RF Median ubRMSE from $0.0249$ to $0.0195\text{ m}^3/\text{m}^3$, increases day-to-day dynamic tracking $\Delta r$ to $+0.2110$, and increases anomaly tracking $r_{\text{anom}}$ to $0.3810$.

---

## 7. End-to-End Pipeline & Reproduction

### 7.1 Installation
```bash
git clone https://github.com/tinkuraviteja/GNSS-IR-Soil-Moisture.git
cd GNSS-IR-Soil-Moisture
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 7.2 Reproduction Workflow

#### Stage 0: Upstream Data Ingestion (Prerequisites & Reproducibility Boundary)
> [!IMPORTANT]
> **Complete raw-to-results reproduction is not yet available from tracked files alone.**
> Generating the intermediate feature and validation tables requires local raw archives (`data/raw/` SNR files and the ISMN archive `E:/MTP Data/P041.zip`) and two upstream preprocessing scripts currently housed in `scratch/`:
> - `scratch/process_full_year_2016.py` currently generates `data/processed/p041/p041_2016_annual_gnss_ir_features.csv`
> - `scratch/prepare_daily_validation_dataset.py` currently generates `data/processed/p041/p041_2016_daily_gnss_cosmos_validation.csv`
>
> These upstream scripts are not yet part of the tracked reproducible pipeline. The published scripts in `scripts/` (Stages 1 through 5 below) reproduce all subsequent daily feature aggregations, temporal cross-validation, baseline construction, final Random Forest test evaluation, and benchmark audits directly from those intermediate files.

#### Stage 1: Daily Feature Preparation
Generates circular and arithmetic daily aggregations, filters valid QC arcs, aligns with COSMOS targets, and establishes the 80/20 chronological partition:
```bash
python scripts/prepare_daily_features.py
```
*Outputs: `data/processed/p041/p041_2016_daily_mean_dataset.csv`, `p041_2016_daily_median_dataset.csv`*

#### Stage 2: Temporal Cross-Validation & Hyperparameter Tuning
Executes 5-fold expanding-window cross-validation across 24 parameter grids strictly on the 252 training dates:
```bash
python scripts/tune_rf_temporal_cv.py
```
*Outputs: `data/processed/p041/p041_2016_rf_temporal_cv_summary.csv`, `p041_2016_rf_temporal_cv_fold_metrics.csv`*

#### Stage 3: Baseline Construction
Fits Climatology and OLS Precipitation models on the training partition:
```bash
python scripts/run_climatology_baseline.py
python scripts/run_precipitation_baseline.py
```
*Outputs: `data/processed/p041/p041_2016_climatology_predictions.csv`, `p041_2016_precipitation_predictions.csv`*

#### Stage 4: Locked Final Test Evaluation
Fits the locked RF models on the 252 training dates and evaluates on the 63 held-out dates:
```bash
python scripts/evaluate_rf_test.py
```
*Outputs: `data/processed/p041/p041_2016_rf_final_test_predictions.csv`, `p041_2016_rf_final_test_metrics.csv`*

#### Stage 5: Final Benchmark Audit
Recomputes full-precision metrics, error decompositions, difference correlations, and anomaly metrics:
```bash
python scripts/audit_final_benchmark.py
```
*Outputs: `data/processed/p041/p041_2016_final_benchmark_metrics.csv`, `p041_2016_final_benchmark_predictions_aligned.csv`*

### 7.3 Automated Test Suite
Run pytest to verify package integrity and unit test passes:
```bash
pytest
```
*Expected: 40 passed in ~5 seconds.*

---

## 8. Repository Structure

```
GNSS-IR-Soil-Moisture/
├── .gitignore                          # Clean ignore rules (protects data/, scratch/, venv/)
├── README.md                           # Master publication and benchmark documentation
├── pytest.ini                          # Test configuration
├── requirements.txt                    # Python package dependencies
├── src/
│   └── gnss_ir/
│       ├── __init__.py
│       ├── arc_detection.py            # GNSS satellite arc extraction
│       ├── data_quality.py             # QC filtering and SNR validation
│       ├── fresnel.py                  # First Fresnel zone spatial footprints
│       ├── ismn.py                     # ISMN archive parsing & stream reading
│       ├── lsp.py                      # Lomb-Scargle periodogram estimation
│       ├── processing.py               # Single-arc NLS feature extraction
│       └── refraction.py               # Atmospheric tropospheric delay corrections
├── scripts/
│   ├── prepare_daily_features.py       # Stage 1: Circular daily feature aggregation & 80/20 split
│   ├── tune_rf_temporal_cv.py          # Stage 2: 5-fold expanding temporal CV grid search
│   ├── run_climatology_baseline.py     # Stage 3a: Climatology baseline generation
│   ├── run_precipitation_baseline.py   # Stage 3b: OLS rolling precipitation baseline
│   ├── evaluate_rf_test.py             # Stage 4: Locked RF evaluation on held-out test set
│   └── audit_final_benchmark.py        # Stage 5: Comprehensive multi-metric benchmark audit
└── tests/
    ├── test_climatology_baseline.py
    ├── test_data_quality.py
    ├── test_ismn.py
    ├── test_precipitation_baseline.py
    ├── test_refraction.py
    └── test_single_arc_processing.py
```

---

## 9. Key Scientific Conclusions

1. **Machine Learning vs. Baselines:** Under leakage-safe temporal validation, Random Forest substantially outperforms both the zero-information climatology baseline and the precipitation-driven empirical baseline in total error (RMSE $0.0337$ vs. $0.0500\text{ m}^3/\text{m}^3$) and dynamic variation tracking ($r = 0.4249$, $r_{\text{anom}} = 0.3148$, $\Delta r = +0.1590$).
2. **Precipitation Baseline Insufficiency:** Antecedent precipitation alone is unable to track soil moisture dynamics during the late-autumn dry-down ($r = 0.0368$, $\Delta r = +0.0072$), proving that GNSS-IR observables provide independent surface information.
3. **Median vs. Mean Aggregation:** Median aggregation provides superior physical robustness across the training period and temporal CV, while Mean aggregation performs well on the test period; both confirm the predictive value of GNSS-IR.
4. **Physical Retrieval Status:** Direct empirical inversion remains pending due to mathematical instability in uncalibrated single-layer phasor models during winter conditions.
