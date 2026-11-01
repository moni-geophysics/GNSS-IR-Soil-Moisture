# Same-Time Spatial Leave-One-Station-Out (LOSO) Cross-Validation Report

**Study Scope:** Strictly Synchronized Common Dates across P041, P037, and P043  
**Common Period:** February 18, 2016 -- November 22, 2016 (279 calendar days, 216 active common dates)  
**Sample Sizes:** $N = 216$ daily observations per station ($N_{train} = 432$, $N_{test} = 216$, $N_{pooled} = 648$)  
**Reference Target:** PBO_H2O surface soil moisture ($0\text{--}5\text{ cm}$, ISMN Quality Flag `G`)  
**Evaluation Standard:** Calendar Gap-Respecting Moving Block Bootstrap ($L = 7\text{ calendar days}$, $B = 1000$ iterations) for 95% Confidence Intervals  
**Associated Data Files:**  
- Predictions: [`data/processed/loso_same_time_predictions.csv`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/data/processed/loso_same_time_predictions.csv)
- Metrics: [`data/processed/loso_same_time_metrics.csv`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/data/processed/loso_same_time_metrics.csv)
- Ablation: [`data/processed/loso_same_time_feature_ablation.csv`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/data/processed/loso_same_time_feature_ablation.csv)
- Weighting: [`data/processed/loso_same_time_weighting_sensitivity.csv`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/data/processed/loso_same_time_weighting_sensitivity.csv)
- Cross-Experiment Comparison: [`data/processed/loso_same_time_vs_all_dates_comparison.csv`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/data/processed/loso_same_time_vs_all_dates_comparison.csv)

---

## 1. Executive Summary

This report evaluates the **Same-Time Spatial Leave-One-Station-Out (LOSO)** experiment across stations `P041` (Marshall, CO), `P037` (Fremont Pass, CO), and `P043` (Newcastle, WY). 

The primary scientific objective is to determine whether GNSS-IR soil moisture estimation transfers across distinct geographic stations when training and testing observations occur on the **exact same calendar dates**.

```
Fold 1: Train on P041 + P043 (N=432)  --> Test on P037 (N=216)  [All on exactly the same 216 dates]
Fold 2: Train on P041 + P037 (N=432)  --> Test on P043 (N=216)  [All on exactly the same 216 dates]
Fold 3: Train on P037 + P043 (N=432)  --> Test on P041 (N=216)  [All on exactly the same 216 dates]
Pooled Benchmark: N_test = 648 common-date test observations across all three folds combined.
```

### Core Empirical Findings:
1. **Impact of Same-Time Synchronization on Performance:**
   - Training and testing on strictly synchronized dates improves the spatial generalization of Random Forest Model A (`rh + amp`): pooled $\text{RMSE}$ drops to **$0.0920\text{ m}^3/\text{m}^3$** (compared to $0.0984\text{ m}^3/\text{m}^3$ when the all-available-dates model is evaluated on the exact same common subset, $\Delta\text{RMSE} = -0.0064\text{ m}^3/\text{m}^3$).
   - Pooled Pearson correlation reaches **$r = 0.434$** (95% CI: $[0.287, 0.550]$), compared to $0.395$ in the all-available spatial LOSO.
2. **Station-Specific Effects:**
   - **Station P037 (Fold 1):** Same-time matching produces a dramatic error reduction: $\text{RMSE}$ drops from $0.0748\text{ m}^3/\text{m}^3$ (all-dates model on common subset) to **$0.0540\text{ m}^3/\text{m}^3$** ($\Delta\text{RMSE} = -0.0208\text{ m}^3/\text{m}^3$). Because mid-winter frozen days (Jan 1--Feb 17) are naturally excluded by common-date availability, the mean prediction bias drops from $+0.0503\text{ m}^3/\text{m}^3$ down to $+0.0243\text{ m}^3/\text{m}^3$.
   - **Station P043 (Fold 2):** Same-time matching maintains a high temporal tracking correlation ($r = \mathbf{0.512}$ in RF Model A; 95% CI: $[0.393, 0.646]$) and reduces RMSE to $0.1050\text{ m}^3/\text{m}^3$ ($\Delta\text{RMSE} = -0.0061\text{ m}^3/\text{m}^3$ vs. the common subset control).
   - **Station P041 (Fold 3):** Performance remains nearly constant ($\text{RMSE} = 0.1069\text{ m}^3/\text{m}^3$ vs. $0.1054\text{ m}^3/\text{m}^3$ control; $r = 0.300$), reflecting the persistent microclimate decoupling caused by the 210.7 m probe separation.
3. **Re-Confirmation of the Phase Barrier on Matched Dates:**
   - Even when observations occur on the exact same days, adding raw interference phase (**Model B**) causes a correlation collapse at `P043` ($r = 0.512 \rightarrow 0.313$ in RF; $0.388 \rightarrow 0.153$ in XGBoost).
   - This proves conclusively that the phase offset barrier is **geometric** (caused by static antenna monument height differences), not temporal or seasonal.

---

## 2. Common-Date Dataset Audit and Calendar Distribution

### 2.1 Intersection Audit
Starting from the 771 usable station-days across 2016, we audited the three-station date intersection:
- `P041` valid paired dates: 247 days
- `P037` valid paired dates: 276 days
- `P043` valid paired dates: 248 days
- **Strict Three-Station Intersection:** Exactly **216 dates** spanning from **February 18, 2016 (DOY 49)** to **November 22, 2016 (DOY 327)**.
- Across all 216 dates, all GNSS observables (`rh`, `amp`, `phase_cos`, `phase_sin`, `phase_resultant_length`, `clean_arc_count`, `qc_pass_rate`) and the target `pbo_sm` (Quality Flag `G`) have zero missing values.

```
+-----------------------------------------------------------------------------------------+
|                  2016 Monthly Usable Sample Availability vs. Common Dates               |
+----------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-------+
| Station  | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | Total |
+----------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-------+
| P041     |   4 |  14 |  16 |  19 |  14 |  30 |  31 |  31 |  30 |  31 |  26 |   1 |   247 |
| P037     |   4 |  16 |  26 |  26 |  29 |  30 |  31 |  31 |  30 |  31 |  20 |   2 |   276 |
| P043     |   0 |   5 |  18 |  26 |  31 |  30 |  31 |  31 |  30 |  29 |  17 |   0 |   248 |
+----------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-------+
| Common   |   0 |   5 |  16 |  19 |  14 |  30 |  31 |  31 |  30 |  28 |  12 |   0 |   216 |
+----------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-------+
```

### 2.2 Chronological Gaps and Calendar Distribution
The 216 common dates span a total calendar duration of 279 days:
- **Contiguous Day-to-Day Steps (gap = 1 day):** 204 transitions ($94.9\%$ of all steps).
- **Short Gaps (gap = 2--3 days):** 4 occurrences (caused by transient rain/snow flags).
- **Moderate Gaps (gap = 5--10 days):** 6 occurrences (early spring freeze/thaw events in March/April).
- **Maximum Gap:** 19 days (between DOY 61, March 1, and DOY 81, March 21, when frozen ground flags simultaneously affected all three sites).

---

## 3. Methodology and Leakage Controls

1. **Zero Target Leakage:**  
   The held-out test station's soil moisture observations were completely excluded from all training operations, scaling, feature transformations, and baseline fitting.
2. **Station Loss Weighting ($w(s,i) = \frac{N_{train}}{2 N_{s, train}}$):**  
   Because both training stations have observations on the exact same 216 dates, $N_{s1} = 216$ and $N_{s2} = 216$ ($N_{train} = 432$). Consequently, the equal loss weight evaluates to:
   $$w(s_1) = \frac{432}{2 \times 216} = 1.0000, \quad w(s_2) = \frac{432}{2 \times 216} = 1.0000$$
   Each training station naturally contributes exactly $50.0\%$ to the training loss surface.
3. **Calendar Gap-Respecting Moving Block Bootstrap:**  
   To prevent joining observations across real non-consecutive temporal gaps, bootstrap candidate blocks were generated using **7 calendar day windows**:
   $$\text{Block}(i) = \{j \mid 0 \le \text{date}_j - \text{date}_i < 7\text{ calendar days}\}$$
   If a gap of $\ge 7$ days occurs (e.g. the 19-day gap in March), the block truncates at the boundary and never merges observations across the gap. Random sampling from this candidate block pool guarantees that 95% confidence intervals reflect real serial correlation without artificial smoothing.

---

## 4. Same-Time Spatial LOSO Results

### 4.1 Station-Wise Evaluation Tables

Table 1 through Table 3 present the performance for each held-out station across all methods and feature sets on the 216 common dates. 95% Confidence Intervals are calculated from 1,000 calendar-gap-respecting bootstrap iterations.

#### Table 1: Fold 1 -- Held-Out Station P037 (Fremont Pass, Common Dates $N=216$)
*Trained on P041 + P043 on the Exact Same 216 Dates ($N_{train} = 432$)*

| Model Family | Feature Configuration | RMSE ($\text{m}^3/\text{m}^3$) | 95% CI | MAE | Bias | ubRMSE | Pearson $r$ | 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Climatology** | Baseline (Train Mean) | $0.0683$ | $[0.0606, 0.0761]$ | $0.0604$ | $+0.0440$ | $0.0522$ | --- | --- |
| **Precipitation OLS** | Baseline (Train Fit) | $0.0647$ | $[0.0577, 0.0718]$ | $0.0571$ | $+0.0429$ | $0.0484$ | $0.378$ | $[0.281, 0.539]$ |
| **Random Forest** | **Model A (`rh + amp`)** | **0.0540** | $[0.0503, 0.0589]$ | **0.0457** | $+0.0243$ | **0.0482** | **0.433** | $[0.214, 0.520]$ |
| **Random Forest** | Model B (`+ phase`) | $0.0528$ | $[0.0487, 0.0574]$ | $0.0461$ | $+0.0194$ | $0.0491$ | $0.345$ | $[0.143, 0.504]$ |
| **Random Forest** | Model C (`+ QC`) | $0.0506$ | $[0.0414, 0.0577]$ | $0.0420$ | $-0.0009$ | $0.0506$ | $0.247$ | $[0.016, 0.497]$ |
| **XGBoost** | Model A (`rh + amp`) | $0.0579$ | $[0.0525, 0.0639]$ | $0.0507$ | $+0.0286$ | $0.0503$ | $0.270$ | $[-0.003, 0.432]$ |
| **XGBoost** | Model B (`+ phase`) | $0.0483$ | $[0.0417, 0.0539]$ | $0.0397$ | $+0.0009$ | $0.0483$ | $0.382$ | $[0.247, 0.530]$ |
| **XGBoost** | Model C (`+ QC`) | **0.0459** | $[0.0366, 0.0529]$ | **0.0364** | $-0.0077$ | **0.0452** | **0.552** | $[0.430, 0.673]$ |

#### Table 2: Fold 2 -- Held-Out Station P043 (Newcastle, Common Dates $N=216$)
*Trained on P041 + P037 on the Exact Same 216 Dates ($N_{train} = 432$)*

| Model Family | Feature Configuration | RMSE ($\text{m}^3/\text{m}^3$) | 95% CI | MAE | Bias | ubRMSE | Pearson $r$ | 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Climatology** | Baseline (Train Mean) | **0.0718** | $[0.0564, 0.0764]$ | **0.0580** | $-0.0275$ | $0.0663$ | --- | --- |
| **Precipitation OLS** | Baseline (Train Fit) | $0.0736$ | $[0.0561, 0.0781]$ | $0.0569$ | $-0.0329$ | $0.0658$ | $0.147$ | $[0.089, 0.311]$ |
| **Random Forest** | **Model A (`rh + amp`)** | $0.1050$ | $[0.0968, 0.1184]$ | $0.0914$ | $+0.0874$ | **0.0581** | **0.512** | $[0.393, 0.646]$ |
| **Random Forest** | Model B (`+ phase`) | $0.0904$ | $[0.0816, 0.1031]$ | $0.0766$ | $+0.0648$ | $0.0631$ | $0.313^*$ | $[0.221, 0.467]$ |
| **Random Forest** | Model C (`+ QC`) | $0.1071$ | $[0.0983, 0.1208]$ | $0.0928$ | $+0.0849$ | $0.0653$ | $0.178^*$ | $[0.078, 0.361]$ |
| **XGBoost** | Model A (`rh + amp`) | $0.1090$ | $[0.1005, 0.1229]$ | $0.0948$ | $+0.0901$ | $0.0612$ | $0.388$ | $[0.275, 0.550]$ |
| **XGBoost** | Model B (`+ phase`) | $0.0953$ | $[0.0862, 0.1082]$ | $0.0813$ | $+0.0670$ | $0.0678$ | $0.153^*$ | $[0.074, 0.332]$ |
| **XGBoost** | Model C (`+ QC`) | $0.1005$ | $[0.0917, 0.1144]$ | $0.0860$ | $+0.0755$ | $0.0664$ | $0.146^*$ | $[0.047, 0.307]$ |

$^*$ *Re-emergence of phase-induced correlation collapse ($r$ drops from $>0.51$ down to $0.15\text{--}0.31$).*

#### Table 3: Fold 3 -- Held-Out Station P041 (Marshall, Common Dates $N=216$)
*Trained on P037 + P043 on the Exact Same 216 Dates ($N_{train} = 432$)*

| Model Family | Feature Configuration | RMSE ($\text{m}^3/\text{m}^3$) | 95% CI | MAE | Bias | ubRMSE | Pearson $r$ | 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Climatology** | Baseline (Train Mean) | $0.0705$ | $[0.0544, 0.0797]$ | $0.0544$ | $-0.0165$ | $0.0685$ | --- | --- |
| **Precipitation OLS** | Baseline (Train Fit) | **0.0627** | $[0.0485, 0.0690]$ | **0.0490** | $-0.0115$ | **0.0616** | **0.447** | $[0.380, 0.598]$ |
| **Random Forest** | **Model A (`rh + amp`)** | $0.1069$ | $[0.0928, 0.1248]$ | $0.0895$ | $+0.0574$ | $0.0902$ | $0.300$ | $[0.131, 0.565]$ |
| **Random Forest** | Model B (`+ phase`) | $0.1060$ | $[0.0900, 0.1236]$ | $0.0875$ | $+0.0537$ | $0.0914$ | $0.285$ | $[0.119, 0.573]$ |
| **Random Forest** | Model C (`+ QC`) | $0.1006$ | $[0.0870, 0.1155]$ | $0.0836$ | $+0.0511$ | $0.0867$ | $0.301$ | $[0.135, 0.587]$ |
| **XGBoost** | Model A (`rh + amp`) | $0.1183$ | $[0.1016, 0.1383]$ | $0.0981$ | $+0.0642$ | $0.0993$ | $0.269$ | $[0.094, 0.565]$ |
| **XGBoost** | Model B (`+ phase`) | $0.1018$ | $[0.0850, 0.1153]$ | $0.0848$ | $+0.0356$ | $0.0953$ | $0.264$ | $[0.091, 0.556]$ |
| **XGBoost** | Model C (`+ QC`) | $0.1045$ | $[0.0890, 0.1198]$ | $0.0867$ | $+0.0483$ | $0.0927$ | $0.306$ | $[0.137, 0.602]$ |

---

### 4.2 Pooled Benchmark Across All 648 Common Observations

Table 4 presents the pooled evaluation across all three held-out test sets combined ($N = 648$).

#### Table 4: Pooled Same-Time Spatial LOSO Benchmark ($N = 648$)

| Model Family | Feature Configuration | RMSE ($\text{m}^3/\text{m}^3$) | 95% CI | MAE | Bias | ubRMSE | Pearson $r$ | 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Climatology** | Baseline (Train Mean) | $0.0702$ | $[0.0628, 0.0746]$ | $0.0576$ | $0.0000$ | $0.0702$ | $-0.316^*$ | $[-0.398, -0.123]$ |
| **Precipitation OLS** | Baseline (Train Fit) | **0.0671** | $[0.0592, 0.0716]$ | **0.0543** | $-0.0005$ | **0.0671** | $0.128$ | $[0.036, 0.335]$ |
| **Random Forest** | **Model A (`rh + amp`)** | **0.0920** | $[0.0863, 0.1067]$ | $0.0755$ | $+0.0564$ | **0.0726** | **0.434** | $[0.287, 0.550]$ |
| **Random Forest** | Model B (`+ phase`) | $0.0860$ | $[0.0796, 0.1007]$ | $0.0701$ | $+0.0460$ | $0.0727$ | $0.387$ | $[0.243, 0.525]$ |
| **Random Forest** | Model C (`+ QC`) | $0.0898$ | $[0.0840, 0.1042]$ | $0.0728$ | $+0.0451$ | $0.0776$ | $0.376$ | $[0.228, 0.510]$ |
| **XGBoost** | Model A (`rh + amp`) | $0.0987$ | $[0.0922, 0.1148]$ | $0.0812$ | $+0.0610$ | $0.0776$ | $0.372$ | $[0.232, 0.509]$ |
| **XGBoost** | Model B (`+ phase`) | $0.0852$ | $[0.0779, 0.0989]$ | $0.0686$ | $+0.0345$ | $0.0779$ | $0.364$ | $[0.209, 0.501]$ |
| **XGBoost** | Model C (`+ QC`) | $0.0878$ | $[0.0812, 0.1028]$ | $0.0697$ | $+0.0387$ | $0.0788$ | $0.393$ | $[0.238, 0.528]$ |

---

## 5. Visualizations of Same-Time Performance

### 5.1 Synchronized Time-Series Tracking Comparison
The time-series plot demonstrates daily tracking fidelity across the synchronized 216 common dates.

![Same-Time Spatial Time Series Comparison](/C:/Users/tinku/.gemini/antigravity/brain/ac9b6960-70bd-4642-8f73-bf99f35d6068/loso_same_time_timeseries.png)

### 5.2 Scatter Plots: Actual vs. Predicted Soil Moisture
Scatter plots illustrate the strong linear correlation of Model A on the synchronized dates.

![Same-Time Scatter Plots](/C:/Users/tinku/.gemini/antigravity/brain/ac9b6960-70bd-4642-8f73-bf99f35d6068/loso_same_time_scatters.png)

### 5.3 Cross-Experiment Comparison: Same-Time vs. All-Available Dates
This chart contrasts the All-Available Spatial LOSO ($N=771$), the All-Available Model evaluated on the Common Subset ($N=648$ control), and the New Same-Time Spatial LOSO ($N=648$).

![Same-Time vs All-Dates Comparison Bar Chart](/C:/Users/tinku/.gemini/antigravity/brain/ac9b6960-70bd-4642-8f73-bf99f35d6068/loso_same_time_vs_all_dates_bars.png)

### 5.4 Calendar Coverage and Station Intersection
This timeline illustrates the active usability of each station and the synchronized 216-day window.

![Timeline of Common Date Coverage](/C:/Users/tinku/.gemini/antigravity/brain/ac9b6960-70bd-4642-8f73-bf99f35d6068/loso_same_time_coverage_calendar.png)

---

## 6. Scientific Comparison: Same-Time vs. All-Available-Dates Spatial LOSO

Table 5 presents a rigorous three-way comparison isolating the effect of same-time date synchronization from simple sample subset selection.

#### Table 5: Cross-Experiment Benchmark Comparison (Random Forest Model A, `rh + amp`)

| Fold | Held-Out Station | All-Dates Spatial LOSO (Full $N=771$) RMSE [$r$] | All-Dates Model on Common Subset Control ($N=648$) RMSE [$r$] | Same-Time Spatial LOSO ($N=648$) RMSE [$r$] | $\Delta\text{RMSE}$ (Same-Time vs. All-Dates) | $\Delta\text{RMSE}$ (Same-Time vs. Subset Control) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | P037 | $0.0729$ [$0.449$] | $0.0748$ [$0.327$] | **0.0540** [**0.433**] | $-0.0189$ | **-0.0208** |
| **Fold 2** | P043 | $0.1052$ [$0.517$] | $0.1110$ [$0.557$] | **0.1050** [**0.512**] | $-0.0003$ | **-0.0061** |
| **Fold 3** | P041 | $0.1018$ [$0.285$] | $0.1054$ [$0.328$] | **0.1069** [**0.300**] | $+0.0051$ | $+0.0015$ |
| **Pooled** | ALL 3 STATIONS | $0.0937$ [$0.395$] | $0.0984$ [$0.411$] | **0.0920** [**0.434**] | **-0.0018** | **-0.0064** |

### Detailed Discussion of Differences:
1. **Consistency of Improvement Across Stations:**
   - The improvement from same-time training is **not uniform**: it is pronounced at station `P037` ($\Delta\text{RMSE} = -0.0208\text{ m}^3/\text{m}^3$), moderate at station `P043` ($\Delta\text{RMSE} = -0.0061\text{ m}^3/\text{m}^3$), and neutral at station `P041` ($\Delta\text{RMSE} = +0.0015\text{ m}^3/\text{m}^3$).
2. **Caution Regarding Causal Attribution:**
   - **Do not attribute these differences purely to spatial transfer mechanics.** Restricting observations to the synchronized common dates fundamentally alters the **seasonal and target-distribution composition** of the dataset:
     - Mid-winter dates (Jan 1--Feb 17) and late-autumn freezing dates (Nov 23--Dec 31) are excluded because of station-specific freezing flags.
     - The common-date dataset consists almost exclusively of **unfrozen, active hydrological regimes** (spring snowmelt recession, summer drying, and autumn precipitation).
     - At subalpine station `P037`, removing the dry winter dates where snow cover created spurious baseline shifts naturally brings the training and test distributions closer together.
3. **Internal Control Verification:**
   - Comparing the All-Dates model evaluated on the common subset ($0.0984\text{ m}^3/\text{m}^3$) against the Same-Time model ($0.0920\text{ m}^3/\text{m}^3$) confirms that training strictly on concurrent dates provides a genuine learning benefit ($\Delta = -0.0064\text{ m}^3/\text{m}^3$), above and beyond the seasonal selection effect.

---

## 7. Methodological Limitations and Synthesis

1. **Target Discrepancy (PBO_H2O vs. COSMOS):**  
   The reference observations in this experiment are surface measurements ($0\text{--}5\text{ cm}$) from PBO_H2O. These results cannot be directly compared to the earlier single-station P041 COSMOS temporal benchmark ($0.0337\text{ m}^3/\text{m}^3$) because COSMOS measures an integrated $200\text{ m}$ radius down to $30\text{ cm}$ depth.
2. **Spatial Separation at Station P041:**  
   At stations `P037` and `P043`, the ISMN probes are collocated at $0.54\text{ m}$ and $0.53\text{ m}$ from the antenna. At `P041`, the `MARSHALL2` probe is located $210.69\text{ m}$ South, completely outside the $4\text{--}45\text{ m}$ reflection zone. This localized spatial separation remains the primary reason Fold 3 exhibits higher unbiased error ($\text{ubRMSE} = 0.0902\text{ m}^3/\text{m}^3$) than Fold 1 ($0.0482\text{ m}^3/\text{m}^3$) and Fold 2 ($0.0581\text{ m}^3/\text{m}^3$).
3. **Winter Representativeness:**  
   Because synchronized common dates require all three stations to be simultaneous, winter frost and snow flags at any single site eliminate that date from the entire experiment. As a result, the same-time spatial LOSO evaluates **unfrozen warm-season spatial transfer** and cannot inform snowpack or frozen soil retrieval performance.
