# Audit F: Metric and Experiment Consistency Audit

**Audit Status:** `COMPLETED`  
**Classification:** `VERIFIED`  
**Deliverable Path:** [experiment_and_metric_consistency_audit.md](file:///C:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/results/final_scientific_audit/experiment_and_metric_consistency_audit.md)

---

## 1. Executive Summary

This audit rigorously verifies the mathematical definitions, computational implementations, masking consistency, and statistical integrity of all performance metrics and experimental designs used across the GNSS-IR Soil Moisture research project.

### Key Audit Findings:
1. **Metric Mathematical Consistency (`VERIFIED`):**  
   All scripts adhere strictly to standard hydrological error conventions: $\text{Error} = \hat{y} - y$ and $\text{Bias} = \text{mean}(\hat{y} - y)$. Total $\text{RMSE}$, $\text{MAE}$, $\text{Bias}$, and unbiased $\text{ubRMSE}$ satisfy the theoretical Pythagorean identity $\text{RMSE}^2 = \text{Bias}^2 + \text{ubRMSE}^2$ within numerical precision.
2. **Pooled Metrics Computation (`VERIFIED`):**  
   Pooled metrics across the three stations are computed from the concatenated vector of all held-out predictions and ground-truth observations, **not** by averaging station-specific summary statistics.
3. **Bootstrap Implementation Progression (`VERIFIED`):**  
   - In the initial spatial LOSO experiment (`run_primary_spatial_loso.py`), a standard 7-sample moving block bootstrap was used on the active sample index array.
   - In the subsequent same-time and spatio-temporal experiments (`run_same_time_spatial_loso.py`, `run_spatiotemporal_loso.py`), this was upgraded to `calendar_block_bootstrap_ci`, which constructs continuous 7-day calendar-time blocks and explicitly truncates at missing-data gaps $\ge 7\text{ days}$ to avoid artificial temporal joins across gaps.
4. **Experimental Design Clarity (`VERIFIED`):**  
   The four project experiments answer four distinct, hierarchically structured scientific questions. Each experiment has distinct boundaries that must be clearly stated during the Master Progress Presentation.

---

## 2. Mathematical Verification of Hydrological Metrics

Let $y_i$ denote the ground-truth reference soil moisture ($\text{m}^3/\text{m}^3$) and $\hat{y}_i$ denote the model prediction for sample $i \in \{1, \dots, n\}$.

### 2.1 Metric Formulations

$$\text{Error}_i = \hat{y}_i - y_i$$

$$\text{Bias} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)$$

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2}$$

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |\hat{y}_i - y_i|$$

$$\text{ubRMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n \left( (\hat{y}_i - y_i) - \text{Bias} \right)^2}$$

$$\text{Pearson } r = \frac{\sum_{i=1}^n (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^n (y_i - \bar{y})^2 \sum_{i=1}^n (\hat{y}_i - \bar{\hat{y}})^2}}$$

### 2.2 Numerical Invariance Check
In all project output tables, the identity holds:
$$\text{RMSE}^2 \equiv \text{Bias}^2 + \text{ubRMSE}^2$$
For example, in Spatio-Temporal LOSO Pooled RF Model A:
- $\text{Bias} = +0.0986$
- $\text{ubRMSE} = 0.0543$
- $\sqrt{(0.0986)^2 + (0.0543)^2} = \sqrt{0.009722 + 0.002948} = \sqrt{0.012671} = \mathbf{0.11256} \approx \mathbf{0.1126}$, matching the reported RMSE exactly.

---

## 3. Bootstrap Procedure and Temporal Autocorrelation

Hydrological time series exhibit strong temporal persistence (autocorrelation in daily soil moisture typically exceeds $\rho_1 \approx 0.85$). Standard i.i.d. bootstrapping severely underestimates uncertainty by breaking temporal dependence.

### 3.1 Standard Moving Block Bootstrap (`run_primary_spatial_loso.py`)
- Selected random starting indices in the sorted test sequence and extracted continuous chunks of 7 consecutive observations.
- Effectively preserved 1-week autocorrelation structures within contiguous observation runs.

### 3.2 Calendar-Gap-Respecting Block Bootstrap (`calendar_block_bootstrap_ci`)
Implemented in `run_same_time_spatial_loso.py` and `run_spatiotemporal_loso.py`:
- Converts sample dates to calendar timestamps.
- Formulates candidate blocks of span $\Delta t \le 7\text{ days}$.
- If an observational gap between consecutive dates exceeds 7 days (e.g., due to freezing flag exclusions or missing days), the candidate block terminates immediately at the gap boundary.
- Preserves genuine calendar-time temporal correlation without splicing disconnected seasonal intervals together.
- 1,000 bootstrap resamples generated the reported 95% empirical percentile confidence intervals $[q_{0.025}, q_{0.975}]$.

---

## 4. Synthesis of the Four Project Experiments

The four experiments form a logical hierarchy testing different dimensions of model validity:

### Table 1: Matrix of Experimental Designs

| Experiment | Target Product | Temporal Split | Spatial Split | Primary Question Addressed | What the Experiment Proves | What It Does NOT Prove |
|---|---|---|---|---|---|---|
| **1. P041 Temporal Benchmark** | COSMOS ($0\text{--}25\text{ cm}$) | 60% train / 40% test (chronological) | Single station (P041) | Can GNSS-IR features track soil moisture dynamics over time at a single site? | Demonstrates strong temporal feature correlation ($R^2 \approx 0.81$, $r \approx 0.90$) at one local monument. | Does **not** prove generalizability to other geographic locations, soils, or antenna geometries. |
| **2. All-Date Spatial LOSO** | PBO_H2O ($0\text{--}5\text{ cm}$) | None (uses all 2016 dates, $N=771$) | Leave-one-station-out (3 folds) | Can a model trained on other stations predict soil moisture at an unseen station across all available seasons? | Proves spatial transferability ($r \approx 0.50$, $\text{ubRMSE} \approx 0.052$) when training and test periods span identical annual cycles. | Does **not** prove predictive skill into an unseen future season (training included future dates from other sites). |
| **3. Same-Time Spatial LOSO** | PBO_H2O ($0\text{--}5\text{ cm}$) | Matched common dates only ($N=648$) | Leave-one-station-out (3 folds) | Does spatial transfer hold when all stations are evaluated on exactly the same calendar dates? | Confirms spatial transfer is not an artifact of asynchronous date sampling ($\text{RMSE} = 0.0883$, $\text{ubRMSE} = 0.0520$, virtually identical to all-date LOSO). | Does **not** test forward temporal extrapolation. |
| **4. Spatio-Temporal LOSO** | PBO_H2O ($0\text{--}5\text{ cm}$) | Jan–Jun train / Jul–Dec test | Leave-one-station-out (3 folds) | Can a model trained on earlier observations at two stations transfer to an unseen station in a later period? | Tests joint spatial and temporal transfer. Proves that Model A maintains dynamic tracking ($r \approx 0.51$, $\text{ubRMSE} \approx 0.054$). | Does **not** represent a pure spatial test; high total RMSE ($0.113$) is driven by seasonal domain shift (wet spring to dry summer). |

---

## 5. Audit Recommendations for Presentation

1. **Explicitly Differentiate the 4 Experiments:**  
   In the Master Progress Presentation, introduce the experimental progression in the sequence shown in Table 1. Clearly state what each experiment was designed to isolate.
2. **Clarify Pooled vs. Station Metrics:**  
   Highlight that pooled metrics reflect total cross-station performance evaluated on all combined test samples, while individual fold metrics show station-specific performance.
3. **Transparently Report the Role of Domain Shift:**  
   Explain that the increased RMSE in Experiment 4 ($0.113$ vs $0.088$) is not a failure of spatial transfer, but an expected outcome of the seasonal regime shift from wet spring training to dry late-summer testing.
