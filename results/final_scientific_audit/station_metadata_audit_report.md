# Station Metadata Scientific Audit Report

**Audit Directory:** `results/final_scientific_audit/`  
**Associated Data File:** [`station_metadata_audit.csv`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/results/final_scientific_audit/station_metadata_audit.csv)  
**Audit Scope:** Station identity, geographic coordinates, vertical datums, antenna height conventions, and propagation of erroneous altitude metadata across reports and scripts.

---

## 1. Authoritative Station Metadata Summary

By cross-referencing authoritative project files:
1. The gnssrefl station database: [`data/raw/gnss/Files/station_pos_2024.db`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/data/raw/gnss/Files/station_pos_2024.db)
2. Station runtime configuration JSONs: `p041.json`, `p037.json`, `p043.json` in [`data/raw/gnss/input/`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/data/raw/gnss/input/)
3. Authoritative ISMN metadata and STM observation headers in [`data/raw/ismn/`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/data/raw/ismn/)
4. EarthScope / PBO permanent station directory

The verified station parameters are established in Table 1.

### Table 1: Verified Authoritative Metadata vs. Reported Values

| Station ID | Network / Station Name | Verified Latitude (°N) | Verified Longitude (°E) | Ellipsoidal Height (m) | Orthometric Height MSL (m) | Climate (Köppen) | Land Cover | Audit Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P041** | Marshall (Field Site 2), CO | `39.949492042` | `-105.194266387` | `1728.86 m` | `1744.72 m` | BSk (Cold semi-arid) | Grassland | **VERIFIED** |
| **P037** | FreemontAPCO2004 (Fremont Co. Airport), CO | `38.421753832` | `-105.104683735` | `1616.89 m` | `1633.66 m` | BSk (Cold semi-arid) | Grassland | **CORRECTION REQUIRED** |
| **P043** | NewcastleWY2005 (Newcastle), WY | `43.881145655` | `-104.185702780` | `1490.95 m` | `1504.50 m` | BSk (Cold semi-arid) | Grassland | **CORRECTION REQUIRED** |

---

## 2. Critical Metadata Finding: The P037 "Fremont Pass" Anomaly

### 2.1 Nature of the Error
In earlier reports ([`SPATIOTEMPORAL_LOSO_REPORT.md`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/results/SPATIOTEMPORAL_LOSO_REPORT.md), [`SAME_TIME_SPATIAL_LOSO_REPORT.md`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/results/SAME_TIME_SPATIAL_LOSO_REPORT.md)) and publication figures ([`loso_spatial_timeseries_comparison.png`](file:///C:/Users/tinku/.gemini/antigravity/brain/ac9b6960-70bd-4642-8f73-bf99f35d6068/loso_spatial_timeseries_comparison.png), [`loso_spatiotemporal_timeseries.png`](file:///C:/Users/tinku/.gemini/antigravity/brain/ac9b6960-70bd-4642-8f73-bf99f35d6068/loso_spatiotemporal_timeseries.png)), station `P037` was repeatedly described as:
> *"P037 — Fremont Pass, CO (subalpine montane site at approximately 3514 m elevation)"*

This description is **factually incorrect** by **$1,897\text{ meters}$**:
- **Real Fremont Pass, Colorado:** A high-altitude Rocky Mountain pass on the Continental Divide in Lake and Summit Counties (approx. lat $39.37^\circ\text{N}$, lon $-106.19^\circ\text{W}$), with an elevation of $3,514\text{ m}$ ($11,532\text{ ft}$).
- **Actual Station P037:** Located at **Fremont County Airport (4V2)** near Cañon City / Florence in Fremont County, Colorado (lat $38.42175^\circ\text{N}$, lon $-105.10468^\circ\text{W}$). Its verified ellipsoidal elevation is **$1,616.89\text{ m}$** ($1,633.66\text{ m}$ orthometric height MSL / $5,360\text{ ft}$).

### 2.2 Root Cause of the Discrepancy
1. In the initial data download script ([`scripts/download_loso_data.py`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/scripts/download_loso_data.py)), P037 was correctly identified in the docstring as `P037 (FreemontAPCO2004, Fremont County Airport, CO)`.
2. In the reference table preparation script ([`scripts/prepare_loso_reference_tables.py`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/scripts/prepare_loso_reference_tables.py)), the ISMN station code `FREEMONTAP` was mistakenly interpreted as "Fremont Alpine Pass" or "Fremont Pass", hardcoding `"name": "Fremont Pass, CO"`.
3. In subsequent plotting scripts ([`scripts/plot_loso_results.py`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/scripts/plot_loso_results.py), [`scripts/plot_spatiotemporal_results.py`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/scripts/plot_spatiotemporal_results.py)), an author searched for "Fremont Pass, CO" on the web, found the Wikipedia elevation ($3514\text{ m}$), and inserted `Alt: 3514m` into the figure titles.
4. This false elevation led to physical interpretations in earlier reports claiming that P037 tested an "elevation gradient from semi-arid plains (1728 m) to high subalpine montane conditions (3514 m)" with "subalpine snowmelt dynamics".

### 2.3 Physical Consequence of the Correction
- Station P037 is **not** an alpine snow station.
- As confirmed by the ISMN HWSD soil database and Köppen classification in `loso_harmonized_dataset_median.csv`, P037 is a **grassland plains site** in an arid-steppe regime (`BSk`), with $58\%$ sand, $24\%$ clay, and an elevation of $1,616.9\text{ m}$.
- All three stations (`P041`: $1728.9\text{ m}$, `P037`: $1616.9\text{ m}$, `P043`: $1490.9\text{ m}$) are semi-arid grassland environments spanning a modest elevation range of $238\text{ meters}$ across Colorado and Wyoming.
- **Physical Interpretation Impact:** Variations in performance across stations cannot be attributed to alpine snowpack or extreme mountain elevation gradients.

---

## 3. Discrepancies at Station P043 (Newcastle, WY)

1. **Elevation Discrepancy:**
   - Authoritative height from `station_pos_2024.db` and `p043.json`: **$1,490.95\text{ m}$** ellipsoidal height ($1,504.50\text{ m}$ orthometric height MSL).
   - In [`plot_spatiotemporal_results.py`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/scripts/plot_spatiotemporal_results.py): Labeled in the title as `Alt: 1304m` (an error of $-187\text{ m}$).
   - In [`generate_comprehensive_presentation.py`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/scripts/generate_comprehensive_presentation.py) Slide 45: Labeled as `elev 3020.4 m` and associated with "Niwot Ridge, CO".
2. **Correction:** P043 is NewcastleWY2005, Newcastle, Wyoming, located at lat $43.88115^\circ\text{N}$, lon $-104.18570^\circ\text{W}$, elevation $1,490.95\text{ m}$ in semi-arid plains grassland.

---

## 4. Verification of Antenna Heights vs. Reflector Heights

1. **Convention Verification:**
   - In PBO/EarthScope deep-drilled braced monuments, the vertical distance from ground level to the antenna base / antenna reference point (ARP) is fixed during monumentation, typically around $\sim 1.8\text{ to }2.0\text{ m}$.
   - The reported values in `build_harmonized_loso_dataset.py`:
     - P041: `nominal_rh_m = 1.93 m`
     - P037: `nominal_rh_m = 1.91 m`
     - P043: `nominal_rh_m = 1.89 m`
   - These are **electromagnetic effective reflector heights ($H_{eff}$)** estimated via Lomb-Scargle periodograms from the antenna phase center (APC) to the mean reflecting soil plane.
   - They should be explicitly referred to as **mean effective reflector heights**, not structural monument heights.

---

## 5. Audit Audit Log: Files Containing Metadata Conflicts

| File Path | Location in File | Erroneous Entry | Verified Correction |
| :--- | :--- | :--- | :--- |
| [`scripts/prepare_loso_reference_tables.py`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/scripts/prepare_loso_reference_tables.py) | Line 129 | `"name": "Fremont Pass, CO"` | `"name": "FreemontAP, CO"` |
| [`scripts/build_harmonized_loso_dataset.py`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/scripts/build_harmonized_loso_dataset.py) | Line 50 | `"name": "Fremont Pass, CO"` | `"name": "FreemontAP, CO"` |
| [`scripts/plot_loso_results.py`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/scripts/plot_loso_results.py) | Line 36 | `Alt: 3514m` | `Alt: 1617m` |
| [`scripts/plot_same_time_results.py`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/scripts/plot_same_time_results.py) | Line 37 | `Alt: 3514m` | `Alt: 1617m` |
| [`scripts/plot_spatiotemporal_results.py`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/scripts/plot_spatiotemporal_results.py) | Lines 37, 38 | `P037 Alt: 3514m`, `P043 Alt: 1304m` | `P037 Alt: 1617m`, `P043 Alt: 1491m` |
| [`scripts/generate_comprehensive_presentation.py`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/scripts/generate_comprehensive_presentation.py) | Slide 45 (lines 1142-1143) | `P037 Superior, CO 1727m`, `P043 Niwot Ridge 3020m` | `P037 Fremont Co. Airport 1617m`, `P043 Newcastle, WY 1491m` |
| [`results/SPATIOTEMPORAL_LOSO_REPORT.md`](file:///c:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/results/SPATIOTEMPORAL_LOSO_REPORT.md) | Sections 1 & 4 | Descriptions of P037 as subalpine alpine pass | Semi-arid airport grassland |

---

## 6. Audit Verdict

- **Classification:** **CORRECTION REQUIRED**
- **Action Plan:** In the upcoming MTP presentation and thesis manuscript, remove all references to "Fremont Pass" and "3514 m alpine elevation". Describe P037 accurately as **Fremont County Airport, Colorado ($1,617\text{ m}$)**. Note that the three stations represent three distinct semi-arid grassland sites across a modest elevation range ($1491\text{--}1729\text{ m}$).
