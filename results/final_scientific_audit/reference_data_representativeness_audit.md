# Audit E: Reference Data Representativeness and Spatial Footprint Audit

**Audit Status:** `COMPLETED`  
**Classification:** `QUALIFICATION REQUIRED` & `VERIFIED`  
**Deliverable Path:** [reference_data_representativeness_audit.md](file:///C:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/results/final_scientific_audit/reference_data_representativeness_audit.md)

---

## 1. Executive Summary

This audit evaluates the ground-truth soil-moisture reference datasets used across the project:
1. **The Single-Station P041 Temporal Benchmark:** Validated against **COSMOS** (Cosmic-Ray Neutron Probe).
2. **The Multi-Station LOSO Experiments (P041, P037, P043):** Evaluated against the **PBO_H2O** product obtained via the International Soil Moisture Network (ISMN).

### Critical Audit Findings:
1. **PBO_H2O is GPS Reflectometry, Not an In-Situ Probe (`QUALIFICATION REQUIRED`):**  
   The PBO_H2O reference dataset is not an invasive in-situ sensor (such as a Campbell Scientific TDR or Decagon capacitance probe). As recorded in the ISMN `.stm` file headers, the sensor type is explicitly `'GPS'`. It is an empirical GPS-IR reflectometry product developed by the University of Colorado Boulder (Larson & Small). Consequently, the LOSO experiment is an evaluation of machine-learning-based GNSS-IR against an authoritative physics-calibrated GPS-IR reference product.
2. **P041 MARSHALL2 Spatial Offset of 210.69 m (`VERIFIED`):**  
   The P041 GNSS tower (`39.94949°N`, `-105.19427°W`) and the MARSHALL2 reference probe coordinate (`39.94760°N`, `-105.19440°W`) are separated by exactly **210.69 meters**. Because the GNSS-IR first Fresnel zone footprint spans an annulus of **4 to 45 meters** from the antenna mast, the MARSHALL2 coordinate lies completely outside the antenna's physical reflection footprint. In contrast, P037's probe is 5.33 m away, and P043's probe is 0.53 m away.
3. **Discrepancy Between COSMOS (0–25 cm) and PBO_H2O (0–5 cm) (`QUALIFICATION REQUIRED`):**  
   The initial temporal benchmark on P041 used COSMOS, which integrates moisture over a 200–300 m radius down to a 25 cm depth. The multi-station LOSO experiments used PBO_H2O, which reflects surface skin depth (0–5 cm). Direct comparison of metrics between the single-station temporal benchmark ($R^2 \approx 0.81$) and multi-station LOSO ($R^2 \approx 0.25$, $r \approx 0.50$) is therefore confounded by the fundamental difference in sensing depth and reference definition.

---

## 2. Comprehensive Comparison of Reference Datasets

### Table 1: Reference Product Specifications Across Project Stages

| Parameter | P041 Temporal Benchmark (COSMOS) | Multi-Station LOSO (PBO_H2O via ISMN) |
|---|---|---|
| **Network / Source** | COSMOS (Cosmic-Ray Soil Moisture Observing System) | PBO_H2O (Plate Boundary Observatory / UNAVCO / ISMN) |
| **Sensor Principle** | Epithermal neutron count moderation by hydrogen atoms | GPS carrier SNR multipath phase/frequency reflectometry |
| **Sensor Type in Metadata** | Cosmic-ray neutron probe | `'GPS'` (empirical reflectometry inversion) |
| **Nominal Depth** | **$0\text{--}25\text{ cm}$** (effective depth varies $10\text{--}30\text{ cm}$ with wetness) | **$0\text{--}5\text{ cm}$** (surface dielectric skin depth) |
| **Spatial Footprint** | Circular footprint, radius $\approx 200\text{--}300\text{ m}$ centered at probe | Annular footprint, radial distance $\approx 4\text{--}45\text{ m}$ around antenna |
| **Temporal Frequency** | Hourly integrated counts, aggregated to daily mean | Daily single value reported at 12:00 UTC |
| **Quality Flags** | Station QC flags (snow, error flags removed) | `G` (Good), `D01`/`D02`/`D03` (frozen ground / model flags), `M` (missing) |
| **Invasive / Disturbed?**| Completely non-invasive | Non-invasive (remote microwave sensing) |

---

## 3. Spatial Collocation and Footprint Geometry

GNSS-IR multipath interference fringes originate from specular reflection zones on the surrounding ground plane. For an antenna height $h \approx 1.90\text{ m}$ and elevation angles $5^\circ \le e \le 25^\circ$:
- Minimum reflection radius ($e = 25^\circ$): $d_{\text{min}} \approx \frac{h}{\tan 25^\circ} \approx 4.1\text{ m}$
- Maximum reflection radius ($e = 5^\circ$): $d_{\text{max}} \approx \frac{h}{\tan 5^\circ} \approx 21.7\text{ m}$ (with elongation along the major axis of the first Fresnel ellipse extending up to $\approx 45\text{ m}$).

### Table 2: Station Antenna vs. Reference Sensor Coordinate Separation

| Station | GNSS Antenna (WGS84) | Reference Probe (ISMN Header) | Separation Distance | Location Relative to Footprint |
|---|---|---|---:|---|
| **P041** | `39.949492°N`, `-105.194266°W`, $1728.86\text{ m}$ | `39.94760°N`, `-105.19440°W`, $1727.30\text{ m}$ | **$210.69\text{ m}$** | **Completely outside footprint** ($> 4.5\times$ max radius) |
| **P037** | `38.421754°N`, `-105.104684°W`, $1616.89\text{ m}$ | `38.42180°N`, `-105.10470°W`, $1616.90\text{ m}$ | **$5.33\text{ m}$** | **Inside inner footprint** ($4\text{--}45\text{ m}$) |
| **P043** | `43.881146°N`, `-104.185703°W`, $1490.95\text{ m}$ | `43.88115°N`, `-104.18570°W`, $1490.90\text{ m}$ | **$0.53\text{ m}$** | **Collocated directly at antenna mast** |

### Implications of P041 Spatial Mismatch ($210.69\text{ m}$):
1. **Micro-Topographic Decoupling:**  
   The $210.69\text{ m}$ southward offset places the MARSHALL2 coordinate in a slightly lower topographic swale ($\Delta z \approx -1.56\text{ m}$). After heavy precipitation or convective thunderstorms, surface runoff and soil drying rates can differ substantially between the P041 antenna plateau and the MARSHALL2 sensor site.
2. **Impact on Correlation:**  
   In the spatio-temporal LOSO experiment, P041 exhibited the lowest correlation ($r = 0.341$), compared to P037 ($r = 0.486$) and P043 ($r = 0.555$). A significant portion of this lower correlation is attributable to spatial representativeness error rather than model failure.

---

## 4. Are the Three-Station LOSO Targets Sufficiently Comparable?

Yes, but under clear qualifications:
1. **Consistent Target Definition:**  
   All three stations in the LOSO experiments use the exact same PBO_H2O $0\text{--}5\text{ cm}$ target definitions and quality flag filtering (retaining only flag `G` and valid positive readings).
2. **Different Physical Soil Baselines:**  
   Soil types and bulk densities vary across the sites:
   - P041: Sandy loam, bulk density $\approx 1.4\text{ g}/\text{cm}^3$, annual mean $\text{SM} \approx 0.125\text{ m}^3/\text{m}^3$.
   - P037: Coarse sandy loam / gravelly alluvium at Fremont County Airport, annual mean $\text{SM} \approx 0.088\text{ m}^3/\text{m}^3$.
   - P043: Silt loam / clay loam near Newcastle WY, annual mean $\text{SM} \approx 0.134\text{ m}^3/\text{m}^3$.
3. **Reference Product Self-Consistency:**  
   Because PBO_H2O itself was derived using empirical regressions from GPS multipath, it contains its own systematic retrieval uncertainties ($\pm 0.03\text{--}0.04\text{ m}^3/\text{m}^3$). Therefore, an unbiased RMSE of $\text{ubRMSE} \approx 0.0537\text{ m}^3/\text{m}^3$ represents excellent agreement with the reference product, approaching the intrinsic noise floor of the reference itself.

---

## 5. Audit Recommendations for Presentation

1. **Be Scientifically Precise About PBO_H2O:**  
   Never claim that the model was validated against "in-situ buried moisture probes" across the three stations. Correctly define PBO_H2O as an authoritative, quality-controlled GPS reflectometry surface soil moisture benchmark product.
2. **Explicitly Mention the P041 Footprint Mismatch:**  
   Include a slide or note mentioning that P041's reference location is $210\text{ m}$ south of the tower, explaining why P041 exhibits lower test correlation ($r = 0.34$) than P043 ($r = 0.56$).
3. **Separate COSMOS Benchmark from LOSO Experiments:**  
   Do not present the single-station P041 COSMOS results ($R^2 = 0.81$, $0\text{--}25\text{ cm}$) and multi-station PBO_H2O LOSO results ($R^2 = 0.26$, $r = 0.50$, $0\text{--}5\text{ cm}$) as a direct drop in performance. State clearly that they evaluate different depths, footprints, and reference sensors.
