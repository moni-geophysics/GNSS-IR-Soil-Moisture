# Audit D: GNSS-IR Phase Physics and Antenna Height Interpretation

**Audit Status:** `COMPLETED`  
**Classification:** `CORRECTION REQUIRED` (for the 178° static rotation claim) & `VERIFIED` (for the uncalibrated cross-station phase barrier)  
**Deliverable Path:** [gnss_ir_phase_physics_audit.md](file:///C:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/results/final_scientific_audit/gnss_ir_phase_physics_audit.md)

---

## 1. Executive Summary

In earlier project reports (notably `results/SPATIOTEMPORAL_LOSO_REPORT.md`, Section 1.4), it was asserted that:
> *"static monument height differences between stations ($\approx 5\text{ cm}$, rotating phase by $\approx 178^\circ$) create an uncalibrated barrier that cannot be bridged without station-specific physical calibration."*

This audit presents a rigorous mathematical and physical evaluation of the GNSS-IR multipath interference model implemented in `src/gnss_ir/processing.py`. 

### Key Audit Findings:
1. **The $178^\circ$ Rotation Claim is Mathematically False (`CORRECTION REQUIRED`):**  
   The calculation $\Delta \theta = \frac{4\pi \Delta h}{\lambda} \approx 178^\circ$ conflates a **spatial frequency change** with a **constant phase shift**. In GNSS-IR, $h$ determines the oscillation frequency with respect to $\sin e$, where $e$ is satellite elevation. An antenna height difference $\Delta h$ creates an elevation-dependent phase change $\Delta \Theta(e) = \frac{4\pi \Delta h}{\lambda} \sin e$. Over the active observation window ($5^\circ \le e \le 25^\circ$), $\sin e \in [0.087, 0.423]$, so the phase shift varies from $15.5^\circ$ to $75.2^\circ$. It only reaches $178^\circ$ at $\sin e = 1$ ($e = 90^\circ$, zenith), where no multipath reflections occur.
2. **The Cross-Station Phase Barrier is Verified (`VERIFIED`):**  
   Empirical analysis of the harmonized dataset confirms that raw phase cannot transfer across stations without calibration. The circular mean phases of the three stations point in completely different directions:
   - **P041:** $\bar{\phi} = -150.3^\circ$ (Quadrant III, Circular $R = 0.900$)
   - **P037:** $\bar{\phi} = +67.0^\circ$ (Quadrant I, Circular $R = 0.746$)
   - **P043:** $\bar{\phi} = +120.0^\circ$ (Quadrant II, Circular $R = 0.833$)
   Because each station's phase clusters tightly around its own station-specific offset (driven by local topography, antenna tilt, and dielectric baselines), training a model on P041/P037 and applying it to P043 causes feature confusion, collapsing test correlation from $r = 0.555$ down to $r = 0.150$.

---

## 2. Mathematical Derivation of the GNSS-IR Interference Model

### 2.1 The Two-Path Geometric Interference Model
For a planar horizontal reflecting surface located at depth $h$ below a right-hand circularly polarized (RHCP) GNSS antenna, the received signal is the coherent superposition of the direct electromagnetic wave and the ground-reflected wave:
$$E_{\text{net}}(t) = E_d(t) + E_r(t)$$
The geometric path length difference between the reflected path and the direct path is:
$$\delta = 2h \sin e$$
where:
- $h$ is the vertical reflector height (m) between the antenna phase center and the effective reflecting planar surface.
- $e$ is the satellite elevation angle (radians).

The electromagnetic phase difference between the direct and reflected waves is:
$$\psi(e) = \frac{2\pi}{\lambda} \delta = \frac{4\pi h}{\lambda} \sin e$$
where $\lambda$ is the carrier wavelength (for GPS L1, $f = 1575.42\text{ MHz}$, $\lambda = c / f \approx 0.19029367\text{ m}$).

### 2.2 Carrier-to-Noise Ratio (SNR) Multipath Modulation
In the presence of specular reflection, the composite signal power (SNR in linear scale, $\text{volt}^2/\text{volt}^2$) oscillates as the satellite changes elevation:
$$\text{SNR}(e) = A_d^2(e) + A_r^2(e) + 2 A_d(e) A_r(e) \cos\left( \frac{4\pi h}{\lambda} \sin e + \phi \right)$$
After removing the direct signal trend $A_d^2(e)$ with a low-order polynomial, the detrended interference fringe pattern is modeled as:
$$\delta\text{SNR}(e) = A \cos\left( \frac{4\pi h}{\lambda} \sin e + \phi \right)$$
where:
- $A = 2 A_d A_r |R|$ is the multipath amplitude, modulated by the Fresnel reflection coefficient $|R(\epsilon_r, e)|$.
- $\phi$ is the net phase intercept at $\sin e = 0$.

### 2.3 Implementation in Codebase (`src/gnss_ir/processing.py`)
In `fit_fixed_height` ([processing.py:319-382](file:///C:/Users/tinku/PycharmProjects/GNSS-IR-Soil-Moisture/src/gnss_ir/processing.py#L319-L382)), the non-linear cosine is decomposed into linear quadrature basis functions:
$$\delta\text{SNR}(x) = c \cos(\omega x) + s \sin(\omega x)$$
where:
- $x = \sin e$
- $\omega = 2\pi f_x = \frac{4\pi h}{\lambda}$
- $c = A \cos \phi$
- $s = -A \sin \phi$

The linear system is solved via ordinary least squares (`np.linalg.lstsq`):
$$A = \sqrt{c^2 + s^2}$$
$$\phi = \text{arctan2}(-s, c)$$
This exact equation confirms that $\phi$ is extracted as the mathematical intercept at $\sin e = 0$.

---

## 3. Dissecting the "$178^\circ$ Phase Rotation" Fallacy

### 3.1 Where the Number $178^\circ$ Came From
The claim originated from taking an antenna height difference $\Delta h \approx 4.75\text{ cm} = 0.0475\text{ m}$ between stations and multiplying it by $\frac{4\pi}{\lambda}$:
$$\Delta \theta_{\text{claimed}} = \frac{4\pi \Delta h}{\lambda} = \frac{4\pi \times 0.0475\text{ m}}{0.190294\text{ m}} = 3.136\text{ radians} \approx 179.7^\circ \approx 178^\circ$$

### 3.2 Why This Formula Does Not Represent a Phase Shift
In the argument of the interference fringe:
$$\Theta(e) = \left( \frac{4\pi h}{\lambda} \right) \sin e + \phi$$
The factor $\frac{4\pi h}{\lambda}$ is **NOT a constant angle**! It is a **spatial angular frequency** with units of radians per unit $\sin e$.
If reflector height changes by $\Delta h$, the phase argument at elevation $e$ changes by:
$$\Delta \Theta(e) = \left( \frac{4\pi \Delta h}{\lambda} \right) \sin e$$

Notice the presence of $\sin e$:
1. **At the intercept ($\sin e = 0$):**
   $$\left. \Delta \Theta(e) \right|_{\sin e = 0} = 0^\circ$$
   The static intercept phase $\phi$ is **completely independent of $\Delta h$**!
2. **Across the Actual GNSS-IR Elevation Window ($5^\circ \le e \le 25^\circ$):**
   - At $e = 5^\circ$ ($\sin 5^\circ = 0.08716$): $\Delta \Theta = 178^\circ \times 0.08716 = \mathbf{15.5^\circ}$
   - At $e = 10^\circ$ ($\sin 10^\circ = 0.1736$): $\Delta \Theta = 178^\circ \times 0.1736 = \mathbf{30.9^\circ}$
   - At $e = 15^\circ$ ($\sin 15^\circ = 0.2588$): $\Delta \Theta = 178^\circ \times 0.2588 = \mathbf{46.1^\circ}$
   - At $e = 20^\circ$ ($\sin 20^\circ = 0.3420$): $\Delta \Theta = 178^\circ \times 0.3420 = \mathbf{60.9^\circ}$
   - At $e = 25^\circ$ ($\sin 25^\circ = 0.4226$): $\Delta \Theta = 178^\circ \times 0.4226 = \mathbf{75.2^\circ}$
3. **Where Does $178^\circ$ Actually Occur?**
   Only at $\sin e = 1.0$ ($e = 90^\circ$, the astronomical zenith).
   In GNSS-IR, satellites at zenith produce zero multipath interference fringes because the reflected ray bounces directly back along the antenna axis where antenna backlobes attenuate it, and the elevation mask explicitly truncates all data above $25^\circ$ or $30^\circ$.

**Conclusion:** Describing a $4.7\text{ cm}$ height difference as "rotating the phase circle by $178^\circ$" is mathematically unsound. A height difference alters the **fringe frequency** (expanding or compressing the multipath sinusoid), not the phase intercept.

---

## 4. Empirical Phase Analysis of P041, P037, and P043

Using the harmonized 3-station dataset ($N = 366$ days per station in 2016), we computed the circular statistics of daily extracted phase:

### Table 1: Station-Wise Empirical Phase and Feature Statistics

| Metric | Station P041 | Station P037 | Station P043 |
|---|---:|---:|---:|
| **Sample Count ($N$)** | 366 | 366 | 366 |
| **Reflector Height ($h$)** | $1.9193 \pm 0.0509\text{ m}$ | $1.9181 \pm 0.0168\text{ m}$ | $1.8867 \pm 0.0271\text{ m}$ |
| **Multipath Amplitude ($A$)** | $12.77 \pm 2.12\text{ volt}$ | $14.29 \pm 1.14\text{ volt}$ | $12.81 \pm 0.76\text{ volt}$ |
| **Phase Cosine Mean ($\overline{\cos\phi}$)** | $-0.7820 \pm 0.2537$ | $+0.2913 \pm 0.4822$ | $-0.4171 \pm 0.4387$ |
| **Phase Sine Mean ($\overline{\sin\phi}$)** | $-0.4461 \pm 0.3544$ | $+0.6873 \pm 0.4599$ | $+0.7211 \pm 0.3383$ |
| **Circular Mean Angle ($\bar{\phi}$)** | **$-150.3^\circ$** ($-2.623\text{ rad}$) | **$+67.0^\circ$** ($+1.170\text{ rad}$) | **$+120.0^\circ$** ($+2.095\text{ rad}$) |
| **Circular Resultant Length ($R$)** | **$0.9003$** | **$0.7464$** | **$0.8330$** |
| **Correlation: $h$ vs. Soil Moisture** | $-0.467$ | $-0.470$ | $+0.050$ |
| **Correlation: $A$ vs. Soil Moisture** | $-0.191$ | $-0.707$ | $-0.629$ |
| **Correlation: $\cos\phi$ vs. Soil Moisture** | $-0.349$ | $+0.206$ | $-0.012$ |
| **Correlation: $\sin\phi$ vs. Soil Moisture** | $+0.488$ | $-0.013$ | $+0.064$ |

### Table 2: Cross-Station Angular Phase Discrepancies

| Station Pair | Angular Difference ($\Delta\bar{\phi}$) | Physical Relationship |
|---|---:|---|
| **P041 vs. P037** | $\mathbf{-142.7^\circ}$ | Near-opposite hemispheres ($\approx 217^\circ$) |
| **P041 vs. P043** | $\mathbf{+89.7^\circ}$ | Orthogonal phase vectors ($\approx 90^\circ$) |
| **P037 vs. P043** | $\mathbf{-53.0^\circ}$ | Moderate offset in upper quadrants |

### Key Physical Insights from Data:
1. **Intra-Station Coherence ($R > 0.74$):**  
   At any single station, the daily phase does not wander randomly across the unit circle; it is strongly clustered around a characteristic baseline ($R = 0.90$ at P041, $R = 0.83$ at P043). Within P041, daily phase variations correlate significantly with soil moisture ($r = +0.488$ for $\sin\phi$, $r = -0.349$ for $\cos\phi$).
2. **Inter-Station Incoherence:**  
   Across stations, the characteristic baselines point into completely different quadrants:
   - P041 points to Quadrant III ($-150^\circ$, negative cosine, negative sine).
   - P037 points to Quadrant I ($+67^\circ$, positive cosine, positive sine).
   - P043 points to Quadrant II ($+120^\circ$, negative cosine, positive sine).
3. **True Source of the Phase Offset:**  
   The $143^\circ$ and $90^\circ$ differences between stations are caused by:
   - **Topographic Slope:** Local surface tilts in the first Fresnel zone alter the path-length geometry by non-zero azimuth-dependent terms.
   - **Antenna Phase Center Patterns:** Small differences in antenna model, radome type, or orientation.
   - **Soil Permittivity Baselines:** Differences in soil texture, bulk density, and mineralogy alter the baseline Fresnel phase intercept $\phi_0$.

---

## 5. Why Model B Fails Under Spatial LOSO

In a tree-based machine learning model (Random Forest / XGBoost):
- If trained on **P041 and P037**, the tree splits learn decision rules on `phase_cos` and `phase_sin`. For example, at P041, high soil moisture requires $\sin\phi > -0.2$ and $\cos\phi \approx -0.9$.
- When evaluated on **P043**, $\sin\phi$ is perpetually positive ($+0.72 \pm 0.34$) and $\cos\phi$ is moderately negative ($-0.42$). The decision trees map these unseen coordinate pairs into arbitrary, out-of-distribution leaf nodes.
- Consequently, at P043, Model A ($h + A$) achieves $r = \mathbf{0.555}$, but adding raw uncalibrated phase (Model B) degrades correlation to $r = \mathbf{0.150}$ (a complete collapse of predictive skill).

---

## 6. Recommendations for Master Progress Presentation

1. **Retract the $178^\circ$ Antenna-Height Claim:**  
   Remove any statement claiming that a $5\text{ cm}$ antenna height difference rotates the phase circle by $178^\circ$. State accurately that $h$ controls spatial frequency, while the phase intercept is sensitive to topographic tilt and uncalibrated instrumental/dielectric offsets.
2. **Defend the Phase Barrier Using Empirical Circular Means:**  
   Present Table 1 and Table 2. Emphasize that the phase barrier is an **empirical and physical reality**—the three stations have phase baselines that differ by $90^\circ\text{--}143^\circ$.
3. **Highlight Model A as the Robust Transfer Model:**  
   Demonstrate that Reflector Height ($h$) and Multipath Amplitude ($A$) are scale-consistent physical observables that transfer across stations without station-specific calibration, whereas Phase requires local calibration before it can be used across stations.
