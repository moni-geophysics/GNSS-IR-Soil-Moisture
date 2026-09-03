from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lombscargle


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SNR_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "gnss"
    / "2016"
    / "snr"
    / "p041"
    / "p0410010.16.snr66"
)


# =========================================================
# SNR FILE COLUMNS
# =========================================================

columns = [
    "satellite",
    "elevation_deg",
    "azimuth_deg",
    "seconds_of_day",
    "elevation_rate",
    "S6",
    "S1",
    "S2",
    "S5",
    "S7",
    "S8",
]


data = pd.read_csv(
    SNR_FILE,
    sep=r"\s+",
    header=None,
    names=columns,
)


# =========================================================
# SELECT GPS PRN 1
# =========================================================

satellite_number = 1

sat = data[
    data["satellite"] == satellite_number
].copy()


# =========================================================
# SELECT THE RISING ARC
# =========================================================

sat = sat[
    (sat["elevation_deg"] >= 5.0)
    & (sat["elevation_deg"] <= 25.0)
    & (sat["elevation_rate"] > 0)
    & (sat["seconds_of_day"] >= 4.6 * 3600)
    & (sat["seconds_of_day"] <= 6.0 * 3600)
].copy()


sat = sat.sort_values(
    "seconds_of_day"
).reset_index(drop=True)


# =========================================================
# REMOVE INVALID SNR VALUES
# =========================================================

sat = sat[
    np.isfinite(sat["S1"])
    & (sat["S1"] > 0)
].copy()


# =========================================================
# CONVERT ELEVATION TO sin(E)
# =========================================================

elevation_rad = np.deg2rad(
    sat["elevation_deg"]
)

sat["sin_elevation"] = np.sin(
    elevation_rad
)


# =========================================================
# SORT BY sin(E)
# =========================================================

sat = sat.sort_values(
    "sin_elevation"
).reset_index(drop=True)


x = sat["sin_elevation"].to_numpy()

snr_db = sat["S1"].to_numpy()


# =========================================================
# CONVERT SNR FROM dB-Hz TO LINEAR REPRESENTATION
# =========================================================

snr_linear = 10 ** (
    snr_db / 20.0
)


# =========================================================
# REMOVE THE SLOWLY VARYING BACKGROUND
# =========================================================

polynomial_degree = 2

coefficients = np.polyfit(
    x,
    snr_linear,
    polynomial_degree
)

trend_linear = np.polyval(
    coefficients,
    x
)

detrended_linear = (
    snr_linear - trend_linear
)


# Remove the mean before Lomb-Scargle analysis

detrended_linear = (
    detrended_linear
    - np.mean(detrended_linear)
)


# =========================================================
# GPS L1 WAVELENGTH
# =========================================================

wavelength = 0.19029367  # metres


# =========================================================
# DEFINE REFLECTOR-HEIGHT SEARCH RANGE
# =========================================================

minimum_height = 0.5  # metres
maximum_height = 5.0  # metres

number_of_heights = 5000

reflector_heights = np.linspace(
    minimum_height,
    maximum_height,
    number_of_heights
)


# =========================================================
# CONVERT REFLECTOR HEIGHT TO SPECTRAL FREQUENCY
# =========================================================
#
# GNSS-IR relation:
#
# f = 2H / lambda
#
# Therefore:
#
# H = f lambda / 2
#
# =========================================================

frequencies = (
    2.0
    * reflector_heights
    / wavelength
)


# Lomb-Scargle requires angular frequency

angular_frequencies = (
    2.0
    * np.pi
    * frequencies
)


# =========================================================
# LOMB-SCARGLE PERIODOGRAM
# =========================================================

power = lombscargle(
    x,
    detrended_linear,
    angular_frequencies,
    normalize=True
)


# =========================================================
# FIND DOMINANT REFLECTOR HEIGHT
# =========================================================

peak_index = np.argmax(power)

dominant_height = reflector_heights[
    peak_index
]

dominant_frequency = frequencies[
    peak_index
]

dominant_power = power[
    peak_index
]


# =========================================================
# PRINT RESULTS
# =========================================================

print("GNSS-IR REFLECTOR-HEIGHT ANALYSIS")
print("=================================")

print(
    "Satellite:",
    satellite_number
)

print(
    "Number of observations:",
    len(sat)
)

print(
    "Elevation range:",
    f"{sat['elevation_deg'].min():.2f}°",
    "to",
    f"{sat['elevation_deg'].max():.2f}°"
)

print(
    "sin(E) range:",
    f"{x.min():.4f}",
    "to",
    f"{x.max():.4f}"
)

print(
    "Wavelength:",
    f"{wavelength:.8f} m"
)

print(
    "Reflector-height search:",
    f"{minimum_height:.2f} m",
    "to",
    f"{maximum_height:.2f} m"
)

print()
print("SPECTRAL RESULT")
print("---------------")

print(
    "Dominant reflector height:",
    f"{dominant_height:.3f} m"
)

print(
    "Corresponding frequency:",
    f"{dominant_frequency:.3f}",
    "cycles per unit sin(E)"
)

print(
    "Maximum Lomb-Scargle power:",
    f"{dominant_power:.4f}"
)


# =========================================================
# PLOT 1 — DETRENDED SIGNAL
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    x,
    detrended_linear,
    ".",
    markersize=3
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "sin(Elevation)"
)

plt.ylabel(
    "Detrended Linear SNR"
)

plt.title(
    "GPS PRN 1 — Detrended SNR"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# =========================================================
# PLOT 2 — SPECTRUM VS REFLECTOR HEIGHT
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    reflector_heights,
    power
)

plt.axvline(
    dominant_height,
    linestyle="--",
    linewidth=1,
    label=(
        f"Peak = {dominant_height:.2f} m"
    )
)

plt.xlabel(
    "Reflector Height (m)"
)

plt.ylabel(
    "Lomb-Scargle Power"
)

plt.title(
    "GPS PRN 1 — GNSS-IR Reflector-Height Spectrum"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()